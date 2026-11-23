// Perf experiment: matmul_tiled_fp8_128x128 (e4m3 single, BF16 out) split into NCHUNKS loop_ws_spad calls
// along I so chunk c's C store can overlap chunk c+1's compute.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_128x128.h"
#include "include/mx_perf.h"


#define DIM 16

// Scale positions are loop-local (MxScalePos): each chunk loads its own scales and issues its own CONFIG_SCALE_MEM
// (a barrier: non-loop cmds wait for prior loops), so chunks no longer overlap their stores with the next compute.
#ifndef NCHUNKS
#define NCHUNKS 2
#endif
#define CHUNK_STR_(x) #x
#define CHUNK_STR(x) CHUNK_STR_(x)

#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_M / BF16_PER_WORD)

typedef uint8_t  elem_t;
typedef uint64_t out_t;

#define CHUNK_M (MATMUL_M / NCHUNKS)

static uint8_t A_scales_chunked[NCHUNKS][MATMUL_GK][CHUNK_M] __attribute__((aligned(64)));
static uint8_t B_scales_rep[NCHUNKS][MATMUL_GK][MATMUL_N] __attribute__((aligned(64)));

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  static out_t C_hw[MATMUL_M][OUT_COLS];
  uint32_t scale_factors[512] = {0};
  memset(C_hw, 0, sizeof(C_hw));

  int tiles_I = MATMUL_M / DIM;
  int tiles_J = MATMUL_N / DIM;
  int tiles_K = MATMUL_K / DIM;
  int chunk_I = tiles_I / NCHUNKS;

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;

  // A: chunk c's rows as [GK][CHUNK_M] (loop_bound_i = chunk_I); B: same weights every chunk, repeated
  for (int c = 0; c < NCHUNKS; c++)
    for (int kb = 0; kb < MATMUL_GK; kb++) {
      memcpy(A_scales_chunked[c][kb], &A_scales_row[kb][c * CHUNK_M], CHUNK_M);
      memcpy(B_scales_rep[c][kb], B_scales_col[kb], MATMUL_N);
    }

  mx_perf_t perf = {0};
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

  MX_PERF_MARK(perf.t0);

  gemmini_config_ld(MATMUL_M * sizeof(elem_t));
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
      gemmini_extended_mvin((void *) dram_ptr, a_base + (i * tiles_K + k) * DIM, DIM, DIM);
    }
  }
  for (int j = 0; j < tiles_J; j++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
      gemmini_extended_mvin((void *) dram_ptr, b_base + (j * tiles_K + k) * DIM, DIM, DIM);
    }
  }

  int SPAD_DEST = a_base + tiles_I * tiles_K * DIM;   // just past A
  gemmini_config_st(OUT_COLS * sizeof(out_t));

  // ---- Compute: one loop per I-chunk; C of chunk c lands where the single loop would put rows c*CHUNK_M.. ----
  MX_PERF_MARK(perf.t_ld);
  mx_perf_ctr_start(&perf);
  for (int c = 0; c < NCHUNKS; c++) {
    // chunk c's scales: A = its CHUNK_M rows, B = all of N; then its own config
#ifdef SCALES_2D
    gemmini_mx_load_scales_2d(&A_scales_row[0][c * CHUNK_M], CHUNK_M, MATMUL_GK, MATMUL_M, 0, 0);
    gemmini_mx_load_scales_2d(B_scales_col, sizeof(B_scales_col), 1, 0, 0, 1);
#else
    gemmini_mx_load_scales((uint64_t)A_scales_chunked[c], sizeof(A_scales_chunked[c]), 0);
    gemmini_mx_load_scales((uint64_t)B_scales_rep[c], sizeof(B_scales_rep[c]), 1);
#endif
    gemmini_fence();
    gemmini_mxquant_config_mvout((uint64_t)scale_factors, chunk_I, tiles_J, tiles_K, 0, 0, 1);
    gemmini_loop_ws_spad(
        chunk_I, tiles_J, tiles_K,
        0, 0, 0,
        a_base + c * chunk_I * tiles_K * DIM,
        BANK_NUM * BANK_ROWS,
        0,
        SPAD_DEST + c * CHUNK_M * MATMUL_N * 2 / DIM,
        false, false,
        false, false, false,
        NO_ACTIVATION,
        0, 0,
        false,
        0x38 | 0x100);   // bit8 inc_acc_addr: alternate acc halves so chunk c+1 doesn't overwrite chunk c's C mid-store
    gemmini_fence();
  }
  MX_PERF_MARK(perf.t_ex);
  mx_perf_ctr_stop(&perf);

  // ---- Flat spad->DRAM readback (BF16, row-major) ----
  gemmini_config_st(DIM * sizeof(uint8_t));
  uint8_t *c_base = (uint8_t *) C_hw;
  int total_spad_rows = MATMUL_M * MATMUL_N * 2 / DIM;
  for (int r = 0; r < total_spad_rows; r += DIM)
    gemmini_extended_mvout(c_base + r * DIM, SPAD_DEST + r, DIM, DIM);
  MX_PERF_MARK(perf.t_st);

  int errors = 0;
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t got = C_hw[i][j];
      for (int lane = 0; lane < BF16_PER_WORD; lane++) {
        uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
        uint16_t exp_bf16 = C_out_bf16[i][j * BF16_PER_WORD + lane];
        if (got_bf16 != exp_bf16) {
          if (errors < 20)
            printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n", i, j * BF16_PER_WORD + lane, got_bf16, exp_bf16);
          errors++;
        }
      }
    }
  }

  if (errors == 0)
    printf("fp8 WS chunked matmul test PASSED (no mismatches).\n");
  else
    printf("fp8 WS chunked matmul test FAILED with %d mismatches.\n", errors);

  mx_perf_report("fp8_128x128_chunk" CHUNK_STR(NCHUNKS), MATMUL_M, MATMUL_N, MATMUL_K, DIM, &perf);

#ifndef BAREMETAL
  exit(errors != 0);
#else
  return errors != 0;
#endif
}
