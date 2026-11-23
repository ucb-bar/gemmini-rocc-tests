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

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define SMEM 0x40000000

#define DIM 16

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)

#define ADDR_LEN 32

// BF16 values packed 4 per uint64_t output word
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_M / BF16_PER_WORD)

typedef uint8_t  elem_t;   // fp8:e4m3 input elements
typedef uint8_t  welem_t;  // fp8:e4m3 weight elements
typedef uint64_t out_t;    // 4x bf16 packed per word

// ---- Scale factor loader ----
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int INDIM, int K) {
  for (size_t k = 0; k < K/32; k++) {
    for (size_t i = 0; i < INDIM / 8; i++) {
//        printf("loading: %lx\n", ((uint64_t*) scale_factors)[k * INDIM/8 + i]);
        sf_mem[k*INDIM/8 + i] = ((uint64_t*) scale_factors)[k * INDIM/8 + i];
    }
  }
}

// Spin until Gemmini reports not busy
static inline void gemmini_poll_until_ready() {
    volatile uint32_t *status = (volatile uint32_t*)(GEMMINI_BUSY_ADDR); // check your offset
    while (*status & 0x1) {
        // busy — keep polling
    }
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  // ---- Output buffer ----
  static out_t C_hw[MATMUL_M][OUT_COLS];
  uint32_t scale_factors[512] = {0};
  memset(C_hw, 0, sizeof(C_hw));

  // ---- Tile dimensions ----
  int tiles_I = MATMUL_M / DIM;
  int tiles_J = MATMUL_N / DIM;
  int tiles_K = MATMUL_K / DIM;

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  uint32_t acc_addr = (1u << (ADDR_LEN - 1));

  // ---- Gemmini setup ----
  mx_perf_t perf = {0};
  // L2 experiment: CPU touches every 64B line of A_in so mvin A hits in L2 (B stays cold) -> A vs B
  // isolates the L2 miss (MSHR) limit. -DWARM_A=0 disables.
#ifndef WARM_A
#define WARM_A 1
#endif
#if WARM_A
  {
    volatile const uint8_t *pa = (volatile const uint8_t *) A_in;
    uint32_t sink = 0;
    for (size_t off = 0; off < sizeof(A_in); off += 64)
      sink += pa[off];
    if (sink == 0xFFFFFFFFu) printf("");
  }
#endif
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

  MX_PERF_MARK(perf.t0);
  // Unified real-RoCC path (Spike AND RTL): funct-27 MX_LOAD_SCALES. The fence orders the async
  // scale DMA on the RTL and is a no-op on Spike, so both emit the identical instruction stream.
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();
  MX_PERF_MARK(perf.t_sc);
  mx_perf_ld_ctr_start(&perf);

  // ---- MVIN A: tile (i,k) -> a_base + (i*tiles_K + k)*DIM ----
  gemmini_config_ld(MATMUL_M * sizeof(elem_t));

#ifndef MVIN_A_TILES
#define MVIN_A_TILES 4   // k-tiles per mvin (64B rows at 4; block stride DIM keeps the layout)
#endif
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k += MVIN_A_TILES) {
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM * MVIN_A_TILES, DIM);
    }
  }

  MX_PERF_MARK(perf.t_lda);
  mx_perf_ld_ctr_read(perf.ld_a);

  // ---- MVIN B: tile (k,j) -> b_base + (j*tiles_K + k)*DIM ----
#ifndef MVIN_B_TILES
#define MVIN_B_TILES 4
#endif
  for (int j = 0; j < tiles_J; j++) {
    for (int k = 0; k < tiles_K; k += MVIN_B_TILES) {
      elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
      uint32_t sp_addr = b_base + (j * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM * MVIN_B_TILES, DIM);
    }
  }

  int SPAD_DEST = a_base + tiles_I * tiles_K * DIM;   // just past A: no C/A overlap, store overlaps compute

  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

  // Rocket has ex_write_to_spad=false, so the BF16 mesh output never reaches the
  // scratchpad; it lives in the accumulator. Compute into the accumulator and mvout from there.
  // flag 0xb8 skips the spad store; 0x38 keeps it for radiance/smem.
  // Spad path (matches Spike + radiance + the non-requant fp4/fp6): F2c full-width BF16 store to the
  // internal scratchpad (0x38), then a flat spad->DRAM mvout. Unified across all targets.
  uint32_t out_dest = SPAD_DEST;
  uint32_t out_flag = 0x38;

  // ---- Compute ----
  MX_PERF_MARK(perf.t_ld);
  mx_perf_ld_ctr_read(perf.ld_b);
  mx_perf_ctr_start(&perf);
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base,
      BANK_NUM * BANK_ROWS,
      0,
      out_dest,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      out_flag);
  MX_PERF_MARK(perf.t_ex);
  mx_perf_ctr_stop(&perf);

//  for (int i = 0; i < tiles_I; i++) {
//    for (int j = 0; j < tiles_J; j++) {
//      uint32_t acc_tile_addr = acc_addr + (i * tiles_J + j) * DIM;
//      out_t *dram_ptr = &C_hw[i * DIM][j * DIM];
//      gemmini_mvout((void *) dram_ptr, acc_tile_addr);
//    }
//  }


//  gemmini_mvout((void*)&C_hw[0][0], 128 )

  gemmini_fence();
  gemmini_fence();

  // Non-requant BF16 in internal spad (F2c full-width store, row-major). Flat spad->DRAM readback:
  // BF16 = 2 bytes/elem -> M*N*2/DIM spad rows. Identical instruction stream on Spike and RTL.
  gemmini_fence();
  gemmini_config_st(DIM * sizeof(uint8_t));
  uint8_t *c_base = (uint8_t *) C_hw;
  int total_spad_rows = MATMUL_M * MATMUL_N * 2 / DIM;
  for (int r = 0; r < total_spad_rows; r += DIM) {
    gemmini_extended_mvout(c_base + r * DIM, SPAD_DEST + r, DIM, DIM);
  }
  gemmini_fence();
  MX_PERF_MARK(perf.t_st);

  // ---- Debug print tile (0,0) ----
//  printf("=== Tile (0,0) - acc_addr=0x%08x ===\n", acc_addr);
//  for (int i = 0; i < DIM; i++) {
//    for (int j = 0; j < DIM; j++) {
//      printf("C_hw[%d][%d] = 0x%016lx\n", i, j, (unsigned long)C_hw[i][j]);
//    }
//  }

  // ---- Elementwise check against C_out_bf16 ----
  int errors = 0;

  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t got = C_hw[i][j];

      // Pack 4 consecutive bf16 golden values into expected uint64_t
      uint64_t exp = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                     ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                     ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                     ((uint64_t)C_out_bf16[i][j*4 + 0]);

      if (got != exp) {
        for (int lane = 0; lane < BF16_PER_WORD; lane++) {
          uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
          uint16_t exp_bf16 = C_out_bf16[i][j * BF16_PER_WORD + lane];
          if (got_bf16 != exp_bf16) {
            printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n",
                   i, j * BF16_PER_WORD + lane, got_bf16, exp_bf16);
            errors++;
          }
        }
      }
    }
  }

  if (errors == 0) {
    printf("fp8 WS matmul test PASSED (no mismatches).\n");
  } else {
    printf("fp8 WS matmul test FAILED with %d mismatches.\n", errors);
  }

  mx_perf_report("fp8_128x128", MATMUL_M, MATMUL_N, MATMUL_K, DIM, &perf);
  mx_perf_report_load("fp8_128x128", (int)(sizeof(A_scales_row) + sizeof(B_scales_col)),
                      MATMUL_M * MATMUL_K, MATMUL_K * MATMUL_N, &perf);

#ifndef BAREMETAL
  exit(errors != 0);
#else
  return errors != 0;
#endif
}