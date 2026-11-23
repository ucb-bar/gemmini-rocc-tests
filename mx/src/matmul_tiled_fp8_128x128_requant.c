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
#undef GEMMINI_BUSY_ADDR
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


static inline int popcount8(uint8_t x) {
  int count = 0;
  while (x) {
    count += x & 1;
    x >>= 1;
  }
  return count;
}


int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  // ---- Output buffer ----
  static out_t C_hw[MATMUL_M][MATMUL_N/8];
  uint32_t scale_factors[512] __attribute__((aligned(32))) = {0};
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
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 0);

  MX_PERF_MARK(perf.t0);
  // Unified real-RoCC path (Spike AND RTL): funct-27 MX_LOAD_SCALES. The fence orders the async
  // scale DMA on the RTL and is a no-op on Spike, so both emit the identical instruction stream.
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();
  MX_PERF_MARK(perf.t_sc);

  // ---- MVIN A: tile (i,k) -> a_base + (i*tiles_K + k)*DIM ----
  gemmini_config_ld(MATMUL_M * sizeof(elem_t));

  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  MX_PERF_MARK(perf.t_lda);

  // ---- MVIN B: tile (k,j) -> b_base + (j*tiles_K + k)*DIM ----
  for (int j = 0; j < tiles_J; j++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
      uint32_t sp_addr = b_base + (j * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  int SPAD_DEST = a_base + tiles_I * tiles_K * DIM;   // just past A: no C/A overlap, store overlaps compute

  gemmini_config_st(1 * sizeof(out_t));   // V1 spad path: match the passing 64x64 requant
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

  // ---- Compute ----
  MX_PERF_MARK(perf.t_ld);
  mx_perf_ctr_start(&perf);
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base,
      BANK_NUM * BANK_ROWS,
      0,
      SPAD_DEST,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      0x38);
  MX_PERF_MARK(perf.t_ex);
  mx_perf_ctr_stop(&perf);

 gemmini_fence();

//  for (int i = 0; i < tiles_I; i++) {
//    for (int j = 0; j < tiles_J; j++) {
//      uint32_t acc_tile_addr = acc_addr + (i * tiles_J + j) * DIM;
//      out_t *dram_ptr = &C_hw[i * DIM][j * DIM];
//      gemmini_mvout((void *) dram_ptr, acc_tile_addr);
//    }
//  }

  // V1: the requant FP8 output lives in the INTERNAL scratchpad (ex_write_to_spad=true routes the
  // requantizer output to the banks). Read it back with a plain spad->DRAM mvout -- spad reads are
  // full-width (16 FP8/row), byte-aligned, NOT chunked like the acc path.
  // PROVISIONAL for N>64: flat row-major-contiguous mvout from SPAD_DEST (proven at 64x64). If the
  // sim shows a scrambled 128-wide pattern, the spad layout interleaves like the acc path did and
  // needs the loop-store geometry (LoopMatmul dst_addr/chunk) instead of this flat loop.
  gemmini_fence();
  gemmini_config_st(DIM * sizeof(uint8_t));               // one spad row = DIM (16) FP8 bytes
  uint8_t *c_base = (uint8_t *) C_hw;
  int total_spad_rows = MATMUL_M * MATMUL_N / DIM;         // 128*128/16 = 1024 spad rows
  for (int r = 0; r < total_spad_rows; r += DIM) {
    gemmini_extended_mvout(c_base + r * DIM, SPAD_DEST + r, DIM, DIM);  // DIM rows x DIM FP8
  }
  gemmini_fence();
  MX_PERF_MARK(perf.t_st);


//  gemmini_mvout((void*)&C_hw[0][0], 128 )



  // ---- Debug print tile (0,0) ----
//  printf("=== Tile (0,0) - acc_addr=0x%08x ===\n", acc_addr);
//  for (int i = 0; i < DIM; i++) {
//    for (int j = 0; j < DIM; j++) {
//      printf("C_hw[%d][%d] = 0x%016lx\n", i, j, (unsigned long)C_hw[i][j]);
//    }
//  }

  // ---- Elementwise check against C_out_bf16 ----

// ---- Elementwise check against C_out (fp8, byte-by-byte) ----



// ---- Elementwise check against C_out (fp8, byte-by-byte) ----
  int errors = 0;
  int diff1 = 0, diff2 = 0, diff3plus = 0;
  uint8_t *hw_bytes = (uint8_t *)C_hw;

  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < MATMUL_N; j++) {
      uint8_t got = hw_bytes[i * MATMUL_N + j];
      uint8_t exp = C_out[i][j];
      if (got != exp) {
        errors++;
        printf("Output[%d][%d], Got: %x, Exp: %x\n", i, j, got, exp);
        int bits = popcount8(got ^ exp);
        if      (bits == 1) diff1++;
        else if (bits == 2) diff2++;
        else                diff3plus++;
      }
    }
  }

  // ---- Check the E8M0 block scales the requantizer wrote to scale_factors ----
  // The codes alone cannot catch a wrong scale convention: shifting every block scale by a
  // constant leaves the codes untouched. C_scales_out is in the layout the requantizer writes,
  // one byte per row per 32 output columns.
  int scale_errors = 0;
  uint8_t *sf_bytes = (uint8_t *) scale_factors;
  for (int i = 0; i < MATMUL_M; i++) {
    for (int b = 0; b < MATMUL_GN; b++) {
      uint8_t got = sf_bytes[b * MATMUL_M + i];
      uint8_t exp = C_scales_out[i][b];
      if (got != exp) {
        scale_errors++;
        if (scale_errors <= 20)
          printf("Scale[%d][%d], Got: %x, Exp: %x\n", i, b, got, exp);
      }
    }
  }

  if (errors == 0 && scale_errors == 0) {
    printf("fp8 WS requant test PASSED (codes and scales both exact).\n");
  } else {
    printf("fp8 WS requant test FAILED: %d code, %d scale mismatches.\n", errors, scale_errors);
    printf("  differ by 1 bit:   %d\n", diff1);
    printf("  differ by 2 bits:  %d\n", diff2);
    printf("  differ by 3+ bits: %d\n", diff3plus);
  }

  mx_perf_report("fp8_128x128_requant", MATMUL_M, MATMUL_N, MATMUL_K, DIM, &perf);
  mx_perf_report_load("fp8_128x128_requant", (int)(sizeof(A_scales_row) + sizeof(B_scales_col)),
                      MATMUL_M * MATMUL_K, MATMUL_K * MATMUL_N, &perf);

#ifndef BAREMETAL
  exit(errors != 0 || scale_errors != 0);
#else
  return errors != 0 || scale_errors != 0;
#endif
}