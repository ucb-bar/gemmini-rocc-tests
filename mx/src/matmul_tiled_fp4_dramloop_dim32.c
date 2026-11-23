// FP4 x FP4 quad matmul on the 32x32 mesh through one native DRAM loop (A, B, C in DRAM, loop-managed scales), BF16
// out: each 64-column quad C row is 128 B, drained by the writer as two 64-B Puts. Built through the npu-exploration
// wrapper with the DIM=32 params header (-iquote dim32/vpu). Golden: gen_matmul_llama.py (C_out_bf16).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#ifndef FP4_HDR
#define FP4_HDR "include/matmul_fp4_64x64_dim32.h"
#endif
#include FP4_HDR

#if DIM != 32
#error "needs the DIM=32 params header (-iquote dim32/vpu)"
#endif


static uint16_t C[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));

int main() {
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 2, 2, 3, 0);
  gemmini_mxquant_config_mvout(0, 1, 1, 1, 0, 0, 0);
  gemmini_fence();
  gemmini_extended3_config_ld(MATMUL_K, MVIN_SCALE_IDENTITY, false, 0);       // A [M/2][K] row pairs
  gemmini_extended3_config_ld(MATMUL_N / 2, MVIN_SCALE_IDENTITY, false, 1);   // B [K][N/2]
  gemmini_config_st(2 * MATMUL_N * sizeof(uint16_t));                         // a quad acc row = 2 C rows

  uint64_t t0 = read_cycles();
  gemmini_loop_ws_mx(MATMUL_M / (2 * DIM), MATMUL_N / (2 * DIM), MATMUL_K / DIM,
                     (uint8_t *)A_in_hw, (uint8_t *)B_in, C, MATMUL_K, MATMUL_N / 2, MATMUL_N,
                     &A_scales_row[0][0], &B_scales_col[0][0], MATMUL_M, MATMUL_N, false, 1, 1);
  gemmini_fence();
  uint64_t t1 = read_cycles();

  int bad = 0;
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < MATMUL_N; j++)
      if (C[i][j] != C_out_bf16[i][j] && bad++ < 20)
        printf("C[%d][%d] got %04x exp %04x\n", i, j, C[i][j], C_out_bf16[i][j]);
  printf("fp4 dramloop %dx%dx%d dim32: %d mismatches; %llu cycles\n", MATMUL_M, MATMUL_N, MATMUL_K, bad,
         (unsigned long long)(t1 - t0));
  printf(bad ? "fp4 dramloop dim32 test FAILED\n" : "fp4 dramloop dim32 test PASSED\n");
  return bad != 0;
}
