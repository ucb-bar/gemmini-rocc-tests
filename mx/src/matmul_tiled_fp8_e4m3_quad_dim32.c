// Native 8-bit E4M3 quad (mode9) matmul, BF16 out, on MxDim32E4M3QuadGemminiRocketConfig (branch large-mesh-fp8): a
// 32x32 mesh doing a 64x64 output tile per K=32 pass. One native DRAM loop (A, B, C in DRAM, loop-managed scales).
// Built only through npu-exploration wrappers with the DIM=32 quad params header (elem_t = one 16-bit lane = two
// E4M3 codes, 64-B spad rows). Golden: gen_e4m3_quad_native.py (QUAD_HDR, default 64x64).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#ifndef QUAD_HDR
#define QUAD_HDR "include/matmul_data_e4m3_quad_64x64_dim32.h"
#endif
#include QUAD_HDR

#if DIM != 32
#error "needs the DIM=32 E4M3 quad params header (-iquote dim32/e4m3_quad)"
#endif


static uint16_t C[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));

int main() {
  gemmini_flush(0);
  // A / B format 0 (E4M3), altfmt 0, no LUT (native quad build: E4M3 is always mode9), C format 3 (BF16)
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16) | (3 << 14) | ((WEIGHT_STATIONARY) << 2) | CONFIG_EX,
    ((uint64_t)(1) << 48), k_CONFIG);
  gemmini_mxquant_config_mvout(0, 1, 1, 1, 0, 0, 0);
  gemmini_fence();

  gemmini_extended3_config_ld(MATMUL_K * sizeof(uint16_t), MVIN_SCALE_IDENTITY, false, 0);       // A [M/2][K] lanes
  gemmini_extended3_config_ld((MATMUL_N / 2) * sizeof(uint16_t), MVIN_SCALE_IDENTITY, false, 1); // B [K][N/2] lanes
  gemmini_config_st(2 * MATMUL_N * sizeof(uint16_t));                                           // a quad acc row = 2 C rows

  uint64_t t0 = read_cycles();
  gemmini_loop_ws_mx(MATMUL_M / (2 * DIM), MATMUL_N / (2 * DIM), MATMUL_K / DIM,
                     A_hw, B_hw, C, MATMUL_K, MATMUL_N / 2, MATMUL_N,
                     &A_sc[0][0], &B_sc[0][0], MATMUL_M, MATMUL_N, false, 1, 1);
  gemmini_fence();
  uint64_t t1 = read_cycles();

  int bad = 0;
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < MATMUL_N; j++)
      if (C[i][j] != C_out_bf16[i][j] && bad++ < 20)
        printf("C[%d][%d] got %04x exp %04x\n", i, j, C[i][j], C_out_bf16[i][j]);
  printf("e4m3 quad %dx%dx%d dim32: %d mismatches; %llu cycles\n", MATMUL_M, MATMUL_N, MATMUL_K, bad,
         (unsigned long long)(t1 - t0));
  printf(bad ? "e4m3 quad test FAILED\n" : "e4m3 quad test PASSED\n");
  return bad != 0;
}
