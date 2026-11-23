// A8W4 (E4M3-single act x FP4 E2M1 quad wei, mode6) through native DRAM loops with loop-managed scales.
// C = A @ B as NCH N-chunks x KT K-tiles of gemmini_loop_ws_mx (split-K accumulate, C on the last K-tile),
// spad/acc halves alternating -- the mxn_matmul schedule. Defaults: one loop for the whole matmul.
//   -DA8W4_HDR='"include/matmul_data_asym_e4m3s_fp4_32x256x512_dim8.h"' -DA8W4_NC=128 -DA8W4_KT=128
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#ifndef A8W4_HDR
#define A8W4_HDR "include/matmul_data_asym_e4m3s_fp4.h"
#endif
#include A8W4_HDR
#ifndef A8W4_NC
#define A8W4_NC MATMUL_N
#endif
#ifndef A8W4_KT
#define A8W4_KT MATMUL_K
#endif


static uint16_t C[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));

int main() {
  gemmini_flush(0);
  // A fp8 code0 + no LUT = E4M3 single; B fp4 (code2) quad; out BF16
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 2, 3, 0);
  gemmini_mxquant_config_mvout(0, 1, 1, 1, 0, 0, 0);
  gemmini_fence();
  gemmini_extended3_config_ld(MATMUL_K, MVIN_SCALE_IDENTITY, false, 0);       // A [M][K]
  gemmini_extended3_config_ld(MATMUL_N / 2, MVIN_SCALE_IDENTITY, false, 1);   // B [K][N/2] nibbles
  gemmini_config_st(MATMUL_N * sizeof(uint16_t));                             // C [M][N] BF16

  const int NC = A8W4_NC, KT = A8W4_KT, nk = MATMUL_K / KT;
  int loops = 0;
  uint64_t t0 = read_cycles();
  for (int c = 0; c < MATMUL_N / NC; c++)
    for (int t = 0; t < nk; t++, loops++) {
      int h = 1 + (loops & 1);
      gemmini_loop_ws_mx(MATMUL_M / DIM, NC / (2 * DIM), KT / DIM,
                         (uint8_t *) A_in + t * KT, (uint8_t *) B_in + t * KT * (MATMUL_N / 2) + c * NC / 2,
                         t == nk - 1 ? (void *) &C[0][c * NC] : NULL,
                         MATMUL_K, MATMUL_N / 2, MATMUL_N,
                         &A_scales_row[t * KT / 32][0], &B_scales_col[t * KT / 32][c * NC], MATMUL_M, MATMUL_N,
                         t > 0, h, h);
    }
  gemmini_fence();
  uint64_t t1 = read_cycles();

  int bad = 0;
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < MATMUL_N; j++)
      if (C[i][j] != C_out_bf16[i][j] && bad++ < 20)
        printf("C[%d][%d] got %04x exp %04x\n", i, j, C[i][j], C_out_bf16[i][j]);
  // ideal: M*N*K MACs at 2 per PE per cycle
  unsigned long long ideal = (unsigned long long) MATMUL_M * MATMUL_N * MATMUL_K / (2 * DIM * DIM);
  printf("a8w4 dramloop %dx%dx%d dim%d NC=%d KT=%d (%d loops): %d mismatches; %llu cycles (ideal %llu)\n",
         MATMUL_M, MATMUL_N, MATMUL_K, DIM, NC, KT, loops, bad, (unsigned long long) (t1 - t0), ideal);
  printf(bad ? "a8w4 dramloop test FAILED\n" : "a8w4 dramloop test PASSED\n");
  return bad != 0;
}
