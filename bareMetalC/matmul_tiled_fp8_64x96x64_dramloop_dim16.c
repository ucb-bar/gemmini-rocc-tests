// DIM=32 LOOP_WS with A, B, C in DRAM: E4M3 single, BF16 out, stored by LoopMatmulStC (acc -> DRAM). Port of
// matmul_tiled_fp8_128x128_dramloop.c at 64x96 (N=96: partial j-group); exercises the shared MxStoreIter on the DRAM store path.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x96x64.h"

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: standalone/Spike-only test\n"); return 0; }
#else

#define DIM 16

typedef uint8_t elem_t;

static uint16_t C_hw[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));
static uint32_t scale_sink[512] __attribute__((aligned(32)));

int main() {
  memset(C_hw, 0, sizeof(C_hw));
  const int I = MATMUL_M / DIM, J = MATMUL_N / DIM, K = MATMUL_K / DIM;

  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
  gemmini_mx_load_scales((uint64_t) &A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t) &B_scales_col, sizeof(B_scales_col), 1);

  gemmini_extended3_config_ld(MATMUL_K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(MATMUL_N * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(MATMUL_N * sizeof(uint16_t));
  gemmini_mxquant_config_mvout((uint64_t) scale_sink, I, J, K, 0, 0, 1);

  gemmini_loop_ws(I, J, K, 0, 0, 0,
                  A_in, B_in, NULL, C_hw,
                  MATMUL_K, MATMUL_N, 0, MATMUL_N,
                  false, false, false, false, false, NO_ACTIVATION,
                  1, 1, false);
  gemmini_fence();

  int errors = 0;
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < MATMUL_N; j++)
      if (C_hw[i][j] != C_out_bf16[i][j]) {
        if (errors < 16) printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n", i, j, C_hw[i][j], C_out_bf16[i][j]);
        errors++;
      }
  if (errors == 0) printf("fp8 DIM16 N96 native-loop DRAM-out test PASSED (no mismatches).\n");
  else printf("fp8 DIM16 N96 native-loop DRAM-out test FAILED with %d mismatches.\n", errors);
  return errors != 0;
}
#endif
