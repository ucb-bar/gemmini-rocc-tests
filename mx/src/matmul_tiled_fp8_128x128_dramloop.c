// Native-loop MX bring-up: matmul_tiled_fp8_128x128 (e4m3 single, BF16 out) as ONE gemmini_loop_ws with
// A, B, C in DRAM -- LoopMatmul's own ldA/ldB mvins and stC (acc -> DRAM) store, no manual mvin/mvout.
// Scales are loaded up front (fenced). Checked against the header golden (Spike: mx_loop_ws_dram).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_128x128.h"
#include "include/mx_perf.h"


#define DIM 16

typedef uint8_t elem_t;

static uint16_t C_hw[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));
static uint32_t scale_sink[512] __attribute__((aligned(32)));

int main() {
  memset(C_hw, 0, sizeof(C_hw));
  const int I = MATMUL_M / DIM, J = MATMUL_N / DIM, K = MATMUL_K / DIM;
  mx_perf_t perf = {0};

  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

  MX_PERF_MARK(perf.t0);
  gemmini_mx_load_scales((uint64_t) &A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t) &B_scales_col, sizeof(B_scales_col), 1);
  MX_PERF_MARK(perf.t_ld);

  // A[M][K] via mvin (id 0), B[K][N] via mvin2 (id 1): row pitch in bytes. C: BF16 row pitch.
  gemmini_extended3_config_ld(MATMUL_K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(MATMUL_N * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(MATMUL_N * sizeof(uint16_t));
  gemmini_mxquant_config_mvout((uint64_t) scale_sink, I, J, K, 0, 0, 1);

  mx_perf_ctr_start(&perf);
  gemmini_loop_ws(I, J, K, 0, 0, 0,
                  A_in, B_in, NULL, C_hw,
                  MATMUL_K, MATMUL_N, 0, MATMUL_N,
                  false, false, false, false, false, NO_ACTIVATION,
                  1, 1, false);
  MX_PERF_MARK(perf.t_ex);
  mx_perf_ctr_stop(&perf);
  perf.t_st = perf.t_ex;   // no separate mvout phase: the loop stores C itself

  int errors = 0, first_bad_row = -1;
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < MATMUL_N; j++)
      if (C_hw[i][j] != C_out_bf16[i][j]) {
        if (errors < 16)
          printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n", i, j, C_hw[i][j], C_out_bf16[i][j]);
        if (first_bad_row < 0) first_bad_row = i;
        errors++;
      }
  // Where did row-tile 0's data land? Helps localize a store-layout (dram_offset) bug.
  if (errors) {
    int found = -1;
    for (int r = 0; r < MATMUL_M && found < 0; r++) {
      int same = 1;
      for (int j = 0; j < MATMUL_N && same; j++) same = C_hw[r][j] == C_out_bf16[1][j];
      if (same) found = r;
    }
    printf("golden row 1 found at HW row %d (first bad row %d)\n", found, first_bad_row);
  }

  if (errors == 0)
    printf("fp8 WS native-loop matmul test PASSED (no mismatches).\n");
  else
    printf("fp8 WS native-loop matmul test FAILED with %d mismatches.\n", errors);

  mx_perf_report("fp8_128x128_dramloop", MATMUL_M, MATMUL_N, MATMUL_K, DIM, &perf);
  return errors != 0;
}
