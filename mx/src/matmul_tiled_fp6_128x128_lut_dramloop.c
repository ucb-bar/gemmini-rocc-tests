// N4: FP6 E3M2 LUT x LUT 128x128x128 on the NATIVE DRAM loop (A/B/scales/C in DRAM, BF16 out), tiled like
// mx_native.h mxn_matmul_lut: N-chunks of NC columns x K-tiles of KT. Act codebooks loaded flat once; weight codebooks
// are loop-managed (LOOP_WS_CONFIG_LUTS rs2): chunk c's loops load its NC >> G codebooks into their slot's weight-LUT
// half. G is set once by CONFIG_SCALE_MEM and carried by every loop's in-stream config. Checked vs C_out_bf16.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/matmul_data_mx_lut_e3m2_128x128.h"
#include "include/mx_perf.h"

#define DIM 16
#define G 1
#ifndef NC
#define NC 64                            // (NC >> G) = 32 codebooks = one weight-LUT half
#endif
#ifndef KT
#define KT 64
#endif
#define LUT_BYTES 12                     // FP6 codebook: 16 x 6 bit


static uint32_t A_lut_al[64][3] __attribute__((aligned(64)));
static uint32_t B_lut_al[64][3] __attribute__((aligned(64)));
static uint8_t A_sc_al[MATMUL_GK][MATMUL_M] __attribute__((aligned(64)));
static uint8_t B_sc_al[MATMUL_GK][MATMUL_N] __attribute__((aligned(64)));
static uint16_t C[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));

int main() {
  printf("fp6 e3m2 LUT native DRAM loop: M=N=K=%d, NC=%d, KT=%d (%d loops)\n", MATMUL_M, NC, KT,
         (MATMUL_N / NC) * (MATMUL_K / KT));
  memcpy(A_lut_al, A_lut, sizeof(A_lut_al));
  memcpy(B_lut_al, B_lut, sizeof(B_lut_al));
  memcpy(A_sc_al, A_scales_row, sizeof(A_sc_al));
  memcpy(B_sc_al, B_scales_col, sizeof(B_sc_al));
  memset(C, 0, sizeof(C));

  gemmini_flush(0);
  // fp6 (code 1) E3M2 (altfmt 0) via LUT on A and B, BF16 out
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16) | (3 << 14) | (1 << 12) | (1 << 10) | (1 << 4) | ((WEIGHT_STATIONARY) << 2) | CONFIG_EX,
    ((uint64_t)(1) << 48), k_CONFIG);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 1, 1, 3, 1);
  gemmini_mx_load_lut_dt((uint64_t)&A_lut_al[0][0], MATMUL_M >> G, 1, 6);   // act codebooks: flat, shared by all loops
  gemmini_mxquant_config_mvout(0, 1, 1, 1, 0, 0, G);                        // the matmul's G (loops carry it)
  gemmini_fence();

  gemmini_extended3_config_ld(MATMUL_K, MVIN_SCALE_IDENTITY, false, 0);       // A [M/2][K] row pairs
  gemmini_extended3_config_ld(MATMUL_N / 2, MVIN_SCALE_IDENTITY, false, 1);   // B [K][N/2]
  gemmini_config_st(2 * MATMUL_N * sizeof(uint16_t));                         // a quad acc row holds 2 C rows

  mx_perf_t p = {0};
  mx_perf_ctr_start(&p);
  MX_PERF_MARK(p.t0);
  p.t_ld = p.t0;
  int loops = 0;
  const int kts = MATMUL_K / KT;
  for (int c = 0; c < MATMUL_N / NC; c++)
    for (int t = 0; t < kts; t++) {
      const int h = 1 + (loops++ & 1);
      gemmini_loop_ws_mx_lut(MATMUL_M / (2 * DIM), NC / (2 * DIM), KT / DIM,
                             (uint8_t *) A_in_hw + (size_t) t * KT,
                             (uint8_t *) B_in + (size_t) t * KT * (MATMUL_N / 2) + (size_t) c * NC / 2,
                             t == kts - 1 ? &C[0][c * NC] : NULL,
                             MATMUL_K, MATMUL_N / 2, MATMUL_N,
                             &A_sc_al[t * KT / 32][0], &B_sc_al[t * KT / 32][c * NC], MATMUL_M, MATMUL_N,
                             0, (uint8_t *) &B_lut_al[0][0] + (size_t) ((c * NC) >> G) * LUT_BYTES,
                             /*accumulate*/ t > 0, h, h);
    }
  MX_PERF_MARK(p.t_ex);
  mx_perf_ctr_stop(&p);
  p.t_st = p.t_ex;

  int bad = 0;
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < MATMUL_N; j++)
      if (C[i][j] != C_out_bf16[i][j] && bad++ < 10)
        printf("C[%d][%d] got %04x exp %04x\n", i, j, C[i][j], C_out_bf16[i][j]);
  mx_perf_report_ideal("fp6_e3m2_lut_dramloop", MATMUL_M, MATMUL_N, MATMUL_K, DIM,
                       MX_PERF_IDEAL_QUAD(MATMUL_M, MATMUL_N, MATMUL_K, DIM), &p);
  printf("%d bad of %d\n", bad, MATMUL_M * MATMUL_N);
  printf("fp6 lut dramloop test %s\n", bad ? "FAILED" : "PASSED");
  return bad != 0;
}
