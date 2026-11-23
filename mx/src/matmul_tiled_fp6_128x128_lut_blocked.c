// S3a: FP6 E3M2 LUT 128x128x128 (BF16 out) tiled into NBLK row blocks with loop-managed metadata: each block's
// loop loads its own act-in codebooks (LOOP_WS_CONFIG_LUTS, LUT half = loop slot) and scale slices
// (LOOP_WS_CONFIG_SCALES, scale half = loop slot). Weight codebooks are shared by all blocks (one flat MX_LOAD_LUT).
//   run 1: one loop, flat codebooks + software scale loads (today's path, reference)
//   run 2: NBLK blocks on alternating acc halves
// Both checked bit-exact against C_out_bf16 and timed (quad ideal).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/matmul_data_mx_lut_e3m2_128x128.h"
#include "include/mx_perf.h"

#ifndef NBLK
#define NBLK 4
#endif
#define BLK_STR_(x) #x
#define BLK_STR(x) BLK_STR_(x)

#define DIM 16
#define G 1                                 // LUT update granularity (managed scales run with G = 1)
#define LOOP_WS_INC_ACC_ADDR (1u << 8)


// the scale and LUT loaders need 8 B-aligned DRAM addresses
static uint32_t A_lut_al[64][3] __attribute__((aligned(64)));
static uint32_t B_lut_al[64][3] __attribute__((aligned(64)));
static uint8_t A_sc_al[MATMUL_GK][MATMUL_M] __attribute__((aligned(64)));
static uint8_t B_sc_al[MATMUL_GK][MATMUL_N] __attribute__((aligned(64)));
static uint16_t C_hw[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));
static uint32_t scale_factors[MATMUL_M * MATMUL_N / 32] __attribute__((aligned(64)));

// BF16 C lands row-major in the scratchpad from spad row `base` (one C row = N*2/16 rows)
static int count_bad(uint32_t base, const char *tag) {
  gemmini_config_st(DIM * sizeof(uint8_t));
  for (int r = 0; r < MATMUL_M * MATMUL_N * 2 / DIM; r += DIM)
    gemmini_extended_mvout((uint8_t *) C_hw + r * DIM, base + r, DIM, DIM);
  gemmini_fence();
  int bad = 0;
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < MATMUL_N; j++)
      if (C_hw[i][j] != C_out_bf16[i][j] && bad++ < 10)
        printf("%s C[%d][%d] got %04x exp %04x\n", tag, i, j, C_hw[i][j], C_out_bf16[i][j]);
  return bad;
}

int main() {
  const int tiles_I = MATMUL_M / 32, tiles_J = MATMUL_N / 32, tiles_K = MATMUL_K / 16;
  const int blk_I = tiles_I / NBLK, blk_rows = blk_I * 32;
  const uint32_t a_base = 0, b_end = 8192, b_base = b_end - tiles_K * tiles_J * K_TILE;
  const uint32_t C_ONE = 8192, C_BLK = 12288;            // banks 2 / 3: clear of A (0..511) and B (7680..8191)
  const int c_rows_per_row = MATMUL_N * 2 / DIM;        // scratchpad rows per BF16 C row
  printf("fp6 e3m2 LUT blocked: M=N=K=%d, NBLK=%d (%d rows per block)\n", MATMUL_M, NBLK, blk_rows);
  if (tiles_I % NBLK) { printf("NBLK must divide %d\n", tiles_I); return 1; }

  memcpy(A_lut_al, A_lut, sizeof(A_lut_al));
  memcpy(B_lut_al, B_lut, sizeof(B_lut_al));
  memcpy(A_sc_al, A_scales_row, sizeof(A_sc_al));
  memcpy(B_sc_al, B_scales_col, sizeof(B_sc_al));

  gemmini_flush(0);
  // fp6 (code 1) E3M2 (altfmt 0) via LUT on A and B, BF16 out
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16) | (3 << 14) | (1 << 12) | (1 << 10) | (1 << 4) | ((WEIGHT_STATIONARY) << 2) | CONFIG_EX,
    ((uint64_t)(1) << 48), k_CONFIG);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 1, 1, 3, 1);
  gemmini_extended_config_st(DIM * sizeof(uint8_t), NO_ACTIVATION, 1);
  gemmini_mx_load_lut_dt((uint64_t)&B_lut_al[0][0], MATMUL_N >> G, 0, 6);   // weight codebooks: shared, flat

  gemmini_config_ld(MATMUL_K * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void *) ((uint8_t *) A_in_hw + i * DIM * MATMUL_K + k * DIM), a_base + (i * tiles_K + k) * DIM, DIM, DIM);
  gemmini_config_ld((MATMUL_N / 2) * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++)
      gemmini_extended_mvin((void *) ((uint8_t *) B_in + k * K_TILE * (MATMUL_N / 2) + j * DIM), b_base + (k * tiles_J + j) * K_TILE, DIM, DIM);
  gemmini_fence();

  // ---- run 1: one loop, flat act-in codebooks, software scale loads ----
  gemmini_mx_load_lut_dt((uint64_t)&A_lut_al[0][0], MATMUL_M >> G, 1, 6);
  gemmini_mx_load_scales((uint64_t)A_sc_al, sizeof(A_sc_al), 0);
  gemmini_mx_load_scales((uint64_t)B_sc_al, sizeof(B_sc_al), 1);
  gemmini_fence();
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, G);
  mx_perf_t p1 = {0};
  mx_perf_ctr_start(&p1);
  MX_PERF_MARK(p1.t0);
  p1.t_ld = p1.t0;
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, b_end, 0, C_ONE,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
  MX_PERF_MARK(p1.t_ex);
  mx_perf_ctr_stop(&p1);
  p1.t_st = p1.t_ex;
  const int bad_one = count_bad(C_ONE, "one");

  // ---- run 2: NBLK blocks, loop-managed codebooks + scales, alternating acc halves ----
  mx_perf_t p2 = {0};
  mx_perf_ctr_start(&p2);
  MX_PERF_MARK(p2.t0);
  p2.t_ld = p2.t0;
  for (int c = 0; c < NBLK; c++)
    gemmini_loop_ws_spad_managed(blk_I, tiles_J, tiles_K,
        a_base + c * blk_I * tiles_K * DIM, b_end, C_BLK + c * blk_rows * c_rows_per_row,
        &A_sc_al[0][c * blk_rows], &B_sc_al[0][0], MATMUL_M, MATMUL_N,
        &A_lut_al[(c * blk_rows) >> G][0], 0x38 | LOOP_WS_INC_ACC_ADDR);
  MX_PERF_MARK(p2.t_ex);
  mx_perf_ctr_stop(&p2);
  p2.t_st = p2.t_ex;
  const int bad_blk = count_bad(C_BLK, "blk");

  const uint64_t ideal = MX_PERF_IDEAL_QUAD(MATMUL_M, MATMUL_N, MATMUL_K, DIM);
  mx_perf_report_ideal("fp6_e3m2_lut_bf16_one_loop", MATMUL_M, MATMUL_N, MATMUL_K, DIM, ideal, &p1);
  mx_perf_report_ideal("fp6_e3m2_lut_bf16_blk" BLK_STR(NBLK), MATMUL_M, MATMUL_N, MATMUL_K, DIM, ideal, &p2);
  printf("one loop: %d bad, blocked: %d bad (of %d)\n", bad_one, bad_blk, MATMUL_M * MATMUL_N);
  const int fail = bad_one || bad_blk;
  printf("fp6 lut blocked test %s\n", fail ? "FAILED" : "PASSED");
  return fail;
}
