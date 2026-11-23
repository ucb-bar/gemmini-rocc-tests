// Chained native 8-bit E4M3 quad matmuls on MxDim32E4M3QuadGemminiRocketConfig (branch large-mesh-fp8):
// C2 = requant(requant(A1 @ B1) @ B2). MM1's requant output C1 lands in the scratchpad in the block-tiled operand-A
// layout (LOOP_WS_REQUANT_TILED) with its block scales resident in the act-scale window; MM2 reads C1 in place as
// operand A (no A mvin, no A-scale load). Checks C1 and C2 codes + scales. Golden: gen_e4m3_quad_native.py --chain.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#ifndef QUAD_HDR
#define QUAD_HDR "include/matmul_data_e4m3_quad_chain_64x64_dim32.h"
#endif
#include QUAD_HDR

#if DIM != 32
#error "needs the DIM=32 E4M3 quad params header (-iquote dim32/e4m3_quad)"
#endif


#define LOOP_WS_REQUANT_TILED (1u << 10)
#define CHAIN_FLAGS (0x38 | LOOP_WS_REQUANT_TILED)
#define GN (MATMUL_N / 32)
static uint16_t tiled[MATMUL_M / 2 * MATMUL_N] __attribute__((aligned(64)));
static uint16_t C_hw[MATMUL_M / 2][MATMUL_N];
static uint8_t c1_scales[MATMUL_M * GN] __attribute__((aligned(64)));
static uint8_t c2_scales[MATMUL_M * GN] __attribute__((aligned(64)));

// read a tiled operand-A result (tile (p/DIM, n/DIM) at base + ((p/DIM)*(N/DIM) + n/DIM)*DIM, row p%DIM) contiguously,
// then de-tile into C_hw[M/2][N]
static void mvout_detile(uint32_t base) {
  const int rows = MATMUL_M * MATMUL_N / (2 * DIM);
  gemmini_config_st(DIM * sizeof(uint16_t));
  for (int r = 0; r < rows; r += DIM) gemmini_extended_mvout(tiled + r * DIM, base + r, DIM, DIM);
  gemmini_fence();
  for (int p = 0; p < MATMUL_M / 2; p++)
    for (int n = 0; n < MATMUL_N; n++)
      C_hw[p][n] = tiled[(((p / DIM) * (MATMUL_N / DIM) + n / DIM) * DIM + p % DIM) * DIM + n % DIM];
}

static int check(const char *name, const uint16_t (*exp)[MATMUL_N], const uint8_t *sc, const uint8_t (*exp_sc)[GN]) {
  int bad = 0, sbad = 0;
  for (int p = 0; p < MATMUL_M / 2; p++)
    for (int n = 0; n < MATMUL_N; n++)
      if (C_hw[p][n] != exp[p][n] && bad++ < 8) printf("%s[%d][%d] got %04x exp %04x\n", name, p, n, C_hw[p][n], exp[p][n]);
  for (int m = 0; m < MATMUL_M; m++)
    for (int b = 0; b < GN; b++)
      if (sc[m * GN + b] != exp_sc[m][b] && sbad++ < 8)
        printf("%s scale[%d][%d] got %02x exp %02x\n", name, m, b, sc[m * GN + b], exp_sc[m][b]);
  printf("%s: %d code, %d scale mismatches\n", name, bad, sbad);
  return bad + sbad;
}

int main() {
  const int tI = MATMUL_M / (2 * DIM), tJ = MATMUL_N / (2 * DIM), tK1 = MATMUL_K / DIM, tK2 = MATMUL_N / DIM;
  const uint32_t a_base = 0, C1 = BANK_ROWS, C2 = 2 * BANK_ROWS;
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 0);

  // ---- MM1: C1 = requant(A1 @ B1), tiled + resident scales ----
  gemmini_mx_load_scales((uint64_t)&A_sc[0][0], sizeof(A_sc), 0);
  gemmini_mx_load_scales((uint64_t)&B_sc[0][0], sizeof(B_sc), 1);
  gemmini_fence();
  uint32_t b_base = BANK_NUM * BANK_ROWS - tK1 * tJ * DIM;
  gemmini_config_ld(MATMUL_K * sizeof(uint16_t));
  for (int i = 0; i < tI; i++)
    for (int k = 0; k < tK1; k++)
      gemmini_extended_mvin(&A_hw[i * DIM][k * DIM], a_base + (i * tK1 + k) * DIM, DIM, DIM);
  gemmini_config_ld((MATMUL_N / 2) * sizeof(uint16_t));
  for (int k = 0; k < tK1; k++)
    for (int j = 0; j < tJ; j++)
      gemmini_extended_mvin(&B_hw[k * DIM][j * DIM], b_base + (k * tJ + j) * DIM, DIM, DIM);
  gemmini_config_st(1);
  gemmini_mxquant_config_mvout_resident((uint64_t)c1_scales, tI, tJ, tK1, 0, 0, 1);
  uint64_t t0 = read_cycles();
  gemmini_loop_ws_spad(tI, tJ, tK1, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0, C1,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false, CHAIN_FLAGS);
  gemmini_fence();
  uint64_t t1 = read_cycles();
  mvout_detile(C1);
  int bad = check("C1", C_q_hw, c1_scales, C_scales_out);

  // ---- MM2: C2 = requant(C1 @ B2), C1 read in place as operand A, its scales resident ----
  gemmini_mx_load_scales((uint64_t)&B2_sc[0][0], sizeof(B2_sc), 1);
  gemmini_fence();
  b_base = BANK_NUM * BANK_ROWS - tK2 * tJ * DIM;
  gemmini_config_ld((MATMUL_N / 2) * sizeof(uint16_t));
  for (int k = 0; k < tK2; k++)
    for (int j = 0; j < tJ; j++)
      gemmini_extended_mvin(&B2_hw[k * DIM][j * DIM], b_base + (k * tJ + j) * DIM, DIM, DIM);
  gemmini_config_st(1);
  gemmini_mxquant_config_mvout_resident((uint64_t)c2_scales, tI, tJ, tK2, 0, 0, 1);
  uint64_t t2 = read_cycles();
  gemmini_loop_ws_spad(tI, tJ, tK2, 0, 0, 0, C1, BANK_NUM * BANK_ROWS, 0, C2,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false, CHAIN_FLAGS);
  gemmini_fence();
  uint64_t t3 = read_cycles();
  mvout_detile(C2);
  bad += check("C2", C2_q_hw, c2_scales, C2_scales_out);

  printf("e4m3 quad chain %dx%dx%d dim32: MM1 %llu, MM2 %llu cycles\n", MATMUL_M, MATMUL_N, MATMUL_K,
         (unsigned long long)(t1 - t0), (unsigned long long)(t3 - t2));
  printf(bad ? "e4m3 quad chain test FAILED\n" : "e4m3 quad chain test PASSED\n");
  return bad != 0;
}
