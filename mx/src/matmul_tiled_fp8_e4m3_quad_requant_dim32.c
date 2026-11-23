// Native 8-bit E4M3 quad (mode9) matmul, E4M3 REQUANT output into the internal scratchpad, on
// MxDim32E4M3QuadGemminiRocketConfig (branch large-mesh-fp8). The requantizer coalesces each output row pair into
// 16-bit lanes (row 2r code low byte, row 2r+1 high byte): the result lands in the spad as the next matmul's operand A
// (flat: lane index p*N + n -> spad row (p*N + n)/DIM). Checks the codes (C_q_hw) and the E8M0 block scales.
// Built only through npu-exploration wrappers with the DIM=32 quad params header. Golden: gen_e4m3_quad_native.py.
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


#define GN (MATMUL_N / 32)
static uint16_t C_hw[MATMUL_M / 2][MATMUL_N] __attribute__((aligned(64)));
static uint8_t scale_factors[MATMUL_M * GN] __attribute__((aligned(64)));

int main() {
  const int tiles_I = MATMUL_M / (2 * DIM), tiles_J = MATMUL_N / (2 * DIM), tiles_K = MATMUL_K / DIM;
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 0);
  gemmini_mx_load_scales((uint64_t)&A_sc[0][0], sizeof(A_sc), 0);
  gemmini_mx_load_scales((uint64_t)&B_sc[0][0], sizeof(B_sc), 1);
  gemmini_fence();

  // A tile (i, k): 32 row pairs x 32 lanes at a_base + (i*tiles_K + k)*DIM; B tile (k, j) at b_base + (k*tiles_J + j)*DIM
  const uint32_t a_base = 0, b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  gemmini_config_ld(MATMUL_K * sizeof(uint16_t));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin(&A_hw[i * DIM][k * DIM], a_base + (i * tiles_K + k) * DIM, DIM, DIM);
  gemmini_config_ld((MATMUL_N / 2) * sizeof(uint16_t));
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++)
      gemmini_extended_mvin(&B_hw[k * DIM][j * DIM], b_base + (k * tiles_J + j) * DIM, DIM, DIM);

  const uint32_t SPAD_DEST = BANK_ROWS;   // bank 1
  gemmini_config_st(1);
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
  uint64_t t0 = read_cycles();
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0, SPAD_DEST,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
  gemmini_fence();
  uint64_t t1 = read_cycles();

  // readback: M/2 * N lanes = M*N/(2*DIM) spad rows of DIM lanes
  gemmini_config_st(DIM * sizeof(uint16_t));
  const int rows = MATMUL_M * MATMUL_N / (2 * DIM);
  for (int r = 0; r < rows; r += DIM)
    gemmini_extended_mvout((uint16_t *)C_hw + r * DIM, SPAD_DEST + r, DIM, DIM);
  gemmini_fence();

  int bad = 0, sbad = 0;
  for (int p = 0; p < MATMUL_M / 2; p++)
    for (int n = 0; n < MATMUL_N; n++)
      if (C_hw[p][n] != C_q_hw[p][n] && bad++ < 16)
        printf("C_q[%d][%d] got %04x exp %04x\n", p, n, C_hw[p][n], C_q_hw[p][n]);
  for (int m = 0; m < MATMUL_M; m++)
    for (int b = 0; b < GN; b++)
      if (scale_factors[m * GN + b] != C_scales_out[m][b] && sbad++ < 16)
        printf("scale[%d][%d] got %02x exp %02x\n", m, b, scale_factors[m * GN + b], C_scales_out[m][b]);
  printf("e4m3 quad requant %dx%dx%d dim32: %d code, %d scale mismatches; %llu cycles\n", MATMUL_M, MATMUL_N, MATMUL_K,
         bad, sbad, (unsigned long long)(t1 - t0));
  printf(bad || sbad ? "e4m3 quad requant test FAILED\n" : "e4m3 quad requant test PASSED\n");
  return bad || sbad;
}
