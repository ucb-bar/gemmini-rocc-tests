// Asymmetric FP8_E4M3-QUAD-activation x FP8_E4M3-SINGLE-weight 64x64 matmul, BF16 (mode12, dual throughput).
// A (activation) = E3M2 4-bit LUT indices, quad (2 rows/lane, act LUT loaded). B (weight) = E4M3 DIRECT
// 8-bit, single (1 col/lane, no weight LUT). => 2 products/PE, 32x16 output tile.
// Output is BF16, read back from the internal scratchpad, compared to golden C_out_bf16.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_data_asym_e4m3_e4m3s_dim32.h"

#define DIM 32
#define VALUES_PER_BYTE 2
#define USE_LUT 1                   // activation E4M3 deproject (act LUT -> quad). Weight is direct single (no LUT).
#define MX_ALTFMT 0                 // E4M3 within fp8/code0 (act); weight altfmt diff 0 -> E4M3
#define QUANT_LUT_UPDATE_GRANULARITY 1
#define ADDR_LEN 32
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_M / BF16_PER_WORD)

typedef uint8_t out_t;

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  static uint32_t scale_factors[MATMUL_M * MATMUL_N / 32] __attribute__((aligned(32))) = {0};
  static uint64_t C_hw[MATMUL_M][OUT_COLS];
  memset(C_hw, 0, sizeof(C_hw));

  gemmini_flush(0);

  // Dual throughput: quad act -> 32 rows/tile (tiles_I = M/32); single wei -> 16 cols/tile (tiles_J = N/16).
  int tiles_I = MATMUL_M / (2 * DIM);
  int tiles_K = MATMUL_K / DIM;
  int tiles_J = MATMUL_N / DIM;

  gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, QUANT_LUT_UPDATE_GRANULARITY);

  // config_ex: A (act) = fp8/E4M3 (code0, altfmt0) quad LUT. B (wgt) = fp8/E4M3 (code0, altfmt diff 0) single.
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
      ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16)         // A stride
    | ((uint64_t)(3) << 14)         // C (out) format = BF16
    | ((uint64_t)(0) << 12)         // B (weight) format = fp8 (code0) -> E4M3 single
    | ((uint64_t)(0) << 10)         // A (activation) format = fp8 (code0) -> E4M3 (altfmt0)
    | ((uint64_t)(0) << 9)          // B transpose
    | ((uint64_t)(0) << 8)          // A transpose
    | ((uint64_t)(0) << 7)          // set only strides
    | ((uint64_t)(MX_ALTFMT) << 6)  // mx_fp8_altfmt
    | ((uint64_t)(USE_LUT) << 5)    // uselut
    | ((uint64_t)(0) << 3)          // activation
    | ((uint64_t)(WEIGHT_STATIONARY) << 2)
    | CONFIG_EX,
      ((uint64_t)(1) << 48) | (0),
      k_CONFIG);

  // Load ONLY the activation LUT (sel=1). No weight LUT: E4M3 weight is direct single. Output is BF16.
  gemmini_mx_load_lut_dt((uint64_t)&A_lut[0][0], (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY), 1, 8);
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;

  // MVIN A: E3M2 LUT indices nibble-packed HW-tiled [M/2][K], 32 rows/tile; stride = MATMUL_K bytes.
  gemmini_config_ld((MATMUL_K) * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      uint8_t *dram_ptr = (uint8_t *)A_in_hw + i * DIM * MATMUL_K + k * DIM;
      uint32_t sp_addr  = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
      gemmini_fence();
    }
  }

  // MVIN B: E4M3 DIRECT 8-bit [K][N], 16 cols/tile; stride = MATMUL_N bytes.
  gemmini_config_ld((MATMUL_N) * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j++) {
      uint8_t *dram_ptr = (uint8_t *)B_in + k * DIM * MATMUL_N + j * DIM;
      uint32_t sp_addr  = b_base + (k * tiles_J + j) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
      gemmini_fence();
    }
  }

  int SPAD_DEST = tiles_I * tiles_K * DIM;
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base, BANK_NUM * BANK_ROWS, 0, SPAD_DEST,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      0x38);

  gemmini_fence();
  gemmini_config_st(DIM * sizeof(uint8_t));
  uint8_t *c_base = (uint8_t *) C_hw;
  int total_spad_rows = MATMUL_M * MATMUL_N * 2 / DIM;
  for (int r = 0; r < total_spad_rows; r += DIM) {
    gemmini_extended_mvout(c_base + r * DIM, SPAD_DEST + r, DIM, DIM);
  }
  gemmini_fence();

  int errors = 0;
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t got = C_hw[i][j];
      uint64_t exp = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                     ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                     ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                     ((uint64_t)C_out_bf16[i][j*4 + 0]);
      if (got != exp) {
        for (int lane = 0; lane < BF16_PER_WORD; lane++) {
          uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
          uint16_t exp_bf16 = C_out_bf16[i][j * BF16_PER_WORD + lane];
          if (got_bf16 != exp_bf16) {
            printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n",
                   i, j * BF16_PER_WORD + lane, got_bf16, exp_bf16);
            errors++;
          }
        }
      }
    }
  }

  if (errors == 0) printf("e4m3xe4m3s(single) asym WS matmul test PASSED (no mismatches).\n");
  else             printf("e4m3xe4m3s(single) asym WS matmul test FAILED with %d mismatches.\n", errors);
  return errors == 0 ? 0 : 1;
}
