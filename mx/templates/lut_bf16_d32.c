// Template: same-format LUT matmul, BF16 out (DIM32 layout). Built per line of mx/tests.txt.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include MX_DATA_H
#include "include/mx_test_common.h"

#define TILE 16
#undef DIM
#define DIM MESH_DIM
#define CODE MX_CODE(FMT)
#define LUT_BITS (CODE == 0 ? 8 : 6)
#define VALUES_PER_BYTE 2
#define USE_LUT 1
#define MX_ALTFMT MX_ALT(FMT)
#define QUANT_LUT_UPDATE_GRANULARITY 1
#define ADDR_LEN 32
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_N / BF16_PER_WORD)

typedef uint64_t out_t;   // 4x bf16 packed per word

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  static uint32_t scale_factors[MATMUL_M * MATMUL_N / 32] __attribute__((aligned(32))) = {0};
  static out_t C_hw[MATMUL_M][OUT_COLS];
  memset(C_hw, 0, sizeof(C_hw));

  gemmini_flush(0);

  // DIM=32 quad tiling: one tile = 2*DIM rows x 2*DIM cols, K-depth DIM.
  int tiles_I = MATMUL_M / (2 * DIM);
  int tiles_K = MATMUL_K / DIM;
  int tiles_J = MATMUL_N / (2 * DIM);

  gemmini_extended_config_st(DIM * sizeof(uint8_t), NO_ACTIVATION, 1);
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, QUANT_LUT_UPDATE_GRANULARITY);

  // config_ex: E4M3-quad = fp8/code0 (A/B format 0) + altfmt0 + lut_en; OUT format = 3 (BF16, non-requant).
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
      ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16)
    | ((uint64_t)(3) << 14)         // C (out) format = 3 (BF16)
    | ((uint64_t)(CODE) << 12)      // B format
    | ((uint64_t)(CODE) << 10)      // A format
    | ((uint64_t)(0) << 9)
    | ((uint64_t)(0) << 8)
    | ((uint64_t)(0) << 7)
    | ((uint64_t)(MX_ALTFMT) << 6)
    | ((uint64_t)(USE_LUT) << 5)
    | ((uint64_t)(0) << 3)
    | ((uint64_t)(WEIGHT_STATIONARY) << 2)
    | CONFIG_EX,
      ((uint64_t)(1) << 48) | (0),
      k_CONFIG);

  gemmini_mx_load_lut_dt((uint64_t)&B_lut[0][0], (MATMUL_N >> QUANT_LUT_UPDATE_GRANULARITY), 0, LUT_BITS);
  gemmini_mx_load_lut_dt((uint64_t)&A_lut[0][0], (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY), 1, LUT_BITS);
  gemmini_mx_load_lut_dt((uint64_t)&C_lut[0][0], (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY), 2, LUT_BITS);
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;

  // MVIN A: HW-tiled [M/2][K], 4-bit indices nibble-packed 2 m-rows/byte; stride = MATMUL_K bytes.
  gemmini_config_ld((MATMUL_K) * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++) {
      uint8_t *dram_ptr = (uint8_t *)A_in_hw + i * DIM * MATMUL_K + k * DIM;
      gemmini_extended_mvin((void *) dram_ptr, a_base + (i * tiles_K + k) * DIM, DIM, DIM);
      gemmini_fence();
    }

  // MVIN B: [K][N/2], 4-bit indices; stride = MATMUL_N/2 bytes. Scratchpad granularity = DIM.
  gemmini_config_ld((MATMUL_N / VALUES_PER_BYTE) * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++) {
      uint8_t *dram_ptr = (uint8_t *)B_in + k * DIM * (MATMUL_N / VALUES_PER_BYTE) + j * DIM;
      gemmini_extended_mvin((void *) dram_ptr, b_base + (k * tiles_J + j) * DIM, DIM, DIM);
      gemmini_fence();
    }

  int SPAD_DEST = 128;
  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K,
      0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0, SPAD_DEST,
      false, false, false, false, false,
      NO_ACTIVATION, 0, 0, false, 0x38);

  // Non-requant BF16 in internal spad (full-width store, row-major). Flat spad->DRAM readback.
  // BF16 = 2 bytes/elem -> M*N*2/DIM spad rows.
  gemmini_fence();
  gemmini_config_st(DIM * sizeof(uint8_t));
  uint8_t *c_base = (uint8_t *) C_hw;
  int total_spad_rows = MATMUL_M * MATMUL_N * 2 / DIM;
  for (int r = 0; r < total_spad_rows; r += DIM)
    gemmini_extended_mvout(c_base + r * DIM, SPAD_DEST + r, DIM, DIM);
  gemmini_fence();

  // ---- Elementwise check against C_out_bf16 ----
  int errors = 0;
  int printed = 0;
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
            errors++;
            if (printed++ < 40)
              printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n",
                     i, j * BF16_PER_WORD + lane, got_bf16, exp_bf16);
          }
        }
      }
    }
  }

  if (errors == 0) printf(MX_STR(FMT) " LUT WS matmul test PASSED (no mismatches).\n");
  else             printf(MX_STR(FMT) " LUT WS matmul test FAILED with %d mismatches.\n", errors);
  return errors == 0 ? 0 : 1;
}
