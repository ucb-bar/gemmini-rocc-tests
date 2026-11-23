// Asymmetric FP4-activation x FP6_E3M2-weight, 64x64 matmul, BF16 output — DIM=32 (32x32 mesh),
// MxDim32AllAsymGemminiRocketConfig. DIM=32 port of matmul_tiled_asym_fp4_fp6_64x64.c: A=FP4 direct (fmt2),
// B=FP6 E3M2 LUT-deprojected (fmt1), OUT=BF16 (fmt3), uselut=1. Tiling/addressing = the proven symmetric
// dim32 port (2*DIM quad tiles, B granularity DIM, SPAD_DEST clears the A footprint). DIM-aware golden header.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_data_asym_fp4_fp6_128x128x256_dim32.h"

#define TILE 16
#define DIM 32
#define VALUES_PER_BYTE 2
#define USE_LUT 1
#define QUANT_LUT_UPDATE_GRANULARITY 1
#define ACT_MX_FMT 2                 // activation = FP4 (direct)
#define WGT_MX_FMT 1                 // weight     = FP6 E3M2 (LUT-deprojected)
#define OUT_MX_FMT 3                 // output     = BF16
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

  int tiles_I = MATMUL_M / (2 * DIM);
  int tiles_K = MATMUL_K / DIM;
  int tiles_J = MATMUL_N / (2 * DIM);

  gemmini_extended_config_st(DIM * sizeof(uint8_t), NO_ACTIVATION, 1);
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, QUANT_LUT_UPDATE_GRANULARITY);
  // Asymmetric: act=FP4(2), wgt=FP6(1), out=BF16(3), uselut=1.
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
      ACT_MX_FMT, WGT_MX_FMT, OUT_MX_FMT, USE_LUT);

  // B (weight) LUT is the real fp6 codebook; A_lut/C_lut are placeholders (fp4 activation takes the direct
  // path, output is bf16). 6-bit LUT entries (fp6).
  gemmini_mx_load_lut_dt((uint64_t)&B_lut[0][0], (MATMUL_N >> QUANT_LUT_UPDATE_GRANULARITY), 0, 6);
  gemmini_mx_load_lut_dt((uint64_t)&A_lut[0][0], (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY), 1, 6);
  gemmini_mx_load_lut_dt((uint64_t)&C_lut[0][0], (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY), 2, 6);
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;

  // MVIN A: HW-tiled [M/2][K], 4-bit codes nibble-packed 2 m-rows/byte; stride = MATMUL_K bytes.
  gemmini_config_ld((MATMUL_K) * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++) {
      uint8_t *dram_ptr = (uint8_t *)A_in_hw + i * DIM * MATMUL_K + k * DIM;
      gemmini_extended_mvin((void *) dram_ptr, a_base + (i * tiles_K + k) * DIM, DIM, DIM);
      gemmini_fence();
    }

  // MVIN B: [K][N/2], 4-bit LUT indices; stride = MATMUL_N/2 bytes. Scratchpad granularity = DIM.
  gemmini_config_ld((MATMUL_N / VALUES_PER_BYTE) * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++) {
      uint8_t *dram_ptr = (uint8_t *)B_in + k * DIM * (MATMUL_N / VALUES_PER_BYTE) + j * DIM;
      gemmini_extended_mvin((void *) dram_ptr, b_base + (k * tiles_J + j) * DIM, DIM, DIM);
      gemmini_fence();
    }

  int SPAD_DEST = tiles_I * tiles_K * DIM;   // clear the A footprint
  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K,
      0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0, SPAD_DEST,
      false, false, false, false, false,
      NO_ACTIVATION, 0, 0, false, 0x38);

  // Non-requant BF16 in internal spad. Flat spad->DRAM readback. BF16 = 2 bytes/elem -> M*N*2/DIM rows.
  gemmini_fence();
  gemmini_config_st(DIM * sizeof(uint8_t));
  uint8_t *c_base = (uint8_t *) C_hw;
  int total_spad_rows = MATMUL_M * MATMUL_N * 2 / DIM;
  for (int r = 0; r < total_spad_rows; r += DIM)
    gemmini_extended_mvout(c_base + r * DIM, SPAD_DEST + r, DIM, DIM);
  gemmini_fence();

  // ---- Elementwise check against C_out_bf16 ----
  int errors = 0, printed = 0;
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t got = C_hw[i][j];
      uint64_t exp = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                     ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                     ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                     ((uint64_t)C_out_bf16[i][j*4 + 0]);
      if (got != exp) {
        for (int lane = 0; lane < BF16_PER_WORD; lane++) {
          uint16_t g = (got >> (lane * 16)) & 0xFFFF;
          uint16_t e = C_out_bf16[i][j * BF16_PER_WORD + lane];
          if (g != e) {
            errors++;
            if (printed++ < 40) printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n", i, j*BF16_PER_WORD+lane, g, e);
          }
        }
      }
    }
  }

  if (errors == 0) printf("fp4xfp6 asym WS matmul dim32 test PASSED (no mismatches).\n");
  else             printf("fp4xfp6 asym WS matmul dim32 test FAILED with %d mismatches.\n", errors);
  return errors == 0 ? 0 : 1;
}
