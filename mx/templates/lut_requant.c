// Same-format LUT matmul with fp4-index requant output. Built per line of mx/tests.txt.
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

#undef DIM
#define DIM MESH_DIM
#define VALUES_PER_BYTE 2
#define QUANT_LUT_UPDATE_GRANULARITY 1
#define CODE MX_CODE(FMT)
#define LUT_BITS (CODE == 0 ? 8 : 6)

// Original fixed layouts, kept so the binaries stay identical: DIM16 = 8192-row spad with C at row 0;
// DIM32 64x64 puts C at row 128; everything else puts C right after A.
#if DIM == 16
#define SPAD_ROWS 8192
#define SPAD_DEST_ROW(ti, tk) 0
#elif DIM == 32 && MATMUL_M == 64 && MATMUL_N == 64 && MATMUL_K == 64
#define SPAD_ROWS (BANK_NUM * BANK_ROWS)
#define SPAD_DEST_ROW(ti, tk) 128
#else
#define SPAD_ROWS (BANK_NUM * BANK_ROWS)
#define SPAD_DEST_ROW(ti, tk) ((ti) * (tk) * DIM)
#endif

typedef uint8_t out_t;

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  static uint32_t scale_factors[MATMUL_M * MATMUL_N / 32] __attribute__((aligned(32))) = {0};
  static uint8_t C_hw[MATMUL_M / 2][MATMUL_N];
  memset(C_hw, 0, sizeof(C_hw));

  gemmini_flush(0);

  int tiles_I = MATMUL_M / (2 * DIM);
  int tiles_K = MATMUL_K / DIM;
  int tiles_J = MATMUL_N / (2 * DIM);

  gemmini_config_st(1 * sizeof(uint64_t));
  gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, QUANT_LUT_UPDATE_GRANULARITY);

  // LUT indices in, 4-bit LUT indices out (requant).
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
      ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16)
    | ((uint64_t)(CODE) << 14)      // C (out) format
    | ((uint64_t)(CODE) << 12)      // B format
    | ((uint64_t)(CODE) << 10)      // A format
    | ((uint64_t)(0) << 9)
    | ((uint64_t)(0) << 8)
    | ((uint64_t)(0) << 7)
    | ((uint64_t)(MX_ALT(FMT)) << 6)
    | ((uint64_t)(1) << 5)
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
  uint32_t b_base = SPAD_ROWS - tiles_K * tiles_J * DIM;

  // MVIN A: HW-tiled [M/2][K], 4-bit indices nibble-packed 2 m-rows/byte; stride = MATMUL_K bytes.
  gemmini_config_ld((MATMUL_K) * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++) {
      uint8_t *dram_ptr = (uint8_t *)A_in_hw + i * DIM * MATMUL_K + k * DIM;
      gemmini_extended_mvin((void *) dram_ptr, a_base + (i * tiles_K + k) * DIM, DIM, DIM);
      gemmini_fence();
    }

  // MVIN B: [K][N/2], 4-bit indices; stride = MATMUL_N/2 bytes.
  gemmini_config_ld((MATMUL_N / VALUES_PER_BYTE) * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++) {
      uint8_t *dram_ptr = (uint8_t *)B_in + k * DIM * (MATMUL_N / VALUES_PER_BYTE) + j * DIM;
      gemmini_extended_mvin((void *) dram_ptr, b_base + (k * tiles_J + j) * DIM, DIM, DIM);
      gemmini_fence();
    }

  int SPAD_DEST = SPAD_DEST_ROW(tiles_I, tiles_K);
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K,
      0, 0, 0, a_base, SPAD_ROWS, 0, SPAD_DEST,
      false, false, false, false, false,
      NO_ACTIVATION, 0, 0, false, 0x38);

  // Readback: requant = 4-bit indices, 2/byte -> M*N/2 bytes -> M*N/2/DIM spad rows.
  gemmini_fence();
  gemmini_config_st(DIM * sizeof(uint8_t));
  uint8_t *c_base = (uint8_t *) C_hw;
  int total_spad_rows = MATMUL_M * MATMUL_N / 2 / DIM;
  for (int r = 0; r < total_spad_rows; r += DIM)
    gemmini_extended_mvout(c_base + r * DIM, SPAD_DEST + r, DIM, DIM);
  gemmini_fence();

  int errors = 0;
  uint8_t *hw_bytes = (uint8_t *)C_hw;
  int printed = 0;
  for (int i = 0; i < MATMUL_M / 2; i++) {
    for (int j = 0; j < MATMUL_N; j++) {
      uint8_t got = hw_bytes[i * MATMUL_N + j];
      uint8_t exp = C_proj_hw[i][j];
      if ((got & 0x0F) != (exp & 0x0F)) {
        errors++;
        if (printed++ < 40) printf("C_proj_hw[%d][%d] lo: got %x exp %x\n", i, j, got & 0xF, exp & 0xF);
      }
      if (((got >> 4) & 0x0F) != ((exp >> 4) & 0x0F)) {
        errors++;
        if (printed++ < 40) printf("C_proj_hw[%d][%d] hi: got %x exp %x\n", i, j, (got >> 4) & 0xF, (exp >> 4) & 0xF);
      }
    }
  }

  int scale_errors = 0;
  uint8_t *sf_bytes = (uint8_t *) scale_factors;
  for (int i = 0; i < MATMUL_M; i++) {
    for (int b = 0; b < MATMUL_GN; b++) {
      uint8_t got = sf_bytes[b * MATMUL_M + i];
      uint8_t exp = C_scales_row[b][i];
      if (got != exp) {
        scale_errors++;
        if (scale_errors <= 20) printf("Scale[%d][%d] got %x exp %x\n", i, b, got, exp);
      }
    }
  }

  if (errors == 0 && scale_errors == 0)
    printf(MX_STR(FMT) " LUT requant test PASSED (no mismatches).\n");
  else
    printf(MX_STR(FMT) " LUT requant test FAILED: %d code, %d scale mismatches.\n", errors, scale_errors);
  return (errors == 0 && scale_errors == 0) ? 0 : 1;
}
