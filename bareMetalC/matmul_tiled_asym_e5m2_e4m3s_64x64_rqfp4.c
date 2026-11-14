// Asymmetric FP8_E5M2-activation x FP8_E4M3-SINGLE-weight 64x64 matmul, FP4 requant output (mode5, dual throughput 32x16).
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
#include "include/matmul_data_asym_e5m2_e4m3s_rqfp4.h"

#define DIM 16
#define VALUES_PER_BYTE 2
#define USE_LUT 1                   // activation E3M2 deproject (act LUT). Weight is direct single (no LUT).
#define MX_ALTFMT 1                 // E5M2 within fp8/code0 (act, altfmt1); weight altfmt diff 1 -> E4M3
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
  int tiles_I = MATMUL_M / 32;
  int tiles_K = MATMUL_K / DIM;
  int tiles_J = MATMUL_N / DIM;

  gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, QUANT_LUT_UPDATE_GRANULARITY);

  // config_ex: A (act) = fp8/E5M2 (code0, altfmt1) quad LUT. B (wgt) = fp8/E4M3 (code0, altfmt diff 1 -> E4M3) single.
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
      ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16)         // A stride
    | ((uint64_t)(2) << 14)         // C (out) format = fp4 (requant)
    | ((uint64_t)(0) << 12)         // B (weight) format = fp8 (code0) -> E4M3 single
    | ((uint64_t)(1) << 31)         // weight_altfmt_diff: weight altfmt = act_altfmt(1) XOR 1 = 0 -> E4M3
    | ((uint64_t)(0) << 10)         // A (activation) format = fp8 (code0); altfmt=1 -> E5M2
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
  uint32_t b_base = 8192 - tiles_K * tiles_J * DIM;

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

  int SPAD_DEST = 128;
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base, 8192, 0, SPAD_DEST,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      0x38);

  gemmini_fence();
  gemmini_config_st(DIM * sizeof(uint8_t));
  uint8_t *c_base = (uint8_t *) C_hw;
  int total_spad_rows = MATMUL_M * MATMUL_N / 2 / DIM;   // fp4: M*N/2 bytes
  for (int r = 0; r < total_spad_rows; r += DIM) {
    gemmini_extended_mvout(c_base + r * DIM, SPAD_DEST + r, DIM, DIM);
  }
  gemmini_fence();

  // FP4 requant: 2 output rows per byte-row (bits[3:0] = row 2r, bits[7:4] = row 2r+1), HW-tiled
  int errors = 0;
  uint8_t *hw_bytes = (uint8_t *)C_hw;
  for (int i = 0; i < MATMUL_M / 2; i++) {
    for (int j = 0; j < MATMUL_N; j++) {
      uint8_t got = hw_bytes[i * MATMUL_N + j];
      uint8_t exp = C_out[i][j];
      for (int h = 0; h < 2; h++) {
        uint8_t g = (got >> (4 * h)) & 0xF, e = (exp >> (4 * h)) & 0xF;
        if (g != e) {
          if (errors < 32) printf("C_out[%d][%d] %s: got %x exp %x\n", i, j, h ? "hi" : "lo", g, e);
          errors++;
        }
      }
    }
  }
  int scale_errors = 0;
  uint8_t *sf_bytes = (uint8_t *) scale_factors;
  for (int i = 0; i < MATMUL_M; i++) {
    for (int b = 0; b < MATMUL_N / 32; b++) {
      uint8_t got = sf_bytes[b * MATMUL_M + i];
      uint8_t exp = C_scales_out[i][b];
      if (got != exp) {
        if (scale_errors < 20) printf("Scale[%d][%d] got %x exp %x\n", i, b, got, exp);
        scale_errors++;
      }
    }
  }
  if (errors == 0 && scale_errors == 0) printf("e5m2_e4m3s fp4-requant WS matmul test PASSED (no mismatches).\n");
  else printf("e5m2_e4m3s fp4-requant WS matmul test FAILED: %d code, %d scale mismatches.\n", errors, scale_errors);
  return (errors == 0 && scale_errors == 0) ? 0 : 1;
}
