// BF16-arithmetic variant (MxE4M3LutBf16Dim* configs, spike --extension=gemmini_mxbf16_dim*): golden from gen_asym.py --bf16-arith; prints phase cycles.
// Asymmetric FP8_E4M3-SINGLE-activation x FP4 E2M1-weight 64x64 matmul, BF16 (mode6, dual throughput).
// A (activation) = E4M3 DIRECT 8-bit codes, single (1 row/lane, NO LUT). B (weight) = FP4 E2M1 DIRECT 4-bit
// codes, quad (2 col/lane, NO LUT). => 2 products/PE, DIMx2DIM output tile.
// Output is BF16, read back from the internal scratchpad, compared to golden C_out_bf16.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_data_asym_e4m3s_fp4_dim32_bf16.h"

#define DIM 32
#define VALUES_PER_BYTE 2
#define USE_LUT 0                   // no LUT: E4M3 act direct single, FP4 wei direct
#define MX_ALTFMT 0                 // E4M3 within fp8/code0 (act)
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

  // Dual throughput: single act -> 16 rows/tile (tiles_I = M/16); quad wei -> 32 cols/tile (tiles_J = N/32).
  int tiles_I = MATMUL_M / DIM;
  int tiles_K = MATMUL_K / DIM;
  int tiles_J = MATMUL_N / (2 * DIM);

  gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, QUANT_LUT_UPDATE_GRANULARITY);

  // config_ex: A (act) = fp8/E4M3 (code0, altfmt0) DIRECT single. B (wgt) = fp4 (code2) direct.
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
      ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16)         // A stride
    | ((uint64_t)(3) << 14)         // C (out) format = BF16
    | ((uint64_t)(2) << 12)         // B (weight) format = fp4 (code2), direct
    | ((uint64_t)(0) << 10)         // A (activation) format = fp8 (code0); altfmt=0 -> E4M3
    | ((uint64_t)(0) << 9)          // B transpose
    | ((uint64_t)(0) << 8)          // A transpose
    | ((uint64_t)(0) << 7)          // set only strides
    | ((uint64_t)(MX_ALTFMT) << 6)  // mx_fp8_altfmt (act)
    | ((uint64_t)(USE_LUT) << 5)    // uselut
    | ((uint64_t)(0) << 3)          // activation
    | ((uint64_t)(WEIGHT_STATIONARY) << 2)
    | CONFIG_EX,
      ((uint64_t)(1) << 48) | (0),
      k_CONFIG);

  // No LUTs. Output is BF16.
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();

  uint64_t t0 = read_cycles();
  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;

  // MVIN A: DIRECT 8-bit E4M3 codes, [M][K], 16x16 tiles; stride = MATMUL_K bytes.
  gemmini_config_ld((MATMUL_K) * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      uint8_t *dram_ptr = (uint8_t *)A_in + i * DIM * MATMUL_K + k * DIM;
      uint32_t sp_addr  = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
      gemmini_fence();
    }
  }

  // MVIN B: FP4 codes nibble-packed [K][N/2], 32 cols/tile (16 bytes); stride = MATMUL_N/2 bytes.
  gemmini_config_ld((MATMUL_N / VALUES_PER_BYTE) * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j++) {
      uint8_t *dram_ptr = (uint8_t *)B_in + k * DIM * (MATMUL_N / VALUES_PER_BYTE) + j * DIM;
      uint32_t sp_addr  = b_base + (k * tiles_J + j) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
      gemmini_fence();
    }
  }

  uint64_t t1 = read_cycles();
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
  uint64_t t2 = read_cycles();
  gemmini_config_st(DIM * sizeof(uint8_t));
  uint8_t *c_base = (uint8_t *) C_hw;
  int total_spad_rows = MATMUL_M * MATMUL_N * 2 / DIM;
  for (int r = 0; r < total_spad_rows; r += DIM) {
    gemmini_extended_mvout(c_base + r * DIM, SPAD_DEST + r, DIM, DIM);
  }
  gemmini_fence();
  uint64_t t3 = read_cycles();
  printf("CYCLES load=%llu compute=%llu store=%llu total=%llu\n", (unsigned long long)(t1 - t0),
         (unsigned long long)(t2 - t1), (unsigned long long)(t3 - t2), (unsigned long long)(t3 - t0));

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

  if (errors == 0) printf("e4m3s(single)xfp4 asym WS matmul test PASSED (no mismatches).\n");
  else             printf("e4m3s(single)xfp4 asym WS matmul test FAILED with %d mismatches.\n", errors);
  return errors == 0 ? 0 : 1;
}
