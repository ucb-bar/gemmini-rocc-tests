// Asymmetric MX weight-stationary matmul. Built per line of mx/tests.txt (see mxtests.py).
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

#ifndef OUT_FP4
#define OUT_FP4 0
#endif
#ifndef CYCLES
#define CYCLES 0
#endif

// Operand paths. Quad (LUT or fp4) operands pack 2 rows per byte, so a tile covers 2*DIM rows/cols;
// e4m3s operands are 8-bit direct and cover DIM.
#define A_SINGLE MX_SINGLE(ACT)
#define B_SINGLE MX_SINGLE(WEI)
#define A_FP4 (MX_CODE(ACT) == 2)
#define B_FP4 (MX_CODE(WEI) == 2)
#define A_DIRECT (A_SINGLE || A_FP4)
#define B_DIRECT (B_SINGLE || B_FP4)
#define USE_LUT (!(A_DIRECT && B_DIRECT))

// altfmt: activation's, or the weight's when the activation is fp4. wdiff: 0 if either side is fp4.
#ifndef ALT
#define ALT (A_FP4 ? MX_ALT(WEI) : MX_ALT(ACT))
#endif
#ifndef WDIFF
#define WDIFF ((A_FP4 || B_FP4) ? 0 : (MX_ALT(ACT) ^ MX_ALT(WEI)))
#endif

// LUT bit width when both operands use LUTs: 8 if either is an 8-bit format, else 6.
#define LUT_BITS ((MX_CODE(ACT) == 0 || MX_CODE(WEI) == 0) ? 8 : 6)
#define QUANT_LUT_UPDATE_GRANULARITY 1
#define VALUES_PER_BYTE 2
#define OUT_COLS (MATMUL_N / 4)

// DIM16 tests keep their original fixed layout (B at the top of an 8192-row spad, C at row 128).
#if DIM == 16
#define SPAD_ROWS 8192
#define SPAD_DEST_ROW(ti, tk) 128
#else
#define SPAD_ROWS (BANK_NUM * BANK_ROWS)
#define SPAD_DEST_ROW(ti, tk) ((ti) * (tk) * DIM)
#endif

#if OUT_FP4
#define MSG MX_STR(ACT) "_" MX_STR(WEI) " fp4-requant WS matmul test"
#else
#define MSG MX_LABEL(ACT) "x" MX_LABEL(WEI) " asym WS matmul test"
#endif

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  static uint32_t scale_factors[MATMUL_M * MATMUL_N / 32] __attribute__((aligned(32))) = {0};
  static uint64_t C_hw[MATMUL_M][OUT_COLS];
  memset(C_hw, 0, sizeof(C_hw));

  gemmini_flush(0);

  int tiles_I = MATMUL_M / (A_SINGLE ? DIM : 2 * DIM);
  int tiles_K = MATMUL_K / DIM;
  int tiles_J = MATMUL_N / (B_SINGLE ? DIM : 2 * DIM);

  gemmini_extended_config_st(DIM * sizeof(uint8_t), NO_ACTIVATION, 1);
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, QUANT_LUT_UPDATE_GRANULARITY);

  mx_config_ex_ws(OUT_FP4 ? MX_OUT_FP4 : MX_OUT_BF16, MX_CODE(WEI), MX_CODE(ACT), ALT, WDIFF, USE_LUT);

  // LUTs: B = sel 0, A = sel 1, C = sel 2 (placeholder; output is not LUT-encoded).
#if !A_SINGLE && !B_SINGLE
  // neither side e4m3s: all three LUTs (an fp4 side is direct but its LUT is still loaded)
  gemmini_mx_load_lut_dt((uint64_t)&B_lut[0][0], (MATMUL_N >> QUANT_LUT_UPDATE_GRANULARITY), 0, LUT_BITS);
  gemmini_mx_load_lut_dt((uint64_t)&A_lut[0][0], (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY), 1, LUT_BITS);
  gemmini_mx_load_lut_dt((uint64_t)&C_lut[0][0], (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY), 2, LUT_BITS);
#elif A_SINGLE && !B_DIRECT
  gemmini_mx_load_lut_dt((uint64_t)&B_lut[0][0], (MATMUL_N >> QUANT_LUT_UPDATE_GRANULARITY), 0, 8);
#elif B_SINGLE && !A_DIRECT
  gemmini_mx_load_lut_dt((uint64_t)&A_lut[0][0], (MATMUL_M >> QUANT_LUT_UPDATE_GRANULARITY), 1, 8);
#endif
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();

#if CYCLES
  uint64_t t0 = read_cycles();
#endif
  uint32_t a_base = 0;
  uint32_t b_base = SPAD_ROWS - tiles_K * tiles_J * DIM;

  // MVIN A: e4m3s [M][K] 8-bit, else HW-tiled [M/2][K] nibble-packed; stride = MATMUL_K bytes.
  gemmini_config_ld((MATMUL_K) * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
#if A_SINGLE
      uint8_t *dram_ptr = (uint8_t *)A_in + i * DIM * MATMUL_K + k * DIM;
#else
      uint8_t *dram_ptr = (uint8_t *)A_in_hw + i * DIM * MATMUL_K + k * DIM;
#endif
      uint32_t sp_addr  = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
      gemmini_fence();
    }
  }

  // MVIN B: e4m3s [K][N] 8-bit, else [K][N/2] nibble-packed.
#if B_SINGLE
  gemmini_config_ld((MATMUL_N) * sizeof(uint8_t));
#else
  gemmini_config_ld((MATMUL_N / VALUES_PER_BYTE) * sizeof(uint8_t));
#endif
  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j++) {
#if B_SINGLE
      uint8_t *dram_ptr = (uint8_t *)B_in + k * DIM * MATMUL_N + j * DIM;
#else
      uint8_t *dram_ptr = (uint8_t *)B_in + k * DIM * (MATMUL_N / VALUES_PER_BYTE) + j * DIM;
#endif
      uint32_t sp_addr  = b_base + (k * tiles_J + j) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
      gemmini_fence();
    }
  }

#if CYCLES
  uint64_t t1 = read_cycles();
#endif
  int SPAD_DEST = SPAD_DEST_ROW(tiles_I, tiles_K);
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base, SPAD_ROWS, 0, SPAD_DEST,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      0x38);                        // skip ldA/ldB/ldD; keep compute + acc->spad store

  // Readback: output in the internal spad (BF16: 2 bytes/elem, fp4: 1/2 byte/elem) -> DRAM.
  gemmini_fence();
#if CYCLES
  uint64_t t2 = read_cycles();
#endif
  gemmini_config_st(DIM * sizeof(uint8_t));
  uint8_t *c_base = (uint8_t *) C_hw;
  int total_spad_rows = OUT_FP4 ? MATMUL_M * MATMUL_N / 2 / DIM : MATMUL_M * MATMUL_N * 2 / DIM;
  for (int r = 0; r < total_spad_rows; r += DIM) {
    gemmini_extended_mvout(c_base + r * DIM, SPAD_DEST + r, DIM, DIM);
  }
  gemmini_fence();
#if CYCLES
  uint64_t t3 = read_cycles();
  printf("CYCLES load=%llu compute=%llu store=%llu total=%llu\n", (unsigned long long)(t1 - t0),
         (unsigned long long)(t2 - t1), (unsigned long long)(t3 - t2), (unsigned long long)(t3 - t0));
#endif

#if OUT_FP4
  // fp4 requant: 2 output rows per byte (bits[3:0] = row 2r, bits[7:4] = row 2r+1), HW-tiled
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
  if (errors == 0 && scale_errors == 0) printf(MSG " PASSED (no mismatches).\n");
  else printf(MSG " FAILED: %d code, %d scale mismatches.\n", errors, scale_errors);
  return (errors == 0 && scale_errors == 0) ? 0 : 1;
#else
  int errors = 0;
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t got = C_hw[i][j];
      uint64_t exp = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                     ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                     ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                     ((uint64_t)C_out_bf16[i][j*4 + 0]);
      if (got != exp) {
        for (int lane = 0; lane < 4; lane++) {
          uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
          uint16_t exp_bf16 = C_out_bf16[i][j * 4 + lane];
          if (got_bf16 != exp_bf16) {
            printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n",
                   i, j * 4 + lane, got_bf16, exp_bf16);
            errors++;
          }
        }
      }
    }
  }
  if (errors == 0) printf(MSG " PASSED (no mismatches).\n");
  else             printf(MSG " FAILED with %d mismatches.\n", errors);
  return errors == 0 ? 0 : 1;
#endif
}
