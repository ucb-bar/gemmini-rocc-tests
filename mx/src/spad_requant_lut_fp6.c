// SPAD_REQUANT LUT test (FP6 E3M2): the BF16 matmul result C_out_bf16 of matmul_data_mx_lut_e3m2_128x128 is mvin'd
// row-major into the scratchpad and requantized to 4-bit indices into the per-row output codebooks C_lut (G = 1),
// two rows per byte (row 2p low nibble), flat [M/2][N] bytes and then operand-A tiled, + E8M0 scales in DRAM [M][N/32].
// Checked bit-exact against the header's in-loop requant golden (C_proj_hw, C_scales_row), which requantizes the same
// BF16 values with the same codebooks -- so the SPAD_REQUANT and in-loop encoders must agree.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/matmul_data_mx_lut_e3m2_128x128.h"


#define M MATMUL_M
#define N MATMUL_N
#define GN (N / 32)
#define G 1
#define SP_SRC   0x0000   // bank 0: M*N/8 = 2048 rows
#define SP_FLAT  (1 * BANK_ROWS)   // bank 1 (0x1000 at DIM16): M*N/32 = 512 rows
#define SP_TILED (2 * BANK_ROWS)   // bank 2 (0x2000 at DIM16): (M/32)*(N/16)*16 = 512 rows

static uint32_t C_lut_al[64][3] __attribute__((aligned(64)));
static uint16_t X[M][N] __attribute__((aligned(64)));
static uint8_t codes_hw[M * N / 2] __attribute__((aligned(64)));
static uint8_t scales_hw[M * GN] __attribute__((aligned(64)));
static uint8_t scales_hw2[M * GN] __attribute__((aligned(64)));

static int check_scales(const char *name, const uint8_t *hw) {
  int bad = 0;
  for (int m = 0; m < M; m++)
    for (int b = 0; b < GN; b++)
      if (hw[b * M + m] != C_scales_row[b][m] && bad++ < 4)
        printf("  %s scale[%d][%d]: hw %02x ref %02x\n", name, m, b, hw[b * M + m], C_scales_row[b][m]);
  return bad;
}

static int check_codes(const char *name, int tiled) {
  int bad = 0;
  for (int p = 0; p < M / 2; p++)
    for (int n = 0; n < N; n++) {
      const int row = tiled ? ((p / DIM) * (N / DIM) + n / DIM) * DIM + p % DIM : p * (N / DIM) + n / DIM;
      const uint8_t got = codes_hw[row * DIM + n % DIM], exp = C_proj_hw[p][n];
      if (got != exp && bad++ < 6) printf("  %s byte[%d][%d]: hw %02x ref %02x\n", name, p, n, got, exp);
    }
  return bad;
}

int main() {
  memcpy(C_lut_al, C_lut, sizeof(C_lut_al));
  memcpy(X, C_out_bf16, sizeof(X));
  gemmini_flush(0);
  gemmini_mx_load_lut_dt((uint64_t)&C_lut_al[0][0], M >> G, 2, 6);   // output codebooks (act-out LUT), flat
  gemmini_fence();
  gemmini_config_ld(DIM);
  gemmini_config_st(DIM);
  memset(scales_hw, 0xa5, sizeof(scales_hw));
  memset(scales_hw2, 0xa5, sizeof(scales_hw2));
  for (int r = 0; r < M * N * 2 / DIM; r += DIM) gemmini_extended_mvin((uint8_t *)X + r * DIM, SP_SRC + r, DIM, DIM);

  // ---- flat: byte (p, n) at SP_FLAT + p*N/DIM + n/DIM, col n%DIM ----
  uint64_t t0 = read_cycles();
  gemmini_spad_requant_lut(SP_FLAT, SP_SRC, M, N, 0, (uint64_t)scales_hw, 0, /*fp6*/ 1, /*alt*/ 0, G);
  memset(codes_hw, 0xa5, sizeof(codes_hw));
  for (int r = 0; r < M * N / (2 * DIM); r += DIM) gemmini_extended_mvout(codes_hw + r * DIM, SP_FLAT + r, DIM, DIM);
  gemmini_fence();
  uint64_t t1 = read_cycles();
  const int bad_flat = check_codes("flat", 0), bad_sc = check_scales("flat", scales_hw);

  // ---- operand-A tiled: tile (i, kt) row r (= row pair) at SP_TILED + (i*N/DIM + kt)*DIM + r ----
  gemmini_spad_requant_lut(SP_TILED, SP_SRC, M, N, 1, (uint64_t)scales_hw2, 0, 1, 0, G);
  memset(codes_hw, 0xa5, sizeof(codes_hw));
  for (int r = 0; r < M * N / (2 * DIM); r += DIM) gemmini_extended_mvout(codes_hw + r * DIM, SP_TILED + r, DIM, DIM);
  gemmini_fence();
  const int bad_tiled = check_codes("tiled", 1), bad_sc2 = check_scales("tiled", scales_hw2);

  printf("spad_requant_lut_fp6 %dx%d: flat %d, tiled %d byte mismatches; scales %d, %d mismatches; %llu cycles (requant+mvout)\n",
         M, N, bad_flat, bad_tiled, bad_sc, bad_sc2, (unsigned long long)(t1 - t0));
  const int fail = bad_flat || bad_tiled || bad_sc || bad_sc2;
  printf("spad_requant_lut_fp6 %s\n", fail ? "FAILED" : "PASSED");
  return fail;
}
