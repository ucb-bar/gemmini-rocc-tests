// SPAD_REQUANT FP4 test (MxFp4E3M2E4M3VpuGemminiRocketConfig / Spike): a row-major BF16 tile mvin'd into the scratchpad is
// requantized into FP4 (E2M1) codes, two rows per byte (row 2r low nibble), flat [M/2][N] bytes and then FP4
// operand-A tiled, + E8M0 scales in DRAM [M][N/32]. Checked bit-exact against the E8M0 block scale of mx_e4m3_ref.h
// and the requantizer's BF16 -> E3M1 -> E2M1 element. No fences between mvin, requant and mvout.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/mx_e4m3_ref.h"

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: SPAD_REQUANT config / Spike-only test\n"); return 0; }
#else

#define M 64
#define N 128
#define GN (N / 32)
#define SP_SRC   0x0000   // bank 0: M*N/8 = 1024 rows
#define SP_FLAT  (1 * BANK_ROWS)   // bank 1 (0x1000 at DIM16): M*N/32 = 256 rows
#define SP_TILED (2 * BANK_ROWS)   // bank 2 (0x2000 at DIM16): (M/32)*(N/16)*16 = 256 rows

static uint16_t X[M][N] __attribute__((aligned(64)));
static uint8_t codes_hw[M * N / 2] __attribute__((aligned(64)));
static uint8_t scales_hw[M * GN] __attribute__((aligned(64)));
static uint8_t scales_hw2[M * GN] __attribute__((aligned(64)));
static uint8_t codes_ref[M][N], scales_ref[M][GN];

static uint32_t lcg = 4343;
static uint32_t rnd(void) { lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }

// BF16 bits -> E3M1 (RNE) -> E2M1 code, as the requantizer (fp4_matmul_model.hw_bf16_to_e2m1)
static uint8_t fp4_code(uint16_t bf) {
  int s = bf >> 15, E = (bf >> 7) & 0xff, Mt = bf & 0x7f, e = E - 127;
  if (E == 0) return 0;
  if (E == 255) return (uint8_t)((s << 3) | 7);
  double mag;
  if (e >= -2) {
    int S = 128 + Mt, q = (S >> 6) & 1, r = (S >> 5) & 1, st = (S & 0x1f) != 0;
    int sig = q + (r & (st | q)), eo = sig >= 2 ? e + 1 : e, mo = sig >= 2 ? 0 : sig;
    if (eo > 3) return (uint8_t)((s << 3) | 7);
    mag = (1 + 0.5 * mo) * mxr_pow2(eo);
  } else if (e == -3) mag = Mt >= 64 ? 0.25 : 0.125;
  else if (e == -4) mag = Mt > 0 ? 0.125 : 0.0;
  else mag = 0.0;
  int mc = mag <= 0.25 ? 0 : mag <= 0.5 ? 1 : mag <= 1 ? 2 : mag == 1.5 ? 3 : mag == 2 ? 4 : mag == 3 ? 5 : mag == 4 ? 6 : 7;
  return mc ? (uint8_t)((s << 3) | mc) : 0;
}

static int check_scales(const char *name, const uint8_t *hw) {
  int bad = 0;
  for (int m = 0; m < M; m++)
    for (int b = 0; b < GN; b++)
      if (hw[b * M + m] != scales_ref[m][b] && bad++ < 4)
        printf("  %s scale[%d][%d]: hw %02x ref %02x\n", name, m, b, hw[b * M + m], scales_ref[m][b]);
  return bad;
}

int main() {
  // every 32-value block gets its own magnitude range; block (0,0) is all zero, (3,1) mixes in tiny values
  for (int m = 0; m < M; m++)
    for (int b = 0; b < GN; b++) {
      int ebase = 100 + (int)(rnd() % 50);
      for (int k = 0; k < 32; k++) {
        int e = ebase - (int)(rnd() % ((m == 3 && b == 1) ? 20 : 6));
        X[m][32 * b + k] = (uint16_t)(((rnd() & 1) << 15) | (e << 7) | (rnd() & 0x7f));
        if (m == 0 && b == 0) X[m][32 * b + k] = 0;
      }
    }
  for (int m = 0; m < M; m++)
    for (int b = 0; b < GN; b++) {
      uint8_t sc = mxr_scale(&X[m][32 * b], 32);
      scales_ref[m][b] = sc;
      for (int k = 0; k < 32; k++) {
        float x = (float)(mxr_bf16(X[m][32 * b + k]) * mxr_pow2(127 - (int)sc));
        uint32_t u; memcpy(&u, &x, 4);
        codes_ref[m][32 * b + k] = fp4_code((uint16_t)(u >> 16));
      }
    }

  gemmini_flush(0);
  gemmini_config_ld(DIM);
  gemmini_config_st(DIM);
  memset(scales_hw, 0xa5, sizeof(scales_hw));
  memset(scales_hw2, 0xa5, sizeof(scales_hw2));

  // ---- flat: byte (m/2, n) at SP_FLAT + (m/2)*N/DIM + n/DIM, col n%DIM ----
  for (int r = 0; r < M * N * 2 / DIM; r += DIM) gemmini_extended_mvin((uint8_t *)X + r * DIM, SP_SRC + r, DIM, DIM);
  uint64_t t0 = read_cycles();
  gemmini_spad_requant_fp4(SP_FLAT, SP_SRC, M, N, 0, (uint64_t)scales_hw, 0);
  memset(codes_hw, 0xa5, sizeof(codes_hw));
  for (int r = 0; r < M * N / (2 * DIM); r += DIM) gemmini_extended_mvout(codes_hw + r * DIM, SP_FLAT + r, DIM, DIM);
  gemmini_fence();
  uint64_t t1 = read_cycles();
  int bad_flat = 0;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      uint8_t byte = codes_hw[(m / 2) * N + n], got = (m & 1) ? byte >> 4 : byte & 0xf;
      if (got != codes_ref[m][n] && bad_flat++ < 4)
        printf("  flat code[%d][%d]: hw %x ref %x\n", m, n, got, codes_ref[m][n]);
    }
  int bad_sc = check_scales("flat", scales_hw);

  // ---- FP4 operand-A tiled: tile (i, kt) row r (= row pair) at SP_TILED + (i*N/DIM + kt)*DIM + r ----
  gemmini_spad_requant_fp4(SP_TILED, SP_SRC, M, N, 1, (uint64_t)scales_hw2, 0);
  memset(codes_hw, 0xa5, sizeof(codes_hw));
  for (int r = 0; r < M * N / (2 * DIM); r += DIM) gemmini_extended_mvout(codes_hw + r * DIM, SP_TILED + r, DIM, DIM);
  gemmini_fence();
  int bad_tiled = 0;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      int p = m / 2, row = ((p / DIM) * (N / DIM) + n / DIM) * DIM + p % DIM;
      uint8_t byte = codes_hw[row * DIM + n % DIM], got = (m & 1) ? byte >> 4 : byte & 0xf;
      if (got != codes_ref[m][n] && bad_tiled++ < 4)
        printf("  tiled code[%d][%d]: hw %x ref %x\n", m, n, got, codes_ref[m][n]);
    }
  int bad_sc2 = check_scales("tiled", scales_hw2);

  printf("spad_requant_fp4 %dx%d: flat %d, tiled %d code mismatches; scales %d, %d mismatches; %llu cycles (requant+mvout)\n",
         M, N, bad_flat, bad_tiled, bad_sc, bad_sc2, (unsigned long long)(t1 - t0));
  int fail = bad_flat || bad_tiled || bad_sc || bad_sc2;
  printf("spad_requant_fp4 %s\n", fail ? "FAILED" : "PASSED");
  return fail;
}
#endif
