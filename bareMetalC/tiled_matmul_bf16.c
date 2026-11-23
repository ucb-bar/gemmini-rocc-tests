// See LICENSE for license details.

// BF16 tiled matmul for the plain BF16 Gemmini (BF16Dim*BaselineConfig, gemmini_params_bf16_dim*_vpu.h), two phases:
// rounding (random BF16 data) and layout (multi-tile, padded, exact integer data; see L_DIM_*).
// C = A * B + D with BF16 A, B, C and an FP32 bias D, checked bit-exactly against a CPU model of the hardware:
//   - each mesh column sums one DIM-long K chunk through BF16 FMA PEs (hardfloat MulAddRecFN, one RNE rounding to BF16
//     per PE, k ascending, starting from +0),
//   - the FP32 accumulator adds the chunks in K order on top of D,
//   - mvout rounds the FP32 accumulator to BF16 (RNE).
// elem_t is the raw BF16 bit pattern (uint16_t) in the generated header; acc_t is float.

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#if !defined(ELEM_T_IS_FLOAT) || ELEM_T_EXP_BITS != 8 || ELEM_T_SIG_BITS != 8
#error "tiled_matmul_bf16 needs a BF16 Gemmini header (gemmini_params_bf16_dim*_vpu.h)"
#endif

// RTL sim: the CPU reference (software BF16 FMAs, I*J*K of them) dominates the run, so keep it small while still
// covering two K chunks into the accumulator and, at DIM 16 / 32, a partial row tile. Spike runs the larger problem.
#define MAT_DIM_I 4
#define MAT_DIM_K (2 * DIM)
#define MAT_DIM_J DIM

#define MAX_PRINTED_ERRORS 10

static uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, sizeof(u)); return u; }
static float f32_from_bits(uint32_t u) { float f; memcpy(&f, &u, sizeof(f)); return f; }
static float bf16_to_f32(uint16_t b) { return f32_from_bits((uint32_t)b << 16); }

// RNE FP32 -> BF16, keeping -0 (as hardfloat's RecFNToRecFN)
static uint16_t f32_to_bf16(float f) {
  const uint32_t u = f32_bits(f);
  if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu)) return (uint16_t)((u >> 16) | 0x40);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}

static int msb64(uint64_t x) { int m = -1; while (x) { x >>= 1; m++; } return m; }

// Exact a*b + c for BF16 a, b, c, rounded once to BF16 (RNE). Inputs are finite here (the test data has no Inf/NaN).
static uint16_t bf16_fma(uint16_t a, uint16_t b, uint16_t c) {
  // value = (-1)^s * m * 2^e, m an integer significand (hidden bit included for normals)
  const int sa = a >> 15, sb = b >> 15, sc = c >> 15;
  const int ea = (a >> 7) & 0xFF, eb = (b >> 7) & 0xFF, ec = (c >> 7) & 0xFF;
  const uint64_t ma = ea ? (0x80 | (a & 0x7F)) : (a & 0x7F);
  const uint64_t mb = eb ? (0x80 | (b & 0x7F)) : (b & 0x7F);
  const uint64_t mc = ec ? (0x80 | (c & 0x7F)) : (c & 0x7F);
  const int sp = sa ^ sb;
  uint64_t mp = ma * mb;
  int ep = (ea ? ea : 1) + (eb ? eb : 1) - 2 * (127 + 7);
  const int e_c = (ec ? ec : 1) - (127 + 7);

  if (mp == 0 && mc == 0) return (uint16_t)((sp & sc) << 15);   // +0 unless both are -0 (RNE)
  if (mp == 0) return c;

  // align: the operand with the higher LSB exponent is shifted left (exact up to 40 bits); a farther-away operand only
  // matters as a sticky bit, so it is replaced by 1 unit below the shifted operand's LSB
  uint64_t mx, my; int sx, sy, ex, ey;
  if (mc == 0 || (mp != 0 && ep >= e_c)) { mx = mp; sx = sp; ex = ep; my = mc; sy = sc; ey = e_c; }
  else { mx = mc; sx = sc; ex = e_c; my = mp; sy = sp; ey = ep; }
  int e;
  if (my == 0) { e = ex; my = 0; }
  else if (ex - ey <= 40) { mx <<= (ex - ey); e = ey; }
  else { mx <<= 40; my = 1; e = ex - 40; }

  int64_t s = (sx ? -(int64_t)mx : (int64_t)mx) + (sy ? -(int64_t)my : (int64_t)my);
  if (s == 0) return 0;   // exact cancellation is +0 under RNE
  const int sign = s < 0;
  uint64_t m = sign ? (uint64_t)(-s) : (uint64_t)s;

  // keep 8 significant bits (fewer for subnormals: the LSB exponent may not go below 2^-133)
  const int top = msb64(m);                 // value = m * 2^e, leading bit at 2^(top + e)
  int lsb_exp = top + e - 7;
  if (lsb_exp < -133) lsb_exp = -133;
  const int shift = lsb_exp - e;
  uint64_t q;
  if (shift <= 0) {
    q = m << (-shift);
  } else if (shift > 63) {
    q = 0;   // far below the smallest subnormal: rounds to zero (half-way impossible here)
  } else {
    q = m >> shift;
    const uint64_t rem = m & ((1ull << shift) - 1);
    const uint64_t half = 1ull << (shift - 1);
    if (rem > half || (rem == half && (q & 1))) q++;
  }
  // q * 2^lsb_exp, q <= 0x100 (a carry out of 8 bits moves to the next binade)
  int exp_field;
  uint32_t frac;
  if (q == 0) return (uint16_t)(sign << 15);
  if (q >= 0x100) { q >>= 1; lsb_exp++; }
  if (q & 0x80) {
    exp_field = lsb_exp + 7 + 127;
    frac = q & 0x7F;
    if (exp_field >= 0xFF) return (uint16_t)((sign << 15) | 0x7F80);   // overflow to Inf
  } else {
    exp_field = 0;   // subnormal (lsb_exp == -133)
    frac = q & 0x7F;
  }
  return (uint16_t)((sign << 15) | (exp_field << 7) | frac);
}

static uint32_t lcg_state = 12345;
static uint32_t lcg(void) { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state >> 8; }

// random BF16 in roughly [-16, 16]: random sign, exponent 2^-4..2^3, random 7-bit fraction
static uint16_t rand_bf16(void) {
  const uint32_t r = lcg();
  const uint32_t sign = r & 1;
  const uint32_t exp = 127 - 4 + ((r >> 1) % 8);
  const uint32_t frac = (r >> 4) & 0x7F;
  return (uint16_t)((sign << 15) | (exp << 7) | frac);
}

// small integer as BF16 / FP32 (|v| <= 256 is exact in BF16)
static uint16_t int_to_bf16(int v) { return f32_to_bf16((float)v); }

static int check(const char *phase, int rows, int cols, const elem_t *C, const elem_t *gold) {
  int errors = 0;
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      if (C[i * cols + j] != gold[i * cols + j]) {
        if (errors < MAX_PRINTED_ERRORS)
          printf("%s MISMATCH [%d][%d]: got 0x%04x, expected 0x%04x\n", phase, i, j, C[i * cols + j], gold[i * cols + j]);
        errors++;
      }
  if (errors) printf("%s FAIL: %d of %d outputs differ\n", phase, errors, rows * cols);
  else printf("%s PASS\n", phase);
  return errors;
}

static void run_matmul(int I, int J, int K, const elem_t *A, const elem_t *B, const acc_t *D, elem_t *C) {
  const unsigned long start = read_cycles();
  tiled_matmul_auto(I, J, K,
          (elem_t*)A, (elem_t*)B, (acc_t*)D, C,
          K, J, J, J,
          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
          false, false,
          false, false,
          0,
          WS);
  gemmini_fence();
  printf("Cycles taken: %lu\n", read_cycles() - start);
}

// Phase 2 (layout): 2 tiles in I and J, 3 in K, none a multiple of DIM, so the loop's tile addressing, padding and
// K accumulation are all exercised. Values in {-1, 0, 1} (bias -4..4) keep every BF16 FMA and FP32 add exact, so the
// reference is a plain integer matmul and stays cheap in RTL simulation.
#define L_DIM_I (DIM + 3)
#define L_DIM_J (DIM + 5)
#define L_DIM_K (2 * DIM + 7)

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif
  gemmini_flush(0);
  int errors = 0;

  // Phase 1 (rounding): random BF16 data, bit-exact against the BF16 FMA / FP32 accumulator model
  {
    printf("BF16 rounding: DIM %d, I %d, J %d, K %d\n", DIM, MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
    static elem_t A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static acc_t D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

    for (size_t i = 0; i < MAT_DIM_I; i++)
      for (size_t k = 0; k < MAT_DIM_K; k++)
        A[i][k] = rand_bf16();
    for (size_t k = 0; k < MAT_DIM_K; k++)
      for (size_t j = 0; j < MAT_DIM_J; j++)
        B[k][j] = rand_bf16();
    for (size_t i = 0; i < MAT_DIM_I; i++)
      for (size_t j = 0; j < MAT_DIM_J; j++)
        D[i][j] = bf16_to_f32(rand_bf16());

    run_matmul(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K, &A[0][0], &B[0][0], &D[0][0], &C[0][0]);

    for (size_t i = 0; i < MAT_DIM_I; i++)
      for (size_t j = 0; j < MAT_DIM_J; j++) {
        float acc = D[i][j];
        for (size_t k0 = 0; k0 < MAT_DIM_K; k0 += DIM) {
          uint16_t s = 0;
          for (size_t k = k0; k < k0 + DIM && k < MAT_DIM_K; k++)
            s = bf16_fma(A[i][k], B[k][j], s);
          acc = acc + bf16_to_f32(s);
        }
        gold[i][j] = f32_to_bf16(acc);
      }
    errors += check("rounding", MAT_DIM_I, MAT_DIM_J, &C[0][0], &gold[0][0]);
  }

  // Phase 2 (layout)
  {
    printf("BF16 layout: DIM %d, I %d, J %d, K %d\n", DIM, L_DIM_I, L_DIM_J, L_DIM_K);
    static int8_t Ai[L_DIM_I][L_DIM_K], Bi[L_DIM_K][L_DIM_J];
    static elem_t A[L_DIM_I][L_DIM_K] row_align(1);
    static elem_t B[L_DIM_K][L_DIM_J] row_align(1);
    static elem_t C[L_DIM_I][L_DIM_J] row_align(1);
    static acc_t D[L_DIM_I][L_DIM_J] row_align_acc(1);
    static elem_t gold[L_DIM_I][L_DIM_J];

    for (int i = 0; i < L_DIM_I; i++)
      for (int k = 0; k < L_DIM_K; k++) { Ai[i][k] = (int8_t)(lcg() % 3) - 1; A[i][k] = int_to_bf16(Ai[i][k]); }
    for (int k = 0; k < L_DIM_K; k++)
      for (int j = 0; j < L_DIM_J; j++) { Bi[k][j] = (int8_t)(lcg() % 3) - 1; B[k][j] = int_to_bf16(Bi[k][j]); }
    for (int i = 0; i < L_DIM_I; i++)
      for (int j = 0; j < L_DIM_J; j++) D[i][j] = (float)((int)(lcg() % 9) - 4);

    run_matmul(L_DIM_I, L_DIM_J, L_DIM_K, &A[0][0], &B[0][0], &D[0][0], &C[0][0]);

    for (int i = 0; i < L_DIM_I; i++)
      for (int j = 0; j < L_DIM_J; j++) {
        int acc = (int)D[i][j];
        for (int k = 0; k < L_DIM_K; k++) acc += Ai[i][k] * Bi[k][j];
        gold[i][j] = int_to_bf16(acc);
      }
    errors += check("layout", L_DIM_I, L_DIM_J, &C[0][0], &gold[0][0]);
  }

  if (errors) exit(1);
  printf("PASS\n");
  exit(0);
}
