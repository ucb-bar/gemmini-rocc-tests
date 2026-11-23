// Integer-exact MX E4M3 block quantizer (no libm): the requantizer's E4M3-single convention, as in the Spike golden.
// Block scale E8M0 = clamp(exponent field of the block's max |finite x|, 104 (FLT_EPSILON), 254); element code =
// E4M3 RNE of x / 2^(scale - 127), saturating to +-448, subnormals kept. Finite inputs only.
#ifndef MX_E4M3_REF_H
#define MX_E4M3_REF_H

#include <stdint.h>
#include <string.h>

static inline double mxr_pow2(int e) { uint64_t u = (uint64_t)(e + 1023) << 52; double d; memcpy(&d, &u, 8); return d; }
static inline double mxr_bf16(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return (double)f; }
static inline int64_t mxr_rne(double x) {   // x >= 0
  int64_t f = (int64_t)x; double fr = x - (double)f;
  if (fr > 0.5 || (fr == 0.5 && (f & 1))) f++;
  return f;
}

static inline uint8_t mxr_e4m3(double v) {
  if (v == 0.0) return 0;
  uint8_t s = v < 0 ? 0x80 : 0;
  double av = v < 0 ? -v : v;
  uint64_t u; memcpy(&u, &av, 8);
  int E = (int)((u >> 52) & 0x7ff) - 1023;
  if (av <= mxr_pow2(-150)) return 0;   // x / 2^(scale-127) underflows FP32 -> +0 (sign dropped, as the RTL)
  if (E < -6) {
    int64_t k = mxr_rne(av * 512.0);
    if (k <= 0) return s;
    if (k >= 8) return (uint8_t)(s | 8);
    return (uint8_t)(s | k);
  }
  if (E > 8) return (uint8_t)(s | 0x7E);
  int64_t k = mxr_rne((av / mxr_pow2(E) - 1.0) * 8.0);
  if (k >= 8) { E++; k = 0; if (E > 8) { E = 8; k = 6; } }
  else { int hi = E == 8 ? 6 : 7; if (k > hi) k = hi; }
  return (uint8_t)(s | ((E + 7) << 3) | k);
}

static inline uint8_t mxr_scale(const uint16_t *x, int n) {
  int emax = 0;
  for (int k = 0; k < n; k++) { int e = (x[k] >> 7) & 0xff; if (e != 0xff && e > emax) emax = e; }
  return (uint8_t)(emax < 104 ? 104 : emax > 254 ? 254 : emax);
}

static inline void mxr_quant_block(const uint16_t *x, uint8_t *codes, uint8_t *scale) {
  uint8_t sc = mxr_scale(x, 32);
  double inv = mxr_pow2(127 - (int)sc);
  for (int k = 0; k < 32; k++) codes[k] = mxr_e4m3(mxr_bf16(x[k]) * inv);
  *scale = sc;
}

#endif
