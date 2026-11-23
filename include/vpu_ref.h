// Bit-exact reference of the Gemmini VPU (src/main/scala/gemmini/vpu/{Vpu,VpuMath}.scala).
// Shared by the bare-metal tests (golden on the host) and the Spike model (libgemmini).
// One VPU row = one scratchpad row = VPU_LANES (8, or 4 at DIM=8) BF16 lanes; lane l is bytes [2l, 2l+1] (little-endian) of the row.
#ifndef VPU_REF_H
#define VPU_REF_H

#include <stdint.h>
#include <string.h>

// the bare-metal tests build with -ffast-math: keep IEEE semantics (no reassociation of the FP32 sum tree)
#pragma GCC push_options
#pragma GCC optimize("no-fast-math")

#ifndef VPU_LANES   // one DIM-byte spad row of BF16: 8 at DIM 16, 4 on the 8x8 mesh
#define VPU_LANES 8
#endif

enum { VPU_ADD = 0, VPU_SUB = 1, VPU_MUL = 2, VPU_ADDS = 3, VPU_MULS = 4,
       VPU_EXP = 5, VPU_RCP = 6, VPU_RSQRT = 7, VPU_RMAX = 8, VPU_RSUM = 9, VPU_RAMAX = 10, VPU_MAX = 11,
       VPU_EXPSUB = 12,   // optional (VpuParams.expSub): exp(src1 - src2)
       VPU_EXPSUM = 13 }; // optional (VpuParams.expSum): EXPSUB, plus row sums of the result at dst2 (imm[13:0])
static inline int vpu_is_reduction(int op) { return op == VPU_RMAX || op == VPU_RSUM || op == VPU_RAMAX; }
static inline int vpu_uses_src2(int op) { return op <= VPU_MUL || op == VPU_MAX || op == VPU_EXPSUB || op == VPU_EXPSUM; }

// LUTs = the RTL's Scala tables (round-half-up of the double value)
static const uint32_t vpu_rcp_tab[128] = {65536, 65028, 64528, 64035, 63550, 63072, 62602, 62138, 61681, 61231, 60787, 60350, 59919, 59494, 59075, 58662, 58254, 57852, 57456, 57065, 56680, 56299, 55924, 55554, 55188, 54828, 54471, 54120, 53773, 53431, 53092, 52759, 52429, 52103, 51782, 51464, 51150, 50840, 50534, 50231, 49932, 49637, 49345, 49056, 48771, 48489, 48210, 47935, 47663, 47393, 47127, 46864, 46603, 46346, 46091, 45839, 45590, 45344, 45100, 44859, 44620, 44384, 44151, 43919, 43691, 43464, 43240, 43019, 42799, 42582, 42367, 42154, 41943, 41734, 41528, 41323, 41121, 40920, 40721, 40525, 40330, 40137, 39946, 39756, 39569, 39383, 39199, 39017, 38836, 38657, 38480, 38304, 38130, 37958, 37787, 37617, 37449, 37283, 37118, 36954, 36792, 36631, 36472, 36314, 36158, 36003, 35849, 35696, 35545, 35395, 35246, 35099, 34953, 34808, 34664, 34521, 34380, 34239, 34100, 33962, 33825, 33689, 33554, 33421, 33288, 33157, 33026, 32897};
static const uint32_t vpu_sqrt_tab[128] = {65536, 65792, 66046, 66300, 66552, 66804, 67054, 67304, 67553, 67801, 68048, 68294, 68539, 68784, 69027, 69270, 69511, 69752, 69992, 70232, 70470, 70708, 70945, 71181, 71416, 71651, 71885, 72118, 72350, 72581, 72812, 73042, 73271, 73500, 73728, 73955, 74182, 74408, 74633, 74857, 75081, 75304, 75527, 75748, 75969, 76190, 76410, 76629, 76848, 77066, 77283, 77500, 77716, 77932, 78147, 78361, 78575, 78788, 79001, 79213, 79424, 79635, 79846, 80056, 80265, 80474, 80682, 80890, 81097, 81303, 81509, 81715, 81920, 82125, 82329, 82532, 82735, 82938, 83140, 83341, 83542, 83743, 83943, 84143, 84342, 84540, 84739, 84936, 85134, 85331, 85527, 85723, 85918, 86113, 86308, 86502, 86696, 86889, 87082, 87275, 87467, 87658, 87849, 88040, 88231, 88420, 88610, 88799, 88988, 89176, 89364, 89552, 89739, 89926, 90112, 90298, 90484, 90669, 90854, 91038, 91222, 91406, 91589, 91772, 91955, 92137, 92319, 92501};
static const uint32_t vpu_exp_tab[32] = {65536, 66971, 68438, 69936, 71468, 73032, 74632, 76266, 77936, 79642, 81386, 83169, 84990, 86851, 88752, 90696, 92682, 94711, 96785, 98905, 101070, 103283, 105545, 107856, 110218, 112631, 115098, 117618, 120194, 122825, 125515, 128263};

static inline double vpu_bf16_to_d(uint16_t b) {
  uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return (double)f;
}

// RNE to BF16 incl. subnormals; NaN -> 0x7fc0 (hardfloat canonical NaN)
static inline uint16_t vpu_d_to_bf16(double d) {
  uint64_t u; memcpy(&u, &d, 8);
  uint16_t sign = (uint16_t)((u >> 48) & 0x8000);
  int de = (int)((u >> 52) & 0x7ff);
  uint64_t frac = u & ((1ULL << 52) - 1);
  if (de == 0x7ff) return frac ? 0x7fc0 : (uint16_t)(sign | 0x7f80);
  if (de == 0) return sign;   // |d| < 2^-1022: far below the BF16 range
  int e = de - 1023;
  uint64_t mant = (1ULL << 52) | frac;
  int s = (e >= -126) ? 45 : (-81 - e);
  uint64_t n;
  if (s >= 54) n = 0;
  else {
    n = mant >> s;
    uint64_t rem = mant & ((1ULL << s) - 1), half = 1ULL << (s - 1);
    if (rem > half || (rem == half && (n & 1))) n++;
  }
  uint32_t bits = (e >= -126) ? (uint32_t)(((e + 127) << 7) + (int)(n - 128)) : (uint32_t)n;
  if (bits >= 0x7f80) bits = 0x7f80;
  return (uint16_t)(sign | bits);
}

static inline float vpu_bf16_to_f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t vpu_f_to_bf16(float f) { return vpu_d_to_bf16((double)f); }

static inline uint16_t vpu_add(uint16_t a, uint16_t b) { return vpu_d_to_bf16(vpu_bf16_to_d(a) + vpu_bf16_to_d(b)); }
static inline uint16_t vpu_sub(uint16_t a, uint16_t b) { return vpu_d_to_bf16(vpu_bf16_to_d(a) - vpu_bf16_to_d(b)); }
static inline uint16_t vpu_mul(uint16_t a, uint16_t b) { return vpu_d_to_bf16(vpu_bf16_to_d(a) * vpu_bf16_to_d(b)); }

static inline uint16_t vpu_ord(uint16_t x) { return (x & 0x8000) ? (uint16_t)~x : (uint16_t)(x ^ 0x8000); }
static inline uint16_t vpu_max(uint16_t a, uint16_t b) { return vpu_ord(a) > vpu_ord(b) ? a : b; }

static inline int vpu_log2_17(uint32_t y) { int h = 0; while (y >> (h + 1)) h++; return h; }

static inline uint16_t vpu_rcp(uint16_t x) {
  int neg = x >> 15, e = (x >> 7) & 0xff, f = x & 0x7f;
  uint32_t y = vpu_rcp_tab[f];
  int hi = vpu_log2_17(y);
  int bfExp = 254 + (hi - 16) - e;
  uint32_t frac = (y >> (hi > 7 ? hi - 7 : 0)) & 0x7f;
  uint16_t sgn = (uint16_t)(neg << 15);
  if (e == 255) return sgn;
  if (e == 0 || bfExp >= 255) return (uint16_t)(sgn | 0x7f80);
  if (bfExp <= 0) return sgn;
  return (uint16_t)(sgn | ((bfExp & 0xff) << 7) | frac);
}

static inline uint16_t vpu_sqrt(uint16_t x) {
  int e = (x >> 7) & 0xff, f = x & 0x7f;
  uint32_t t = vpu_sqrt_tab[f];
  uint32_t base = !(e & 1) ? (uint32_t)(((uint64_t)t * 92682) >> 16) & 0x1ffff : t;
  int hi = vpu_log2_17(base);
  int sum = hi - 16 + e - 127;
  int bfExp = 127 + (sum >> 1);   // arithmetic shift (floor)
  uint32_t frac = (base >> (hi > 7 ? hi - 7 : 0)) & 0x7f;
  if (e == 0 || (e == 255 && f != 0)) return 0;
  if ((e == 255 && f == 0) || bfExp >= 255) return 0x7f80;
  return (uint16_t)(((bfExp & 0xff) << 7) | frac);
}

static inline uint16_t vpu_rsqrt(uint16_t x) { return vpu_rcp(vpu_sqrt(x)); }

static inline uint16_t vpu_exp(uint16_t x) {
  int sign = x >> 15, e = (x >> 7) & 0xff, f = x & 0x7f;
  if (e == 255 && f) return (uint16_t)((sign << 15) | 0x7f80 | ((!(f & 0x40)) << 6) | 0x3f);   // NaN
  if (e == 0) return 0x3f80;                                   // 0 and subnormals -> 1
  if (e == 255) return sign ? 0 : 0x7f80;                      // -inf, +inf
  if (!sign && (e > 0x85 || (e == 0x85 && f > 0x31))) return 0x7f80;
  if (sign && e >= 0x86) return 0;
  // Q9.12 fixed point of x, * 1/ln2 (5909 / 2^12) -> k + r/2^12
  int shift = e - 127 + 5;
  int64_t mag = shift < 0 ? (int64_t)((0x80 | f) >> (-shift > 20 ? 20 : -shift)) : (int64_t)((0x80 | f) << shift);
  int64_t q = sign ? -mag : mag;
  int64_t prod = (q * 5909) >> 12;   // floor
  int k = (int)(prod >> 12);
  int r = (int)(prod & 0xfff);
  int addr = r >> 7, rLow = r & 0x7f;
  uint32_t y0 = vpu_exp_tab[addr];
  uint32_t y1 = addr == 31 ? 131071 : vpu_exp_tab[addr + 1];
  uint32_t interp = (y0 + (((y1 - y0) * (uint32_t)rLow) >> 7)) & 0x1ffff;
  uint32_t m = interp >> 7, sticky = (interp & 0x7f) != 0;
  // value = (2m + sticky) * 2^(k - 10), rounded once to BF16
  double v = (double)(2 * m + sticky);
  int ex = k - 10;
  while (ex > 0) { v *= 2; ex--; }
  while (ex < 0) { v *= 0.5; ex++; }
  return vpu_d_to_bf16(v);
}

// dst[i] = op(src1[i], src2[bcast ? i/rlen : i] | imm); reductions write dst[g] = reduce(src1[g*rlen .. +rlen])
// replicated in every lane. rows are scratchpad rows of VPU_LANES BF16. src/dst may only alias exactly (in place).
static inline void vpu_ref_exec(int op, uint16_t (*dst)[VPU_LANES], const uint16_t (*src1)[VPU_LANES],
                                const uint16_t (*src2)[VPU_LANES], int rows, int rlen, int bcast, uint16_t imm) {
  if (vpu_is_reduction(op)) {
    for (int g = 0; g < rows / rlen; g++) {
      uint16_t mx = 0; float acc = 0;
      for (int j = 0; j < rlen; j++) {
        const uint16_t *a = src1[g * rlen + j];
        uint16_t rm = 0;
        float s[VPU_LANES];
        for (int l = 0; l < VPU_LANES; l++) {
          uint16_t v = op == VPU_RAMAX ? (uint16_t)(a[l] & 0x7fff) : a[l];
          rm = l == 0 ? v : vpu_max(rm, v);
          s[l] = vpu_bf16_to_f(a[l]);
        }
        // the RTL's pairwise lane tree (Vpu.scala tree: grouped(2) recursively); 8 lanes: ((0+1)+(2+3))+((4+5)+(6+7))
        float t[VPU_LANES];
        for (int l = 0; l < VPU_LANES; l++) t[l] = s[l];
        for (int w = VPU_LANES; w > 1; w >>= 1)
          for (int l = 0; l < w / 2; l++) t[l] = t[2 * l] + t[2 * l + 1];
        float rs = t[0];
        mx = j == 0 ? rm : vpu_max(mx, rm);
        acc = j == 0 ? rs : acc + rs;
      }
      uint16_t out = op == VPU_RSUM ? vpu_f_to_bf16(acc) : mx;
      for (int l = 0; l < VPU_LANES; l++) dst[g][l] = out;
    }
    return;
  }
  for (int i = 0; i < rows; i++) {
    const uint16_t *a = src1[i];
    const uint16_t *b = vpu_uses_src2(op) ? src2[bcast ? i / rlen : i] : 0;
    uint16_t out[VPU_LANES];
    for (int l = 0; l < VPU_LANES; l++) {
      switch (op) {
        case VPU_ADD:   out[l] = vpu_add(a[l], b[l]); break;
        case VPU_SUB:   out[l] = vpu_sub(a[l], b[l]); break;
        case VPU_MUL:   out[l] = vpu_mul(a[l], b[l]); break;
        case VPU_MAX:   out[l] = vpu_max(a[l], b[l]); break;
        case VPU_ADDS:  out[l] = vpu_add(a[l], imm); break;
        case VPU_MULS:  out[l] = vpu_mul(a[l], imm); break;
        case VPU_EXP:   out[l] = vpu_exp(a[l]); break;
        case VPU_EXPSUB:
        case VPU_EXPSUM: out[l] = vpu_exp(vpu_sub(a[l], b[l])); break;   // EXPSUM's sums: vpu_ref_exec(VPU_RSUM, ..) on dst
        case VPU_RCP:   out[l] = vpu_rcp(a[l]); break;
        case VPU_RSQRT: out[l] = vpu_rsqrt(a[l]); break;
        default:        out[l] = a[l]; break;
      }
    }
    memcpy(dst[i], out, sizeof(out));
  }
}

#pragma GCC pop_options

#endif
