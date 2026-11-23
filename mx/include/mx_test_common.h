// Shared helpers for the MX matmul test templates (mx/templates/*.c).
#ifndef MX_TEST_COMMON_H
#define MX_TEST_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#define MX_STR_(x) #x
#define MX_STR(x) MX_STR_(x)
#define MX_CAT_(a, b) a##b
#define MX_CAT(a, b) MX_CAT_(a, b)

// Element formats, by name. code: config_ex format field (0 fp8, 1 fp6, 2 fp4); alt: altfmt bit;
// single: e4m3s = 8-bit E4M3 on the direct single-throughput path (no LUT); label: name in messages.
#define MX_CODE_e4m3  0
#define MX_CODE_e5m2  0
#define MX_CODE_e4m3s 0
#define MX_CODE_e2m3  1
#define MX_CODE_e3m2  1
#define MX_CODE_fp4   2
#define MX_ALT_e4m3   0
#define MX_ALT_e5m2   1
#define MX_ALT_e4m3s  0
#define MX_ALT_e2m3   1
#define MX_ALT_e3m2   0
#define MX_ALT_fp4    0
#define MX_SINGLE_e4m3  0
#define MX_SINGLE_e5m2  0
#define MX_SINGLE_e4m3s 1
#define MX_SINGLE_e2m3  0
#define MX_SINGLE_e3m2  0
#define MX_SINGLE_fp4   0
#define MX_LABEL_e4m3  "e4m3"
#define MX_LABEL_e5m2  "e5m2"
#define MX_LABEL_e4m3s "e4m3s(single)"
#define MX_LABEL_e2m3  "e2m3"
#define MX_LABEL_e3m2  "e3m2"
#define MX_LABEL_fp4   "fp4"

#define MX_CODE(f)   MX_CAT(MX_CODE_, f)
#define MX_ALT(f)    MX_CAT(MX_ALT_, f)
#define MX_SINGLE(f) MX_CAT(MX_SINGLE_, f)
#define MX_LABEL(f)  MX_CAT(MX_LABEL_, f)

// config_ex output format codes.
#define MX_OUT_FP4  2
#define MX_OUT_BF16 3

// Weight-stationary MX config_ex. wdiff (rs1[31]): weight altfmt = alt XOR wdiff.
static inline void mx_config_ex_ws(int out_fmt, int b_fmt, int a_fmt, int alt, int wdiff, int use_lut) {
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
      ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(wdiff) << 31)
    | ((uint64_t)(1) << 16)               // A stride
    | ((uint64_t)(out_fmt) << 14)
    | ((uint64_t)(b_fmt) << 12)
    | ((uint64_t)(a_fmt) << 10)
    | ((uint64_t)(alt) << 6)
    | ((uint64_t)(use_lut) << 5)
    | ((uint64_t)(WEIGHT_STATIONARY) << 2)
    | CONFIG_EX,
      ((uint64_t)(1) << 48) | (0),
      k_CONFIG);
}

#endif
