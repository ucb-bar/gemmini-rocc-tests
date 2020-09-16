#ifndef GEMMINI_PARAMS_H
#define GEMMINI_PARAMS_H

#include <stdint.h>
#include <limits.h>

#define DIM 16
#define ADDR_LEN 32
#define BANK_NUM 4
#define BANK_ROWS 4096
#define ACC_ROWS 1024
#define MAX_BYTES 64
#define MAX_BLOCK_LEN (MAX_BYTES/(DIM*1))
#define MAX_BLOCK_LEN_ACC (MAX_BYTES/(DIM*4))

typedef int8_t elem_t;
elem_t elem_t_max = 127;
elem_t elem_t_min = -128;
typedef int32_t acc_t;
typedef int64_t full_t;

#define HAS_MVIN_SCALE
typedef int32_t scale_t;
typedef uint32_t scale_t_bits;

#define HAS_MVIN_ACC_SCALE
typedef int32_t scale_acc_t;
typedef uint32_t scale_acc_t_bits;

#define row_align(blocks) __attribute__((aligned(blocks*DIM*sizeof(elem_t))))
#define row_align_acc(blocks) __attribute__((aligned(blocks*DIM*sizeof(acc_t))))

#define MVIN_SCALE_IDENTITY 0

// Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
#ifndef ELEM_T_IS_FLOAT
#define ROUNDING_RIGHT_SHIFT(x, shift) \
    ({(shift) > 0 ? (((x) >> (shift)) + \
        (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
             ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)))) : ((x) << (-(shift)));})
#else
#define ROUNDING_RIGHT_SHIFT(x, shift) \
    ((x) / (1 << (shift)))
#endif

#endif // GEMMINI_PARAMS_H
