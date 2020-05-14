// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_H
#define SRC_MAIN_C_GEMMINI_H

#undef abs

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#include "include/gemmini_params.h"

#define GEMMINI_ASSERTIONS

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

// Accelerator interface
#include "rocc-software/src/xcustom.h"

#define k_CONFIG 0
#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 6
#define k_FLUSH 7
#define k_LOOP_WS 8

#define CONFIG_EX 0
#define CONFIG_LD 1
#define CONFIG_ST 2

#define XCUSTOM_ACC 3

#define GARBAGE_ADDR ((uint32_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define NO_ACTIVATION 0
#define RELU 1
#define RELU6 2

#ifdef ELEM_T_IS_FLOAT
elem_t elem_t_bits_to_elem_t(elem_t_bits x) {
    union {
        elem_t_bits b;
        elem_t f;
    } un;

    un.b = x;
    return un.f;
}

elem_t_bits elem_t_to_elem_t_bits(elem_t x) {
    union {
        elem_t_bits b;
        elem_t f;
    } un;

    un.f = x;
    return un.b;
}

acc_t acc_t_bits_to_acc_t(acc_t_bits x) {
    union {
        acc_t_bits b;
        acc_t f;
    } un;

    un.b = x;
    return un.f;
}

acc_t_bits acc_t_to_acc_t_bits(acc_t x) {
    union {
        acc_t_bits b;
        acc_t f;
    } un;

    un.f = x;
    return un.b;
}

bool elem_t_isnan(elem_t x) {
    elem_t_bits bits = elem_t_to_elem_t_bits(x);
    uint64_t exp = (bits >> (ELEM_T_SIG_BITS-1)) & (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
    uint64_t sig = bits & (((uint64_t)1 << ELEM_T_SIG_BITS) - 1);
    bool is_nan_or_inf = exp == (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
    bool is_not_inf = sig != 0;
    return is_nan_or_inf && is_not_inf;
}

bool acc_t_isnan(acc_t x) {
    acc_t_bits bits = acc_t_to_acc_t_bits(x);
    uint64_t exp = (bits >> (ACC_T_SIG_BITS-1)) & (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
    uint64_t sig = bits & (((uint64_t)1 << ACC_T_SIG_BITS) - 1);
    bool is_nan_or_inf = exp == (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
    bool is_not_inf = sig != 0;
    return is_nan_or_inf && is_not_inf;
}
#endif

#ifdef HAS_MVIN_SCALE
scale_t scale_t_bits_to_scale_t(scale_t_bits x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.b = x;
    return un.f;
}

scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.f = x;
    return un.b;
}
#endif

#ifdef HAS_MVIN_ACC_SCALE
scale_acc_t scale_acc_t_bits_to_scale_acc_t(scale_acc_t_bits x) {
    union {
        scale_acc_t_bits b;
        scale_acc_t f;
    } un;

    un.b = x;
    return un.f;
}

scale_acc_t_bits scale_acc_t_to_scale_acc_t_bits(scale_acc_t x) {
    union {
        scale_acc_t_bits b;
        scale_acc_t f;
    } un;

    un.f = x;
    return un.b;
}
#endif

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)

// mvin and mvout
#define gemmini_extended_mvin(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN)

#define gemmini_block_mvin(dram_addr, spad_addr, len) \
  gemmini_extended_mvin(dram_addr, spad_addr, (len) * DIM, DIM)

#define gemmini_mvin(dram_addr, spad_addr) \
  gemmini_extended_mvin(dram_addr, spad_addr, DIM, DIM)

#define gemmini_extended_mvout(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(spad_addr), k_MVOUT)

#define gemmini_mvout(dram_addr, spad_addr) \
  gemmini_extended_mvout(dram_addr, spad_addr, DIM, DIM)

// compute
#define gemmini_extended_compute_preloaded(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_PRELOADED)

#define gemmini_extended_compute_accumulated(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_ACCUMULATE)

#define gemmini_compute_preloaded(A, BD) \
  gemmini_extended_compute_preloaded(A, BD, DIM, DIM, DIM, DIM)

#define gemmini_compute_accumulated(A, BD) \
  gemmini_extended_compute_accumulated(A, BD, DIM, DIM, DIM, DIM)

// preload
#define gemmini_extended_preload(BD, C, BD_cols, BD_rows, C_cols, C_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), ((uint64_t)(C_rows) << (ADDR_LEN + 16)) | ((uint64_t)(C_cols) << ADDR_LEN) | (uint64_t)(C), k_PRELOAD)

#define gemmini_preload(BD, C) \
  gemmini_extended_preload(BD, C, DIM, DIM, DIM, DIM)

#define gemmini_preload_zeros(C) \
  gemmini_preload(GARBAGE_ADDR, C)

// weight-stationary matmul loop
// #define gemmini_loop_ws(A, B, I, J, K, bias) \
    // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(B) << 32) | (A), ((uint64_t)(bias) << 48) | ((uint64_t)(K) << 32) | ((J) << 16) | (I), k_LOOP_WS)

// config
/*
#define gemmini_extended_config_ex(mode, act, sys_shift, acc_shift, relu6_shift, A_stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(acc_shift) << 32) | ((uint64_t)(A_stride) << 16) | ((act) << 3) | ((mode) << 2) | CONFIG_EX, ((uint64_t)(relu6_shift) << 32) | (sys_shift), k_CONFIG)
*/
#define gemmini_extended_config_ex(mode, act, sys_shift, acc_shift, relu6_shift, ocol, orow, kcol, krow, stride, channel) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(acc_shift) << 32) |  ((act) << 3) | ((mode) << 2) | CONFIG_EX, ((uint64_t)ocol << 56) | ((uint64_t)orow << 48) | ((uint64_t)krow << 45) | ((uint64_t)kcol << 42) | ((uint64_t)(relu6_shift) << 32) | ((uint64_t)channel << 25) | ((uint64_t)stride << 22) | (sys_shift), k_CONFIG)

#define gemmini_config_ex(mode, act, sys_shift, acc_shift, relu6_shift) \
    gemmini_extended_config_ex(mode, act, sys_shift, acc_shift, relu6_shift, 0,0,0,0,0,1)

/*
#define gemmini_config_ex(mode, act, sys_shift, acc_shift, relu6_shift) \
    gemmini_extended_config_ex(mode, act, sys_shift, acc_shift, relu6_shift, 1)
*/
#if defined(HAS_MVIN_SCALE) || defined(HAS_MVIN_ACC_SCALE)
#define gemmini_extended_config_ld(stride, scale) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | CONFIG_LD, stride, k_CONFIG)
#else
#define gemmini_extended_config_ld(stride, scale) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_LD, stride, k_CONFIG)
#endif

#define gemmini_config_ld(stride) \
  gemmini_extended_config_ld(stride, MVIN_SCALE_ONE)

#define gemmini_extended_config_st(stride, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, upad, lpad) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(ocols) << 56) | ((uint64_t)(orows) << 48) | ((uint64_t)(pocols) << 40) | ((uint64_t)(porows) << 32) | ((uint64_t)(pool_out_dim) << 24) | ((uint64_t)(lpad) << 10) | ((uint64_t)(upad) << 8) | ((uint64_t)(pool_size) << 6) | ((uint64_t)(pool_stride) << 4) | CONFIG_ST, stride, k_CONFIG)

#define gemmini_config_st(stride) \
    gemmini_extended_config_st(stride, 0, 0, 0, 0, 0, 0, 0, 0, 0)

// flush
#define gemmini_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

// fence
#define gemmini_fence() asm volatile("fence")



uint64_t read_cyclesh() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}



// Tiling functions
static void sp_tiled_matmul_os(const elem_t * A, const elem_t * B, const acc_t * D, elem_t * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool no_bias, bool repeating_bias) {

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

  const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
  const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
  const int D_blocks = J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC;

  // Move-in D
  if (D != NULL && !no_bias) {
    const size_t D_stride = repeating_bias ? 0 : D_row_stride * sizeof(acc_t);
    gemmini_extended_config_ld(D_stride, D_scale_factor);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const acc_t * const D_dram_addr = (acc_t *)D + (bias_row * D_row_stride + j)*DIM;

        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;

        const size_t blocks = j + D_blocks <= J ? D_blocks : J-j;

        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        const size_t rows = DIM - (i == I-1 ? pad_I : 0);

        gemmini_extended_mvin(D_dram_addr, D_sp_addr_acc, cols, rows);
      }
    }
  }

  // Move-in B
  gemmini_extended_config_ld(B_row_stride * sizeof(elem_t), B_scale_factor);
  for (size_t j = 0; j < J; j += B_blocks) {
    for (size_t k = 0; k < K; k++) {
      const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
      const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
      const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
      const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
      const size_t rows = DIM - (k == K-1 ? pad_K : 0);
      gemmini_extended_mvin(B_dram_addr, B_sp_addr, cols, rows);
    }
  }

  // Move-in A
  gemmini_extended_config_ld(A_row_stride * sizeof(elem_t), A_scale_factor);
  for (size_t i = 0; i < I; i++) {
    for (size_t k = 0; k < K; k += A_blocks) {
      const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
      const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
      const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
      const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
    }
  }

  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j++) {
      const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

      for (size_t k = 0; k < K; k++) {

        const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

        uint32_t out_sp_addr = k == K-1 ? C_sp_addr : GARBAGE_ADDR;

        // If we're not using a bias, then we want to overwrite what's in the
        // accumulator, rather than writing over it
        int no_bias_new_matrix = no_bias && D != NULL && k == K-1;
        if (no_bias_new_matrix) {
          out_sp_addr &= ~(1 << (ADDR_LEN-2));
        }

        const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
        const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
        const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);

        if (k == 0) { // First iteration
          gemmini_extended_compute_preloaded(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
        } else { // All other iterations
          gemmini_extended_compute_accumulated(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
        }
      }
    }
  }

  // Move-out C
  if (C != NULL) {
    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j++) {
        elem_t * const C_dram_addr = C + (i*C_row_stride + j)*DIM;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        gemmini_extended_mvout(C_dram_addr, C_sp_addr, C_cols, C_rows);
      }
    }
  }
}

static void sp_tiled_matmul_ws(const elem_t * A, const elem_t * B,
        const acc_t * D, elem_t * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool no_bias, bool repeating_bias) {

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

  const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
  const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
  const int D_blocks = J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC;

  // Move-in D
  if (D != NULL && !no_bias) {
    const size_t D_stride = repeating_bias ? 0 : D_row_stride * sizeof(acc_t);
    gemmini_extended_config_ld(D_stride, D_scale_factor);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const acc_t * const D_dram_addr = (acc_t *)D + (bias_row * D_row_stride + j)*DIM;

        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;

        size_t blocks = j + D_blocks <= J ? D_blocks : J-j;
        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        const size_t rows = DIM - (i == I-1 ? pad_I : 0);

        gemmini_extended_mvin(D_dram_addr, D_sp_addr_acc, cols, rows);
      }
    }
  }

  // Move-in B
  gemmini_extended_config_ld(B_row_stride * sizeof(elem_t), B_scale_factor);
  for (size_t j = 0; j < J; j += B_blocks) {
    for (size_t k = 0; k < K; k++) {
      const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
      const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
      const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
      const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
      const size_t rows = DIM - (k == K-1 ? pad_K : 0);
      gemmini_extended_mvin(B_dram_addr, B_sp_addr, cols, rows);
    }
  }

  // Move-in A
  gemmini_extended_config_ld(A_row_stride * sizeof(elem_t), A_scale_factor);
  for (size_t k = 0; k < K; k += A_blocks) {
    for (size_t i = 0; i < I; i++) {
      const elem_t * const A_dram_addr = A + (i * A_row_stride + k)*DIM;
      const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
      const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
      const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
    }
  }

  // Compute
  // gemmini_loop_ws(A_sp_addr_start, B_sp_addr_start, I, J, K, !no_bias || D == NULL);

  // The above "gemmini_loop_ws" command will be unrolled in hardware into the
  // following loop:
  for (size_t j = 0; j < J; j++) {
    for (size_t k = 0; k < K; k++) {
      const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

      for (size_t i = 0; i < I; i++) {
        const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
        uint32_t out_sp_addr = C_sp_addr;

        // If we're not using a bias, then we want to overwrite what's in the
        // accumulator, rather than writing over it
        int no_bias_new_matrix = no_bias && D != NULL && k == 0;
        if (no_bias_new_matrix) {
          out_sp_addr &= ~(1 << (ADDR_LEN-2));
        }

        const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
        const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
        const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        gemmini_extended_preload(pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);

        if (i == 0) { // First iteration
          gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
        } else { // All other iterations
          gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
        }
      }
    }
  }

  // Move-out C
  if (C != NULL) {
    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j++) {
        elem_t * const C_dram_addr = C + (i*C_row_stride + j)*DIM;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        gemmini_extended_mvout(C_dram_addr, C_sp_addr, C_cols, C_rows);
      }
    }
  }
}

static void tiled_matmul_outer(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const acc_t * D, elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t tile_I, size_t tile_J, size_t tile_K,
        int act, int shift, size_t relu6_shift, bool repeating_bias,
        int dataflow) {

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const size_t I0 = dim_I_padded / (tile_I*DIM) + (dim_I_padded % (tile_I*DIM) != 0);
  const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
  const size_t K0 = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (tile_I*DIM) == 0 ? tile_I : (dim_I_padded/DIM) % tile_I;
  const size_t last_J = dim_J_padded % (tile_J*DIM) == 0 ? tile_J : (dim_J_padded/DIM) % tile_J;
  const size_t last_K = dim_K_padded % (tile_K*DIM) == 0 ? tile_K : (dim_K_padded/DIM) % tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - dim_I;
  const size_t padding_J = dim_J_padded - dim_J;
  const size_t padding_K = dim_K_padded - dim_K;

  const bool no_bias = D == NULL;

  if (no_bias) {
    D = (acc_t*) 1; // Dummy address which isn't NULL
  }

  gemmini_config_ex(dataflow, act, 0, shift, relu6_shift);
  gemmini_config_st(stride_C * sizeof(elem_t));

  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++) {

        const acc_t * pre;
        if (k0 != 0) {
          pre = NULL;
        } else {
          size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
          pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
        }
        elem_t * out = k0 == K0-1 ? C + i0*tile_I*DIM*stride_C + j0*tile_J*DIM : NULL;

        const size_t I = i0 < I0-1 ? tile_I : last_I;
        const size_t J = j0 < J0-1 ? tile_J : last_J;
        const size_t K = k0 < K0-1 ? tile_K : last_K;

        const size_t pad_I = i0 == I0-1 ? padding_I : 0;
        const size_t pad_J = j0 == J0-1 ? padding_J : 0;
        const size_t pad_K = k0 == K0-1 ? padding_K : 0;

        if (dataflow == OUTPUT_STATIONARY) {
          sp_tiled_matmul_os(A + i0*tile_I*DIM*stride_A + k0*tile_K*DIM,
              B + k0*tile_K*DIM*stride_B + j0*tile_J*DIM,
              pre, out,
              A_scale_factor, B_scale_factor, D_scale_factor,
              I, J, K,
              pad_I, pad_J, pad_K,
              stride_A, stride_B, stride_D, stride_C,
              no_bias, repeating_bias);
        } else {
          sp_tiled_matmul_ws(A + i0*tile_I*DIM*stride_A + k0*tile_K*DIM,
              B + k0*tile_K*DIM*stride_B + j0*tile_J*DIM,
              pre, out,
              A_scale_factor, B_scale_factor, D_scale_factor,
              I, J, K,
              pad_I, pad_J, pad_K,
              stride_A, stride_B, stride_D, stride_C,
              no_bias, repeating_bias);
        }
      }

  gemmini_fence();
}

static void matmul_cpu(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B, const acc_t * D,
        elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, size_t shift, size_t relu6_shift, bool repeating_bias) {

  const bool no_bias = D == NULL;

  for (size_t i = 0; i < dim_I; i++) {
    for (size_t j = 0; j < dim_J; j++) {
      size_t bias_row = repeating_bias ? 0 : i;
      acc_t result = no_bias ? 0 : D_scale_factor* *(D + bias_row*stride_D + j);

      for (size_t k = 0; k < dim_K; k++) {
        result += A_scale_factor * *(A + i*stride_A + k) * B_scale_factor * *((elem_t*)B + k*stride_B + j);
      }

      // Shift while rounding to nearest integer (ties round to negative infinity)
      result = ROUNDING_RIGHT_SHIFT(result, shift);

      // Clip result
      result = result > elem_t_max ? elem_t_max : (result < elem_t_min ? elem_t_min : result);

      // Apply activation function
      if (act == RELU) {
        result = result < 0 ? 0 : result;
      } else if (act == RELU6) {
        int max = 6 << relu6_shift;
        result = result < 0 ? 0 : (result > max ? max : result);
      }

      *(C + i*stride_C + j) = (elem_t)result;
    }
  }
}

// General matmul which can be run with different dataflows, or on the CPU
enum tiled_matmul_type_t {OS, WS, CPU};

// This function runs a tiled matrix multiplication, with hardcoded tiling
// factors
void tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const acc_t * D, elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, size_t shift, size_t relu6_shift, bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        enum tiled_matmul_type_t tiled_matmul_type) {

#ifdef GEMMINI_ASSERTIONS
  // Make sure that the tiling factors make sense
  if (tile_I <= 0) {
    printf("tile_I is non-positive\n");
    exit(1);
  } else if (tile_J <= 0) {
    printf("tile_J is non-positive\n");
    exit(1);
  } else if (tile_K <= 0) {
    printf("tile_K is non-positive\n");
    exit(1);
  }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  if (tile_I * DIM > dim_I_padded) {
    printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
    exit(1);
  } else if (tile_J * DIM > dim_J_padded) {
    printf("tile_J is too large (tile_J * DIM > dim_J_padded)\n");
    exit(1);
  } else if (tile_K * DIM > dim_K_padded) {
    printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
    exit(1);
  }

  const size_t total_spad_rows =
      (tile_I * tile_K * DIM) +   // Rows to store A
      (tile_K * tile_J * DIM);    // Rows to store B

  if (total_spad_rows > BANK_NUM * BANK_ROWS) {
    printf("Not enough space in scratchpad to store A and B matrices\n");
    exit(1);
  }

  const size_t total_acc_rows =
      tile_I * tile_J * DIM;      // Rows to store C

  if (total_acc_rows > ACC_ROWS) {
    printf("Not enough space in accumulator to store C\n");
    exit(1);
  }

  if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535) {
    printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
    exit(1);
  }
#endif

  // Run a tiled matrix multiplication on either Gemmini or the CPU
  if (tiled_matmul_type == OS || tiled_matmul_type == WS) {
      tiled_matmul_outer(dim_I, dim_J, dim_K,
              A, B, D, C,
              stride_A, stride_B, stride_D, stride_C,
              A_scale_factor, B_scale_factor, D_scale_factor,
              tile_I, tile_J, tile_K,
              act, shift, relu6_shift, repeating_bias,
              (int)tiled_matmul_type);
  } else /*if (tiled_matmul_type == CPU)*/ {
      matmul_cpu(dim_I, dim_J, dim_K,
              A, B, D, C,
              stride_A, stride_B, stride_D, stride_C,
              A_scale_factor, B_scale_factor, D_scale_factor,
              act, shift, relu6_shift, repeating_bias);
  }
}

// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const acc_t * D, elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, size_t shift, size_t relu6_shift, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type) {
#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

    const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
    const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
    const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

    const size_t tile_I = dim_I_padded/DIM < max_tile_i_j ? dim_I_padded/DIM : max_tile_i_j;
    const size_t tile_J = dim_J_padded/DIM < max_tile_i_j ? dim_J_padded/DIM : max_tile_i_j;
    const size_t tile_K = dim_K_padded/DIM < max_tile_k ? dim_K_padded/DIM : max_tile_k;

    tiled_matmul(dim_I, dim_J, dim_K,
        A, B, D, C, 
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        act, shift, relu6_shift, repeating_bias,
        tile_I, tile_J, tile_K,
        tiled_matmul_type);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}

void sp_tiled_conv(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim, int pool_out_dim,

        int stride, int padding, int kernel_dim,

        int pool_size, int pool_stride, int pool_padding,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs,

        int lpad, int rpad, int upad, int dpad,
        int plpad, int prpad, int pupad, int pdpad,

        elem_t * input,
        elem_t * weights,
        elem_t * output,
        acc_t * bias,

        bool no_bias, bool no_pool, bool no_1d) {

    const int orows = porows * pool_stride + pool_size - 1 - pupad - pdpad;
    const int ocols = pocols * pool_stride + pool_size - 1 - plpad - prpad;
    const int ochs = pochs;

    // Calculate image dimensions
//    const int irows = orows * stride + krows - 1; // - 2 * padding;
//    const int icols = ocols * stride + kcols - 1; // - 2 * padding;
    const int irows = (orows - 1) * stride + krows;
    const int icols = (ocols - 1) * stride + kcols; 
    const int irows_unpadded = irows - upad - dpad;
    const int icols_unpadded = icols - lpad - rpad;
    const int ichs = kchs;

    // Calculate spad address offsets
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
    const int B_rows = out_channels_per_bank * kcols * krows * kchs;

    const uint32_t A_sp_addr_start = 0;
    const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - B_rows;
    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN - 1);
    const uint32_t C_sp_addr_start = 3 << (ADDR_LEN - 2);

    // printf("mvin bias\n");
    // mvin bias
    if (!no_bias && bias != NULL) {
        // TODO we probably don't need quite this many nested loops for this part
        gemmini_config_ld(0);
        for (int b = 0; b < batches; b++)
            for (int orow = 0; orow < orows; orow++)
                for (int ocol = 0; ocol < ocols; ocol += DIM) {
                    const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

                    for (int och = 0; och < ochs; och += DIM) {
                        const int J = ochs - och > DIM ? DIM : ochs - och;

                        const uint32_t D_sp_addr = D_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

                        gemmini_extended_mvin(bias + och,
                                D_sp_addr,
                                J, I);
                    }
                }
    }

    // mvin input
    // printf("mvin inputs\n");
    gemmini_config_ld(in_channels * sizeof(elem_t));
    gemmini_fence(); // TODO fix ROB to get rid of this requirement
    for (int b = 0; b < batches; b++) {
        for (int irow = -upad; irow < irows_unpadded + dpad; irow++) {
            const int irow_padded = irow + upad;

            for (int icol = -lpad; icol < icols_unpadded + rpad;) {
                int I = icols_unpadded - icol > DIM ? DIM : icols_unpadded - icol;

                if (icol < 0) {
                    I = -icol > DIM ? DIM : -icol;
                } else if (icol >= icols_unpadded) {
                    I = icols_unpadded + rpad - icol > DIM ? DIM : icols_unpadded + rpad - icol;
                }

                const int icol_padded = icol + lpad;

                for (int ich = 0; ich < ichs; ich += DIM) {
                    const int K = ichs - ich > DIM ? DIM : ichs - ich;

                    elem_t * in = input + (b*in_dim*in_dim + irow*in_dim + icol) * in_channels + ich;

                    const bool is_zeros = irow < 0 || irow >= irows_unpadded || icol < 0 || icol >= icols_unpadded;
                    if (is_zeros) {
                        gemmini_config_ld(0);
                        static elem_t zeros[MAX_BYTES / sizeof(elem_t)] = {0};
                        in = &zeros[0];
                    }

                    const uint32_t A_sp_addr = A_sp_addr_start + (ich / DIM) * batches * irows * icols + b * irows * icols + irow_padded * icols + icol_padded;

                    gemmini_extended_mvin(in,
                            A_sp_addr,
                            K, I);

                    if (is_zeros) {
                        gemmini_config_ld(in_channels * sizeof(elem_t));
                    }
                }

                icol += I;
            }
        }
    }
    gemmini_fence(); // TODO fix ROB to get rid of this requirement


    int odims = orows*ocols;
    int kdims = krows*kcols; 
    // mvin weights
//    printf("weight move in \n");
    gemmini_config_ld(out_channels * sizeof(elem_t));
    for (int och = 0; och < ochs; och += DIM) {
        const int J = ochs - och > DIM ? DIM : ochs - och;

	for (int kch = 0; kch < kchs; kch += DIM) {
             const int K = kchs - kch > DIM ? DIM : kchs - kch;
		for(int kkdim = 0; kkdim < K*kdims; kkdim += DIM){
			const int KK = K*kdims - kkdim > DIM ? DIM : K*kdims - kkdim;
			const uint32_t B_sp_addr = B_sp_addr_start + (och/DIM)*kdims*kchs + kch*kdims + kkdim;
			gemmini_extended_mvin(weights+och+out_channels*(kch*kdims + kkdim), B_sp_addr, J, KK);

//printf("%d %d \n", J, KK);
/*
        	for (int krow = 0; krow < krows; krow++){
       		     for (int kcol = 0; kcol < kcols; kcol++){
//                for (int kch = 0; kch < kchs; kch += DIM) {
//                    const int K = kchs - kch > DIM ? DIM : kchs - kch;

//                    const uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow * kcols * kchs + kcol * kchs + kch;
			const uint32_t B_sp_addr = B_sp_addr_start + (och/DIM)*krows*kcols*kchs + kch*kcols*krows + krow*kcols*K + kcol*K;

//                    gemmini_extended_mvin(weights + (krow*kernel_dim*in_channels + kcol*in_channels + kch) * out_channels + och,
			gemmini_extended_mvin(weights + och + out_channels*((krow*kcols+kcol)*K + krows*kcols*kch),
                        B_sp_addr,
                        J, K);
                }
*/	   }
	}
    }

    if(!no_1d && no_pool) gemmini_extended_config_st(out_channels * sizeof(elem_t), 0, 1, 0, 0, 0, orows, ocols, 0, 0);


   // Compute
    for (int b = 0; b < batches; b++){
        for (int och = 0; och < ochs; och += DIM) {
            const int J = ochs - och > DIM ? DIM : ochs - och;
 	    for (int kch = 0; kch < kchs; kch += DIM) {
	        const int K = kchs - kch > DIM ? DIM : kchs - kch;
//		gemmini_extended_config_ex(WEIGHT_STATIONARY, act, 0, shift, relu6_shift, ocols, orows, kcols, krows, stride, K);

		const uint32_t A_sp_addr = A_sp_addr_start + (kch / DIM)*batches*irows*icols + b*irows*icols;           
            	for(int odim = 0; odim < odims; odim += DIM){ //both dimension at the same time
			const int I = odims - odim > DIM ? DIM : odims - odim;
     	       	    	const int C_sp_addr = C_sp_addr_start + (och / DIM) * batches * odims + b * odims + odim;

			for(int kkdim = 0; kkdim < K*kdims; kkdim += DIM){
				const int kk = K*kdims - kkdim > DIM ? DIM : K*kdims - kkdim;
				const uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * kdims * kchs + kch*kdims + kkdim;//kkdim;
				const uint32_t out_sp_addr =
                                    (bias != NULL && no_bias) && kkdim == 0 && kch == 0 ?
                                    C_sp_addr & ~((uint32_t)(1 << (ADDR_LEN - 2))) :
                                    C_sp_addr;
//				printf("\n, batch: %d A_sp_addr: %lu, out_sp_addr: %lu \n", b, A_sp_addr, out_sp_addr);
                                gemmini_extended_preload(B_sp_addr, out_sp_addr,
                                        J, kk, J, I);
                                gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, kk, I, J, I);
				gemmini_fence();
 
			}
                    }
                }
            }
       }





    // mvout output
   // printf("no_pool %d, no_1d %d \n", no_pool, no_1d);
//    gemmini_fence();
//    uint64_t start_mvout = read_cyclesh();
    if (output != NULL) {
        if (no_pool) {
	   if(no_1d){
            for (int b = 0; b < batches; b++)
                for (int orow = 0; orow < orows; orow++)
                    for (int ocol = 0; ocol < ocols; ocol += DIM) {
                        const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

                        for (int och = 0; och < ochs; och += DIM) {
                            const int J = ochs - och > DIM ? DIM : ochs - och;

                            const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * odims + b * odims + orow * ocols + ocol;

                            gemmini_extended_mvout(output + (b*out_dim*out_dim + orow*out_dim + ocol) * out_channels + och,
                                    C_sp_addr,
                                    J, I);
                        }
                    }
	   } else{
//		gemmini_extended_config_st(out_channels * sizeof(elem_t), 0, 1, 0, 0, 0, orows, ocols, 0, 0);
//		gemmini_fence();
		for(int b = 0; b < batches; b++)
			for(int och = 0; och < ochs; och += DIM){
				const int J = ochs - och > DIM ? DIM : ochs - och;
				const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * odims + b * odims;
				gemmini_extended_mvout(output + (b * out_dim * out_dim)*out_channels + och, C_sp_addr, J, 0);
			}	
		}

	   } else {
              gemmini_extended_config_st(out_channels * sizeof(elem_t), pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, pupad, plpad);
             gemmini_fence(); // TODO remove this when the ROB can accurately handle these
            for (int b = 0; b < batches; b++) {
                for (int poch = 0; poch < pochs; poch += DIM) {
                    const int channels = poch + DIM >= pochs ? pochs - poch : DIM;

                    elem_t * pout = output + (b * pool_out_dim * pool_out_dim)*out_channels + poch;

                    const uint32_t C_sp_addr = C_sp_addr_start + (poch / DIM) * batches * orows * ocols + b * orows * ocols;

                    gemmini_extended_mvout(pout,
                            C_sp_addr,
                            channels, 0);
                }
            }
            gemmini_fence();
        }
//    	uint64_t end_mvout = read_cyclesh();
//	printf("mvout cycles: %d \n", end_mvout - start_mvout);
   }
}

static int tiled_conv_total_spad_rows(bool acc,
        int stride,
        int batches,
        int porows, int pocols, int ochs,
        int krows, int kcols, int kchs,
        int pool_size, int pool_stride) {

    const int orows = porows * pool_stride + pool_size - 1;
    const int ocols = pocols * pool_stride + pool_size - 1;

    const int irows = orows * stride + krows - 1; // - 2 * padding;
    const int icols = ocols * stride + kcols - 1; // - 2 * padding;
    const int ichs = kchs;

    const int in_channels_per_bank = ichs / DIM + (ichs % DIM != 0);
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);

    const int A_rows = in_channels_per_bank * batches * irows * icols;
    const int B_rows = out_channels_per_bank * kcols * krows * kchs;
    const int C_rows = out_channels_per_bank * batches * orows * ocols;

    if (acc)
        return C_rows;
    else
        return A_rows + B_rows;
}

void tiled_conv(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int padding, int kernel_dim,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs,

        elem_t * input,
        elem_t * weights,
        acc_t * bias,
        elem_t * output,

        int act, size_t shift, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding) {

    bool no_bias = false;
    if (bias == NULL) {
        bias = (acc_t*)1;
        no_bias = true;
    }

    bool no_1d = pool_stride == 0 && pool_size == 0;
    bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }
    //include padding?
    const int orows = porows * pool_stride + pool_size - 1;
    const int ocols = pocols * pool_stride + pool_size - 1;


//    printf("no_1d: %d, no_pool: %d, pool_size: %d, pool_stride: %d \n", no_1d, no_pool, pool_size, pool_stride);

#ifdef GEMMINI_ASSERTIONS
    {
        // Check that data will fit in scratchpad
        const int spad_rows = tiled_conv_total_spad_rows(false,
            stride, batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);
        const int acc_rows = tiled_conv_total_spad_rows(true,
            stride, batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);

        // printf("spad_rows: %d\n", spad_rows);
        // printf("acc_rows: %d\n", acc_rows);
        // exit(1);

        if (spad_rows > BANK_NUM * BANK_ROWS) {
            printf("not enough scratchpad space to store inputs and weights\n");
            exit(1);
        }
        if (acc_rows > ACC_ROWS) {
            printf("not enough accumulator space to store outputs\n");
            exit(1);
        }
        if (kernel_dim <= padding) {
            printf("kernel_dim must be larger than padding\n");
            exit(1);
        }
    }
#endif

    const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;

//    gemmini_extended_config_ex(WEIGHT_STATIONARY, act, 0, shift, relu6_shift, stride);
//	gemmini_extended_config_ex(WEIGHT_STATIONARY, act, 0, shift, relu6_shift, ocols, orows, kcols, krows, stride, kchs);

    if (no_pool && no_1d) {
        gemmini_config_st(out_channels * sizeof(elem_t));
    }

    for (int b = 0; b < batch_size; b += batches) {
        for (int porow = 0; porow < pool_out_dim; porow += porows) {
            const int orow = porow * pool_stride - pool_padding;

            for (int pocol = 0; pocol < pool_out_dim; pocol += pocols) {
                const int ocol = pocol * pool_stride - pool_padding;

                for (int poch = 0; poch < out_channels; poch += pochs) {
                    for (int krow = 0; krow < kernel_dim; krow += krows) {
                        const int orow_floored = orow < 0 ? 0 : orow;
                        const int irow = orow_floored * stride + krow - padding;
//			printf("irow: %d, orows: %d, orow: %d, orow_floored: %d, krow: %d \n", irow, orows, orow, orow_floored, krow);

                        for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                            const int ocol_floored = ocol < 0 ? 0 : ocol;
                            const int icol = ocol_floored * stride + kcol - padding;
//			    printf("icol: %d, ocols: %d, ocol: %d, ocol_floored: %d, kcol: %d \n", icol, ocols, ocol, ocol_floored, kcol);

                            for (int kch = 0; kch < in_channels; kch += kchs) {
                                elem_t * out = output + (b*pool_out_dim*pool_out_dim + porow*pool_out_dim + pocol) * out_channels + poch;
                                if (krow + krows < kernel_dim ||
                                        kcol + kcols < kernel_dim ||
                                        kch + kchs < in_channels) {
                                    out = NULL;
                                }

                                acc_t * bias_ = bias + poch;
                                if (krow > 0 ||
                                        kcol > 0 ||
                                        kch > 0) {
                                    bias_ = NULL;
                                }

                                const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                                const int porows_ = pool_out_dim - porow > porows ? porows : pool_out_dim - porow;
                                const int pocols_ = pool_out_dim - pocol > pocols ? pocols : pool_out_dim - pocol;
                                const int pochs_ = out_channels - poch > pochs ? pochs : out_channels - poch;
                                const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                                const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;
                                const int kchs_ = in_channels - kch > kchs ? kchs : in_channels - kch;

                                const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                                const int orows_ = porows_ * pool_stride + pool_size - 1;
				gemmini_extended_config_ex(WEIGHT_STATIONARY, act, 0, shift, relu6_shift, ocols_, orows_, kcols_, krows_, stride, kchs_);

                                const int plpad = ocol < 0 ? -ocol : 0;
                                const int prpad = ocol + ocols_ > out_dim ? ocol + ocols_ - out_dim : 0;
                                const int pupad = orow < 0 ? -orow : 0;
                                const int pdpad = orow + orows_ > out_dim ? orow + orows_ - out_dim : 0;

                               // const int icols_ = (ocols_ - plpad - prpad) * stride + kcols_ - 1;
                               // const int irows_ = (orows_ - pupad - pdpad) * stride + krows_ - 1;
				const int icols_ = (ocols_ - 1 - plpad - prpad) * stride + kcols_;
                                const int irows_ = (orows_ - 1 - pupad - pdpad) * stride + krows_;

                                const int lpad = icol < 0 ? -icol : 0;
                                const int rpad = icol + icols_ > in_dim ? icol + icols_ - in_dim : 0;
                                const int upad = irow < 0 ? -irow : 0;
                                const int dpad = irow + irows_ > in_dim ? irow + irows_ - in_dim : 0;

/*
				printf("ocols_: %d \n", ocols_);
				printf("orows_: %d \n", orows_);
				printf("kcols_: %d \n", kcols_);
				printf("krows_: %d \n", krows_);
				printf("icols_: %d \n", icols_);
				printf("irows_: %d \n", irows_);
				printf("kchs_: %d \n", kchs_);
				printf("kch: %d \n", kch);

                                 printf("upad: %d\n", upad);
                                 printf("dpad: %d\n", dpad);
                                 printf("lpad: %d\n", lpad);
                                 printf("rpad: %d\n", rpad);
*/                                // printf("pupad: %d\n", pupad);
                                // printf("pdpad: %d\n", pdpad);
                                // printf("plpad: %d\n", plpad);
                                // printf("prpad: %d\n", prpad);

                               sp_tiled_conv(
                                    batch_size, in_dim, in_channels,
                                    out_channels, out_dim, pool_out_dim,

                                    stride, padding, kernel_dim,

                                    pool_size, pool_stride, pool_padding,

                                    batches_,
                                    porows_, pocols_, pochs_,
                                    krows_, kcols_, kchs_,

                                    lpad, rpad, upad, dpad,
                                    plpad, prpad, pupad, pdpad,

                                    input + (b*in_dim*in_dim + (irow+upad)*in_dim + (icol+lpad)) * in_channels + kch,
				    weights + (kch*kernel_dim*kernel_dim +  krow*kernel_dim + kcol)*out_channels + poch,
                                    //weights + (krow*kernel_dim*in_channels + kcol*in_channels + kch) * out_channels + poch,
                                    out,
                                    bias_,

                                    no_bias, no_pool, no_1d);
                            }
                        }
                    }
                }
            }
        }
    }
}

void tiled_conv_auto(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int padding, int kernel_dim,

        elem_t * input,
        elem_t * weights,
        acc_t * bias,
        elem_t * output,

        int act, size_t shift, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding) {

    const bool no_1d = pool_stride == 0 && pool_size == 0;
    const bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }
//    printf("no_1d: %d, no_pool: %d, pool_size: %d, pool_stride: %d \n", no_1d, no_pool, pool_size, pool_stride);

    const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;

    // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
    int args[] = {batch_size, pool_out_dim, pool_out_dim, out_channels, in_channels};
/*
    int spad_rows = tiled_conv_total_spad_rows(false,
        stride, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    int acc_rows = tiled_conv_total_spad_rows(true,
        stride, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
*/
    int spad_rows = tiled_conv_total_spad_rows(false,
        stride, args[0], args[1], args[2], args[3], kernel_dim, kernel_dim, args[4], pool_size, pool_stride);
    int acc_rows = tiled_conv_total_spad_rows(true,
        stride, args[0], args[1], args[2], args[3], kernel_dim, kernel_dim, args[4], pool_size, pool_stride);

    int kch_floor = (args[4]/DIM) + 1;
//    while (spad_rows > BANK_NUM*BANK_ROWS || acc_rows > ACC_ROWS) {
    while(spad_rows > 4000 || acc_rows > 4000){
        int max_val = -1;
        int max_idx = -1;

        for (int i = 0; i < (sizeof(args)/sizeof(args[0])); i++) {
            if (args[i] > max_val) {
		if(i!=4){
	                max_val = args[i];
        	        max_idx = i;
		} else if(kch_floor > 1){
			max_val = args[4];
			max_idx = i;
		}
            }
        }

	if(max_idx == 4) {
		kch_floor = kch_floor - 1;
		args[4] = kch_floor * DIM;
	}
        else args[max_idx]--;

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, args[0], args[1], args[2], args[3], kernel_dim, kernel_dim, args[4], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, args[0], args[1], args[2], args[3], kernel_dim, kernel_dim, args[4], pool_size, pool_stride);
    }

    int batches = args[0];
    int orows = args[1];
    int ocols = args[2];
    int ochs = args[3];
    int krows = kernel_dim;//args[4];
    int kcols = kernel_dim;//args[5];
    int kchs = args[4];
/*
     printf("batches = %d\n", batches);
     printf("orows = %d\n", orows);
     printf("ocols = %d\n", ocols);
     printf("ochs = %d\n", ochs);
     printf("krows = %d\n", krows);
     printf("kcols = %d\n", kcols);
     printf("kchs = %d\n", kchs);
*/
    tiled_conv(
        batch_size, in_dim, in_channels,
        out_channels, out_dim,
        stride, padding, kernel_dim,

        batches,
        orows, ocols, ochs,
        krows, kcols, kchs,

        input,
        weights,
        bias,
        output,

        act, shift, relu6_shift,
        no_1d ? 0 : pool_size, no_pool ? 0 : pool_stride, pool_padding);
}

#undef abs

#endif // SRC_MAIN_C_GEMMINI_H

