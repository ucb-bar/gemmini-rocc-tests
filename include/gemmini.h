// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_H
#define SRC_MAIN_C_GEMMINI_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#include "include/gemmini_params.h"

// Matmul utility functions
void matmul(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], int64_t C_full[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

void matmul_short(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], elem_t C[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C[r][c] += A[r][k]*B[k][c];
    }
}

void matmul_full(elem_t A[DIM][DIM], elem_t B[DIM][DIM], int64_t D[DIM][DIM], int64_t C_full[DIM][DIM]) {
  // Identical to the other matmul function, but with a 64-bit bias
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

void matadd(int64_t sum[DIM][DIM], int64_t m1[DIM][DIM], int64_t m2[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++)
      sum[r][c] = m1[r][c] + m2[r][c];
}

// Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
#define ROUNDING_RIGHT_SHIFT(x, shift) \
    ({(shift) > 0 ? (((x) >> (shift)) + \
        (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
             ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)))) : ((x) << (-(shift)));})

// THIS IS A ROUNDING SHIFT! It also performs a saturating cast
void matshift(int64_t full[DIM][DIM], elem_t out[DIM][DIM], int shift) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      // Bitshift and round element
      int64_t shifted = ROUNDING_RIGHT_SHIFT(full[r][c], shift);

      // Saturate and cast element
      int64_t elem = shifted > elem_t_max ? elem_t_max : (shifted < elem_t_min ? elem_t_min : shifted);
      out[r][c] = elem;
    }
}

void matrelu(elem_t in[DIM][DIM], elem_t out[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++)
      out[r][c] = in[r][c] > 0 ? in[r][c] : 0;
}

void matrelu6(elem_t in[DIM][DIM], elem_t out[DIM][DIM], int scale) {
  // int max = 6;
  int max = 6 * scale;

  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      elem_t positive = in[r][c] > 0 ? in[r][c] : 0;
      out[r][c] = positive > max ? max : positive;
    }
}

void transpose(elem_t in[DIM][DIM], elem_t out[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++)
      out[c][r] = in[r][c];
}

void printMatrix(elem_t m[DIM][DIM]) {
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int is_equal(elem_t x[DIM][DIM], elem_t y[DIM][DIM]) {
  for (size_t i = 0; i < DIM; ++i)
    for (size_t j = 0; j < DIM; ++j)
      if (x[i][j] != y[i][j])
          return 0;
  return 1;
}

// This is a GNU extension known as statment expressions
#define MAT_IS_EQUAL(dim_i, dim_j, x, y) \
    ({int result = 1; \
      for (size_t i = 0; i < dim_i; i++) \
        for (size_t j = 0; j < dim_j; ++j) \
          if (x[i][j] != y[i][j]) { \
            result = 0; \
            break; \
          } \
      result;})

int rand() {
  static uint32_t x = 777;
  x = x * 1664525 + 1013904223;
  return x >> 24;
}

uint64_t read_cycles() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;

    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbff8);
    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbffc);
    // return *mtime;
}

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

#define GARBAGE_ADDR ((uint64_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define NO_ACTIVATION 0
#define RELU 1
#define RELU6 2

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct, 10, 11)

// mvin and mvout
#define gemmini_mvin(dram_addr, spad_addr) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)1 << ADDR_LEN) | (spad_addr), k_MVIN)

#define gemmini_block_mvin(dram_addr, spad_addr, len) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(len) << ADDR_LEN) | (spad_addr), k_MVIN)

#define gemmini_mvout(dram_addr, spad_addr) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, spad_addr, k_MVOUT)

// compute
#define gemmini_compute_preloaded(A, BD) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, BD, k_COMPUTE_PRELOADED)

#define gemmini_compute_accumulated(A, BD) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, BD, k_COMPUTE_ACCUMULATE)

// preload
#define gemmini_preload(BD, C) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, BD, C, k_PRELOAD)

#define matmul_preload_zeros(C) \
  gemmini_preload(GARBAGE_ADDR, C)

// weight-stationary matmul loop
#define gemmini_loop_ws(A, B, I, J, K, bias) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(B) << 32) | (A), ((uint64_t)(bias) << 48) | ((uint64_t)(K) << 32) | ((J) << 16) | (I), k_LOOP_WS)

// config
#define gemmini_config_ex(mode, act, sys_shift, acc_shift, relu6_shift) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(acc_shift) << 32) | ((act) << 3) | ((mode) << 2) | CONFIG_EX, ((uint64_t)(relu6_shift) << 32) | (sys_shift), k_CONFIG)

#define gemmini_config_ld(stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_LD, stride, k_CONFIG)

#define gemmini_config_st(stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_ST, stride, k_CONFIG)

// flush
#define gemmini_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

// fence
#define gemmini_fence() asm volatile("fence")

// Tiling functions
static void sp_tiled_matmul_ws(const elem_t * A, const elem_t * B,
        const acc_t * D, elem_t * C,
        size_t I, size_t J, size_t K, size_t A_row_len,
        size_t B_row_len, size_t D_row_len, size_t C_row_len,
        bool no_bias, bool repeating_bias) {

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = I * K * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

  const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
  const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
  const int D_blocks = J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC;

  // Move-in D
  if (D != NULL && !no_bias) {
    const size_t D_stride = repeating_bias ? 0 : D_row_len * sizeof(acc_t);
    gemmini_config_ld(D_stride);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const acc_t * const D_dram_addr = (acc_t *)D + (bias_row * D_row_len + j)*DIM;

        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;

        size_t blocks = j + D_blocks <= J ? D_blocks : J-j;

        gemmini_block_mvin(D_dram_addr, D_sp_addr_acc, blocks);
      }
    }
  }

  // Move-in B
  gemmini_config_ld(B_row_len * sizeof(elem_t));
  for (size_t j = 0; j < J; j += B_blocks) {
    for (size_t k = 0; k < K; k++) {
      const elem_t * const B_dram_addr = B + (k * B_row_len + j)*DIM;

      const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

      const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
      gemmini_block_mvin(B_dram_addr, B_sp_addr, blocks);
    }
  }

  // Move-in A
  gemmini_config_ld(A_row_len * sizeof(elem_t));
  for (size_t i = 0; i < I; i++) {
    for (size_t k = 0; k < K; k += A_blocks) {
      const elem_t * const A_dram_addr = A + (i * A_row_len + k)*DIM;

      const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;

      const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
      gemmini_block_mvin(A_dram_addr, A_sp_addr, blocks);
    }
  }

  // Compute
  gemmini_loop_ws(A_sp_addr_start, B_sp_addr_start, I, J, K, !no_bias || D == NULL);

  // The above "gemmini_loop_ws" command will be unrolled in hardware into the
  // following loop
  /*
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

        gemmini_preload(pre_sp_addr, out_sp_addr);

        if (i == 0) { // First iteration
          gemmini_compute_preloaded(A_sp_addr, GARBAGE_ADDR);
        } else { // All other iterations
          gemmini_compute_accumulated(A_sp_addr, GARBAGE_ADDR);
        }
      }
    }
  }
  */

  // Move-out C
  if (C != NULL) {
    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j++) {
        elem_t * const C_dram_addr = C + (i*C_row_len + j)*DIM;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        gemmini_mvout(C_dram_addr, C_sp_addr);
      }
    }
  }
}

static void tiled_matmul_ws(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const acc_t * D, elem_t C[dim_I][dim_J],
        size_t tile_I, size_t tile_J, size_t tile_K,
        int act, int shift, bool repeating_bias) {

  const size_t I0 = dim_I / (tile_I*DIM) + (dim_I % (tile_I*DIM) != 0);
  const size_t J0 = dim_J / (tile_J*DIM) + (dim_J % (tile_J*DIM) != 0);
  const size_t K0 = dim_K / (tile_K*DIM) + (dim_K % (tile_K*DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I % (tile_I*DIM) == 0 ? tile_I : (dim_I/DIM) % tile_I;
  const size_t last_J = dim_J % (tile_J*DIM) == 0 ? tile_J : (dim_J/DIM) % tile_J;
  const size_t last_K = dim_K % (tile_K*DIM) == 0 ? tile_K : (dim_K/DIM) % tile_K;

  const bool no_bias = D == NULL;

  if (no_bias) {
    D = (void*) 1; // Dummy address which isn't NULL
  }

  gemmini_config_ex(WEIGHT_STATIONARY, act, 0, shift, 0);
  gemmini_config_st(dim_J * sizeof(elem_t));

  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++) {

        const acc_t * pre;
        if (k0 != 0) {
          pre = NULL;
        } else {
          size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
          pre = &((acc_t (*)[dim_J])D)[bias_row][j0*tile_J*DIM];
        }
        elem_t * out = k0 == K0-1 ? &C[i0*tile_I*DIM][j0*tile_J*DIM] : NULL;

        const size_t I = i0 < I0-1 ? tile_I : last_I;
        const size_t J = j0 < J0-1 ? tile_J : last_J;
        const size_t K = k0 < K0-1 ? tile_K : last_K;

        sp_tiled_matmul_ws(&A[i0*tile_I*DIM][k0*tile_K*DIM],
            &B[k0*tile_K*DIM][j0*tile_J*DIM],
            pre, out,
            I, J, K,
            dim_K, dim_J, dim_J, dim_J,
            no_bias, repeating_bias);
      }

  gemmini_fence();
}

static void matmul_cpu(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J], const void * D,
        elem_t C[dim_I][dim_J],
        int act, int shift, bool repeating_bias) {

  const bool no_bias = D == NULL;

  for (size_t i = 0; i < dim_I; i++) {
    for (size_t j = 0; j < dim_J; j++) {
      size_t bias_row = repeating_bias ? 0 : i;
      acc_t result = no_bias ? 0 : ((acc_t (*)[dim_J])D)[bias_row][j];

      for (size_t k = 0; k < dim_K; k++) {
        result += A[i][k] * B[k][j];
      }

      // Shift while rounding to nearest integer (ties round to negative infinity)
      result = ROUNDING_RIGHT_SHIFT(result, shift);

      // Clip result
      result = result > elem_t_max ? elem_t_max : (result < elem_t_min ? elem_t_min : result);

      // Apply activation function
      if (act == RELU) {
        result = result < 0 ? 0 : result;
      }

      C[i][j] = (elem_t)result;
    }
  }
}

// General matmul which can be run with different dataflows, or on the CPU
enum tiled_matmul_type_t {OS, WS, CPU};

// This function runs a tiled matrix multiplication, with hardcoded tiling
// factors
void tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const acc_t * D, elem_t C[dim_I][dim_J],
        int act, int shift, bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        enum tiled_matmul_type_t tiled_matmul_type) {

  // Make sure that the tiling factors make sense
  if (tile_I * DIM > dim_I) {
    printf("tile_I is too large (tile_I * DIM > dim_I)\n");
    exit(1);
  } else if (tile_J * DIM > dim_J) {
    printf("tile_J is too large (tile_J * DIM > dim_J)\n");
    exit(1);
  } else if (tile_K * DIM > dim_K) {
    printf("tile_K is too large (tile_K * DIM > dim_K)\n");
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

  // Run a tiled matrix multiplication on either Gemmini or the CPU
  if (tiled_matmul_type == OS) {
      printf("Output-stationary dataflow unsupported for EE290 class\n");
      exit(1);
  } else if (tiled_matmul_type == WS) {
      tiled_matmul_ws(dim_I, dim_J, dim_K,
              A, B, D, C,
              tile_I, tile_J, tile_K,
              act, shift, repeating_bias);
  } else /*if (tiled_matmul_type == CPU)*/ {
      matmul_cpu(dim_I, dim_J, dim_K,
              A, B, D, C,
              act, shift, repeating_bias);
  }
}

// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const acc_t * D, elem_t C[dim_I][dim_J],
        int act, int shift, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type) {
    /*
     * REPLACE THE THREE LINES BELOW IF YOU WANT TO USE THE
     * "tiled_matmul_auto" BELOW
     */
    const size_t tile_I = 1;
    const size_t tile_J = 1;
    const size_t tile_K = 1;

    tiled_matmul(dim_I, dim_J, dim_K,
        A, B, D, C, act, shift, repeating_bias,
        tile_I, tile_J, tile_K,
        tiled_matmul_type);
}

#endif // SRC_MAIN_C_GEMMINI_H

