// See LICENSE for license details.

#ifndef SRC_MAIN_C_SYSTOLIC_H
#define SRC_MAIN_C_SYSTOLIC_H

#include <stdint.h>
#include <assert.h>
#include <limits.h>

// Dimension of the systolic array
// Should be tileColumns*meshColumns
// #define DIM 8
// #define ADDR_LEN 32
// #define BANK_ROWS (8*16) // 16
// #define BANK_NUM 2 // 7
// #define ACC_ROWS 64
#define DIM 16
#define ADDR_LEN 32
#define BANK_ROWS (64 * 1024 * 8 / (4 * 16*8))
#define BANK_NUM 4
#define ACC_ROWS (16 * 1024 * 8 / (16*32))

// Datatype of the systolic array
// typedef int16_t elem_t;
// elem_t elem_t_max = SHRT_MAX;
// elem_t elem_t_min = SHRT_MIN;
// typedef int32_t acc_t;
typedef int8_t elem_t;
elem_t elem_t_max = SCHAR_MAX;
elem_t elem_t_min = SCHAR_MIN;
typedef int32_t acc_t;

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
  // Identical to the other matmul fuction, but with a 64-bit bias
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

// THIS IS A ROUNDING SHIFT! It also performs a saturating cast
void matshift(int64_t full[DIM][DIM], elem_t out[DIM][DIM], int shift) {
  int divisor = 1 << shift;

  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      // Bitshift and round element
      int64_t abs = full[r][c] > 0 ? full[r][c] : -full[r][c];
      int64_t shifted = (abs + (divisor/2)) / divisor;
      if (full[r][c] < 0)
        shifted = -shifted;

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

void matrelu6(elem_t in[DIM][DIM], elem_t out[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      elem_t positive = in[r][c] > 0 ? in[r][c] : 0;
      out[r][c] = positive > 6 ? 6 : positive;
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


int rand() {
  static uint32_t x = 777;
  x = x * 1664525 + 1013904223;
  return x >> 24;
}

// Accelerator interface
#include "rocc-software/src/xcustom.h"

#define k_CONFIG 0
#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 6

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

#define to_deps(push1, pop1, push2, pop2) \
  (((push1 << 3) | (pop1 << 2) | (push2 << 1) | pop2) << 3)

// mvin and mvout
#define matmul_mvin(dram_addr, spad_addr, push_mvout, pop_mvout, push_ex, pop_ex) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, spad_addr, to_deps(push_mvout, pop_mvout, push_ex, pop_ex) | k_MVIN)

#define matmul_mvout(dram_addr, spad_addr, push_mvin, pop_mvin, push_ex, pop_ex) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, spad_addr, to_deps(push_mvin, pop_mvin, push_ex, pop_ex) | k_MVOUT)

// compute
#define matmul_compute_preloaded(A, BD) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, BD, k_COMPUTE_PRELOADED)

#define matmul_compute_accumulated(A, BD) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, BD, k_COMPUTE_ACCUMULATE)

// preload
#define matmul_preload(BD, C, push_mvin, pop_mvin, push_mvout, pop_mvout) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, BD, C, to_deps(push_mvin, pop_mvin, push_mvout, pop_mvout) | k_PRELOAD)

#define matmul_preload_zeros(C, push_mvin, pop_mvin, push_mvout, pop_mvout) \
  matmul_preload(GARBAGE_ADDR, C, push_mvin, pop_mvin, push_mvout, pop_mvout)

// config
#define matmul_config_ex(mode, act, shift, push_mvin, pop_mvin, push_mvout, pop_mvout) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (act << 3) | (mode << 2) | CONFIG_EX, shift, to_deps(push_mvin, pop_mvin, push_mvout, pop_mvout) | k_CONFIG)

#define matmul_config_ld(stride, push_mvout, pop_mvout, push_ex, pop_ex) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_LD, stride, to_deps(push_mvout, pop_mvout, push_ex, pop_ex) | k_CONFIG)

#define matmul_config_st(stride, push_mvin, pop_mvin, push_ex, pop_ex) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_ST, stride, to_deps(push_mvin, pop_mvin, push_ex, pop_ex) | k_CONFIG)

// fence
#define matmul_fence() asm volatile("fence")

// Tiling functions
void sp_tiled_matmul(elem_t * A, elem_t * B, elem_t * D, elem_t * C, size_t I,
        size_t J, size_t K, size_t A_row_len, size_t B_row_len,
        size_t D_row_len, size_t C_row_len,
        int first, int last) {

  const int A_sp_addr_start = 0;
  const int B_sp_addr_start = BANK_ROWS;
  const int D_sp_addr_start = 2*BANK_ROWS;
  const int C_sp_addr_start = 3*BANK_ROWS;

  matmul_config_ld(A_row_len * sizeof(elem_t), 0, 0, 0, 0);
  matmul_config_st(C_row_len * sizeof(elem_t), 0, 0, 0, 0);

  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j++) {
      elem_t * D_dram_addr = D + (i*D_row_len + j)*DIM;
      elem_t * C_dram_addr = C + (i*C_row_len + j)*DIM;

      int D_sp_addr = D_sp_addr_start + (i*J + j)*DIM;
      int C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

      for (size_t k = 0; k < K; k++) {
        elem_t * A_dram_addr = A + (i*A_row_len + k)*DIM;
        elem_t * B_dram_addr = B + (k*B_row_len + j)*DIM;

        int A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
        int B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

        // Move-in
        {
          int A_already_moved_in = j != 0;
          int B_already_moved_in = i != 0;
          int D_already_moved_in = k != 0;

          if (!A_already_moved_in) {
            matmul_mvin(A_dram_addr, A_sp_addr, 0, 0, 0, 0);
          }

          if (!B_already_moved_in) {
            matmul_config_ld(B_row_len * sizeof(elem_t), 0, 0, 0, 0);
            matmul_mvin(B_dram_addr, B_sp_addr, 0, 0, 0, 0);
          }
          
          if (!D_already_moved_in) {
            matmul_config_ld(D_row_len * sizeof(elem_t), 0, 0, 0, 0);
            matmul_mvin(D_dram_addr, D_sp_addr, 0, 0, 0, 0);
          }

          matmul_config_ld(A_row_len * sizeof(elem_t), 0, 0, 1, 0);
        }
        
        // Compute
        {
          int preload_sp_addr = k == 0 ? D_sp_addr : GARBAGE_ADDR;
          int out_sp_addr = k == K-1 ? C_sp_addr : GARBAGE_ADDR;

          if (k == K-1) { // Last iteration
            if (first) {
              matmul_preload(preload_sp_addr, out_sp_addr, 0, 1, 1, 0);
            } else {
              matmul_preload(preload_sp_addr, out_sp_addr, 0, 1, 1, 1);
            }
          } else { // All other iterations
            matmul_preload(preload_sp_addr, out_sp_addr, 0, 1, 0, 0);
          }

          if (k == 0) { // First iteration
            matmul_compute_preloaded(A_sp_addr, B_sp_addr);
          } else { // All other iterations
            matmul_compute_accumulated(A_sp_addr, B_sp_addr);
          }
        }
      }

      // Move-out
      {
        if (last)  {
          matmul_mvout(C_dram_addr, C_sp_addr, 0, 0, 0, 1);
        } else {
          matmul_mvout(C_dram_addr, C_sp_addr, 0, 0, 1, 1);
        }
      }
    }
  }
}



void sp_tiled_matmul_ws(elem_t * A, elem_t * B, elem_t * D, elem_t * C, size_t I,
        size_t J, size_t K, size_t A_row_len, size_t B_row_len,
        size_t D_row_len, size_t C_row_len,
        int first, int last) {


  const int A_sp_addr_start = 0;
  const int B_sp_addr_start = BANK_ROWS;
  const int D_sp_addr_start = 2*BANK_ROWS;
  const int C_sp_addr_start = 3*BANK_ROWS;

  matmul_config_st(C_row_len * sizeof(elem_t), 0, 0, 0, 0);

  for (size_t k = 0; k < K; k++) {
    for (size_t j = 0; j < J; j++) {
      elem_t * B_dram_addr = B + (k*B_row_len + j)*DIM;
      int B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
      matmul_config_ld(B_row_len * sizeof(elem_t), 0, 0, 0, 0);
      matmul_mvin(B_dram_addr, B_sp_addr, 0, 0, 0, 0);

      matmul_config_ld(A_row_len * sizeof(elem_t), 0, 0, 0, 0);
      for (size_t i = 0; i < I; i++) {

        elem_t * D_dram_addr = D + (i*D_row_len + j)*DIM;
        elem_t * C_dram_addr = C + (i*C_row_len + j)*DIM;

        int D_sp_addr = D_sp_addr_start + (i*J + j)*DIM;
        int C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        elem_t * A_dram_addr = A + (i*A_row_len + k)*DIM;
        int A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;

        // Move-in
        {
          int A_already_moved_in = j != 0;
          int D_already_moved_in = k != 0;

          if (!A_already_moved_in) {
            matmul_mvin(A_dram_addr, A_sp_addr, 0, 0, 0, 0);
          }
          
          if (!D_already_moved_in) {
            matmul_config_ld(D_row_len * sizeof(elem_t), 0, 0, 0, 0);
            matmul_mvin(D_dram_addr, D_sp_addr, 0, 0, 0, 0);
          }

          matmul_config_ld(A_row_len * sizeof(elem_t), 0, 0, 1, 0);
        }


        // Compute
        {
          int preload_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          int out_sp_addr = i == I-1 ? C_sp_addr : GARBAGE_ADDR;

          if (i == I-1) { // Last iteration
            if (first) {
              matmul_preload(preload_sp_addr, out_sp_addr, 0, 1, 1, 0);
            } else {
              matmul_preload(preload_sp_addr, out_sp_addr, 0, 1, 1, 1);
            }
          } else { // All other iterations
            matmul_preload(preload_sp_addr, out_sp_addr, 0, 1, 0, 0);
          }

          if (i == 0) { // First iteration
            matmul_compute_preloaded(A_sp_addr, D_sp_addr);
          } else { // All other iterations
            matmul_compute_accumulated(A_sp_addr, D_sp_addr);
          }
        }


        // Move-out
        {
          if (last)  {
            matmul_mvout(C_dram_addr, C_sp_addr, 0, 0, 0, 1);
          } else {
            matmul_mvout(C_dram_addr, C_sp_addr, 0, 0, 1, 1);
          }
        }
      }
    }
  }

}



#endif  // SRC_MAIN_C_SYSTOLIC_H

