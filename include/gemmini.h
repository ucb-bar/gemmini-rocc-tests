// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_H
#define SRC_MAIN_C_GEMMINI_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
// TODO use stdbool.h as well

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
    ({((x) >> (shift)) + \
        (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
             ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)));})

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

unsigned long read_cycles() {
    unsigned long cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
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
#define matmul_mvin(dram_addr, spad_addr) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)1 << ADDR_LEN) | (spad_addr), k_MVIN)

#define matmul_block_mvin(dram_addr, spad_addr, len) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(len) << ADDR_LEN) | (spad_addr), k_MVIN)

#define matmul_mvout(dram_addr, spad_addr) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, spad_addr, k_MVOUT)

// compute
#define matmul_compute_preloaded(A, BD) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, BD, k_COMPUTE_PRELOADED)

#define matmul_compute_accumulated(A, BD) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, BD, k_COMPUTE_ACCUMULATE)

// preload
#define matmul_preload(BD, C) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, BD, C, k_PRELOAD)

#define matmul_preload_zeros(C) \
  matmul_preload(GARBAGE_ADDR, C)

// config
#define matmul_config_ex(mode, act, sys_shift, acc_shift, relu6_shift) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(acc_shift) << 32) | ((act) << 3) | ((mode) << 2) | CONFIG_EX, ((uint64_t)(relu6_shift) << 32) | (sys_shift), k_CONFIG)

#define matmul_config_ld(stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_LD, stride, k_CONFIG)

#define matmul_config_st(stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_ST, stride, k_CONFIG)

// flush
#define matmul_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

// fence
#define matmul_fence() asm volatile("fence")

// Tiling functions
// static void sp_tiled_matmul_os(elem_t * A, elem_t * B, acc_t * D, elem_t * C,
static void sp_tiled_matmul_os(elem_t * A, elem_t * B, void * D, elem_t * C,
        size_t I, size_t J, size_t K, size_t A_row_len,
        size_t B_row_len, size_t D_row_len, size_t C_row_len,
        int first_mvin, int last_mvout, int no_bias, int full_bias_width) {

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS / 2;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

  const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
  const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
  const int D_blocks_max = full_bias_width ? MAX_BLOCK_LEN_ACC : MAX_BLOCK_LEN;
  const int D_blocks = J <= D_blocks_max ? J : D_blocks_max;

  const int sizeof_bias = full_bias_width ? sizeof(acc_t) : sizeof(elem_t);

  // Move-in D
  if (D != NULL && !no_bias) {
    matmul_config_ld(D_row_len * sizeof_bias);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j++) {
        void * D_dram_addr;
        uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;
        uint32_t D_sp_addr_sp = (i*J + j)*DIM;

        if (full_bias_width) {
          acc_t * const D_ = (acc_t *) D;
          D_dram_addr = (void*)(D_ + (i*D_row_len + j)*DIM);
        } else {
          elem_t * const D_ = (elem_t *) D;
          D_dram_addr = (void*)(D_ + (i*D_row_len + j)*DIM);
        }

        int already_moved_in = j % D_blocks != 0;

        if (!already_moved_in) {
          int blocks = j + D_blocks <= J ? D_blocks : J-j;

          if (full_bias_width) {
            matmul_block_mvin(D_dram_addr, D_sp_addr_acc, blocks);
          } else {
            matmul_block_mvin(D_dram_addr, D_sp_addr_sp, blocks);
          }
        }

        if (!full_bias_width) {
          matmul_preload(D_sp_addr_sp, D_sp_addr_acc);
          matmul_compute_preloaded(GARBAGE_ADDR, GARBAGE_ADDR);
        }
      }
    }
  }

  /*
  for (size_t i = 0; i < I; i++) {
    for (size_t k = 0; k < K; k += A_blocks) {
      elem_t * const A_dram_addr = A + (i*A_row_len + k)*DIM;
      const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;

      const int blocks = k + A_blocks <= K ? A_blocks : K-k;

      matmul_config_ld(A_row_len * sizeof(elem_t));
      matmul_block_mvin(A_dram_addr, A_sp_addr, blocks);
    }
  }

  for (size_t j = 0; j < J; j += B_blocks) {
    for (size_t k = 0; k < K; k++) {
        elem_t * const B_dram_addr = B + (k*B_row_len + j)*DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

        const int blocks = j + B_blocks <= J ? B_blocks : J-j;

        matmul_config_ld(B_row_len * sizeof(elem_t));
        matmul_block_mvin(B_dram_addr, B_sp_addr, blocks);
    }
  }
  */

  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j++) {
      const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

      for (size_t k = 0; k < K; k++) {
        // printf("  i: %u, j: %u, k: %u\n", i, j, k);

        elem_t * const A_dram_addr = A + (i*A_row_len + k)*DIM;
        elem_t * const B_dram_addr = B + (k*B_row_len + j)*DIM;

        const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

        // Move-in A and B
        {
          // printf("    Enter mvin\n");

          int A_already_moved_in = j != 0 || k % A_blocks != 0;
          int B_already_moved_in = i != 0 || j % B_blocks != 0;

          if (!A_already_moved_in) {
            matmul_config_ld(A_row_len * sizeof(elem_t));

            const int blocks = k + A_blocks <= K ? A_blocks : K-k;
            matmul_block_mvin(A_dram_addr, A_sp_addr, blocks);
          }

          if (!B_already_moved_in) {
            matmul_config_ld(B_row_len * sizeof(elem_t));

            const int blocks = j + B_blocks <= J ? B_blocks : J-j;
            matmul_block_mvin(B_dram_addr, B_sp_addr, blocks);
          }

          // printf("    Exit mvin\n");
        }

        // Compute
        {
          // printf("    Enter compute\n");
          uint32_t out_sp_addr = k == K-1 ? C_sp_addr : GARBAGE_ADDR;

          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it
          int no_bias_new_matrix = no_bias && D != NULL && k == K-1;
          if (no_bias_new_matrix) {
            out_sp_addr &= ~(1 << (ADDR_LEN-2));
          }

          matmul_preload(GARBAGE_ADDR, out_sp_addr);

          if (k == 0) { // First iteration
            matmul_compute_preloaded(A_sp_addr, B_sp_addr);
          } else { // All other iterations
            matmul_compute_accumulated(A_sp_addr, B_sp_addr);
          }

          // printf("    Exit compute\n");
        }
      }

      // Move-out C
      // if (C != NULL) {
      //   elem_t * const C_dram_addr = C + (i*C_row_len + j)*DIM;
      //   matmul_mvout(C_dram_addr, C_sp_addr);
      // }
    }
  }

  // printf("Exit main inner loop\n");

  // TODO this should be overlapped with the next "Move-in D"
  // Move-out C
  if (C != NULL) {
    // printf("  Enter mvout loop\n");

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j++) {
        // printf("    i: %lu, j: %lu\n", i, j);

        elem_t * const C_dram_addr = C + (i*C_row_len + j)*DIM;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        matmul_mvout(C_dram_addr, C_sp_addr);
      }
    }

    // printf("  Exit mvout loop\n");
  }
}


// static void sp_tiled_matmul_ws(elem_t * A, elem_t * B, acc_t * D, elem_t * C,
static void sp_tiled_matmul_ws(elem_t * A, elem_t * B, void * D, elem_t * C,
        size_t I, size_t J, size_t K, size_t A_row_len,
        size_t B_row_len, size_t D_row_len, size_t C_row_len,
        int first_mvin, int last_mvout, int no_bias, int full_bias_width) {

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS / 2;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

  const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
  const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
  const int D_blocks_max = full_bias_width ? MAX_BLOCK_LEN_ACC : MAX_BLOCK_LEN;
  const int D_blocks = J <= D_blocks_max ? J : D_blocks_max;

  const int sizeof_bias = full_bias_width ? sizeof(acc_t) : sizeof(elem_t);

  const int I_iterations = I;
  const int J_iterations = (J/B_blocks + (J % B_blocks != 0));
  const int K_iterations = (K/A_blocks + (K % A_blocks != 0));
  const int total_iterations = I_iterations * J_iterations * K_iterations;

  int old_iterations = total_iterations;

  // Move-in D
  if (D != NULL && !no_bias) {
    matmul_config_ld(D_row_len * sizeof_bias);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j++) {
        void * D_dram_addr;
        uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;
        uint32_t D_sp_addr_sp = (i*J + j)*DIM;

        if (full_bias_width) {
          acc_t * const D_ = (acc_t *) D;
          D_dram_addr = (void*)(D_ + (i*D_row_len + j)*DIM);
        } else {
          elem_t * const D_ = (elem_t *) D;
          D_dram_addr = (void*)(D_ + (i*D_row_len + j)*DIM);
        }

        int already_moved_in = j % D_blocks != 0;

        if (!already_moved_in) {
          int blocks = j + D_blocks <= J ? D_blocks : J-j;

          if (full_bias_width) {
            matmul_block_mvin(D_dram_addr, D_sp_addr_acc, blocks);
          } else /*if (!full_bias_width)*/ {
            matmul_block_mvin(D_dram_addr, D_sp_addr_sp, blocks);
          }
        }

        if (!full_bias_width) {
          matmul_preload(GARBAGE_ADDR, D_sp_addr_acc);
          matmul_compute_preloaded(GARBAGE_ADDR, D_sp_addr_sp);
        }
      }
    }
  }

  for (size_t j = 0; j < J; j++) {
    for (size_t k = 0; k < K; k++) {
      const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

      for (size_t i = 0; i < I; i++) {
        elem_t * const A_dram_addr = A + (i*A_row_len + k)*DIM;
        elem_t * const B_dram_addr = B + (k*B_row_len + j)*DIM;

        const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        // Move-in A and B
        {
          int A_already_moved_in = j != 0 || k % A_blocks != 0;
          int B_already_moved_in = i != 0 || j % B_blocks != 0;

          if (!A_already_moved_in) {
            matmul_config_ld(A_row_len * sizeof(elem_t));

            int blocks = k + A_blocks <= K ? A_blocks : K-k;
            // printf("Moving in %d blocks of A: %u\n", blocks, A_sp_addr);
            matmul_block_mvin(A_dram_addr, A_sp_addr, blocks);
          }

          if (!B_already_moved_in) {
            matmul_config_ld(B_row_len * sizeof(elem_t));

            int blocks = j + B_blocks <= J ? B_blocks : J-j;
            // printf("Moving in %d blocks of B: %u\n", blocks, B_sp_addr);
            matmul_block_mvin(B_dram_addr, B_sp_addr, blocks);
          }
        }

        // Compute
        {
          uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          uint32_t out_sp_addr = C_sp_addr;

          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it
          int no_bias_new_matrix = no_bias && D != NULL && k == 0;
          if (no_bias_new_matrix) {
            out_sp_addr &= ~(1 << (ADDR_LEN-2));
          }

          int final_submatrix = i == I-1 && j == J-1 && k == K-1 && C != NULL;

          matmul_preload(pre_sp_addr, out_sp_addr);

          if (i == 0) { // First iteration
            matmul_compute_preloaded(A_sp_addr, GARBAGE_ADDR);
          } else { // All other iterations
            matmul_compute_accumulated(A_sp_addr, GARBAGE_ADDR);
          }
        }
      }
    }
  }

  // TODO this should be overlapped with the next "Move-in D"
  // Move-out C
  if (C != NULL) {
    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j++) {
        elem_t * const C_dram_addr = C + (i*C_row_len + j)*DIM;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        matmul_mvout(C_dram_addr, C_sp_addr);
      }
    }
  }
}

static void tiled_matmul_os(size_t DIM_I, size_t DIM_J, size_t DIM_K,
        // elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], acc_t D[DIM_I][DIM_J],
        elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], void* D,
        elem_t C[DIM_I][DIM_J], size_t TILE_I, size_t TILE_J, size_t TILE_K,
        int act, int shift, int relu6_shift, int full_bias_width) {

    const int I0 = DIM_I / (TILE_I*DIM);
    const int J0 = DIM_J / (TILE_J*DIM);
    const int K0 = DIM_K / (TILE_K*DIM);

    const int no_bias = D == NULL;

    if (no_bias) {
      // D = (acc_t (*)[DIM_J]) 1; // Dummy address which isn't NULL
      D = (void*) 1; // Dummy address which isn't NULL
    }

    matmul_config_ex(OUTPUT_STATIONARY, act, 0, shift, relu6_shift);
    matmul_config_st(DIM_J * sizeof(elem_t));

    for (size_t i0 = 0; i0 < I0; i0++)
      for (size_t j0 = 0; j0 < J0; j0++)
        for (size_t k0 = 0; k0 < K0; k0++) {
          // printf("i0: %lu, j0: %lu, k0: %lu\n", i0, j0, k0);

          int first_mvin = i0 == 0 && j0 == 0 && k0 == 0;
          int last_mvout = (i0 == I0-1) && (j0 == J0-1) && (k0 == K0-1);

          // acc_t * pre = k0 == 0 ? &D[i0*TILE_I*DIM][j0*TILE_J*DIM] : NULL;
          void * pre;
          if (k0 != 0) {
            pre = NULL;
          } else if (full_bias_width) {
            pre = &((acc_t (*)[DIM_J])D)[i0*TILE_I*DIM][j0*TILE_J*DIM];
          } else {
            pre = &((elem_t (*)[DIM_J])D)[i0*TILE_I*DIM][j0*TILE_J*DIM];
          }

          elem_t * out = k0 == K0-1 ? &C[i0*TILE_I*DIM][j0*TILE_J*DIM] : NULL;

          sp_tiled_matmul_os(&A[i0*TILE_I*DIM][k0*TILE_K*DIM],
              &B[k0*TILE_K*DIM][j0*TILE_J*DIM],
              pre, out,
              TILE_I, TILE_J, TILE_K,
              DIM_K, DIM_J, DIM_J, DIM_J,
              first_mvin, last_mvout, no_bias, full_bias_width);
        }

    matmul_fence();
}

static void tiled_matmul_ws(size_t DIM_I, size_t DIM_J, size_t DIM_K,
        // elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], acc_t D[DIM_I][DIM_J],
        elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], void * D,
        elem_t C[DIM_I][DIM_J], size_t TILE_I, size_t TILE_J, size_t TILE_K,
        int act, int shift, int relu6_shift, int full_bias_width) {

    const int I0 = DIM_I / (TILE_I*DIM);
    const int J0 = DIM_J / (TILE_J*DIM);
    const int K0 = DIM_K / (TILE_K*DIM);

    const int no_bias = D == NULL;

    if (no_bias) {
      // D = (acc_t (*)[DIM_J]) 1; // Dummy address which isn't NULL
      D = (void*) 1; // Dummy address which isn't NULL
    }

    matmul_config_ex(WEIGHT_STATIONARY, act, 0, shift, relu6_shift);
    matmul_config_st(DIM_J * sizeof(elem_t));

    for (size_t i0 = 0; i0 < I0; i0++)
      for (size_t j0 = 0; j0 < J0; j0++)
        for (size_t k0 = 0; k0 < K0; k0++) {
          /*
          if (i0 == 0 && j0 == 0) {
            printf("Outer: i0: %u, j0: %u, k0: %u\n", i0, j0, k0);
          }
          */

          int first_mvin = i0 == 0 && j0 == 0 && k0 == 0;
          int last_mvout = (i0 == I0-1) && (j0 == J0-1) && (k0 == K0-1);

          // acc_t * pre = k0 == 0 ? &D[i0*TILE_I*DIM][j0*TILE_J*DIM] : NULL;
          void * pre;
          if (k0 != 0) {
            pre = NULL;
          } else if (full_bias_width) {
            pre = &((acc_t (*)[DIM_J])D)[i0*TILE_I*DIM][j0*TILE_J*DIM];
          } else {
            pre = &((elem_t (*)[DIM_J])D)[i0*TILE_I*DIM][j0*TILE_J*DIM];
          }

          elem_t * out = k0 == K0-1 ? &C[i0*TILE_I*DIM][j0*TILE_J*DIM] : NULL;

          /*
          if (i0 == 0 && j0 == 0) {
            printf("  pre is %p\n", pre);
            if (pre != NULL) {
              printf("    bias is %d\n", ((acc_t (*)[DIM_J])pre)[0][1]);
            }

            printf("  out is %p\n", out);
          }
          */

          sp_tiled_matmul_ws(&A[i0*TILE_I*DIM][k0*TILE_K*DIM],
              &B[k0*TILE_K*DIM][j0*TILE_J*DIM],
              pre, out,
              TILE_I, TILE_J, TILE_K,
              DIM_K, DIM_J, DIM_J, DIM_J,
              first_mvin, last_mvout, no_bias, full_bias_width);

          /*
          if (i0 == 0 && j0 == 0) {
            if (out != NULL) {
              printf("    result is %d\n", ((elem_t (*)[DIM_J])out)[0][1]);
              printf("    actual result is %d\n", C[0][1]);
            }
          }
          */
        }

    matmul_fence();

    // printf("    final result is %d\n", C[0][1]);
}

static void matmul_cpu(size_t DIM_I, size_t DIM_J, size_t DIM_K,
        // elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], acc_t D[DIM_I][DIM_J],
        elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], void * D,
        elem_t C[DIM_I][DIM_J],
        int act, int shift, int relu6_shift, int full_bias_width) {

  const int no_bias = D == NULL;

  for (size_t i = 0; i < DIM_I; i++) {
    for (size_t j = 0; j < DIM_J; j++) {
      acc_t result;
      if (full_bias_width) {
        result = no_bias ? 0 : ((acc_t (*)[DIM_J])D)[i][j];
      } else {
        result = no_bias ? 0 : ((elem_t (*)[DIM_J])D)[i][j];
      }

      for (size_t k = 0; k < DIM_K; k++) {
        result += A[i][k] * B[k][j];
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

      C[i][j] = (elem_t)result;
    }
  }
}

/*
static void matmul_cpu(size_t DIM_I, size_t DIM_J, size_t DIM_K,
        // elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], acc_t D[DIM_I][DIM_J],
        elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], void * D,
        elem_t C[DIM_I][DIM_J],
        int act, int shift, int relu6_shift, int full_bias_width) {
  // TODO This function is incorrect. The activation functions, scaling down,
  // and clipping must be done BEFORE acc_t is cast down to elem_t
  
  const int no_bias = D == NULL;

  if (DIM_I % 4 == 0 && DIM_J % 4 == 0) {
    for (size_t i = 0; i < DIM_I; i += 4) {
      for (size_t j = 0; j < DIM_J; j += 4) {
        acc_t result[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
        for (size_t k = 0; k < DIM_K; k++) {
          result[0][0] += A[i  ][k] * B[k][j  ];
          result[0][1] += A[i  ][k] * B[k][j+1];
          result[0][2] += A[i  ][k] * B[k][j+2];
          result[0][3] += A[i  ][k] * B[k][j+3];

          result[1][0] += A[i+1][k] * B[k][j  ];
          result[1][1] += A[i+1][k] * B[k][j+1];
          result[1][2] += A[i+1][k] * B[k][j+2];
          result[1][3] += A[i+1][k] * B[k][j+3];

          result[2][0] += A[i+2][k] * B[k][j  ];
          result[2][1] += A[i+2][k] * B[k][j+1];
          result[2][2] += A[i+2][k] * B[k][j+2];
          result[2][3] += A[i+2][k] * B[k][j+3];

          result[3][0] += A[i+3][k] * B[k][j  ];
          result[3][1] += A[i+3][k] * B[k][j+1];
          result[3][2] += A[i+3][k] * B[k][j+2];
          result[3][3] += A[i+3][k] * B[k][j+3];
        }
        C[i  ][j  ] = result[0][0];
        C[i  ][j+1] = result[0][1];
        C[i  ][j+2] = result[0][2];
        C[i  ][j+3] = result[0][3];

        C[i+1][j  ] = result[1][0];
        C[i+1][j+1] = result[1][1];
        C[i+1][j+2] = result[1][2];
        C[i+1][j+3] = result[1][3];

        C[i+2][j  ] = result[2][0];
        C[i+2][j+1] = result[2][1];
        C[i+2][j+2] = result[2][2];
        C[i+2][j+3] = result[2][3];

        C[i+3][j  ] = result[3][0];
        C[i+3][j+1] = result[3][1];
        C[i+3][j+2] = result[3][2];
        C[i+3][j+3] = result[3][3];
      }
    }
  } else {
    for (size_t i = 0; i < DIM_I; i++) {
      for (size_t j = 0; j < DIM_J; j++) {
        acc_t result = 0;
        for (size_t k = 0; k < DIM_K; k++) {
          result += A[i][k] * B[k][j];
        }
        C[i][j] = result;
      }
    }
  }
  for (size_t i = 0; i < DIM_I; i++) {
    for (size_t j = 0; j < DIM_J; j++) {
      // acc_t result = C[i][j] + (no_bias ? 0 : D[i][j]);
      acc_t result = C[i][j];
      if (!no_bias && full_bias_width) {
        result += ((acc_t (*)[DIM_J])D)[i][j];
      } else if (!no_bias && !full_bias_width) {
        result += ((elem_t (*)[DIM_J])D)[i][j];
      }

      // Scale value down and round it
      const int divisor = 1 << shift;
      acc_t abs = result > 0 ? result : -result;
      acc_t shifted = (abs + (divisor/2)) / divisor;
      if (result < 0)
          result = -shifted;
      else
          result = shifted;

      // Clip result
      result = result > elem_t_max ? elem_t_max : (result < elem_t_min ? elem_t_min : result);

      // Apply activation function
      if (act == RELU) {
        result = result < 0 ? 0 : result;
      } else if (act == RELU6) {
        int max = 6 << relu6_shift;
        result = result < 0 ? 0 : (result > max ? max : result);
      }

      C[i][j] = (elem_t)result;
    }
  }
}
*/

// General matmul which can be run with different dataflows, or on the CPU
enum tiled_matmul_type_t {OS, WS, CPU};

// TODO add support for non-divisible tiling factors
static size_t tiling_factor(const size_t dimension, const size_t max_tile_factor) {
    const size_t start = dimension < max_tile_factor ? dimension : max_tile_factor;

    for (size_t tile_factor = start; tile_factor >= 1; tile_factor--) {
        if (dimension % tile_factor == 0)
            return tile_factor;
    }
    return 1; // We should never reach here anyway
}

static void __attribute__((unused)) tiled_matmul_option(size_t DIM_I, size_t DIM_J, size_t DIM_K,
        // elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], acc_t D[DIM_I][DIM_J],
        elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], void * D,
        elem_t C[DIM_I][DIM_J],
        int act, int shift, int relu6_shift, int full_bias_width,
        enum tiled_matmul_type_t tiled_matmul_type) {
    // const int partition_rows = BANK_NUM * BANK_ROWS / 2;
    // const int mats_in_partition = partition_rows / DIM;
    // const int mats_in_acc = ACC_ROWS / DIM;
    // const int max_tile_i_j = (int)sqrt(mats_in_acc);
    // const int max_tile_k = mats_in_partition / max_tile_i_j;

    // We use macros here instead of "const int" so that GCC const-folds sqrt
#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((int)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

    const size_t tile_i = tiling_factor(DIM_I/DIM, max_tile_i_j);
    const size_t tile_j = tiling_factor(DIM_J/DIM, max_tile_i_j);
    const size_t tile_k = tiling_factor(DIM_K/DIM, max_tile_k);

    // printf("tile_i: %lu\n", tile_i);
    // printf("tile_j: %lu\n", tile_j);
    // printf("tile_k: %lu\n", tile_k);

    if (tiled_matmul_type == OS) {
        tiled_matmul_os(DIM_I, DIM_J, DIM_K,
                A, B, D, C,
                tile_i, tile_j, tile_k,
                act, shift, relu6_shift, full_bias_width);
    } else if (tiled_matmul_type == WS) {
        tiled_matmul_ws(DIM_I, DIM_J, DIM_K,
                A, B, (acc_t (*)[DIM_J])D, C,
                tile_i, tile_j, tile_k,
                act, shift, relu6_shift, full_bias_width);
    } else /*if (tiled_matmul_type == CPU)*/ {
        matmul_cpu(DIM_I, DIM_J, DIM_K,
                A, B, (acc_t (*)[DIM_J])D, C,
                act, shift, relu6_shift, full_bias_width);
    }/* else {
        printf("unknown tiled matrix type");
        exit(1);
    }*/

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}

#endif  // SRC_MAIN_C_GEMMINI_H
