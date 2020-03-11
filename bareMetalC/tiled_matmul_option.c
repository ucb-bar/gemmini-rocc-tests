// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"

#ifndef BAREMETAL
#define MAT_DIM_I 256
#define MAT_DIM_K 256
#define MAT_DIM_J 256
#else
#define MAT_DIM_I 33
#define MAT_DIM_K 28
#define MAT_DIM_J 32
#endif

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], acc_t D[MAT_DIM_I][MAT_DIM_J],
  int64_t C_full[MAT_DIM_I][MAT_DIM_J], bool repeating_bias)
{
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = D[repeating_bias ? 0 : r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

void full_printMatrix64Bit(int64_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%lld ", m[i][j]);
    printf("\n");
  }
}

void full_matshift(int64_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], int shift) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      // Bitshift and round element
      int64_t shifted = ROUNDING_RIGHT_SHIFT(full[r][c], shift);

      // Saturate and cast element
      int64_t elem = shifted > elem_t_max ? elem_t_max : (shifted < elem_t_min ? elem_t_min : shifted);
      out[r][c] = elem;
    }
}

void full_matrelu(elem_t in[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++)
      out[r][c] = in[r][c] > 0 ? in[r][c] : 0;
}

void full_matrelu6(elem_t in[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], int scale) {
  int max = 6 * scale;

  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      elem_t positive = in[r][c] > 0 ? in[r][c] : 0;
      out[r][c] = positive > max ? max : positive;
    }
}

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  gemmini_flush(0);

#ifdef BAREMETAL
  for (enum tiled_matmul_type_t option = OS; option <= WS; option++) {
    for (int activation = 0; activation <= 1; activation++) {
      for (int shift = 0; shift <= 1; shift += 1) {
#else
  for (enum tiled_matmul_type_t option = OS; option <= CPU; option++) {
    for (int activation = 0; activation <= 2; activation++) {
      for (int shift = 0; shift <= 12; shift += 6) {
#endif
        for (bool no_bias = true; no_bias; no_bias = false) {
          for (bool repeating_bias = true; repeating_bias; repeating_bias = false) {

            size_t relu6_shift = shift + 1;

            static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
            static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
            static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
            static acc_t full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

            static int64_t gold_full[MAT_DIM_I][MAT_DIM_J];
            static elem_t gold[MAT_DIM_I][MAT_DIM_J];

            // printf("Init A\n");
            for (size_t i = 0; i < MAT_DIM_I; ++i) {
              for (size_t j = 0; j < MAT_DIM_K; ++j) {
                full_A[i][j] = (rand() % 3) - 1;
              }
            }

            // printf("Init B\n");
            for (size_t i = 0; i < MAT_DIM_K; ++i) {
              for (size_t j = 0; j < MAT_DIM_J; ++j) {
                full_B[i][j] = (rand() % 3) - 1;
              }
            }

            // printf("Init D\n");
            for (size_t i = 0; i < (repeating_bias ? 1 : MAT_DIM_I); ++i) {
              for (size_t j = 0; j < MAT_DIM_J; ++j) {
                full_D[i][j] = no_bias ? 0 : ((rand() % 3) - 1);
              }
            }

            printf("Starting CPU matmul\n");
            full_matmul(full_A, full_B, full_D, gold_full, repeating_bias);
            full_matshift(gold_full, gold, shift);

            if (activation == RELU) {
              full_matrelu(gold, gold);
            } else if (activation == RELU6) {
              full_matrelu6(gold, gold, 1 << relu6_shift);
            }

            printf("Starting gemmini matmul\n");
            tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
                    full_A, full_B, no_bias ? NULL : &full_D[0][0], full_C,
                    activation, shift, relu6_shift, repeating_bias,
                    option);

            if (!full_is_equal(full_C, gold)) {
              printf("\nINCORRECT!\n");
              printf("option: %d\n", option);
              printf("activation: %d\n", activation);
              printf("shift: %d\n", shift);
              printf("relu_shift: %d\n", relu6_shift);
              printf("no_bias: %d\n", no_bias);
              printf("repeating_bias: %d\n", repeating_bias);

              printf("C:\n");
              full_printMatrix(full_C);
              printf("Gold:\n");
              full_printMatrix(gold);
              printf("Gold full:\n");
              full_printMatrix64Bit(gold_full);
              printf("\n");

              exit(1);
            }
          }
        }
      }
    }
  }

  exit(0);
}

