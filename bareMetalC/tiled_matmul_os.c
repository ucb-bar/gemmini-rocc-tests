// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/systolic.h"
#include "util.h"

#define CHECK_RESULT 0

#define MAT_DIM_I 512
#define MAT_DIM_K 512
#define MAT_DIM_J 512
#define TILE_I 4
#define TILE_J 4
#define TILE_K 4

void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*MAT_DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*MAT_DIM_J + c));
    }
    printf("\n");
  }
}

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], elem_t D[MAT_DIM_I][MAT_DIM_J], int64_t C_full[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = D[r][c];
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

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

void full_matshift(int64_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], int shift) {
  int divisor = 1 << shift;

  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
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

int main() {
    static elem_t ZERO[DIM][DIM];

    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static elem_t full_D[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);

    static int64_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

    // printf("Init A\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        full_A[i][j] = rand() % 2; (rand() % 64) - 32;
      }
    }

    // printf("Init B\n");
    for (size_t i = 0; i < MAT_DIM_K; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_B[i][j] = rand() % 2; (rand() % 64) - 32;
      }
    }

    // printf("Init D\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_D[i][j] = rand() % 2; (rand() % 64) - 32;
      }
    }

#if CHECK_RESULT == 1
    // printf("Starting CPU matmul\n");
    full_matmul(full_A, full_B, full_D, gold_full);
    full_matshift(gold_full, gold, 0);
#endif

    const int I0 = MAT_DIM_I / (TILE_I*DIM);
    const int J0 = MAT_DIM_J / (TILE_J*DIM);
    const int K0 = MAT_DIM_K / (TILE_K*DIM);

    // printf("Starting systolic matmul\n");
    unsigned long start = read_cycles();

    matmul_config_ex(OUTPUT_STATIONARY, NO_ACTIVATION, 0, 0, 0, 0, 0, 0);

    for (size_t i0 = 0; i0 < I0; i0++)
      for (size_t j0 = 0; j0 < J0; j0++)
        for (size_t k0 = 0; k0 < K0; k0++) {
          // printf("i0: %u, j0: %u, k0: %u\n", i0, j0, k0);

          int first = i0 == 0 && j0 == 0 && k0 == 0;
          int last = (i0 == I0-1) && (j0 == J0-1) && (k0 == K0-1);

          elem_t (*preload)[][MAT_DIM_J] = k0 == 0 ? &full_D : &full_C;

          sp_tiled_matmul(&full_A[i0*TILE_I*DIM][k0*TILE_K*DIM],
              &full_B[k0*TILE_K*DIM][j0*TILE_J*DIM],
              &(*preload)[i0*TILE_I*DIM][j0*TILE_J*DIM],
              &full_C[i0*TILE_I*DIM][j0*TILE_J*DIM],
              TILE_I, TILE_J, TILE_K,
              MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
              first, last);
        }

    matmul_fence();

    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);

#if CHECK_RESULT == 1
    if (!full_is_equal(full_C, gold)) {
      printf("C:\n");
      full_printMatrix(full_C);
      printf("Gold:\n");
      full_printMatrix(gold);
      printf("\n");

      exit(1);
    }
#endif

  exit(0);
}

