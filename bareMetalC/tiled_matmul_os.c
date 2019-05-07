// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/systolic.h"
#include "util.h"

#define MAT_DIM_I 16
#define MAT_DIM_K 16
#define MAT_DIM_J 16
#define TILE_I 1
#define TILE_J 1
#define TILE_K 1

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
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++)
        out[r][c] = full[r][c] >> shift;                       
} 

int main() {
    static elem_t ZERO[DIM][DIM];

    static elem_t full_A[MAT_DIM_I][MAT_DIM_K];
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J];
    static elem_t full_D[MAT_DIM_I][MAT_DIM_J];
    static elem_t full_C[MAT_DIM_I][MAT_DIM_J];

    static int64_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        full_A[i][j] = i * (i == j); // rand() % 2;
      }
    }

    for (size_t i = 0; i < MAT_DIM_K; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_B[i][j] = i == j; // rand() % 2;
      }
    }

    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_D[i][j] = 0;
      }
    }

    full_matmul(full_A, full_B, full_D, gold_full);

    sp_tiled_matmul(&full_A[0][0], &full_B[0][0], &full_D[0][0], &full_C[0][0],
        TILE_I, TILE_J, TILE_K,
        MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
        1, 1);

    matmul_fence();

    full_matshift(gold_full, gold, 0);   
    printf("C:\n");
    full_printMatrix(full_C);
    printf("Gold:\n");
    full_printMatrix(gold);
    printf("\n");
 
    if (!full_is_equal(full_C, gold))
      exit(1);
  
  exit(0);
}

