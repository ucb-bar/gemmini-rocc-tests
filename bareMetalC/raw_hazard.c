// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/systolic.h"
#include "util.h"

int main() {
  const int a_additions = 10;
  const int b_additions = 10;
  const int d_additions = 10;

  static elem_t IDENTITY[DIM][DIM];

  static elem_t result_A[DIM][DIM];
  static elem_t result_B[DIM][DIM];
  static elem_t result_D[DIM][DIM];

  static elem_t gold_A[DIM][DIM];
  static elem_t gold_B[DIM][DIM];
  static elem_t gold_D[DIM][DIM];

  for (size_t i = 0; i < DIM; i++) {
    for (size_t j = 0; j < DIM; j++) {
      IDENTITY[i][j] = i == j;
      gold_A[i][j] = i == j ? (a_additions+1) : 0;
      gold_B[i][j] = i == j ? (b_additions+1) : 0;
      gold_D[i][j] = i == j ? (d_additions+1) : 0;
    }
  }

  int IDENTITY1_addr = 0;
  int IDENTITY2_addr = BANK_SIZE;
  int A_addr = 2*BANK_SIZE;
  int B_addr = 3*BANK_SIZE;
  int D_addr = 4*BANK_SIZE;

  // printf("Moving in\n");
  matmul_mvin(IDENTITY, IDENTITY1_addr, 0, 0, 0, 0);
  matmul_mvin(IDENTITY, IDENTITY2_addr, 0, 0, 0, 0);
  matmul_mvin(IDENTITY, A_addr, 0, 0, 1, 0);
  matmul_mvin(IDENTITY, B_addr, 0, 0, 1, 0);
  matmul_mvin(IDENTITY, D_addr, 0, 0, 1, 0);
  
  // printf("Setting mode\n");
  matmul_config_ex(OUTPUT_STATIONARY, 0, 0, 0, 0, 0, 0);

  // printf("RAW with A\n");
  for (int i = 0; i < a_additions; i++) {
    if (i == 0) {
      matmul_preload(IDENTITY1_addr, A_addr, 0, 1, 0, 0);
    } else if (i == a_additions-1) {
      matmul_preload(IDENTITY1_addr, A_addr, 0, 0, 1, 0);
    } else {
      matmul_preload(IDENTITY1_addr, A_addr, 0, 0, 0, 0);
    }
    matmul_compute_preloaded(A_addr, IDENTITY2_addr);
  }

  // printf("RAW with B\n");
  for (int i = 0; i < b_additions; i++) {
    if (i == 0) {
      matmul_preload(IDENTITY1_addr, B_addr, 0, 1, 0, 0);
    } else if (i == b_additions-1) {
      matmul_preload(IDENTITY1_addr, B_addr, 0, 0, 1, 0);
    } else {
      matmul_preload(IDENTITY1_addr, B_addr, 0, 0, 0, 0);
    }
    matmul_compute_preloaded(IDENTITY2_addr, B_addr);
  }

  // printf("RAW with D\n");
  for (int i = 0; i < d_additions; i++) {
    if (i == 0) {
      matmul_preload(D_addr, D_addr, 0, 1, 0, 0);
    } else if (i == d_additions-1) {
      matmul_preload(D_addr, D_addr, 0, 0, 1, 0);
    } else {
      matmul_preload(D_addr, D_addr, 0, 0, 0, 0);
    }
    matmul_compute_preloaded(IDENTITY1_addr, IDENTITY2_addr);
  }

  // printf("Moving out\n");
  matmul_mvout(result_A, A_addr, 0, 0, 0, 1);
  matmul_mvout(result_B, B_addr, 0, 0, 0, 1);
  matmul_mvout(result_D, D_addr, 0, 0, 0, 1);

  matmul_fence();

  /*printf("A:\n");
  printMatrix(result_A);
  printf("\n");
  printMatrix(gold_A);
  printf("\n");

  printf("B:\n");
  printMatrix(result_B);
  printf("\n");
  printMatrix(gold_B);
  printf("\n");

  printf("D:\n");
  printMatrix(result_D);
  printf("\n");
  printMatrix(gold_D);
  printf("\n");*/

  if (!is_equal(result_A, gold_A) || !is_equal(result_B, gold_B) || !is_equal(result_D, gold_D))
    exit(1);

  exit(0);
}

