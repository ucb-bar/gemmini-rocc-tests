// See LICENSE for license details.

#ifndef SRC_MAIN_C_SYSTOLIC_H
#define SRC_MAIN_C_SYSTOLIC_H

// Dimension of the systolic array
// Should be tileColumns*meshColumns
#define DIM 4
// Datatype of the systolic array
// TODO: Should be signed, but then need to add a bias term below
typedef uint16_t elem_t;

// Matmul utility functions
typedef struct matmul_t {
    elem_t A[DIM][DIM];
    elem_t B[DIM][DIM];
    elem_t C[DIM][DIM];
    elem_t D[DIM][DIM];

    elem_t A_tp[DIM][DIM];
    elem_t B_tp[DIM][DIM];

    elem_t gold[DIM][DIM];
} matmul_t;

void matmul(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], elem_t C[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C[r][c] += A[r][k]*B[k][c];
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

int matmul_is_correct(matmul_t * mm) {
  for (size_t i = 0; i < DIM; ++i)
    for (size_t j = 0; j < DIM; ++j)
      if (mm->C[i][j] != mm->gold[i][j])
          return 0;
  return 1;
}

// Run this function after setting the values of A, B, and D
void init_matmul(matmul_t * mm) {
  matmul(mm->A, mm->B, mm->D, mm->gold);
  transpose(mm->A, mm->A_tp);
  transpose(mm->B, mm->B_tp);
}

int rand() {
  static int x = 1;
  x ^= (x << 21);
  x ^= (x >> 35);
  x ^= (x << 4);
  return x >= 0 ? x : -x;
}

// Accelerator interface
#include "rocc-software/src/xcustom.h"

#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 8
#define k_SETMODE 9

#define XCUSTOM_ACC 3

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct, 10, 11)

#define matmul_mvin(dram_addr, spad_addr)                        \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, spad_addr, k_MVIN);
#define matmul_mvout(dram_addr, spad_addr)                              \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, spad_addr, k_MVOUT);

#define matmul_compute_preloaded(A, B)                              \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_COMPUTE_PRELOADED);

#define matmul_preload(rd, D, C)                              \
  ROCC_INSTRUCTION(XCUSTOM_ACC, rd, D, C, k_PRELOAD);
#define matmul_preload_no_rd(D, C)                              \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_PRELOAD);
#define matmul_setmode(mode)                              \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, mode, 0, k_SETMODE);

#endif  // SRC_MAIN_C_SYSTOLIC_H
