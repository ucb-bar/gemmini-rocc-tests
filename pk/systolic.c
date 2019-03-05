// See LICENSE for license details.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "include/systolic.h"

int main() {
  // TODO: Should be signed, but then need to add a bias term below
  static uint8_t A[16][16];
  static uint8_t B[16][16];
  static uint8_t C[16][16];
  static uint8_t D[16][16];
  for (size_t i = 0; i < 16; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      // A = incrementing values row by row
      A[i][j] = i*16 + j;
      // B = identity matrix
      B[i][j] = (i == j) ? 1 : 0;
      // D = zeros, try to use matmul.preload(rd = 1) instead
      D[i][j] = 0;
    }
  }

#ifdef DEBUG_PRINTS
  for (size_t j = 0; j < 16; ++j) {
    printf("i = %lu, j = %lu, A_ij = %d\n", i, j, A[i][j]);
    printf("i = %lu, j = %lu, B_ij = %d\n", i, j, B[i][j]);
  }
#endif

  // TODO: eliminate the need for this dummy variable
  // Need to modify rocc-software/src/xcustom.h
  uint64_t dummy = 0;
  assert(dummy == 0);

  for (size_t i = 0; i < 16; ++i) {
    matmul_mvin(dummy, A[i], i*16*sizeof(uint8_t));
    matmul_mvin(dummy, B[i], 16*16*sizeof(uint8_t) +
        i*16*sizeof(uint8_t));
    matmul_mvin(dummy, D[i], 2*16*16*sizeof(uint8_t) +
        i*16*sizeof(uint8_t));
  }

  matmul_setmode(dummy, 0);
  matmul_preload(dummy, 3*16*16*sizeof(uint8_t), 2*16*16*sizeof(uint8_t));
  matmul_compute_preloaded(dummy, 0x0, 16*16*sizeof(uint8_t));

  for (size_t i = 0; i < 16; ++i) {
    // TODO: mvout rs1 = scratchpad or DRAM address? mismatch bet spike and ISA spec
    matmul_mvout(dummy, C[i], 3*16*16*sizeof(uint8_t) + i*16*sizeof(uint8_t));
  }
#define DEBUG_PRINTS
  for (size_t i = 0; i < 16; ++i) {
    for (size_t j = 0; j < 16; ++j) {
#ifdef DEBUG_PRINTS
      printf("C[%lu][%lu] = %d\n", i, j, C[i][j]);
      printf("A[%lu][%lu] = %d\n", i, j, A[i][j]);
#endif
      assert(C[i][j] == A[i][j]);
    }
  }

  return 0;
}
