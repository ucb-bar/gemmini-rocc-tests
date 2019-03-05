// See LICENSE for license details.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "include/systolic.h"

int main() {
  // TODO: Should be signed
  uint8_t A[16][16];
  uint8_t B[16][16];
  for (size_t i = 0; i < 16; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      A[i][j] = i*16 + j;
      B[i][j] = (i == j) ? 1 : 0;
    }
  }

  uint64_t dummy = 0;
  assert(dummy == 0);

#ifdef DEBUG_PRINTS
  for (size_t j = 0; j < 16; ++j) {
    printf("i = %lu, j = %lu, A_ij = %d\n", i, j, A[i][j]);
    printf("i = %lu, j = %lu, B_ij = %d\n", i, j, B[i][j]);
  }
#endif

  for (size_t i = 0; i < 16; ++i) {
    matmul_mvin(dummy, A[i], i*16*sizeof(uint8_t));
    matmul_mvin(dummy, B[i], 16*16*sizeof(uint8_t) +
        i*16*sizeof(uint8_t));
  }
  

  return 0;
}
