// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include <time.h>
#include "include/gemmini.h"

#define N (2)

void operands(int c, int * a, int * b, int * d) {
  *d = c % N;
  *b = (c / N) % N;
  *a = c / (N*N);
}

#if 3*N*DIM > (BANK_NUM * BANK_ROWS) || N*N*N*DIM > ACC_ROWS
//#error scratchpad or accumulator not big enough
#endif

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  static elem_t ZERO[DIM][DIM];

  gemmini_flush(0);

  static elem_t A[N][DIM][DIM] row_align(1);
  static elem_t B[N][DIM][DIM] row_align(1);

  // We will try out every combination of A, B, D possible
  static elem_t C[N*N*N][DIM][DIM] row_align(1);
  static int64_t gold_full[N*N*N][DIM][DIM];
  static elem_t gold[N*N*N][DIM][DIM];

  // ...taking into account whether we preload new weights or re-use the old ones
  static int preload[N*N*N] = {1};
  for (int i = 1; i < N*N*N; ++i)
    preload[i] = rand() % 2;

  // ...and whether we accumulate on top of the previous result
  static int accumulate[N*N*N] = {0};
  for (int i = 1; i < N*N*N; ++i)
    accumulate[i] = rand() % 2;

  static int no_output[N*N*N];
  for (int i = 0; i < N*N*N-1; ++i)
    no_output[i] = accumulate[i+1];
  no_output[N*N*N-1] = 0;

  // Print the sequence out
  /*printf("Preloads: ");
  for (int i = 0; i < N*N*N; ++i)
    printf("%d, ", preload[i]);
  printf("\n");
  printf("\n");
  printf("Accumulates: ");
  for (int i = 0; i < N*N*N; ++i)
    printf("%d, ", accumulate[i]);
  printf("\n");
  printf("No outputs: ");
  for (int i = 0; i < N*N*N; ++i)
    printf("%d, ", no_output[i]);
  printf("\n");*/

  for (size_t n = 0; n < N; ++n) {
    for (size_t i = 0; i < DIM; ++i) {
      for (size_t j = 0; j < DIM; ++j) {
        A[n][i][j] = (rand() % 64) - 32;
        B[n][i][j] = (rand() % 64) - 32;
      }
    }
  }

  for (size_t g = 0; g < N*N*N; ++g) {
    int a, b, d;
    operands(g, &a, &b, &d);

    // We need to find the last B value in case we aren't preloading new weights
    for (int last_g = g; last_g >= 0; --last_g) {
        int tmp_a, tmp_d;
        if (preload[last_g]) {
            operands(last_g, &tmp_a, &b, &tmp_d);
            break;
        }
    }

    matmul(A[a], B[b], ZERO, gold_full[g]);

    if (accumulate[g])
      matadd(gold_full[g], gold_full[g-1], gold_full[g]);
  }

  for (size_t g = 0; g < N*N*N; ++g) {
    matshift(gold_full[g], gold[g], 0);
  }

  int A_addr = 0;
  int B_addr = N*DIM;
  uint32_t C_addr_acc = 1 << (ADDR_LEN-1);

  // Calculate the proper destination addresses of everything
  int C_addrs[N*N*N];
  for (size_t c = 0; c < N*N*N; ++c)
    C_addrs[c] = C_addr_acc + c*DIM;
  for (size_t c = 0; c < N*N*N; ++c) {
    int last_c;
    for (last_c = c; last_c >= 0; --last_c)
      if (!accumulate[last_c])
        break;
    if (c != last_c)
      C_addrs[c] = C_addrs[last_c] | (1 << (ADDR_LEN-2));
  }

  // printf("Moving in\n");
  for (size_t n = 0; n < N; ++n)
    gemmini_mvin(A[n], A_addr + n*DIM);

  for (size_t n = 0; n < N; ++n)
    gemmini_mvin(B[n], B_addr + n*DIM);

  // printf("Setting mode\n");
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0, 0, 0);

  // printf("Matmulling\n");
  for (size_t c = 0; c < N*N*N; ++c) {
    int a, b, d;
    operands(c, &a, &b, &d);

    if (!preload[c]) {
      matmul_preload_zeros(C_addrs[c]);
      gemmini_compute_accumulated(A_addr + a*DIM, GARBAGE_ADDR);
    } else {
      gemmini_preload(B_addr + b*DIM, C_addrs[c]);
      gemmini_compute_preloaded(A_addr + a*DIM, GARBAGE_ADDR);
    }
  }

  // printf("Moving out\n");
  for (size_t c = 0; c < N*N*N; ++c)
    if (!no_output[c]) {
      gemmini_mvout(C[c], C_addrs[c] & ~(1 << (ADDR_LEN-2)));
    }

  gemmini_fence();

  // printf("Checking\n");
  for (int n = 0; n < N*N*N; ++n)
    if (!no_output[n] && !is_equal(C[n], gold[n])) {
      printf("Actual (matrix %d):\n", n);
      printMatrix(C[n]);
      printf("\nCorrect:\n");
      printMatrix(gold[n]);

      printf("\nFAIL\n");
      exit(1);
    }

  printf("PASS\n");
  exit(0);
}

