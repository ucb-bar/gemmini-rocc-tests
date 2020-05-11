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

#define input_width 4
#define weight_width 2 //make output width 3x3
#define channel 4 //im2col 16x9
#define im2col_en 1
#define N 1

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
  static elem_t D[N][DIM][DIM] row_align(1);

      // We will try out every combination of A, B, D possible
      static elem_t C[N][DIM][DIM] row_align(1);
     static int preload[N*N*N] = {1};

      // ...whether we pass in a D or just use zeros
      static int add_to_zeros[N*N*N] = {0};

      // ...and whether we accumulate on top of the previous result
      static int accumulate[N*N*N] = {0};
      for (int i = 1; i < N*N*N; ++i){
        accumulate[i] = 0;//rand() % 2;
	preload[i] = 1;
	add_to_zeros[i] = 0;
      }

      static int no_output[N*N*N];
      for (int i = 0; i < N*N*N-1; ++i)
        no_output[i] = 0;//accumulate[i+1];
      no_output[N*N*N-1] = 0;

      for (size_t n = 0; n < N; ++n) {
        for (size_t i = 0; i < DIM; ++i) {
          for (size_t j = 0; j < DIM; ++j) {
            A[n][i][j] = i*DIM + j;
            B[n][i][j] = i==j;
            D[n][i][j] = 0;
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

      uint32_t A_addr = 0;
      uint32_t B_addr = BANK_ROWS + N*DIM;
      uint32_t D_addr = 2*BANK_ROWS + N*DIM;
      uint32_t C_addr_acc = 1 << (ADDR_LEN-1);

      // Calculate the proper destination addresses of everything
      uint32_t C_addrs[N*N*N];
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

      for (size_t n = 0; n < N; ++n)
        if (n == N-1) {
          gemmini_mvin(D[n], D_addr + n*DIM);
        } else {
          gemmini_mvin(D[n], D_addr + n*DIM);
        }

      // printf("Setting mode\n");
	 gemmini_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 0, input_width, weight_width, channel, im2col_en);
 
      // printf("Matmulling\n");
      for (size_t c = 0; c < N*N*N; ++c) {
        int a, b, d;
        operands(c, &a, &b, &d);

        uint32_t d_addr = D_addr + d*DIM;
//        if (add_to_zeros[c])
//          d_addr = GARBAGE_ADDR;
	//just always add 0 value

//        if (!preload[c]) {
//          gemmini_preload_zeros(C_addrs[c]);
//          gemmini_compute_accumulated(A_addr + a*DIM, d_addr);
//        } else {
          gemmini_preload(B_addr + b*DIM, C_addrs[c]);
         gemmini_compute_preloaded(A_addr + a*DIM, d_addr);
//        }
//      }

      // printf("Moving out\n");
      for (size_t c = 0; c < N*N*N; ++c)
//        if (!no_output[c]) {
          gemmini_mvout(C[c], C_addrs[c] & ~(1 << (ADDR_LEN-2)));
//        }

      gemmini_fence();

      // printf("Checking\n");
      for (int n = 0; n < N*N*N; ++n){
         printf("Actual (%d):\n", n);
          printMatrix(C[n]);
         exit(1);

	}
    }
  }

  exit(0);
}

