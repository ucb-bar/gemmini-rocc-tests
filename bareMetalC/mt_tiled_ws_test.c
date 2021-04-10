// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "util.h"

#define NO_BIAS 0
#define REPEATING_BIAS 1

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

#define MAT_DIM 512
#ifndef BAREMETAL
#define MAT_DIM_I 512
#define MAT_DIM_K 512
#define MAT_DIM_J 512
#else
#define MAT_DIM_I 512
#define MAT_DIM_K 512
#define MAT_DIM_J 512

#endif

#if A_TRANSPOSE==0
#define A_STRIDE MAT_DIM_K
#else
#define A_STRIDE MAT_DIM_I
#endif

#if B_TRANSPOSE==0
#define B_STRIDE MAT_DIM_J
#else
#define B_STRIDE MAT_DIM_K
#endif

#define WARMUP 1
#define num_thread 4

static elem_t in_A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN) = {1};
static elem_t in_B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {1};
static acc_t bias[MAT_DIM_I][MAT_DIM_J] row_align_acc(MAX_BLOCK_LEN_ACC) = {1};
static elem_t Out[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {1};

void thread_entry(int cid, int nc)
{
  for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
    barrier(nc);
  }
  int gemmini_cid = cid - 1;
  elem_t* A = (elem_t*) in_A + MAT_DIM_K*(MAT_DIM/2)*(gemmini_cid/2);
  elem_t* B = (elem_t*) in_B + (MAT_DIM/2)*(gemmini_cid%2);
  elem_t* C = (elem_t*) Out + (MAT_DIM/2)*(gemmini_cid%2) + MAT_DIM_J*(MAT_DIM/2)*(gemmini_cid/2);
//	 acc_t * D = (acc_t*) bias + (MAT_DIM_J/2)*(cid%2) + MAT_DIM_J*(MAT_DIM_I/2)*(cid/2);
#if WARMUP == 1
  //if(cid > 0 && cid < nc - 1) gemmini_flush(0);
  barrier(nc);
  uint64_t warm_start = read_cycles();

  uint64_t warm_end = read_cycles();
  for(int i = 1; i < nc-1; i++){
	  if (i == cid) {
		 printf("Thread %d warmup Cycles taken: %llu\n", cid, warm_end - warm_start);
		 const int total_macs = MAT_DIM * MAT_DIM * MAT_DIM / num_thread;
		 const int ideal_cycles = total_macs / (DIM * DIM);
		 const int utilization = 100 * ideal_cycles / (warm_end-warm_start);
		 printf("Utilization: %d%%\n", utilization);
	  }
	  barrier(nc);
  }
#endif

  for (int i = 1; i < nc-1; i++) {
    if (i == cid) printf("Starting gemmini tiled_matmul\n");
    barrier(nc);
  }
//  if(cid > 0 && cid < nc - 1) gemmini_flush(0);


  barrier(nc);
  uint64_t start = read_cycles();
  //barrier(nc);


  uint64_t end = read_cycles();

  for(int i = 1; i < nc-1; i++){
     if (i == cid) {
	 printf("Thread %d Cycles taken: %llu\n", cid, end - start);
	 const int total_macs = MAT_DIM * MAT_DIM * MAT_DIM / num_thread;
	 const int ideal_cycles = total_macs / (DIM * DIM);
	 const int utilization = 100 * ideal_cycles / (end-start);
	 printf("Utilization: %d%%\n", utilization);
     }
  barrier(nc);
  }

  exit(0);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

#if A_TRANSPOSE==0
    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
#else
    static elem_t full_A[MAT_DIM_K][MAT_DIM_I] row_align(1);
#endif

#if B_TRANSPOSE==0
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
#else
    static elem_t full_B[MAT_DIM_J][MAT_DIM_K] row_align(1);
#endif

    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static acc_t full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

    printf("Starting gemmini matmul\n");
    printf("I: %d, J: %d, K: %d\n", MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
    printf("NO_BIAS: %d, REPEATING_BIAS: %d\n", NO_BIAS, REPEATING_BIAS);
    printf("A_TRANSPOSE: %d, B_TRANSPOSE: %d\n", A_TRANSPOSE, B_TRANSPOSE);
    unsigned long start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            A_TRANSPOSE, B_TRANSPOSE,
            false, false,
            WS);

    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);

    const int total_macs = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
    const int ideal_cycles = total_macs / (DIM * DIM);
    const int utilization = 100 * ideal_cycles / (end-start);
    printf("Utilization: %d%%\n", utilization);

  exit(0);
}

