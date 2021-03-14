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

#define NO_BIAS 1
#define REPEATING_BIAS 0
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#error variable-bitwidth bias not currently supported
#endif

#define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

#define CHECK_RESULT 0

#define WARMUP 1

#ifndef BAREMETAL
#define MAT_DIM 512
#define MAT_DIM_I 512
#define MAT_DIM_K 512
#define MAT_DIM_J 512
#else
#define MAT_DIM 512
#define MAT_DIM_I MAT_DIM+64
#define MAT_DIM_K MAT_DIM+64
#define MAT_DIM_J MAT_DIM+64
#endif
#define num_thread 4
#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

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


void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*MAT_DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*MAT_DIM_J + c));
    }
    printf("\n");
  }
}

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J]) {
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

void full_matshift(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], int shift) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      // Bitshift and round element
      full_t shifted = ROUNDING_RIGHT_SHIFT(full[r][c], shift);

      // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = shifted > elem_t_max ? elem_t_max : (shifted < elem_t_min ? elem_t_min : shifted);
      out[r][c] = elem;
#else
      out[r][c] = shifted; // TODO should we also saturate when using floats?
#endif
    }
} 

static elem_t in_A[num_thread][MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN) = {1};
static elem_t in_B[num_thread][MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {1};
//static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
//static ACC_T bias[MAT_DIM_I][MAT_DIM_J] row_align_acc(1) = {0};
static elem_t Out[num_thread][MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {1};
//static elem_t gold[MAT_DIM_I][MAT_DIM_J];

void thread_entry(int cid, int nc)
{
  for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
    barrier(nc);
  }
#if CHECK_RESULT == 1
    // printf("Init A\n");
  if(cid == 1){
    for (size_t i = 0; i < MAT_DIM_I/2; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        in_A[i][j] = rand() % 3 - 1;
      }
    }
  }
  if(cid == 3){
    // printf("Init B\n");
    for (size_t i = 0; i < MAT_DIM_K/2; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        in_B[i][j] = rand() % 3 -1;
      }
    }
  }
  if(cid == 0){
    
    for (size_t i = MAT_DIM_I/2; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        in_A[i][j] = rand() % 3 - 1;
      }
    }
  }
  if(cid == 2){
    // printf("Init B\n");
    for (size_t i = MAT_DIM_K/2; i < MAT_DIM_K; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        in_B[i][j] = rand() %  3 - 1;
      }
    }
  }
  barrier(nc);
  
  if(cid == 0){
    // printf("Init D\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        bias[i][j] = NO_BIAS ? 0 : rand() % 2;
      }
    }
	static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    
    printf("Starting slow CPU matmul\n");
    uint64_t cpu_start = read_cycles();
    full_matmul(in_A, in_B, bias, gold_full);
    uint64_t cpu_end = read_cycles();
    printf("Cycles taken: %llu\n", cpu_end-cpu_start);
    full_matshift(gold_full, gold, 0);

  }
  
#endif

	 elem_t* A = (elem_t*) in_A + cid*MAT_DIM_K*MAT_DIM_I;//MAT_DIM_K*(MAT_DIM/2)*(cid/2);
	 elem_t* B = (elem_t*) in_B + cid*MAT_DIM_K*MAT_DIM_J;//(MAT_DIM/2)*(cid%2);
	 elem_t* C = (elem_t*) Out + cid*MAT_DIM_I*MAT_DIM_J;//(MAT_DIM/2)*(cid%2) + MAT_DIM_J*(MAT_DIM/2)*(cid/2);
//	 acc_t * D = (acc_t*) bias + (MAT_DIM_J/2)*(cid%2) + MAT_DIM_J*(MAT_DIM_I/2)*(cid/2);
#if WARMUP == 1
	gemmini_flush(0);
	 barrier(nc);
		if(cid == 0) {
			uint64_t cid0_start = read_cycles();
			uint64_t cid0_cycles = 0;
			while(cid0_cycles < 200000){
				uint64_t new = read_cycles();
				cid0_cycles = new - cid0_start;
			}
		}
		if(cid == 1) {
			uint64_t cid1_start = read_cycles();
			uint64_t cid1_cycles = 0;
			while(cid1_cycles < 50000){
				uint64_t new = read_cycles();
				cid1_cycles = new - cid1_start;
			}
		}
		if(cid == 2) {
		   uint64_t cid2_start = read_cycles();
			uint64_t cid2_cycles = 0;
			while(cid2_cycles < 400000){
				uint64_t new = read_cycles();
				cid2_cycles = new - cid2_start;
			}
		}
		
	 uint64_t warm_start = read_cycles();
  for(int j = 0; j < nc; j++){
	if(j==cid)	{
		 tiled_matmul_auto(MAT_DIM, MAT_DIM, MAT_DIM, 
				A, B, NULL, C,
			   A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            A_TRANSPOSE, B_TRANSPOSE,
            WS);
	}
  }
  uint64_t warm_end = read_cycles();
  for(int i = 0; i < nc; i++){
	  if (i == cid) {
		 printf("Thread %d Cycles taken: %llu\n", cid, warm_end - warm_start);
		 const int total_macs = MAT_DIM * MAT_DIM * MAT_DIM;
		 const int ideal_cycles = total_macs / (DIM * DIM);
		 const int utilization = 100 * ideal_cycles / (warm_end-warm_start);
		 printf("Utilization: %d%%\n", utilization);
	  }
	  barrier(nc);
  }
#endif

  for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Starting gemmini tiled_matmul\n");
    barrier(nc);
  }
  gemmini_flush(0);


  barrier(nc);
		if(cid == 0) {
			uint64_t cid0_start = read_cycles();
			uint64_t cid0_cycles = 0;
			while(cid0_cycles < 200000){
				uint64_t new = read_cycles();
				cid0_cycles = new - cid0_start;
			}
		}
		if(cid == 1) {
			uint64_t cid1_start = read_cycles();
			uint64_t cid1_cycles = 0;
			while(cid1_cycles < 50000){
				uint64_t new = read_cycles();
				cid1_cycles = new - cid1_start;
			}
		}
		if(cid == 2) {
		   uint64_t cid2_start = read_cycles();
			uint64_t cid2_cycles = 0;
			while(cid2_cycles < 400000){
				uint64_t new = read_cycles();
				cid2_cycles = new - cid2_start;
			}
		}
  uint64_t start = read_cycles();
  //barrier(nc);

  for(int j = 0; j < nc; j++){
		//printf("thread: %d, loop: %d \n", cid, j);
//	 if(j == cid && j == 0)
	if(j==cid){	
		 tiled_matmul_auto(MAT_DIM, MAT_DIM, MAT_DIM, 
				A, B, NULL, C,
			   A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            A_TRANSPOSE, B_TRANSPOSE,
            WS);
	}
  }

  uint64_t end = read_cycles();

  for(int i = 0; i < nc; i++){
	  if (i == cid) {
		 printf("Thread %d Cycles taken: %llu\n", cid, end - start);
		 const int total_macs = MAT_DIM * MAT_DIM * MAT_DIM;
		 const int ideal_cycles = total_macs / (DIM * DIM);
		 const int utilization = 100 * ideal_cycles / (end-start);
		 printf("Utilization: %d%%\n", utilization);
	  }
	  barrier(nc);
  }

#if CHECK_RESULT == 1
    if(cid == 0){
		 if (!full_is_equal(Out, gold)) {
			printf("wrong result: thread %d \n", cid);
//			printf("Gold:\n");
//			full_printMatrix(gold);
			printf("C:\n");
			full_printMatrix(Out);
			printf("\n");

	//      exit(1);
		 }
	 }
	 barrier(nc);
#endif


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

    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

#if CHECK_RESULT == 1
    // printf("Init A\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        full_A[i][j] = rand() % 2;
      }
    }

    // printf("Init B\n");
    for (size_t i = 0; i < MAT_DIM_K; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_B[i][j] = rand() % 2;
      }
    }

    // printf("Init D\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_D[i][j] = NO_BIAS ? 0 : rand() % 2;
      }
    }

    printf("Starting slow CPU matmul\n");
    unsigned long cpu_start = read_cycles();
    full_matmul(full_A, full_B, full_D, gold_full);
    unsigned long cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);
    full_matshift(gold_full, gold, 0);
#endif

    printf("Starting gemmini matmul\n");
    unsigned long start = read_cycles();
/*
    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, 0, 0, false,
            WS);
*/
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

