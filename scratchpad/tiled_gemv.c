// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#define FLOAT true
#include "include/gemmini_testutils.h"

#define CHECK_RESULT 1
#define BASE_ADDR 0x70000000L
#define PageSize 4096

#define NO_BIAS 0
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

#ifndef BAREMETAL
#define MAT_DIM_I 512
#define MAT_DIM_K 512
#define MAT_DIM_J 512
#else
#define MAT_DIM_I 75//113
#define MAT_DIM_K 185//179
#define MAT_DIM_J 1
#endif

#define SCALE 1 // -1

#define NUM_INT 4
#define NUM_FP 2

void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*MAT_DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*MAT_DIM_J + c));
    }
    printf("\n");
  }
}

void full_gemv(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], elem_t C_full[MAT_DIM_I]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r] = D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r] += (SCALE * A[r][k]*B[k][c]);
    }
}

void full_printMatrix(elem_t m[MAT_DIM_I]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    //for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i]);
    //printf("\n");
  }
  printf("\n");
}

void full_printMatrixB(elem_t m[MAT_DIM_K][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_K; ++i) {
    //for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i][0]);
    //printf("\n");
  }
  printf("\n");
}
int full_is_equal(elem_t x[MAT_DIM_I], elem_t y[MAT_DIM_I]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    //for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i] != y[i])
        return 0;
  return 1;
}

void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], acc_scale_t scale) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      // Scale element
      full_t scaled = ACC_SCALE(full[r][c], scale);

      // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled; // TODO should we also saturate when using floats?
#endif
    }
} 

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    printf("MAT_DIM_I: %d\n", MAT_DIM_I);
    printf("MAT_DIM_J: %d\n", MAT_DIM_J);
    printf("MAT_DIM_K: %d\n", MAT_DIM_K);

    int cfgid = 0;
    int i = NUM_FP+NUM_INT-1;
    //for(int i = 0; i < 2; i++){
        bool acquired = rr_acquire_single(cfgid, i);
        if(acquired){
            printf("gemmini %d acquired to cfgid %d\n", i, cfgid);
            //break;
        }
    //}
    rr_set_opc(XCUSTOM_ACC, cfgid);
    vega_flush(0);

    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static elem_t full_C[MAT_DIM_I] row_align(1) = {0};
    static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

    static full_t gold_full[MAT_DIM_I];
    static elem_t gold[MAT_DIM_I]= {0};

    uint64_t A_copy_addr = (BASE_ADDR & ~(PageSize-1));
    printf("A copy addr: 0x%08lx\n", A_copy_addr);
    uint64_t B_copy_addr = A_copy_addr + (MAT_DIM_I * MAT_DIM_K) * sizeof(elem_t);// + 64*3;
    uint64_t C_copy_addr = B_copy_addr + (MAT_DIM_K * MAT_DIM_J) * sizeof(elem_t);// + 64*3;
    uint64_t D_copy_addr = C_copy_addr + (MAT_DIM_I) * sizeof(elem_t);
#if CHECK_RESULT == 1
#ifdef FAST
#define RAND 1
#else
#define RAND rand()
#endif
    // printf("Init A\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        full_A[i][j] = RAND % 2;
      }
    }

    // printf("Init B\n");
    for (size_t i = 0; i < MAT_DIM_K; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_B[i][j] = (RAND % 3) - 1;
        //full_B[i][j] = i < MAT_DIM_K /2 ? 1 : -1;//(RAND % 3)-1;
      }
    }

    // printf("Init D\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_D[i][j] = NO_BIAS ? 0 : RAND % 2;
      }
    }
#endif
    printf("perform memcpy\n");
    bool granted = false;
    int index = 0;
    printf("copy A\n");
    memcpy((elem_t*) A_copy_addr, (elem_t*) full_A, sizeof(elem_t)*MAT_DIM_I*MAT_DIM_K);
    printf("copy B\n");
    memcpy((elem_t*) B_copy_addr, (elem_t*) full_B, sizeof(elem_t)*MAT_DIM_K*MAT_DIM_J);
    printf("copy D\n");
    memcpy((acc_t*) D_copy_addr, (acc_t*) full_D, sizeof(acc_t)*MAT_DIM_I*MAT_DIM_J);
    
    printf("clock gating gemmini\n");
    vega_clock_gate(1, 0, 1);

    printf("Starting vega gemv\n");
    unsigned long start = read_cycles();

    tiled_gemv_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)A_copy_addr, (elem_t*)B_copy_addr, NO_BIAS ? NULL : (acc_t*) D_copy_addr, (elem_t*)C_copy_addr,
            MAT_DIM_K, MAT_DIM_K, MAT_DIM_I, MAT_DIM_I,
            SCALE, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);
    rr_fence(cfgid);

    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);
    rr_release(cfgid); 
#if CHECK_RESULT == 1
    printf("Starting slow CPU gemv\n");
    unsigned long cpu_start = read_cycles();
    full_gemv(full_A, full_B, full_D, gold);
    unsigned long cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);
    //full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);
    printf("copy C\n");
    memcpy((elem_t*) full_C, (elem_t*) C_copy_addr, sizeof(elem_t)*MAT_DIM_I);
#endif

#if CHECK_RESULT == 1
    if (!full_is_equal(full_C, gold)) {
      printf("C:\n");
      full_printMatrix(full_C);
      printf("Gold:\n");
      full_printMatrix(gold);
      printf("B:\n");
      full_printMatrixB(full_B);
      printf("\n");

      exit(1);
   }
#endif

  exit(0);
}

