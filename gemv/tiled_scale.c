// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#define FLOAT true
#include "include/gemmini_testutils.h"

#define CHECK_RESULT 1

#define NO_BIAS 1
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
#define MAT_DIM_I 32
#define MAT_DIM_K 32
#define MAT_DIM_J 32
#endif

#define VEC_DIM 30

#define VEC_DIM_I 10//113//98
#define VEC_DIM_K 17//179//69
#define VEC_DIM_J 1

#define SCALE 2

#define NUM_INT 4
#define NUM_FP 2

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

void full_vecscale(elem_t in[VEC_DIM], elem_t out[VEC_DIM], scale_t  scale){
    for(size_t r = 0; r < VEC_DIM; r++)
      out[r]  = MVIN_SCALE(in[r], scale);
}

void full_printVec(elem_t m[VEC_DIM_I]) {
  for (size_t i = 0; i < VEC_DIM_I; ++i) {
    //for (size_t j = 0; j < VEC_DIM_J; ++j)
      printf("%d ", (int)(m[i]*1000));
    //printf("\n");
  }
  printf("\n");
}


void full_printScale(elem_t m[VEC_DIM]) {
  for (size_t i = 0; i < VEC_DIM; ++i) {
    //for (size_t j = 0; j < VEC_DIM_J; ++j)
      printf("%d ", (int)(m[i]*1000));
    //printf("\n");
  }
  printf("\n");
}

int vec_is_equal(elem_t x[VEC_DIM_I], elem_t y[VEC_DIM_I]) {
  for (size_t i = 0; i < VEC_DIM_I; ++i)
    //for (size_t j = 0; j < VEC_DIM_J; ++j)
      if (x[i] != y[i])
        return 0;
  return 1;
}
int scale_is_equal(elem_t x[VEC_DIM], elem_t y[VEC_DIM]) {
  for (size_t i = 0; i < VEC_DIM; ++i)
    //for (size_t j = 0; j < VEC_DIM_J; ++j)
      if (x[i] != y[i])
        return 0;
  return 1;
}
void full_gemv(elem_t A[VEC_DIM_I][VEC_DIM_K], elem_t B[VEC_DIM_K][VEC_DIM_J], ACC_T D[VEC_DIM_I][VEC_DIM_J], elem_t C_full[VEC_DIM_I]) {
  for (size_t r = 0; r < VEC_DIM_I; r++)
    for (size_t c = 0; c < VEC_DIM_J; c++) {
      C_full[r] = D[r][c];
      for (size_t k = 0; k < VEC_DIM_K; k++)
        C_full[r] += (SCALE * A[r][k]*B[k][c]);
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
    printf("VEC_DIM_I: %d\n", VEC_DIM_I);
    printf("VEC_DIM_J: %d\n", VEC_DIM_J);
    printf("VEC_DIM_K: %d\n", VEC_DIM_K);

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
    gemmini_flush(0);

    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

    static full_t gold_full[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
    static elem_t gold[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);

    static elem_t gemv_A[VEC_DIM_I][VEC_DIM_K] row_align(1);
    static elem_t gemv_B[VEC_DIM_K][VEC_DIM_J] row_align(1);
    static elem_t gemv_C[VEC_DIM_I] row_align(1) = {0};
    static ACC_T gemv_D[VEC_DIM_I][VEC_DIM_J] row_align_acc(1);

    static elem_t gold_gemv[VEC_DIM_I]= {0};

    static elem_t vec_in[VEC_DIM] = {0};
    static elem_t vec_out[VEC_DIM] = {0};

    static elem_t vec_gold[VEC_DIM] = {0};
//#if CHECK_RESULT == 1
#ifdef FAST
#define RAND 1
#else
#define RAND rand()
#endif
    for (size_t i = 0 ; i < VEC_DIM; i++){
        vec_in[i] = i;//RAND % 3;
        vec_gold[i] = vec_in[i] * SCALE;
    }
    /*
    // printf("Init A\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        full_A[i][j] = RAND % 2;
      }
    }

    // printf("Init B\n");
    for (size_t i = 0; i < MAT_DIM_K; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_B[i][j] = RAND % 2;
      }
    }

    // printf("Init D\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_D[i][j] = NO_BIAS ? 0 : RAND % 2;
      }
    }

    for (size_t i = 0; i < VEC_DIM_I; ++i) {
      for (size_t j = 0; j < VEC_DIM_K; ++j) {
        gemv_A[i][j] = RAND % 2;
      }
    }

    // printf("Init B\n");
    for (size_t i = 0; i < VEC_DIM_K; ++i) {
      for (size_t j = 0; j < VEC_DIM_J; ++j) {
        gemv_B[i][j] = (RAND % 3) - 1;
        //gemv_B[i][j] = i < VEC_DIM_K /2 ? 1 : -1;//(RAND % 3)-1;
      }
    }

    // printf("Init D\n");
    for (size_t i = 0; i < VEC_DIM_I; ++i) {
      for (size_t j = 0; j < VEC_DIM_J; ++j) {
        gemv_D[i][j] = NO_BIAS ? 0 : RAND % 2;
      }
    }

    for(int i = 0; i < 2; i++){
    printf("Starting gemmini matmul\n");
    // clock gating vega
    vega_clock_gate(1, 1, 0);
    unsigned long start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);

    //rr_release(cfgid);
    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);

    printf("Starting vega gemv\n");
    // clock gating gemmini
    vega_clock_gate(1, 0, 1);
    unsigned long gemv_start = read_cycles();

    tiled_gemv_auto(VEC_DIM_I, VEC_DIM_J, VEC_DIM_K,
            (elem_t*)gemv_A, (elem_t*)gemv_B, NO_BIAS ? NULL : &gemv_D[0][0], (elem_t*)gemv_C,
            VEC_DIM_K, VEC_DIM_K, VEC_DIM_I, VEC_DIM_I,
            SCALE, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);

    unsigned long gemv_end = read_cycles();
    printf("Cycles taken: %u\n", gemv_end-gemv_start);
    vega_fence();
    }

    printf("Starting slow CPU matmul\n");
    uint64_t cpu_start = read_cycles();
#ifdef FAST
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        gold_full[i][j] = MAT_DIM_K + (NO_BIAS ? 0 : (RAND % 2));
      }
    }

#else
    full_matmul(full_A, full_B, full_D, gold_full);
#endif
    uint64_t cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);
    full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);
#endif

#if CHECK_RESULT == 1
    printf("Starting slow CPU gemv\n");
    cpu_start = read_cycles();
    full_gemv(gemv_A, gemv_B, gemv_D, gold_gemv);
    cpu_end = read_cycles();
    printf("cycles taken: %u\n", cpu_end-cpu_start);
#endif
    */

   printf("starting gemv scaling \n");
   vega_clock_gate(1, 0, 1);
   uint64_t scale_start = read_cycles();
   tiled_vector_scale(VEC_DIM, SCALE, (elem_t*) vec_in, (elem_t*) vec_out, false);
   rr_fence(cfgid);
   uint64_t scale_end = read_cycles();
   printf("cycles taken: %u\n", scale_end-scale_start);
   rr_release(cfgid);


#if CHECK_RESULT == 1
    printf("check gemv\n");
    /*
    if (!vec_is_equal(gemv_C, gold_gemv)) {
      printf("C:\n");
      full_printVec(gemv_C);
      printf("Gold:\n");
      full_printVec(gold_gemv);
      printf("\n");

      exit(1);
   }
    printf("check matmul\n");
    if (!full_is_equal(full_C, gold)) {
      printf("C:\n");
      full_printMatrix(full_C);
      printf("Gold:\n");
      full_printMatrix(gold);
      printf("\n");

      exit(1);
    }
    */
    if(!scale_is_equal(vec_gold, vec_out)) {
      printf("C:\n");
      full_printScale(vec_out);
      printf("Gold:\n");
      full_printScale(vec_gold);
      printf("\n");

      exit(1);
   }
#endif

  exit(0);
}

