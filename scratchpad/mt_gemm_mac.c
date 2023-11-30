// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#define SIZE 2

#include "include/gemmini_testutils.h"
#include "util.h"
#include "include/rerocc.h"
#include "include/dma.h"

#define ACTIVATION NO_ACTIVATION

#define NO_BIAS 1
#define REPEATING_BIAS 1

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

#ifndef BAREMETAL

#define MAT_DIM_I 128
#define MAT_DIM_K 512
#define MAT_DIM_J 256

#else

#define MAT_DIM_I 512
#define MAT_DIM_K 128
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

#define PageSize 4096

// 0,1/2,3
#define ACC_ID1 0
#define ACC_ID2 1

#if A_TRANSPOSE==0
static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
static elem_t full_A2[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
#else
static elem_t full_A[MAT_DIM_K][MAT_DIM_I] row_align(MAX_BLOCK_LEN);
#endif

#if B_TRANSPOSE==0
static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static elem_t full_B2[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
#else
static elem_t full_B[MAT_DIM_J][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
#endif

static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static acc_t full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(MAX_BLOCK_LEN_ACC);
static elem_t full_C2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static acc_t full_D2[MAT_DIM_I][MAT_DIM_J] row_align_acc(MAX_BLOCK_LEN_ACC);


void thread_entry(int cid, int nc){
    for(int i = 0; i < nc; i ++){
        if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
        barrier(nc);
    }
    uint64_t A_copy_addr = (BASE_ADDR & ~(PageSize-1))+32;
    uint64_t B_copy_addr = A_copy_addr + (MAT_DIM_I * MAT_DIM_K) * sizeof(elem_t)+64;
    uint64_t C_copy_addr = B_copy_addr + (MAT_DIM_K * MAT_DIM_J) * sizeof(elem_t)+64;
    uint64_t D_copy_addr = C_copy_addr + (MAT_DIM_I * MAT_DIM_J) * sizeof(elem_t)+64;
    uint64_t A_copy_addr2 = D_copy_addr;// + (MAT_DIM_J) * sizeof(acc_t);
    uint64_t B_copy_addr2 = A_copy_addr2 + (MAT_DIM_I * MAT_DIM_K) * sizeof(elem_t)+64;
    uint64_t C_copy_addr2 = B_copy_addr2 + (MAT_DIM_K * MAT_DIM_J) * sizeof(elem_t)+64;
    uint64_t D_copy_addr2 = C_copy_addr2 + (MAT_DIM_I * MAT_DIM_J) * sizeof(elem_t);

    for(int i = 0; i < nc; i ++){
      if(i == cid && i == 0){
        rr_acquire_single(0, ACC_ID1);
        rr_set_opc(XCUSTOM_ACC, 0); 
        gemmini_flush(0);
      }
      else if(i == cid && i == 1){
        rr_acquire_single(0, ACC_ID2);
        rr_set_opc(XCUSTOM_ACC, 0); 
        gemmini_flush(0);
      }
    }

    uint64_t start = read_cycles();
    for(int j = 0; j < nc; j++){
       if(j == cid && j == 0){ 
       
           tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
               (elem_t*)A_copy_addr, (elem_t*)B_copy_addr, NO_BIAS ? NULL : (acc_t*) D_copy_addr, (elem_t*)C_copy_addr,
               A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
               MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
               ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
               A_TRANSPOSE, B_TRANSPOSE,
               false, false,
               0,
               WS);

           rr_fence(0);
       }
       else if(j == cid && j == 1){
         
          tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
              (elem_t*)A_copy_addr2, (elem_t*)B_copy_addr2, NO_BIAS ? NULL : (acc_t*) D_copy_addr2, (elem_t*)C_copy_addr2,
              A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
              MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
              ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
              A_TRANSPOSE, B_TRANSPOSE,
              false, false,
              0,
              WS);

          rr_fence(0);
       }
    }
    uint64_t end = read_cycles();

    for(int i = 0; i < nc; i ++){
       if(i == cid){
          printf("core %d\n", i);
          printf("Cycles taken: %llu\n", end-start);

          const uint64_t total_macs = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
          const uint64_t ideal_cycles = total_macs / (DIM * DIM);
          const uint64_t utilization = 100 * ideal_cycles / (end-start);
          printf("Total macs: %llu\n", total_macs);
          printf("Ideal cycles: %llu\n", ideal_cycles);
          printf("Utilization: %llu%%\n", utilization);
          //barrier(nc);
       }
       barrier(nc);
    }
    
    rr_release(0);
    barrier(nc);
    exit(0);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
    exit(0);
}

