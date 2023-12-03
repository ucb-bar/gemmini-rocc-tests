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
#define FLOAT false
#include "include/gemmini_testutils.h"
#include "include/rerocc.h"
#include "util.h"

#define NO_BIAS 1
#define ACTIVATION NO_ACTIVATION
#define BASE_ADDR 0x70000000L

#define FULL_BIAS_WIDTH 1
#define PageSize 4096

#define ACC_T acc_t
#include "data_matmul.h"
#include "data_matmul2.h"
#define NUM_INT 8
#define NUM_FP 4

#define NUM_ARRAY 4

static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static elem_t full_C2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);

void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i){
    //printf("row %d\n", i);
    for (size_t j = 0; j < MAT_DIM_J; ++j){
      if (x[i][j] != y[i][j])
        return 0;
      else if(j % 8 == 0 && i % 4 == 0){
          printf("j %d ", j);
      }
    }
    printf("\n");
    //if(i % 4 == 0) 
    //printf("row %d pass\n", i);
  }
  return 1;
}
void thread_entry(int cid, int nc){
    for(int i = 0; i < nc; i ++){
        if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
        barrier(nc);
    }
    uint64_t A_copy_addr = (BASE_ADDR & ~(PageSize-1));//+32;
    uint64_t B_copy_addr = A_copy_addr + (MAT_DIM_I * MAT_DIM_K) * sizeof(elem_t);
    uint64_t C_copy_addr = B_copy_addr + (MAT_DIM_K * MAT_DIM_J) * sizeof(elem_t);
    uint64_t D_copy_addr = C_copy_addr + (MAT_DIM_I * MAT_DIM_J) * sizeof(elem_t);
    uint64_t A_copy_addr2 = D_copy_addr + (MAT_DIM_J) * sizeof(acc_t);
    uint64_t B_copy_addr2 = A_copy_addr2 + (MAT_DIM_I * MAT_DIM_K) * sizeof(elem_t);
    uint64_t C_copy_addr2 = B_copy_addr2 + (MAT_DIM_K * MAT_DIM_J) * sizeof(elem_t);
    uint64_t D_copy_addr2 = C_copy_addr2 + (MAT_DIM_I * MAT_DIM_J) * sizeof(elem_t);


    int array[NUM_ARRAY];
    int num_acquired = 0;
    //barrier(nc);
    for(int i = 0; i < NUM_INT; i++){
        bool acquired = rr_acquire_single(num_acquired, i);
        if(acquired){
            array[num_acquired] = i;
            num_acquired ++;
            if(num_acquired == NUM_ARRAY) break;
            for(int j=0; j<10000; j++){
                int k = 1;
                k=k*2;
            }
        }
    }
    for(int i = 0; i < nc; i ++){
        if (i == cid) {
            for(int n = 0; n < NUM_ARRAY; n++)
                printf("Thread %d/%d accel %d acquired to cfgid %d\n", i, nc, array[n], n);
            
        }
        barrier(nc);
    }
    for(int n = 0; n < NUM_ARRAY; n++){
      rr_set_opc(XCUSTOM_ACC, n);
      gemmini_flush(0);
    }
    for(int i = 0; i < nc; i++){
        if(i == 0 && i == cid){
          memcpy((elem_t*) A_copy_addr, (elem_t*) full_A, sizeof(elem_t)*MAT_DIM_I*MAT_DIM_K);
          memcpy((elem_t*) B_copy_addr, (elem_t*) full_B, sizeof(elem_t)*MAT_DIM_K*MAT_DIM_J);
          memcpy((acc_t*) D_copy_addr, (acc_t*) full_D, sizeof(acc_t)*MAT_DIM_J);
        }
        else if(i == 1 && i == cid){
          memcpy((elem_t*) A_copy_addr2, (elem_t*) full_A2, sizeof(elem_t)*MAT_DIM_I*MAT_DIM_K);
          memcpy((elem_t*) B_copy_addr2, (elem_t*) full_B2, sizeof(elem_t)*MAT_DIM_K*MAT_DIM_J);
          memcpy((acc_t*) D_copy_addr2, (acc_t*) full_D2, sizeof(acc_t)*MAT_DIM_J);
        }
    }
    for(int i = 0; i < nc; i++){
        if(i == cid) printf("memcpy done\n");
        barrier(nc);
    }


    uint64_t start = read_cycles();
    for(int j = 0; j < nc; j++){
       if(j == cid && j == 0){ 
       
           multi_tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
               (elem_t*)A_copy_addr, (elem_t*)B_copy_addr, NO_BIAS ? NULL : (acc_t*) D_copy_addr, (elem_t*)C_copy_addr,
               MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
               MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
               ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
               false, false,
               false, false,
               0,
               NUM_ARRAY);

           //rr_fence(cfgid);
       }
       else if(j == cid && j == 1){
         
          multi_tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
              (elem_t*)A_copy_addr2, (elem_t*)B_copy_addr2, NO_BIAS ? NULL : (acc_t*) D_copy_addr2, (elem_t*)C_copy_addr2,
              MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
              MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
              ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
              false, false,
              false, false,
              0,
              NUM_ARRAY);

          //rr_fence(cfgid);
       }
    }
    uint64_t end = read_cycles();

    for(int i = 0; i < nc; i ++){
       if(i == cid){
          for(int cfgid=0;cfgid < NUM_ARRAY; cfgid++)
           rr_release(cfgid);
          printf("core %d\n", i);
          printf("Cycles taken: %llu\n", end-start);

          const uint64_t total_macs = (MAT_DIM_I * MAT_DIM_J * MAT_DIM_K) / NUM_ARRAY;
          const uint64_t ideal_cycles = total_macs / (DIM * DIM);
          const uint64_t utilization = 100 * ideal_cycles / (end-start);
          printf("Total macs: %llu\n", total_macs);
          printf("Ideal cycles: %llu\n", ideal_cycles);
          printf("Utilization: %llu%%\n", utilization);
          //barrier(nc);
       }
       barrier(nc);
    }
    
    //rr_release(0);
    for(int i = 0; i < nc; i++){
        if(i == 0 && i == cid){
          memcpy((elem_t*) full_C, (elem_t*) C_copy_addr, sizeof(elem_t)*MAT_DIM_I*MAT_DIM_J);
        }
        else if(i == 1 && i == cid){
          memcpy((elem_t*) full_C2, (elem_t*) C_copy_addr2, sizeof(elem_t)*MAT_DIM_I*MAT_DIM_J);
        }
    }
    barrier(nc);
    for(int i = 0; i < nc; i++){
        if(i == cid) printf("memcpy done\n");
        barrier(nc);
    }
    for(int i = 0; i < nc; i ++){
        if(i == cid && i == 0){
            if (!full_is_equal(full_C, gold)) {
              printf("full_C:\n");
              full_printMatrix(full_C);
              printf("Gold:\n");
              full_printMatrix(gold);
              printf("\n");

              exit(1);
            }
        }
        barrier(nc);
        if(i == cid && i == 1){
            if (!full_is_equal(full_C2, gold2)) {
              printf("full_C2:\n");
              full_printMatrix(full_C2);
              printf("Gold2:\n");
              full_printMatrix(gold2);
              printf("\n");

              exit(1);
            }
        }
        barrier(nc);
    }
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

