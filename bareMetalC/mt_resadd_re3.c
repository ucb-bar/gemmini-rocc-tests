// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
//#include "include/rerocc.h" 
#include "include/gemmini_testutils.h"
#include "util.h"

#define OP 3
#define MAT_DIM_I 512
#define MAT_DIM_J 576

#define A_SCALE 1
#define B_SCALE MVIN_SCALE_IDENTITY
#define C_SCALE ACC_SCALE_IDENTITY
#define USE_RELU true
#define NUM_ARRAY1 2
#define NUM_ARRAY2 2
#define NUM_ARRAY3 2
#define NUM_ARRAY4 2

static elem_t A1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t B1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t C1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};

static elem_t A2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t B2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t C2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};

static elem_t A3[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t B3[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t C3[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};

static elem_t A4[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t B4[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t C4[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};


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
      if (x[i][j] != y[i][j]){
        printf("i: %d, j: %d, x: %d, y: %d\n", i, j, x[i][j], y[i][j]); 
        //return 0;
      }
  return 1;
}
void thread_entry (int cid, int nc){

    for(int i = 0; i < nc; i++){
      if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
      barrier(nc);
    }
    for(int j = 0; j < nc; j++){
      if(j == cid && j == 0){ 
        for(int i = 0; i < NUM_ARRAY1; i++)
          while(!rerocc_acquire(i, 0xff)){}

        printf("rerocc acquired for cpu %d \n", j);
        for (int i = 0; i < NUM_ARRAY1; i++) {
          rerocc_assign(OP, i);
          gemmini_flush(0);
          printf("cpu %d flushed\n", j);
        } 
      }
      else if(j == cid && j == 1){ 
        for(int i = 0; i < NUM_ARRAY2; i++)
          while(!rerocc_acquire(i, 0xff)){}

        printf("rerocc acquired for cpu %d \n", j);
        for (int i = 0; i < NUM_ARRAY2; i++) {
          rerocc_assign(OP, i);
          gemmini_flush(0);
          printf("cpu %d flushed\n", j);
        } 
      }
      else if(j == cid && j == 2){ 
        for(int i = 0; i < NUM_ARRAY3; i++)
          while(!rerocc_acquire(i, 0xff)){}

        printf("rerocc acquired for cpu %d \n", j);
        for (int i = 0; i < NUM_ARRAY3; i++) {
          rerocc_assign(OP, i);
          gemmini_flush(0);
          printf("cpu %d flushed\n", j);
        } 
      }
      else if(j == cid && j == 3){ 
        for(int i = 0; i < NUM_ARRAY4; i++)
          while(!rerocc_acquire(i, 0xff)){}

        printf("rerocc acquired for cpu %d \n", j);
        for (int i = 0; i < NUM_ARRAY4; i++) {
          rerocc_assign(OP, i);
          gemmini_flush(0);
          printf("cpu %d flushed\n", j);
        } 
      }
      
      barrier(nc);
    }

    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t total_start = read_cycles();
    for(int j = 0; j < nc; j++){
      if(j == cid && j == 0){
        start = read_cycles();
        tiled_opcode_resadd_auto_multi(MAT_DIM_I, MAT_DIM_J, A_SCALE, B_SCALE, C_SCALE, MAT_DIM_J, false, false, false,
                (elem_t*)A1, (elem_t*)B1,
                (elem_t*)C1, USE_RELU, WS, NUM_ARRAY1, 0);
        end = read_cycles();
      }else if(j == cid && j == 1){
        start = read_cycles();
        tiled_opcode_resadd_auto_multi(MAT_DIM_I, MAT_DIM_J, A_SCALE, B_SCALE, C_SCALE, MAT_DIM_J, false, false, false,
                (elem_t*)A2, (elem_t*)B2,
                (elem_t*)C2, USE_RELU, WS, NUM_ARRAY2, 0);
        end = read_cycles();
      }else if(j == cid && j == 2){
        start = read_cycles();
        tiled_opcode_resadd_auto_multi(MAT_DIM_I, MAT_DIM_J, A_SCALE, B_SCALE, C_SCALE, MAT_DIM_J, false, false, false,
                (elem_t*)A3, (elem_t*)B3,
                (elem_t*)C3, USE_RELU, WS, NUM_ARRAY3, 0);
        end = read_cycles();
      }else if(j == cid && j == 3){
        start = read_cycles();
        tiled_opcode_resadd_auto_multi(MAT_DIM_I, MAT_DIM_J, A_SCALE, B_SCALE, C_SCALE, MAT_DIM_J, false, false, false,
                (elem_t*)A4, (elem_t*)B4,
                (elem_t*)C4, USE_RELU, WS, NUM_ARRAY4, 0);
        end = read_cycles();
      }
    }
    uint64_t total_end = read_cycles();
    
    barrier(nc);
    for(int i = 0; i < nc; i++){
      if(i == cid)  printf("Cycles taken: %llu, %llu\n", end-start, total_end-total_start);
      barrier(nc);
    }

    for(int j = 0; j < nc; j++){
      if(j == cid && j == 0){ 
        for(int i = 0; i < NUM_ARRAY1; i++)
          rerocc_release(i);
      }
      else if(j == cid && j == 1){ 
        for(int i = 0; i < NUM_ARRAY2; i++)
          rerocc_release(i);
      }
      else if(j == cid && j == 2){ 
        for(int i = 0; i < NUM_ARRAY3; i++)
          rerocc_release(i);
      }
      else if(j == cid && j == 3){ 
        for(int i = 0; i < NUM_ARRAY4; i++)
          rerocc_release(i);
      }
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

