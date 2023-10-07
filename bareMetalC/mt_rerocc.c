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
#include "include/rerocc.h"

#define MAT_DIM_I 32
#define MAT_DIM_J 32
#define TURN 20

#define CHECK_RESULT 1
#define OP 3

#define A_SCALE 1
#define B_SCALE MVIN_SCALE_IDENTITY
#define C_SCALE ACC_SCALE_IDENTITY
#define USE_RELU true
#define NUM_ARRAY 3
#define NUM_ARRAY1 1
#define NUM_ARRAY2 1

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

static uint64_t gemmini_accels[] = {0, 1, 2};

static elem_t A1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t B1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t A2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t B2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t C1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t C2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};

void thread_entry (int cid, int nc){

    for(int i = 0; i < nc; i++){
      if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
      barrier(nc);
    }
    for(int j = 0; j < nc; j++){
      if(j == cid && j == 0){
        for(int t = 0; t < TURN; t++){ 
          int acquired_array = 0;
		  for(int i = 0 ; i < NUM_ARRAY; i++){
            if (rr_acquire_multi(acquired_array, gemmini_accels, NUM_ARRAY))
              acquired_array ++;
            if (acquired_array == NUM_ARRAY1)
              break;
          }
          for (int i = 0; i < NUM_ARRAY1; i++) {
            rr_set_opc(OP, i);
            gemmini_flush(0);
          } 
          tiled_resadd_auto(MAT_DIM_I, MAT_DIM_J, A_SCALE, B_SCALE, C_SCALE,
                (elem_t*)A1, (elem_t*)B1,
                (elem_t*)C1, USE_RELU, WS);
  
          for(int i = 0; i < NUM_ARRAY1; i++)
            rr_release(i);     
        }
      }
      else if(j == cid && j == 1){ 
        for(int t = 0; t < TURN; t++){
          int acquired_array = 0;
		  for(int i = 0 ; i < NUM_ARRAY; i++){
            if (rr_acquire_multi(acquired_array, gemmini_accels, NUM_ARRAY))
              acquired_array ++;
            if (acquired_array == NUM_ARRAY2)
              break;
          }
          for (int i = 0; i < NUM_ARRAY2; i++) {
            rr_set_opc(OP, i);
            gemmini_flush(0);
          } 
          tiled_resadd_auto(MAT_DIM_I, MAT_DIM_J, A_SCALE, B_SCALE, C_SCALE, 
                (elem_t*)A2, (elem_t*)B2,
                (elem_t*)C2, USE_RELU, WS);
          for(int i = 0; i < NUM_ARRAY2; i++)
            rr_release(i);     
        }
      }
      
      //barrier(nc);
    }
    for(int i = 0; i < nc; i++){
      if (i == cid) printf("Thread %d/%d finished\n", cid, nc);
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

