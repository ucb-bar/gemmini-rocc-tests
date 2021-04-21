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

#define CHECK_RESULT 1

#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#error variable-bitwidth bias not currently supported
#endif

#ifndef BAREMETAL
#define MAT_DIM_I 512
#define MAT_DIM_K 512
#define MAT_DIM_J 512
#else
#define MAT_DIM_I 512
#define MAT_DIM_K 512
#define MAT_DIM_J 512
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

void  dist_matmul_i(elem_t first[MAT_DIM_I][MAT_DIM_K], elem_t second[MAT_DIM_K][MAT_DIM_J], ACC_T full_D[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], int cid, int nc) {

  int gemmini_cid = cid;

  size_t offset_first = MAT_DIM_I*gemmini_cid/nc;
  //set B to 0, iterating only over i.
  size_t offset_second = 0;
  size_t offset_out = MAT_DIM_I*gemmini_cid/nc;

  // if (cid ==0){
  //   offset_out+=2;
  // }

  elem_t * my_A = first + offset_first;
  elem_t * my_B = second + offset_second;
  elem_t * my_out = out + offset_out;

  //memory is loaded, flush and start matmul
  gemmini_flush(0);
  barrier(nc);

  for (int j=0; j < nc; j++){

    if (j==cid){
      tiled_matmul_auto(MAT_DIM_I/nc, MAT_DIM_J, MAT_DIM_K,
              (elem_t*)my_A, (elem_t*)my_B,  NO_BIAS ? NULL : &full_D[0][0], (elem_t*) my_out,
              MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
              MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
              NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
              false, false,
              false, false,
              3,
              CPU);

    }
    //setup to test write
    //multi_out[cid][cid] = cid;
    //barrier(nc);
  }

  barrier(nc);

  return;
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

static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1); // TODO don't use row_align_acc when ACC_T is elem_t
static elem_t multi_out[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);
static elem_t multi_j[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);
static elem_t func_i[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);


static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
static elem_t gold[MAT_DIM_I][MAT_DIM_J];


int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
}

void thread_entry(int cid, int nc){

    gemmini_flush(0);

    for (int i =0; i < nc; i++){

      if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
      barrier(nc);
    }


    if (cid == 0){

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
    full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);
#endif

    printf("Starting fast CPU matmul\n");
    unsigned long start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, false,
            3,
            CPU);

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
    } else {
      printf("Passed equality check for singular.\n");
    }
#endif

}
barrier(nc);

//now  start multithreaded
//each thread must define gemmini cid and grab blocks
int gemmini_cid = cid;

size_t offset_A = MAT_DIM_I*gemmini_cid/nc;
//set B to 0, iterating only over i.
size_t offset_B = 0;
size_t offset_out = MAT_DIM_I*gemmini_cid/nc;

// if (cid ==0){
//   offset_out+=2;
// }

elem_t * my_A = full_A + offset_A;
elem_t * my_B = full_B + offset_B;
elem_t * my_out = multi_out + offset_out;

//memory is loaded, flush and start matmul
gemmini_flush(0);
barrier(nc);

barrier(nc);

// for (int j=0; j <nc; j++){
//   if (j ==cid){
//     printf("Dims for thread %d\n", cid);
//     printf("Ab: %llu\n", offset_A);
//     printf("B: %llu\n", offset_B);
//     printf("Out: %llu\n", offset_out);
//     printf("Dims: %llu %llu %llu\n\n", MAT_DIM_I/nc, MAT_DIM_J, MAT_DIM_K );
//   }
//   barrier(nc);
// }

unsigned long start = read_cycles();
for (int j=0; j < nc; j++){

  if (j==cid){
    tiled_matmul_auto(MAT_DIM_I/nc, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)my_A, (elem_t*)my_B,  NO_BIAS ? NULL : &full_D[0][0], (elem_t*) my_out,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, false,
            3,
            CPU);

  }
  //setup to test write
  //multi_out[cid][cid] = cid;
  //barrier(nc);
}
unsigned long end = read_cycles();
for (int j=0; j < nc; j++){

  if (j==cid){
    printf("Thread %d  Cycles taken: %llu\n", cid, end-start);
  }
barrier(nc);
}

if (cid==0){
  #if CHECK_RESULT == 1
      if (!full_is_equal(multi_out, gold)) {
        printf("multi_out:\n");
        full_printMatrix(multi_out);
        printf("Gold:\n");
        full_printMatrix(gold);
        printf("\n");

        exit(1);
      } else {
        printf("Passed equality check for multi on i.\n");
      }
  #endif
}
barrier(nc);


//func i test
start = read_cycles();
dist_matmul_i(full_A,  full_B, full_D, func_i, cid, nc);
end = read_cycles();

for (int j=0; j < nc; j++){

  if (j==cid){
    printf("Func Thread %d  Cycles taken: %llu\n", cid, end-start);
  }
barrier(nc);
}

if (cid==0){
  #if CHECK_RESULT == 1
      if (!full_is_equal(func_i, gold)) {
        printf("func_out:\n");
        full_printMatrix(func_i);
        printf("Gold:\n");
        full_printMatrix(gold);
        printf("\n");

        exit(1);
      } else {
        printf("Passed equality check for i function.\n");
      }
  #endif
}
barrier(nc);

//now  start multithreaded k
//each thread must define gemmini cid and grab blocks
gemmini_cid = cid;

//set A to 0 always iterate over j;
offset_A = 0;
offset_B = MAT_DIM_J*gemmini_cid/nc;
offset_out = MAT_DIM_J*gemmini_cid/nc;

// if (cid ==0){
//   offset_out+=2;
// }

my_A = full_A + offset_A;
my_B = *(full_B +  0) + offset_B;
my_out = *(multi_j +0) + offset_out;

//memory is loaded, flush and start matmul
gemmini_flush(0);
barrier(nc);

barrier(nc);

// for (int j=0; j <nc; j++){
//   if (j ==cid){
//     printf("Dims for thread %d\n", cid);
//     printf("Ab: %llu\n", offset_A);
//     printf("B: %llu\n", offset_B);
//     printf("Out: %llu\n", offset_out);
//     printf("Dims: %llu %llu %llu\n\n", MAT_DIM_I, MAT_DIM_J/nc, MAT_DIM_K );
//   }
//   barrier(nc);
// }

start = read_cycles();
for (int j=0; j < nc; j++){


  if (j==cid){
    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J/nc, MAT_DIM_K,
            (elem_t*)my_A, (elem_t*)my_B,  NO_BIAS ? NULL : &full_D[0][0], (elem_t*) my_out,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, false,
            3,
            CPU);

  }
  //setup to test write
  //multi_out[cid][cid] = cid;
  //barrier(nc);
}
end = read_cycles();
for (int j=0; j < nc; j++){

  if (j==cid){
    printf("Thread %d  Cycles taken: %llu\n", cid, end-start);
  }
barrier(nc);
}

if (cid==0){
  #if CHECK_RESULT == 1
      if (!full_is_equal(multi_j, gold)) {
        printf("multi_out:\n");
        full_printMatrix(multi_j);
        printf("Gold:\n");
        full_printMatrix(gold);
        printf("\n");

        exit(1);
      } else {
        printf("Passed equality check for multi on j.\n");
      }
  #endif
}
barrier(nc);

//now do j

exit(0);
}
