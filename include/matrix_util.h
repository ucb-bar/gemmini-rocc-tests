// See LICENSE for license details.

#ifndef __MATRIX_UTIL__
#define __MATRIX_UTIL__

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>

#include "gemmini.h"

//============================================================================
// create input-sized matrices
//============================================================================
static elem_t * create_zero_matrix_i(size_t r, size_t c) {
  const size_t bytes = r*c*sizeof(elem_t);
  elem_t *m = (elem_t*) malloc(bytes);
  memset((void*)m, 0, bytes);
  return m;
}

static elem_t * create_diag_matrix_i(size_t r, size_t c) {
  elem_t *m = create_zero_matrix_i(r,c);
  const size_t min_dim = (r<c) ? r : c;
  for(size_t i=0; i<min_dim; i++) {
    m[i*c+i] = ((elem_t)rand() & 0xf) - 8;
  }
  return m;
}

static elem_t * create_rand_matrix_i(size_t r, size_t c) {
  const size_t bytes = r*c*sizeof(elem_t);
  elem_t *m = (elem_t*) malloc(bytes);
  for(size_t i=0; i<r; i++) {
    for(size_t j=0; j<c; j++) {
      m[i*c+j] = ((elem_t)rand() & 0xf) - 8;
    }
  }
  return m;
}

//============================================================================
// create output-sized matrices
//============================================================================
static acc_t * create_zero_matrix_o(size_t r, size_t c) {
  const size_t bytes = r*c*sizeof(acc_t);
  acc_t *m = (acc_t*) malloc(bytes);
  memset((void*)m, 0, bytes);
  return m;
}

static acc_t * create_diag_matrix_o(size_t r, size_t c) {
  acc_t *m = create_zero_matrix_o(r,c);
  const size_t min_dim = (r<c) ? r : c;
  for(size_t i=0; i<min_dim; i++) {
    m[i*c+i] = ((acc_t)rand() & 0xf) - 8;
  }
  return m;
}

static acc_t * create_rand_matrix_o(size_t r, size_t c) {
  const size_t bytes = r*c*sizeof(acc_t);
  acc_t *m = (acc_t*) malloc(bytes);
  for(size_t i=0; i<r; i++) {
    for(size_t j=0; j<c; j++) {
      m[i*c+j] = ((acc_t)rand() & 0xf) - 8;
    }
  }
  return m;
}

//============================================================================
// dump routines
//============================================================================
static void dump_matrix_i(const char *name,const elem_t *m,size_t r,size_t c){
  printf("%s = [", name);
  for(size_t i=0; i<r; i++) {
    printf("\n");
    for(size_t j=0; j<c; j++) {
      printf("  %4d,", m[i*c+j]);
    }
  }
  printf("\n]\n");
}

static void dump_matrix_o(const char *name,const acc_t *m,size_t r,size_t c){
  printf("%s = [", name);
  for(size_t i=0; i<r; i++) {
    printf("\n");
    for(size_t j=0; j<c; j++) {
      printf("  %4d,", m[i*c+j]);
    }
  }
  printf("\n]\n");
}

//============================================================================
// verify routines
//============================================================================
static bool compare_matrices_i(
  elem_t *test, elem_t *gold, size_t r, size_t c) {
  bool success = true;
  for(size_t i=0; i<r; i++) {
    for(size_t j=0; j<c; j++) {
      if(test[i*c+j] != gold[i*c+j]) {
        printf("differ at index [%d][%d].test=%d, gold=%d\n",
            i, j, test[i*c+j], gold[i*c+j]);
        success = false;
      }
    }
  }
  return success;
}

#endif // __MATRIX_UTIL__

