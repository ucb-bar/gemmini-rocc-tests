// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "include/systolic.h"
#include "util.h"

#define MAT_DIM_I 16
#define MAT_DIM_K 16
#define MAT_DIM_J 16
#define SCRATCHPAD_SIZE 4096
#define TILE_DIM 4


void transpose_tile(elem_t* in, elem_t* out, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++)
    for (size_t c = 0; c < tile_dim; c++)
      *(out + c*tile_dim + r) = *(in +r*MAT_DIM_K + c);
      //out[c][r] = in[r][c];
}

void transpose_block(elem_t* in, elem_t out[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++)
      out[c][r] = *(in +r*MAT_DIM_K + c);
}

void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*MAT_DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*MAT_DIM_J + c));
    }
    printf("\n");
  }
}

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], elem_t D[MAT_DIM_I][MAT_DIM_J], int64_t C_full[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

/*
void full_matmul_full(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], int64_t D[MAT_DIM_I][MAT_DIM_J], int64_t C_full[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}
*/

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

void full_matshift(int64_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], int shift) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++)
        out[r][c] = full[r][c] >> shift;                       
} 


int main() {
    static elem_t ZERO[DIM][DIM];

    static elem_t full_A[MAT_DIM_I][MAT_DIM_K];
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J];
    static elem_t full_D[MAT_DIM_I][MAT_DIM_J];
    static elem_t full_C[MAT_DIM_I][MAT_DIM_J];

    static int64_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

    static elem_t A_tp_s[TILE_DIM][TILE_DIM]; //TODO: need to make this a dynamic allocation
    static elem_t* A_tp; //TODO: need to make this a dynamic allocation
    static elem_t A_tps[DIM][DIM]; //TODO: need to make this a dynamic allocation

    A_tp = &A_tp_s;

    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        full_A[i][j] = rand() % 32;
      }
    }

    for (size_t i = 0; i < MAT_DIM_K; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_B[i][j] = rand() % 32;
      }
    }

    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        //full_D[i][j] = rand() % 32;
        full_D[i][j] = 0;
      }
    }

    full_matmul(full_A, full_B, full_D, gold_full);


    //printf("full_A:\n");
    //full_printMatrix(full_A);
    //printf("full_B:\n");
    //full_printMatrix(full_B);



    //need to split into tiles here
    int num_tiles;
    int tile_size = SCRATCHPAD_SIZE / 4;
    int tile_dim = TILE_DIM; //sqrt(tile_size);

    // printf("Setting mode\n");
    matmul_config_ex(OUTPUT_STATIONARY, 0, 0, 0, 0, 0, 0);

    for (int i0=0; i0 < MAT_DIM_I; i0 += tile_dim) {
      for (int j0=0; j0 < MAT_DIM_J; j0 += tile_dim) {

        elem_t* D = &(full_D[i0][j0]);
        elem_t* C = &(full_C[i0][j0]);
        int D_addr = 2*tile_size;
        int C_addr = 3*tile_size;

        for (int k0=0; k0 < MAT_DIM_K; k0 += tile_dim) {

          elem_t* A = &(full_A[i0][k0]); 
          elem_t* B = &(full_B[k0][j0]);

          int A_addr = 0;
          int B_addr = 1*tile_size;

          for (int i=0; i < tile_dim; i += DIM) {
            for (int j=0; j < tile_dim; j += DIM) {

              matmul_config_ld(MAT_DIM_J*sizeof(elem_t), 0, 0, 0, 0)
              matmul_mvin(D + i*MAT_DIM_J*sizeof(elem_t), D_addr, 0, 0, 0, 0);
              
              matmul_config_ld(MAT_DIM_K*sizeof(elem_t), 0, 0, 0, 0)
              matmul_mvin(A, A_addr, 0, 0, 0, 0);
  
              matmul_config_ld(MAT_DIM_J*sizeof(elem_t), 0, 0, 0, 0)
              matmul_mvin(B, B_addr, 0, 0, 1, 0);

              uint64_t out_addr = GARBAGE_ADDR;
              if (tile_dim == DIM) { 
                out_addr = C_addr + j*(tile_dim / DIM) + i;
              }
              if (k0 == 0) {
                //printf("preload from D at: %d\n", D_addr + j*(tile_dim / DIM) + i);
                matmul_preload(D_addr + j*(tile_dim / DIM) + i, out_addr, 0, 1, 0, 0);
              } else {
                //printf("preload from C at: %d\n", C_addr + j*(tile_dim / DIM) + i);
                matmul_preload(C_addr + j*(tile_dim / DIM) + i, out_addr, 0, 1, 0, 0);
              }
              matmul_compute_preloaded(A_addr, B_addr);

              for (int k=DIM; k < tile_dim; k += DIM) {
                matmul_config_ld(MAT_DIM_K*sizeof(elem_t), 0, 0, 0, 0)
                matmul_mvin(A + i*MAT_DIM_K + k, A_addr, 0, 0, 0, 0);  

                matmul_config_ld(MAT_DIM_J*sizeof(elem_t), 0, 0, 0, 0)
                matmul_mvin(B + k*MAT_DIM_J + j, B_addr, 0, 0, 1, 0);

                out_addr = GARBAGE_ADDR;
                matmul_preload_zeros(out_addr, 0, 1, 0, 0);
                matmul_compute_accumulated(A_addr + k, B_addr + k);
              }

              if (tile_dim != DIM) { 
                matmul_config_ld(DIM*sizeof(elem_t), 0, 0, 0, 0)
                matmul_mvin(A + i*MAT_DIM_K + (tile_dim - DIM), A_addr, 0, 0, 0, 0); 
 
                matmul_config_ld(MAT_DIM_J*sizeof(elem_t), 0, 0, 0, 0)
                matmul_mvin(B + (tile_dim - DIM)*MAT_DIM_J + j, B_addr, 0, 0, 1, 0);

                out_addr = C_addr + j*(tile_dim / DIM) + i;
                matmul_preload_zeros(out_addr, 0, 1, 1, 0);
                matmul_compute_accumulated(A_addr + tile_dim - DIM, B_addr + tile_dim - DIM);

              }

            }
          }
        }

        matmul_config_st(MAT_DIM_J*sizeof(elem_t), 0, 0, 0, 0)
        for (size_t m = 0; m < (tile_dim * tile_dim) / (DIM*DIM); ++m) {
          int i = (m % tile_dim) * DIM;
          int j = (m / tile_dim) * DIM;
          if (m == 0) {
            matmul_mvout(C + ((i)*MAT_DIM_J + j)*sizeof(elem_t) , C_addr + m, 0, 0, 0, 1);
          } else {
            matmul_mvout(C + ((i)*MAT_DIM_J + j)*sizeof(elem_t) , C_addr + m, 0, 0, 0, 0);
          }
        }

        //printf("Moved out\n");
        //printf("Tile at i0: %d, j0: %d\n", i0, j0);
        //printf("Tile at starts at address: %p\n", C);
        //print_tile(C, tile_dim); 


      }
    }


    full_matshift(gold_full, gold, 0);   
    printf("C:\n");
    full_printMatrix(full_C);
    printf("Gold:\n");
    full_printMatrix(gold);
    printf("\n");
 
    if (!full_is_equal(full_C, gold))
      exit(1);
  
  exit(0);
}

