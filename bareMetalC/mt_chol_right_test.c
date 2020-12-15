// See LICENSE for license details.
#define _GNU_SOURCE
#include <sched.h>
#include <pthread.h>

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "chol_data1.h"

#define CHECK_RESULT 0
#define num_proc 2

#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#error variable-bitwidth bias not currently supported
#endif

void print_tile(elem_t* in, int tile_dim, int in_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    //printf("row starts at: %p\n", in +r*in_dim);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d.%d ", (int)(*(in +r*in_dim + c)), ((int)((*(in +r*in_dim + c))*100))%100);
    }
    printf("\n");
  }
}

void full_matmul(elem_t A[MAT_DIM][MAT_DIM], elem_t B[MAT_DIM][MAT_DIM], ACC_T D[MAT_DIM][MAT_DIM], full_t C_full[MAT_DIM][MAT_DIM]) {
  for (size_t r = 0; r < MAT_DIM; r++)
    for (size_t c = 0; c < MAT_DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

void full_transposed_matmul(int block_dim, int I_block, int J_block, int K_block, int A_stride, int B_stride, int C_stride, elem_t* A, elem_t* B, elem_t* C, bool sub) {
/*
	bool no_B_tiling = false;
	if(I_block > 1 && J_block == 1 && K_block == 1)
		no_B_tiling = true;

	if(no_B_tiling)
		tiled_matmul_auto_notileB(block_dim*I_block, block_dim, block_dim,
			A, B, sub ? C : NULL, C,
			A_stride, B_stride, C_stride, C_stride,
			MVIN_SCALE_IDENTITY, sub ? (-1) : MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
			NO_ACTIVATION, ACC_SCALE_IDENTITY, 
			0, false, 
			false, true, 
			WS);	
	else
*/
		tiled_matmul_auto(block_dim*I_block, block_dim*J_block, block_dim*K_block,
			A, B, sub ? C : NULL, C,
			A_stride, B_stride, C_stride, C_stride,
			MVIN_SCALE_IDENTITY, sub ? (-1) : MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
			NO_ACTIVATION, ACC_SCALE_IDENTITY, 
			0, false, 
			false, true, 
			WS);	
}

void full_printMatrix(elem_t m[MAT_DIM][MAT_DIM]) {
  for (size_t i = 0; i < MAT_DIM; ++i) {
    for (size_t j = 0; j < MAT_DIM; ++j)
		 printf("%d.%d ", (int)m[i][j], ((int)(m[i][j]*100))%100);
    printf("\n");
  }
}

int full_is_equal(elem_t x[MAT_DIM][MAT_DIM], elem_t y[MAT_DIM][MAT_DIM]) {
  for (size_t i = 0; i < MAT_DIM; ++i)
    for (size_t j = 0; j < MAT_DIM; ++j)
      if (((int)(x[i][j]*50)) != ((int)(y[i][j]*50))){
			printf("i: %d, j: %d, value: %d.%d, %d.%d \n", i, j, (int)x[i][j], ((int)(x[i][j]*1000))%1000 , (int)y[i][j], ((int)(y[i][j]*1000))%1000);
         return 0;
		}
  return 1;
}

void full_matshift(full_t full[MAT_DIM][MAT_DIM], elem_t out[MAT_DIM][MAT_DIM], int shift) {
  for (size_t r = 0; r < MAT_DIM; r++)                             
    for (size_t c = 0; c < MAT_DIM; c++) {
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


void lower_triangle_inverse(int block_dim, elem_t* A, elem_t M[block_dim][block_dim]){
//	for(int i = 0; i < block_dim; i++)
//		for(int j = 0; j < block_dim; j++)
//			M[i][j] = 0;

	for(int i = 0; i < block_dim; i++){
		M[i][i] = 1/(*(A+i*MAT_DIM+i));
		for(int j = 0; j < i; j++)
			M[j][i] = 0;
		for(int j = i+1; j < block_dim; j++){
			elem_t sum = 0;
			M[j][i] = 0;
			for(int k = i; k < j; k++){
	//			if(j == i+1 && k != i) M[k][i] = 0;
				sum += (*(A+j*MAT_DIM+k))*M[k][i];///(*(A+j*MAT_DIM+j));
			}
			M[j][i] = M[j][i] - sum/(*(A+j*MAT_DIM+j));
		}
	}
}


static elem_t LR_block[MAT_DIM][MAT_DIM] row_align(1) = {0};	
struct thread_args{
	int k, cpu_id, block_dim;
	//int A_offset, B_offset, C_offset;
	//elem_t temp[block_dim][block_dim];
	elem_t *temp;
};

void *panel_update1(void *args){
	elem_t * L = (elem_t*) LR_block;
	struct thread_args * chol_args = (struct thread_args*) args;
	int k = chol_args->k;
	int block_dim = chol_args->block_dim;
	int num_block = MAT_DIM/block_dim;
	int cpu_id = chol_args->cpu_id;
	int mid =  (num_block + k)/2 + 1;
	int start = cpu_id == 0 ? k+1 : mid;
	int end = cpu_id == 0 ? mid : num_block;
/*	
	for(int i = k+1; i < num_block; i++){
		if(i%num_proc == cpu_id){
			bool skip_B = (i != k+1) && (i != k+2);
			full_transposed_matmul(block_dim, 1, 1, 1, MAT_DIM, block_dim, MAT_DIM, L+block_dim*(i*MAT_DIM+k), skip_B ? NULL : chol_args->temp, L+block_dim*(i*MAT_DIM+k), false);				
		}
	}
*/
	int length = end - start;
	int i = start;
	full_transposed_matmul(block_dim, length, 1, 1, MAT_DIM, block_dim, MAT_DIM, L+block_dim*(i*MAT_DIM+k), chol_args->temp, L+block_dim*(i*MAT_DIM+k), false);				
	
}

void *panel_update2(void *args){
	elem_t* LR_pt = (elem_t*) LR_block;
	struct thread_args * chol_args = (struct thread_args*) args;
	int k = chol_args->k;
	int block_dim = chol_args->block_dim;
	int num_block = MAT_DIM/block_dim;
	int cpu_id = chol_args->cpu_id;
	for(int j = k+1; j < num_block; j++){
		if(j%num_proc == cpu_id)
			full_transposed_matmul(block_dim, (num_block-j), 1, 1, MAT_DIM, MAT_DIM, MAT_DIM, LR_pt+block_dim*(j*MAT_DIM+k), LR_pt+block_dim*(j*MAT_DIM+k), LR_pt+block_dim*(j*MAT_DIM+j), true);
	}
}

void *print_message(void *ptr){
    int cpu_id = sched_getcpu();
   // char *msg;
   // msg = (char *) ptr;
    printf("print msg - cpu_id: %d \n", cpu_id);
   // printf("%s \n", msg);
}
/*
void lower_triangle_inverse(elem_t* A, elem_t M[block_dim][block_dim]){
//	for(int i = 0; i < block_dim; i++)
//		for(int j = 0; j < block_dim; j++)
//			M[i][j] = 0;

	for(int i = 0; i < block_dim; i++){
		M[i][i] = 1/(*(A+i*MAT_DIM+i));
		for(int j = i+1; j < block_dim; j++){
			elem_t sum = 0;
			for(int k = i; k < j; k++)
				sum += (*(A+j*MAT_DIM+k))*M[k][i];///(*(A+j*MAT_DIM+j));
			M[j][i] = M[j][i] - sum/(*(A+j*MAT_DIM+j));
		}
	}
}

void *lower_triangle_inverse(void *args){
	struct thread_args * chol_args = (struct thread_args*) args;
	int k = chol_args->k;
	elem_t * A = (elem_t*) LR_block + block_dim * (k*MAT_DIM + k);
	elem_t * M = chol_args->temp;
	
	for(int i = 0; i < block_dim; i++){
		*(M+i*block_dim+i) = 1/(*(A+i*MAT_DIM+i));
		for(int j = i+1; j < block_dim; j++){
			elem_t sum = 0;
			for(int k = i; k < j; k++)
				sum += (*(A+j*MAT_DIM+k))*(*(M+k*block_dim+i));///(*(A+j*MAT_DIM+j));
			*(M+j*block_dim+i) -= sum/(*(A+j*MAT_DIM+j));
		}
	}
}
*/
void full_right_chol(int block_dim, elem_t* L){
	for(int k = 0; k < block_dim; k++){
		//printf("%d %d \n", (int)(L[k][k]*100), (int)((float)(sqrt(L[k][k]))*100));
		*(L+k*MAT_DIM+k) = (float)(sqrt(*(L+k*MAT_DIM+k)));
		for(int i = 0; i < block_dim; i++){
			if(i > k) *(L+i*MAT_DIM+k) = (float)(*(L+i*MAT_DIM+k) / *(L+k*MAT_DIM+k));
			else if(i < k) *(L+i*MAT_DIM+k) = 0;
		}
		for(int j = k+1; j < block_dim; j++)
			for(int i = j; i < block_dim; i++){
				//if(i==block_dim-1 && j==block_dim-1) printf("Lkk: %d, Lik:%d, Ljk: %d, mult: %d \n", (int)(L[i][j]*100), (int)(L[i][k]*100), (int)(L[j][k]*100), (int)(L[i][k]*L[j][k]*100));
				*(L+i*MAT_DIM+j) -= (*(L+i*MAT_DIM+k))*(*(L+j*MAT_DIM+k));
			}
				//printf("%d \n", (int)(L[k][k]));
	}
}

int main() {
#ifndef BAREMETAL
	if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);
    uint64_t cpu_start = 0;
    uint64_t cpu_end = 0;

#if CHECK_RESULT == 1

	 int block_size = MAT_DIM/num_block;
/*
	 for(int i = 0; i < num_block; i++)
		 for(int j = 0; j < num_block; j++)
			 for(int ii = 0; ii < block_size; ii++){
				 for(int jj = 0; jj < block_size; jj++){
					 if(j > i){
						 LR_block[i*block_size+ii][j*block_size+jj] = 0;
					 }
					 else{
						 LR_block[i*block_size+ii][j*block_size+jj] = in_A[i*block_size+ii][j*block_size+jj];
					 }
				 }
			 }
*/

    for(int i = 0; i < MAT_DIM; i++)
	for(int j = 0; j <= i; j++)
	   LR_block[i][j] = in_A[i][j];
#endif

 elem_t* LR_pt = (elem_t *) LR_block;

 int cpu_id;
 cpu_id = sched_getcpu();
 cpu_set_t cpuset;
 pthread_t thread[num_proc];
 pthread_attr_t attr;
 pthread_attr_init(&attr);
 for(int i = 0; i < num_proc; i++){
	 CPU_ZERO(&cpuset);
	 CPU_SET(i, &cpuset);
	 pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
	 pthread_create(&thread[i], &attr, print_message, NULL);
 }
// pthread_create(&thread[1], NULL, print_message, NULL);
 pthread_join(thread[0], NULL);
 pthread_join(thread[1], NULL);

 struct thread_args matmul_args[num_proc];
 int block_dim, num_block;
/*
 block_dim = 128;
 printf("block dimension: %d \n", block_dim);
 num_block = MAT_DIM/block_dim;
 printf("Starting block right chol\n");
 cpu_start = read_cycles();

  //set up
	
  for(int k = 0; k < num_block; k++){
	full_right_chol(block_dim, LR_pt+block_dim*(MAT_DIM*k+k));
	elem_t temp[block_dim][block_dim];// = {0};
	lower_triangle_inverse(block_dim, LR_pt+(k*MAT_DIM+k)*block_dim, temp);
	
	for(int i = 0; i < num_proc; i++){
		matmul_args[i].k = k;
		matmul_args[i].cpu_id = i;
		matmul_args[i].block_dim = block_dim;
		matmul_args[i].temp = (elem_t *)temp;
	//	CPU_ZERO(&cpuset); //empty the cpu set
	//	CPU_SET(i, &cpuset); //add each cpu to cpu set
	//	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
	//	pthread_create(&thread[i], &attr, panel_update1, &matmul_args[i]);
		pthread_create(&thread[i], NULL, panel_update1, &matmul_args[i]);
	}
	for(int i = 0; i < num_proc; i++)
		pthread_join(thread[i], NULL);

	for(int i = 0; i < num_proc; i++){
	//	CPU_ZERO(&cpuset);
	//	CPU_SET(i, &cpuset);
	//	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
	//	pthread_create(&thread[i], &attr, panel_update2, &matmul_args[i]);	
		pthread_create(&thread[i], NULL, panel_update2, &matmul_args[i]);		
	}
	for(int i = 0; i < num_proc; i++)
		pthread_join(thread[i], NULL);
  }
	 
    cpu_end = read_cycles();
    printf("Cycles taken:%llu\n", cpu_end-cpu_start);

 block_dim = 256;
 printf("block dimension: %d \n", block_dim);
 num_block = MAT_DIM/block_dim;
 printf("Starting block right chol\n");
 cpu_start = read_cycles();

  //set up
	
  for(int k = 0; k < num_block; k++){
	full_right_chol(block_dim, LR_pt+block_dim*(MAT_DIM*k+k));
	elem_t temp[block_dim][block_dim];// = {0};
	lower_triangle_inverse(block_dim, LR_pt+(k*MAT_DIM+k)*block_dim, temp);
	
	for(int i = 0; i < num_proc; i++){
		matmul_args[i].k = k;
		matmul_args[i].cpu_id = i;
		matmul_args[i].block_dim = block_dim;
		matmul_args[i].temp = (elem_t *)temp;
	//	CPU_ZERO(&cpuset); //empty the cpu set
	//	CPU_SET(i, &cpuset); //add each cpu to cpu set
	//	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
	//	pthread_create(&thread[i], &attr, panel_update1, &matmul_args[i]);
		pthread_create(&thread[i], NULL, panel_update1, &matmul_args[i]);
	}
	for(int i = 0; i < num_proc; i++)
		pthread_join(thread[i], NULL);

	for(int i = 0; i < num_proc; i++){
	//	CPU_ZERO(&cpuset);
	//	CPU_SET(i, &cpuset);
	//	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
	//	pthread_create(&thread[i], &attr, panel_update2, &matmul_args[i]);	
		pthread_create(&thread[i], NULL, panel_update2, &matmul_args[i]);		
	}
	for(int i = 0; i < num_proc; i++)
		pthread_join(thread[i], NULL);
  }
	 
    cpu_end = read_cycles();
    printf("Cycles taken:%llu\n", cpu_end-cpu_start);


 block_dim = 512;
 printf("block dimension: %d \n", block_dim);
 num_block = MAT_DIM/block_dim;
 printf("Starting block right chol\n");
 cpu_start = read_cycles();

  //set up
	
  for(int k = 0; k < num_block; k++){
	full_right_chol(block_dim, LR_pt+block_dim*(MAT_DIM*k+k));
	elem_t temp[block_dim][block_dim];// = {0};
	lower_triangle_inverse(block_dim, LR_pt+(k*MAT_DIM+k)*block_dim, temp);
	
	for(int i = 0; i < num_proc; i++){
		matmul_args[i].k = k;
		matmul_args[i].cpu_id = i;
		matmul_args[i].block_dim = block_dim;
		matmul_args[i].temp = (elem_t *)temp;
	//	CPU_ZERO(&cpuset); //empty the cpu set
	//	CPU_SET(i, &cpuset); //add each cpu to cpu set
	//	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
	//	pthread_create(&thread[i], &attr, panel_update1, &matmul_args[i]);
		pthread_create(&thread[i], NULL, panel_update1, &matmul_args[i]);
	}
	for(int i = 0; i < num_proc; i++)
		pthread_join(thread[i], NULL);

	for(int i = 0; i < num_proc; i++){
	//	CPU_ZERO(&cpuset);
	//	CPU_SET(i, &cpuset);
	//	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
	//	pthread_create(&thread[i], &attr, panel_update2, &matmul_args[i]);	
		pthread_create(&thread[i], NULL, panel_update2, &matmul_args[i]);		
	}
	for(int i = 0; i < num_proc; i++)
		pthread_join(thread[i], NULL);
  }
	 
    cpu_end = read_cycles();
    printf("Cycles taken:%llu\n", cpu_end-cpu_start);

*/
 block_dim = 32;
 printf("block dimension: %d \n", block_dim);
 num_block = MAT_DIM/block_dim;
 printf("Starting block right chol\n");
 cpu_start = read_cycles();

  //set up
	
  for(int k = 0; k < num_block; k++){
	full_right_chol(block_dim, LR_pt+block_dim*(MAT_DIM*k+k));
	elem_t temp[block_dim][block_dim];// = {0};
	lower_triangle_inverse(block_dim, LR_pt+(k*MAT_DIM+k)*block_dim, temp);
	
	for(int i = 0; i < num_proc; i++){
		matmul_args[i].k = k;
		matmul_args[i].cpu_id = i;
		matmul_args[i].block_dim = block_dim;
		matmul_args[i].temp = (elem_t *)temp;
	//	CPU_ZERO(&cpuset); //empty the cpu set
	//	CPU_SET(i, &cpuset); //add each cpu to cpu set
	//	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
	//	pthread_create(&thread[i], &attr, panel_update1, &matmul_args[i]);
		pthread_create(&thread[i], NULL, panel_update1, &matmul_args[i]);
	}
	for(int i = 0; i < num_proc; i++)
		pthread_join(thread[i], NULL);

	for(int i = 0; i < num_proc; i++){
	//	CPU_ZERO(&cpuset);
	//	CPU_SET(i, &cpuset);
	//	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
	//	pthread_create(&thread[i], &attr, panel_update2, &matmul_args[i]);	
		pthread_create(&thread[i], NULL, panel_update2, &matmul_args[i]);		
	}
	for(int i = 0; i < num_proc; i++)
		pthread_join(thread[i], NULL);
  }
	 
    cpu_end = read_cycles();
    printf("Cycles taken:%llu\n", cpu_end-cpu_start);

 block_dim = 64;
 printf("block dimension: %d \n", block_dim);
 num_block = MAT_DIM/block_dim;
 printf("Starting block right chol\n");
 cpu_start = read_cycles();

  //set up
	
  for(int k = 0; k < num_block; k++){
	full_right_chol(block_dim, LR_pt+block_dim*(MAT_DIM*k+k));
	elem_t temp[block_dim][block_dim];// = {0};
	lower_triangle_inverse(block_dim, LR_pt+(k*MAT_DIM+k)*block_dim, temp);
	
	for(int i = 0; i < num_proc; i++){
		matmul_args[i].k = k;
		matmul_args[i].cpu_id = i;
		matmul_args[i].block_dim = block_dim;
		matmul_args[i].temp = (elem_t *)temp;
	//	CPU_ZERO(&cpuset); //empty the cpu set
	//	CPU_SET(i, &cpuset); //add each cpu to cpu set
	//	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
	//	pthread_create(&thread[i], &attr, panel_update1, &matmul_args[i]);
		pthread_create(&thread[i], NULL, panel_update1, &matmul_args[i]);
	}
	for(int i = 0; i < num_proc; i++)
		pthread_join(thread[i], NULL);

	for(int i = 0; i < num_proc; i++){
	//	CPU_ZERO(&cpuset);
	//	CPU_SET(i, &cpuset);
	//	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
	//	pthread_create(&thread[i], &attr, panel_update2, &matmul_args[i]);	
		pthread_create(&thread[i], NULL, panel_update2, &matmul_args[i]);		
	}
	for(int i = 0; i < num_proc; i++)
		pthread_join(thread[i], NULL);
  }
	 
    cpu_end = read_cycles();
    printf("Cycles taken:%llu\n", cpu_end-cpu_start);





#if CHECK_RESULT == 1
	 if (!full_is_equal(LR_block, gold_L)) {
		printf("C:\n");
	//	full_printMatrix(LR_block);
		printf("Block Right Gold:\n");
	//	full_printMatrix(gold_L);
		printf("\n");
		exit(1);

	 }

#endif

  exit(0);
}

