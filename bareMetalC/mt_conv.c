// See LICENSE for license details.
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

// in common
#define num_proc 4 // 2 for each
#define NO_BIAS false
#define REPEATING_BIAS 1
#define FULL_BIAS_WIDTH 1
#define dilation 1
#define OROW_DIVIDE 2
#define BATCH_DIVIDE 1

// how many turns to repeat
#define turns 3
#define rotate 4

// layer 1 config
// resnet layer 7
#define BATCH_SIZE_1 1
#define IN_DIM_1 14
#define IN_CHANNELS_1 512
#define OUT_CHANNELS_1 512
#define KERNEL_DIM_1 3
#define PADDING_1 1
#define STRIDE_1 2

#define OUT_DIM_1 ((IN_DIM_1 + 2*PADDING_1 - KERNEL_DIM_1) / STRIDE_1 + 1)
#define PATCH_SIZE_1 (KERNEL_DIM_1 * KERNEL_DIM_1 * IN_CHANNELS_1)
#define N_PATCHES_1 (BATCH_SIZE_1 * OUT_DIM_1 * OUT_DIM_1)

#define BATCH_SIZE_2 1
#define IN_DIM_2 28
#define IN_CHANNELS_2 128
#define OUT_CHANNELS_2 128
#define KERNEL_DIM_2 3
#define PADDING_2 1
#define STRIDE_2 1

#define OUT_DIM_2 ((IN_DIM_2 + 2*PADDING_2 - KERNEL_DIM_2) / STRIDE_2 + 1)
#define PATCH_SIZE_2 (KERNEL_DIM_2 * KERNEL_DIM_2 * IN_CHANNELS_2)
#define N_PATCHES_2 (BATCH_SIZE_2 * OUT_DIM_2 * OUT_DIM_2)


//stride with software padding
#define IN_STRIDE_1 (IN_CHANNELS_1 % 128 == 0) ? (IN_CHANNELS_1 + 64) : IN_CHANNELS_1
#define OUT_STRIDE_1 (OUT_CHANNELS_1 % 128 == 0) ? (OUT_CHANNELS_1 + 64) : OUT_CHANNELS_1

#define IN_STRIDE_2 (IN_CHANNELS_2 % 128 == 0) ? (IN_CHANNELS_2 + 64) : IN_CHANNELS_2
#define OUT_STRIDE_2 (OUT_CHANNELS_2 % 128 == 0) ? (OUT_CHANNELS_2 + 64) : OUT_CHANNELS_2

#define DESYNC 0
//meaningless for initializing and clearing out cache
#define MAT_DIM_I 1024
#define MAT_DIM_K 512
#define MAT_DIM_J 1024

pthread_barrier_t barrier;
pthread_barrier_t barrier1;
pthread_barrier_t barrier2;

static int args[7];
static int layer1_cycles[turns];
static int layer2_cycles[turns]; // to record cycles
	
void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*MAT_DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*MAT_DIM_J + c));
    }
    printf("\n");
  }
}

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], acc_t D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J]) {
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

void full_matshift(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], int shift) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
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

//static elem_t in0[N_PATCHES][IN_CHANNELS] row_align(MAX_BLOCK_LEN);
//static acc_t bias0[OUT_CHANNELS] row_align_acc(MAX_BLOCK_LEN_ACC);
//static elem_t weights0[PATCH_SIZE][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);
//static elem_t out0[N_PATCHES][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);
static elem_t in1[rotate][N_PATCHES_1][IN_STRIDE_1] row_align(MAX_BLOCK_LEN);
static acc_t bias1[rotate][OUT_STRIDE_1] row_align_acc(MAX_BLOCK_LEN_ACC);
static elem_t weights1[rotate][PATCH_SIZE_1][OUT_STRIDE_1] row_align(MAX_BLOCK_LEN);
static elem_t out1[N_PATCHES_1][OUT_STRIDE_1] row_align(MAX_BLOCK_LEN);
static elem_t in2[rotate][N_PATCHES_2][IN_STRIDE_2] row_align(MAX_BLOCK_LEN);
static acc_t bias2[rotate][OUT_STRIDE_2] row_align_acc(MAX_BLOCK_LEN_ACC);
static elem_t weights2[rotate][PATCH_SIZE_2][OUT_STRIDE_2] row_align(MAX_BLOCK_LEN);
static elem_t out2[rotate][N_PATCHES_2][OUT_STRIDE_2] row_align(MAX_BLOCK_LEN);
static elem_t in3[N_PATCHES_1][IN_CHANNELS_1] row_align(MAX_BLOCK_LEN);
static elem_t in4[N_PATCHES_2][IN_STRIDE_2] row_align(MAX_BLOCK_LEN);
//static acc_t bias3[OUT_CHANNELS] row_align_acc(MAX_BLOCK_LEN_ACC);
//static elem_t weights3[PATCH_SIZE][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);
//static elem_t out3[N_PATCHES][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);

static elem_t in_A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN) = {0};
static elem_t in_B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
//static acc_t bias[MAT_DIM_I][MAT_DIM_J] row_align_acc(1) = {0};
static elem_t Out[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};

struct thread_args{
        uint64_t cycles;
	int target_util;
//	int batches, orows, ocols, ochs, krows, kcols, kchs;
};

// random matmul
void *thread_matmul(void *arg){
        //struct thread_args * matmul_args = (struct thread_args *) arg;
        gemmini_flush(0);
	pthread_barrier_wait(&barrier);
        int cid = sched_getcpu();//matmul_args->i;
//      int b_unit = MAX_BLOCK_LEN;
          elem_t* A = (elem_t*) in_A + MAT_DIM_K*(MAT_DIM_I/2)*(cid/2);
          elem_t* B = (elem_t*) in_B + (MAT_DIM_J/2)*(cid%2);
          elem_t* C = (elem_t*) Out + (MAT_DIM_J/2)*(cid%2) + MAT_DIM_J*(MAT_DIM_I/2)*(cid/2);
          tiled_matmul_auto(MAT_DIM_I/2, MAT_DIM_J/2, MAT_DIM_K,
                                A, B, NULL, C, //NO_BIAS ? NULL : D, C,
                           MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            false, false,
            false, !FULL_BIAS_WIDTH,
            WS);

//    uint64_t end = read_cycles();
//    matmul_args->cycles = end - start;
}

void *thread_stream(void *arg){
	gemmini_flush(0);
	int cid = sched_getcpu();
	int gemmini_cid = cid % OROW_DIVIDE;
	pthread_barrier_wait(&barrier);
	if(cid < OROW_DIVIDE)
		tiled_resadd_auto_stride(N_PATCHES_1, IN_STRIDE_1,
			MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
			IN_STRIDE_1, (elem_t*) in1, (elem_t*) in1, (elem_t*) in3,
			false, WS, false, false, OROW_DIVIDE, 1, gemmini_cid); 
	
	else
		tiled_resadd_auto_stride(N_PATCHES_2, IN_STRIDE_2,
			MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
			IN_STRIDE_2, (elem_t*) in2, (elem_t*) in2, (elem_t*) in4,
			false, WS, false, false, OROW_DIVIDE, 1, gemmini_cid); 
	
}

void *thread_conv1(void *arg){
        struct thread_args * conv_args = (struct thread_args *) arg;
        gemmini_flush(0);
        int cid = sched_getcpu();//matmul_args->i;
	cid = cid % OROW_DIVIDE; 
       elem_t* A = (elem_t*) in1;
       elem_t* B = (elem_t*) weights1;
       elem_t* C = (elem_t*) out1;
       acc_t* D = (acc_t*) bias1;

	int target_util = conv_args->target_util;


	pthread_barrier_wait(&barrier);

//        pthread_barrier_wait(&barrier1);
	uint64_t end = 0;
	uint64_t start = read_cycles();
    for(int R = 0; R < rotate; R++){
         elem_t* A = (elem_t*)(in1);
         elem_t* B = (elem_t*)(weights1);
         elem_t* C = (elem_t*)(out1);
         acc_t* D = (acc_t*)(bias1);
        tiled_conv_A_stride_auto_stride(
	  BATCH_SIZE_1, IN_DIM_1, IN_CHANNELS_1,
	  OUT_CHANNELS_1, OUT_DIM_1,
	  STRIDE_1, 1, PADDING_1, KERNEL_DIM_1,
	  OUT_STRIDE_1, IN_STRIDE_1, OUT_STRIDE_1,

	  A,//+ R * (N_PATCHES_1*IN_STRIDE_1),
	  B + R * (PATCH_SIZE_1 * OUT_STRIDE_1),
	  D + R * (OUT_STRIDE_1),
	  C + R * (N_PATCHES_1 * OUT_STRIDE_1),
	  NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 
	  0, 0, 0, false,
	  WS, OROW_DIVIDE, BATCH_DIVIDE, cid,
	  target_util);
	 if(R == 0) end = read_cycles();
    }
    //	uint64_t end = read_cycles();
    	conv_args->cycles = end - start;
    
}

//need to sweep tiling factors on 2nd layer
void *thread_conv2(void *arg){
	struct thread_args * conv_args = (struct thread_args *) arg;
        gemmini_flush(0);
        int cid = sched_getcpu();//matmul_args->i;
	cid = cid % OROW_DIVIDE;

	int target_util  = conv_args->target_util;
	pthread_barrier_wait(&barrier);

//        pthread_barrier_wait(&barrier2);
	uint64_t end = 0;
	uint64_t start = read_cycles();
    for(int R = 0; R < rotate; R++){
         elem_t* A = (elem_t*)(in2);
         elem_t* B = (elem_t*)(weights2);
         elem_t* C = (elem_t*)(out2);
         acc_t* D = (acc_t*)(bias2);
        tiled_conv_A_stride_auto_stride(
	  BATCH_SIZE_2, IN_DIM_2, IN_CHANNELS_2,
	  OUT_CHANNELS_2, OUT_DIM_2,
	  STRIDE_2, 1, PADDING_2, KERNEL_DIM_2,
	  OUT_STRIDE_2, IN_STRIDE_2, OUT_STRIDE_2,

	  A,//+ R * (N_PATCHES_2*IN_STRIDE_2),
	  B + R * (PATCH_SIZE_2 * OUT_STRIDE_2),
	  D + R * (OUT_STRIDE_2),
	  C + R * (N_PATCHES_2 * OUT_STRIDE_2),
	  NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 
	  0, 0, 0, false,
	  WS, OROW_DIVIDE, BATCH_DIVIDE, cid, 
	  target_util;
	 if(R == 0) end = read_cycles();
    }
    //	uint64_t end = read_cycles();
    	conv_args->cycles = end - start;
    
}


void *print_message(void *ptr){
    int cpu_id = sched_getcpu();
   // char *msg;
   // msg = (char *) ptr;
    printf("print msg - cpu_id: %d \n", cpu_id);
   // printf("%s \n", msg);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
#ifndef BAREMETAL
         int cpu_id;
         cpu_id = sched_getcpu();
         printf("main thread cpuid: %d \n", cpu_id);
	 printf("Gemmini conv...\n");


         cpu_set_t cpuset[num_proc];
         pthread_t thread[num_proc];
         pthread_attr_t attr[num_proc];
         for(int i = 0; i < num_proc; i++)
                pthread_attr_init(&attr[i]);
         struct thread_args matmul_args[num_proc];


 	printf("\nlayer 1 kernel dim: %d, input dim: %d, output dim: %d, batch size: %d, input channel: %d, output channel: %d, in stride: %d, out stride: %d \n", KERNEL_DIM_1, IN_DIM_1, OUT_DIM_1, BATCH_SIZE_1, IN_CHANNELS_1, OUT_CHANNELS_1, IN_STRIDE_1, OUT_STRIDE_1);
  	printf("\nlayer 2 kernel dim: %d, input dim: %d, output dim: %d, batch size: %d, input channel: %d, output channel: %d, in stride: %d, out stride: %d \n", KERNEL_DIM_2, IN_DIM_2, OUT_DIM_2, BATCH_SIZE_2, IN_CHANNELS_2, OUT_CHANNELS_2, IN_STRIDE_2, OUT_STRIDE_2);
 
         for(int i = 0; i < num_proc; i++){
                 CPU_ZERO(&cpuset[i]);
                 CPU_SET(i, &cpuset[i]);
                 pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpuset[i]);
                 pthread_create(&thread[i], &attr[i], print_message, NULL);
         }

         for(int i = 0; i < num_proc; i++){
                pthread_join(thread[i], NULL);
         }

        pthread_barrier_init(&barrier, NULL, num_proc);

  for(int turn = 0; turn < turns; turn++){ 
    //just random turn
    for(int i = 0; i < num_proc; i++){
      pthread_create(&thread[i], &attr[i], thread_matmul, &matmul_args[i]);
    }

    for(int i = 0; i < num_proc; i++)
      pthread_join(thread[i], NULL);

    for(int i = 0; i < num_proc; i++){
      if(i < OROW_DIVIDE) {
        if(turn == 0)
          matmul_args[i].target_util = 0;
        else if(turn == 1)
          matmul_args[i].target_util = 0;
        else if(turn == 2)
          matmul_args[i].target_util = 35;
        pthread_create(&thread[i], &attr[i], thread_conv1, &matmul_args[i]);
      }
      else {
        if(turn == 0)
          matmul_args[i].target_util = 0;
        else if(turn == 1)
          matmul_args[i].target_util = 35;
        else if(turn == 2)
          matmul_args[i].target_util = 0;
         pthread_create(&thread[i], &attr[i], thread_conv2, &matmul_args[i]);
      }
    }

    for(int i = 0; i < num_proc; i++)
      pthread_join(thread[i], NULL);

    uint64_t max_layer1 = matmul_args[0].cycles > matmul_args[1].cycles ? matmul_args[0].cycles : matmul_args[1].cycles;
    uint64_t max_layer2 = matmul_args[2].cycles > matmul_args[3].cycles ? matmul_args[2].cycles : matmul_args[3].cycles;

    layer1_cycles[turn] = max_layer1;
    layer2_cycles[turn] = max_layer2;
    
  }

  for(int i = 0; i < turns; i++){
      printf("layer 1 turn %d cycles taken: %llu\n", i, layer1_cycles[i]);
      printf("layer 2 turn %d cycles taken: %llu\n", i, layer2_cycles[i]);
      // only print out utilization for main
      int total_macs = KERNEL_DIM_1 * KERNEL_DIM_1 * OUT_DIM_1 * OUT_DIM_1 * IN_CHANNELS_1 * OUT_CHANNELS_1 * BATCH_SIZE_1;
      int ideal_cycles = total_macs / (DIM * DIM) / OROW_DIVIDE;
      int utilization = 100 * ideal_cycles / layer1_cycles[i];
      printf("layer 1 turn %d Utilization: %d%%\n", i, utilization);

      total_macs = KERNEL_DIM_2 * KERNEL_DIM_2 * OUT_DIM_2 * OUT_DIM_2 * IN_CHANNELS_2 * OUT_CHANNELS_2 * BATCH_SIZE_2;
      ideal_cycles = total_macs / (DIM * DIM) / OROW_DIVIDE;
      utilization = 100 * ideal_cycles / layer2_cycles[i];
      printf("layer 2 turn %d Utilization: %d%%\n\n", i, utilization);
  }

	pthread_barrier_destroy(&barrier);
	//pthread_barrier_destroy(&barrier2);


#endif
}
