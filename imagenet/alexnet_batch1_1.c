#define _GNU_SOURCE
#ifndef BAREMETAL
#include <pthread.h>
#include <sched.h>
#endif

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

//#include "alexnet_params_1.h"
//#include "alexnet50_mt_images.h"
#define NUM_OUTPUT (5+3+3+4)

#define num_proc 4
#define num_layer 11
#define num_pool 4

#define NUM_CORE num_proc
//pthread_barrier_t barrier[num_proc];

#define THREAD_SYNC 0
#include "funct_alexnet_1.h"

#define BATCH_DIVIDE 1
#define OROW_DIVIDE num_proc // 1: independent, 2: 2+2 collab, 4: sequential

#define ALEXNET_REPEAT 1//7


//struct thread_args{
uint64_t total_thread_cycles, total_cycles, total_conv_cycles, total_matmul_cycles, total_pool_cycles;
uint64_t pool_cycles[num_pool];
uint64_t conv_cycles[num_layer];
uint64_t matmul_cycles[num_layer];
uint64_t other_cycles; //global average
    //uint64_t target_cycles;
    //int barrier_index;
//};

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    printf("alexnet 1 batch script start\n");
    gemmini_flush(0);

    for(int r = 0; r < ALEXNET_REPEAT; r++){
      uint64_t* cycles;
      uint64_t start = read_cycles();
      cycles = alexnet_function_1(0, 0, true, true, OROW_DIVIDE, BATCH_DIVIDE, -1);
      uint64_t end = read_cycles();
	
//	 printf("alexnet repeat %d total cycles with threading overhead: %llu \n", r, end - start);

   /*
	 for(int i = 0; i < OROW_DIVIDE; i++){
		  uint64_t matmul_cycles = nn_args[i].total_matmul_cycles;
		  uint64_t conv_cycles = nn_args[i].total_conv_cycles;
		  uint64_t resadd_cycles = nn_args[i].total_resadd_cycles;
		  //uint64_t other_cycles = nn_args[i].other_cycles;
		  uint64_t total_cycles =  nn_args[i].total_cycles; //conv_cycles + matmul_cycles + resadd_cycles + other_cycles;
		  uint64_t thread_cycles = nn_args[i].total_thread_cycles;
		  
		  thread_alexnet_max = thread_alexnet_max > thread_cycles ? thread_alexnet_max : thread_cycles;
		  total_alexnet_max = total_alexnet_max > total_cycles ? total_alexnet_max : total_cycles;
	 }
   */

      total_thread_cycles = end - start;
      //total_matmul_cycles = cycles[71];
      total_conv_cycles = cycles[11] + cycles[12];
      //other_cycles = other_cycles;
      total_cycles = cycles[14];
      for(int i = 0; i < NUM_OUTPUT; i++){
        if(i < 8) conv_cycles[i] = cycles[i];
        else if(i < 11) pool_cycles[i-8] = cycles[i];
      }
      printf("\nalexnet repeat %d total thread cycles: %llu\n", r, total_thread_cycles);
      printf("alexnet repeat %d total cycles: %llu\n", r, total_cycles);
    


     for(int i = 0; i < 8; i++)    

     { 
        printf("alexnet repeat %d Conv layer %d cycles: %llu \n", r, i, conv_cycles[i]);
     }
     
     for(int i = 0; i < 3; i++)    

     { 
        printf("alexnet repeat %d Pool layer %d cycles: %llu \n", r, i, pool_cycles[i]);

        

     }
    }
    //pthread_barrier_destroy(&barrier[0]);
 
    //printf("==================================\n");
    exit(0);
}

