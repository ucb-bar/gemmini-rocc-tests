#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

//#include "resnet_params_1.h"
//#include "resnet50_mt_images.h"
#define NUM_OUTPUT (20+34+16+3)

#define num_proc 1
#define num_layer 54
#define num_resadd 16

#include "funct_resnet_1.h"

#define NUM_ARRAY 1

#define RESNET_REPEAT 6


#define MAT_DIM_I 512
#define MAT_DIM_J 512
#define MAT_DIM_K 512
#define FULL_BIAS_WIDTH true
#define REPEATING_BIAS true

struct thread_args{
    uint64_t total_thread_cycles, total_cycles, total_conv_cycles, total_matmul_cycles, total_resadd_cycles;
	uint64_t resadd_cycles[num_resadd];
	uint64_t conv_cycles[num_layer];
    uint64_t matmul_cycles[num_layer];
	uint64_t other_cycles; //global average
    //uint64_t target_cycles;
    int num_array;
    bool input_dram, output_dram, weight_dram, bias_dram;
};

void *thread_NN(void *arg){
	int cid = sched_getcpu();
	struct thread_args * nn_args = (struct thread_args *) arg;
    int num_array = nn_args->num_array;
    bool input_direct_dram = nn_args->input_dram;
    bool weight_direct_dram = nn_args->weight_dram;
    bool output_direct_dram = nn_args->output_dram;
    bool bias_direct_dram = nn_args->bias_dram;
	uint64_t* cycles;

    uint64_t thread_start = read_cycles();
    for(int i = 0; i < num_array; i++)
      while(!rerocc_acquire(i, 0xf)){}

    for (int i = 0; i < num_array; i++) {
      rerocc_assign(OP3, i);
      gemmini_flush(0);
    }
    
    cycles = resnet_function_1(true, true, true, true, input_direct_dram, weight_direct_dram, bias_direct_dram, output_direct_dram, num_array);
    
    for(int i = 0; i < num_array; i++)
      rerocc_release(i);   

    uint64_t thread_end = read_cycles();
    nn_args->total_thread_cycles = thread_end - thread_start;
    //nn_args->total_matmul_cycles = *(cycles+71);
    nn_args->total_conv_cycles = *(cycles+70);
    //nn_args->other_cycles = other_cycles;
    nn_args->total_resadd_cycles = *(cycles+71);
    nn_args->total_cycles = *(cycles+72);
    for(int i = 0; i < NUM_OUTPUT; i++){
	  if(i < 54) nn_args->conv_cycles[i] = *(cycles+i);
	  else if(i < 70) nn_args->resadd_cycles[i-54] = *(cycles+i);
    }

}
void *print_message(void *ptr){
    printf("entered message thread\n");
    //gemmini_flush(0); // check whether all have gemmini cores
    int cpu_id = sched_getcpu();
    printf("print msg - cpu_id: %d \n", cpu_id);
}

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
    int cpu_id;
    cpu_id = sched_getcpu();
    printf("main cpu: %d \n", cpu_id);
 
    cpu_set_t cpuset[num_proc];
    pthread_t thread[num_proc];
    pthread_attr_t attr[num_proc];
    for(int i = 0; i < num_proc; i++)
	pthread_attr_init(&attr[i]);
    struct thread_args nn_args[num_proc];

    printf("create threading \n");
    for(int i = 0; i < num_proc; i++){
	 CPU_ZERO(&cpuset[i]);
	 CPU_SET(i, &cpuset[i]);
	 pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpuset[i]);
	 pthread_create(&thread[i], &attr[i], print_message, NULL);
    }

    for(int i = 0; i < num_proc; i++){
        pthread_join(thread[i], NULL);
    }
    printf("thread joined after message printing\n");

   // for(int i = 0; i < OROW_DIVIDE; i++)
   //     nn_args[i].target_cycles = RESNET_TARGET;
    bool input_dram = false;
    bool output_dram = false;
    bool weight_dram = false;
    bool bias_dram = false;
    for(int r = 0; r < RESNET_REPEAT; r++){
     if(r == 2){
         bias_dram = true;
     }
     else if(r == 3){
         bias_dram = false;
         input_dram = true;
         output_dram = true;
     }
     else if(r == 4){
         input_dram = false;
         output_dram = false;
         weight_dram = true;
     }
     else if(r == 5){
         output_dram = true;
         input_dram = true;
         bias_dram = true;
         weight_dram = true;
     }
	 for(int i = 0; i < num_proc; i++){
        nn_args[i].input_dram = input_dram;
        nn_args[i].output_dram = output_dram;
        nn_args[i].weight_dram = weight_dram;
        nn_args[i].bias_dram = bias_dram;
		nn_args[i].num_array = NUM_ARRAY;
		pthread_create(&thread[i], &attr[i], thread_NN, &nn_args[i]);
	 }
	 for(int i = 0; i < num_proc; i++)
		  pthread_join(thread[i], NULL);
	
//	 printf("resnet repeat %d total cycles with threading overhead: %llu \n", r, end - start);
	 uint64_t matmul_cycles = nn_args[0].total_matmul_cycles;
     uint64_t conv_cycles = nn_args[0].total_conv_cycles;
     uint64_t resadd_cycles = nn_args[0].total_resadd_cycles;
     //uint64_t other_cycles = nn_args[0].other_cycles;
     uint64_t total_cycles =  nn_args[0].total_cycles; //conv_cycles + matmul_cycles + resadd_cycles + other_cycles;
     uint64_t thread_cycles = nn_args[0].total_thread_cycles;
		  
	
	 printf("\nresnet repeat %d total thread cycles: %llu\n", r, thread_cycles);
	 printf("resnet repeat %d total cycles: %llu\n", r, total_cycles);
	
     printf("resnet repeat %d input dram: %d, output dram: %d, weight dram: %d, bias dram: %d\n", r, input_dram, output_dram, weight_dram, bias_dram);

	 for(int i = 0; i < 54; i++)    

	 {
		  for(int j = 0; j < num_proc; j++)
            printf("resnet repeat %d Conv layer %d cycles: %llu \n", r, i, nn_args[j].conv_cycles[i]);
	 }
	 

	 for(int i = 0; i < 16; i++)    

	 {
		  for(int j = 0; j < num_proc; j++)
            printf("resnet repeat %d Resadd layer %d cycles: %llu \n", r, i, nn_args[j].resadd_cycles[i]);

	 }
    } 
    printf("==================================\n");
    exit(0);
}

