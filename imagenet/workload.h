#ifndef BAREMETAL
// code for each workload
#include "funct_resnet_1.h"
#include "funct_googlenet_1.h"
#include "funct_fcnnet_1.h"
#include "funct_squeezenet_1.h"
#include "funct_kwsnet_1.h"
#include "funct_alexnet_1.h"
#include "funct_yolonet_1.h"
#include "funct_yololitenet_1.h"
/*
#include "funct_resnet_2.h"
#include "funct_fcnnet_2.h"
#include "funct_googlenet_2.h"
#include "funct_squeezenet_2.h"
#include "funct_kwsnet_2.h"
#include "funct_alexnet_2.h"
#include "funct_yolonet_2.h"
#include "funct_yololitenet_2.h"

#include "funct_resnet_4.h"
#include "funct_fcnnet_4.h"
#include "funct_googlenet_4.h"
#include "funct_squeezenet_4.h"
#include "funct_kwsnet_4.h"
#include "funct_alexnet_4.h"
#include "funct_yolonet_4.h"
#include "funct_yololitenet_4.h"
*/
#endif

#define FCNNET_1 0
#define RESNET_1 1
#define ALEXNET_1 2
#define GOOGLENET_1 3
#define SQUEEZENET_1 4
#define KWSNET_1 5
#define YOLONET_1 6
#define YOLOLITENET_1 7

#define FCNNET_2 8
#define RESNET_2 9
#define ALEXNET_2 10
#define GOOGLENET_2 11
#define SQUEEZENET_2 12
#define KWSNET_2 13
#define YOLONET_2 14
#define YOLOLITENET_2 15

#define FCNNET_4 16
#define RESNET_4 17
#define ALEXNET_4 18
#define GOOGLENET_4 19
#define SQUEEZENET_4 20
#define KWSNET_4 21
#define YOLONET_4 22
#define YOLOLITENET_4 23

#define MAX_WORKLOAD 200
#define NUM_WORKLOAD (8*3) // 1, 2, 4 batches

//[[119760510, 67024601, 40234597], [25727753, 16815039, 12475991], [17927216, 13401393, 9768785], [11874273, 7346737, 5416207], [4222239, 2793391, 1788653], [9195628, 5284644, 4081583], [17273935, 10529885, 9382349], [3847462, 3092522, 3199658]]
//[[239950436, 129882609, 79363161], [50827889, 31787132, 22436942], [24796119, 18453904, 13685471], [23111997, 13703960, 12326887], [8340242, 4903209, 4366495], [18471732, 11301087, 8395059], [33626369, 20244540, 16272834], [7532796, 5960033, 8726273]]
//[[48cap7855, 258891996, 158774922], [102468011, 64136386, 46741738], [38304694, 26072816, 20023558], [45576644, 26416274, 16289487], [16274246, 9360352, 6089778], [39172097, 23574881, 16040738], [66543598, 39096729, 39286233], [14897266, 11421865, 17485051]]

// single program (isolated run) cycles
static uint64_t sp_cycles[NUM_WORKLOAD] = 
 {40234597, 12475991, 9768785, 5416207, 1788653, 4081583, 9382349, 3199658,
  79363161, 22436942, 13685471, 12326887, 4366495, 8395059, 16272834, 8726273, 
  158774922, 46741738, 20023558, 16289487, 6089778, 16040738, 39286233, 17485051};


//#define SEED 10 // to randomize workload more
//#define CAP 0.8 // 0 to 1 (smaller number: shorter time between workload dispatch time)

static int total_queue_type[MAX_WORKLOAD] = {-1};
static uint64_t total_queue_dispatch[MAX_WORKLOAD] = {0}; // dispatched time (in order)
static uint64_t total_queue_finish[NUM_CORE][MAX_WORKLOAD] = {0};
static int total_queue_priority[MAX_WORKLOAD] = {-1}; // 0 - 11
static int total_queue_qos[MAX_WORKLOAD] = {-1}; // latency sensitivity of workload (target: (qos + 1) * 1.2 * sp_cycles)
static uint64_t total_queue_runtime_thread[NUM_CORE][MAX_WORKLOAD] = {0}; // for checking purpose (end to end runtime)
static uint64_t total_queue_runtime_total[NUM_CORE][MAX_WORKLOAD] = {0}; // for checking purpose (end to end runtime)


static int gemmini_workload_assigned[NUM_CORE] = {-1};
static int gemmini_workload_received[NUM_CORE] = {-1};
static int gemmini_runtime[NUM_CORE] = {0}; // to track real runtime without thread create overhead
static int smallest_pointer = 0; // before this pointer, finished executing (for fast search, once the pointer reaches total number of workload, it is finished)


int rand_seed(uint32_t seed) {
  static uint32_t x = 777;
  x = x * (1664525 + seed) + 1013904223;
  return x >> 24;
}

int workload_type_assign(bool batch1, bool batch2, bool batch4, uint32_t seed){
  // currently only batch1
  int rand_mod = 159;
  int rand_base = 0;
  if (batch1 && batch2 && batch4) {
    rand_mod = NUM_WORKLOAD;
  }
  else if (batch1 && batch2){
    rand_mod = 8 * 2;
  }
  if(batch2){
    rand_base = 8;
  }
  else if(batch4){
    rand_base = 16;
  }

  static int id = 1;
  uint32_t rand_out = rand_seed(seed);
  int r = rand_out % rand_mod + rand_base;
  if(r < 8){
    id = 1;
  }
  else if(r < (1+8)){
    id = 0;
  }
  else if(r < (1+8+12)){
    id = 2;
  }
  else if(r < (1+8+12+17)){
    id = 3;
  }
  else if(r < (1+8+12+17+45)){
    id = 4;
  }
  else if(r < (1+8+12+17+45+24)){
    id = 5;
  }
  else if(r < (1+8+12+17+45+24+12)){
    id = 6;
  }
  else if(r < (1+8+12+17+45+24+12+41)){
    id = 7;
  }
  //printf("rand output: %zu, rand output value: %d, workload id: %d \n", rand_out, r, id);
  return id;
}

// mode 1 workload create function (SLA satisfaction)
void workload_mode_1(int qos, int workload, bool batch1, bool batch2, bool batch4, uint32_t seed, float cap){
  // qos < 0 -> mixed
  // qos >= 0 -> workload dispatch qos apart, qos ways at once

  if (qos == 0){ // mixed QoS
    // extremely high QoS (0) should come really rarely
    // 1: 30%, 2: 40%, 3: 30%
    for (int i = 0; i < workload; i++){
      int select = rand_seed(seed) % 10;
      int workload_qos = 1;
      if(select >= 7)
        workload_qos = 3;
      else if (select >= 3)
        workload_qos = 2;
      int workload_type = workload_type_assign(batch1, batch2, batch4, seed);//rand_base + rand_seed(seed) % rand_mod;
      total_queue_type[i] = workload_type;
      total_queue_priority[i] = 5;
      total_queue_qos[i] = workload_qos;
      for (int j = 0; j < NUM_CORE; j++){
        total_queue_finish[j][i] = 0;
        total_queue_runtime_thread[j][i] = 0;
        total_queue_runtime_total[j][i] = 0;
      }
      if(i < 3){
        total_queue_dispatch[i] = 0;
      }
      else{
        total_queue_dispatch[i] = total_queue_dispatch[i - 3] + sp_cycles[total_queue_type[i - 3]] * 3 * cap; // is it enough?
      }
    }

    int num_qos_4 = workload / 100; // < 1% of workload 
    for(int i = 0; i < num_qos_4; i++){
      int qos_4_index = 0;
      while (!(qos_4_index > 3 && qos_4_index < 100 * (i + 1))){
        qos_4_index = rand_seed(seed) % workload;
      }
      total_queue_qos[qos_4_index] = 0;
      total_queue_priority[qos_4_index] = 11;
    }
  }
  else{
    int num_workload_group = ceil_divide_int(workload, qos+1);//(int)(workload / (qos+1));
    for(int i = 0; i < num_workload_group; i++){
      for(int j = 0; j < (qos+1); j++){
        int index = (qos+1) * i + j;
        int workload_type = workload_type_assign(batch1, batch2, batch4, seed);
        //int workload_type = rand_base + rand_seed(seed) % rand_mod;
        total_queue_type[index] = workload_type; 
//printf("index: %d, output workload type: %d, stored type: %d\n", index, workload_type, total_queue_type[index]);
        total_queue_priority[index] = 5; // mode 1 -> same priority 
        total_queue_qos[index] = qos;
        for (int j = 0; j < NUM_CORE; j++){
           total_queue_finish[j][index] = 0;
           total_queue_runtime_thread[j][index] = 0;
           total_queue_runtime_total[j][index] = 0;
      	}
        if(i == 0){
          total_queue_dispatch[index] = 0;
        }
        else{
          total_queue_dispatch[index] = total_queue_dispatch[index - qos - 1] + sp_cycles[total_queue_type[index - qos - 1]] * (qos+1) * cap; // is it enough?
        }
      }
    }
  }  

  for(int i = 0; i < workload; i++){
    for(int j = i+1; j < workload; j++){
      if(total_queue_dispatch[i] > total_queue_dispatch[j]){
        uint64_t a = total_queue_dispatch[i];
        total_queue_dispatch[i] = total_queue_dispatch[j];
        total_queue_dispatch[j] = a;
 
        int b = total_queue_priority[i];
        total_queue_priority[i] = total_queue_priority[j];
        total_queue_priority[j] = b;
  
        b = total_queue_type[i];
        total_queue_type[i] = total_queue_type[j];
        total_queue_type[j] = b;
 
        b = total_queue_qos[i];
        total_queue_qos[i] = total_queue_qos[j];
        total_queue_qos[j] = b;                    
      }
    }
  }

for(int i = 0; i < workload; i++)
	//printf("after mixing entry %d, workload id %d\n", i, total_queue_type[i]);

  for(int i = 0; i < NUM_CORE; i++){
    gemmini_runtime[i] = 0; // initialize time 
  }
}

void workload_mode_2(int workload, bool batch1, bool batch2, bool batch4, uint32_t seed, float cap){
  // priority (0: 15, 1: 18 / 2: 10, 4: 15, 6: 15, 8: 15 / 9: 10, 11: 2)
  int qos = 3; // to lowest QoS
  int group = 8;

  int num_workload_group = ceil_divide_int(workload, group);

  for(int i = 0; i < num_workload_group; i++){
    for(int j = 0; j < group; j++){
      int index = group * i + j;
      int workload_type = workload_type_assign(batch1, batch2, batch4, seed);
      //int workload_type = rand_base + rand_seed(seed) % rand_mod;
      total_queue_type[index] = workload_type; 
      int priority_level = rand_seed(seed) % 100;
      if(priority_level < 15){
          priority_level = 0;
      }
      else if(priority_level < 33){
          priority_level = 1;
      }
      else if(priority_level < 43){
          priority_level = 2;
      }
      else if(priority_level < 58){
          priority_level = 4;
      }
      else if(priority_level < 73){
          priority_level = 6;
      }
      else if(priority_level < 88){
          priority_level = 8;
      }
      else if(priority_level < 98){
          priority_level = 9;
      }
      else{
          priority_level = 11;
      }
      total_queue_priority[index] = priority_level; 
      total_queue_qos[index] = qos;
      for (int j = 0; j < NUM_CORE; j++){
   	total_queue_finish[j][index] = 0;
   	total_queue_runtime_thread[j][index] = 0;
   	total_queue_runtime_total[j][index] = 0;
      }
      if(i == 0){
        total_queue_dispatch[index] = 0;
      }
      else{
        total_queue_dispatch[index] = total_queue_dispatch[index - group] + sp_cycles[total_queue_type[index - group]] * (group) * cap; // is it enough?
      }
    }
  }
 
  for(int i = 0; i < workload; i++){
    for(int j = i+1; j < workload; j++){
      if(total_queue_dispatch[i] > total_queue_dispatch[j]){
        uint64_t a = total_queue_dispatch[i];
        total_queue_dispatch[i] = total_queue_dispatch[j];
        total_queue_dispatch[j] = a;
 
        int b = total_queue_priority[i];
        total_queue_priority[i] = total_queue_priority[j];
        total_queue_priority[j] = b;
  
        b = total_queue_type[i];
        total_queue_type[i] = total_queue_type[j];
        total_queue_type[j] = b;
 
        b = total_queue_qos[i];
        total_queue_qos[i] = total_queue_qos[j];
        total_queue_qos[j] = b;                    
      }
    }
  }

  for(int i = 0; i < NUM_CORE; i++){
    gemmini_runtime[i] = 0; // initialize time 
  }
}

#ifndef BAREMETAL
uint64_t workload_function(int workload_id, int cid, int num_gemmini, pthread_barrier_t *barrier_funct){
  gemmini_flush(0);
  uint64_t* cycles;
  uint64_t total_runtime;

  //uint64_t start = read_cycles();
  if(workload_id < 8){
    int orow_divide = num_gemmini;
    int batch_divide = 1; // 1 batch workload
    if(workload_id == 0){
      cycles = fcnnet_function_1(cid, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+73);
    }
    else if(workload_id == 1){
      cycles = resnet_function_1(cid, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+72);
    }
    else if(workload_id == 2){
      cycles = alexnet_function_1(cid, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+14);
    }
    else if(workload_id == 3){
      cycles = googlenet_function_1(cid, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+71);
    }
    else if(workload_id == 4){
      cycles = squeezenet_function_1(cid, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+29);
    }
    else if(workload_id == 5){
      cycles = kwsnet_function_1(cid, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+40);
    }
    else if(workload_id == 6){
      cycles = yolonet_function_1(cid, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+26);
    }
    else if(workload_id == 7){
      cycles = yololitenet_function_1(cid, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+14);
    }
  }


  //uint64_t runtime = read_cycles() - start;
  return total_runtime;

}
#endif
