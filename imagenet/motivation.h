#ifndef NUM_CORE
#define NUM_CORE 8 // for 8 cores
#endif

#ifndef BAREMETAL
// code for each workload
#if BATCH1 == true
#include "funct_resnet_1.h"
#include "funct_googlenet_1.h"
#include "funct_fcnnet_1.h"
#include "funct_squeezenet_1.h"
#include "funct_kwsnet_1.h"
#include "funct_alexnet_1.h"
#include "funct_yolonet_1.h"
#include "funct_yololitenet_1.h"
#endif

#if BATCH8 == true
#include "funct_resnet_8.h"
#include "funct_fcnnet_8.h"
#include "funct_googlenet_8.h"
#include "funct_squeezenet_8.h"
#include "funct_kwsnet_8.h"
#include "funct_alexnet_8.h"
#include "funct_yolonet_8.h"
#include "funct_yololitenet_8.h"
#endif

#if BATCH4 == true
#include "funct_resnet_4.h"
#include "funct_fcnnet_4.h"
#include "funct_googlenet_4.h"
#include "funct_squeezenet_4.h"
#include "funct_kwsnet_4.h"
#include "funct_alexnet_4.h"
#include "funct_yolonet_4.h"
#include "funct_yololitenet_4.h"
#endif

#endif

#define FCNNET_1 0
#define RESNET_1 1 // 4 blocks: [12, 25, 44, 54 (mem)] -> with squeezenet(4), yololitenet(7), kwsnet group1(5)
#define ALEXNET_1 2 // 2 blocks: conv, fc -> googlenet(3), resnet group1 (1), kwsnet (5), yolonet group1&2(6)
#define GOOGLENET_1 3 
#define SQUEEZENET_1 4 
#define KWSNET_1 5 // 2 blocks: [13, 25] just divided almost equally based on runtime
#define YOLONET_1 6 // 3 blocks: [4, 13, 19 (mem)] same as ResNet
#define YOLOLITENET_1 7 

#define FCNNET_4 8
#define RESNET_4 9
#define ALEXNET_4 10
#define GOOGLENET_4 11
#define SQUEEZENET_4 12
#define KWSNET_4 13
#define YOLONET_4 14
#define YOLOLITENET_4 15

#define FCNNET_8 16
#define RESNET_8 17
#define ALEXNET_8 18
#define GOOGLENET_8 19
#define SQUEEZENET_8 20
#define KWSNET_8 21
#define YOLONET_8 22
#define YOLOLITENET_8 23

#define MAX_WORKLOAD 500
#define NUM_WORKLOAD (8*3) // 1, 2, 4 batches

#ifndef total_workloads
#define total_workloads 200
#define QUEUE_DEPTH 10
#endif


// no interference thread cycles
//[62623637, 15070506, 8382324, 7070440, 2608024, 5132036, 9458914, 1978161]
//[34187290, 9829820, 5539067, 4923351, 1538641, 3314037, 6203714, 1998578]
// no interference total cycles
//[60300978, 12929163, 8180201, 16387230489, 2152710, 4005437, 2865948763, 1652242]
//[31627086, 7664272, 5298022, 12706574663, 947793, 1992228, 2026144604, 1588663]

static int total_queue_type[NUM_GROUP][MAX_WORKLOAD] = {-1};
static uint64_t total_queue_target[NUM_GROUP][MAX_WORKLOAD] = {0};
static uint64_t total_queue_runtime_thread[NUM_GROUP][WORKLOAD_CORE][MAX_WORKLOAD] = {0}; // for checking purpose (end to end runtime)
static uint64_t total_queue_runtime_total[NUM_GROUP][WORKLOAD_CORE][MAX_WORKLOAD] = {0}; // for checking purpose (end to end runtime)

//static int gemmini_workload_assigned[NUM_GROUP][SUB_GROUP][MAX_ITER][QUEUE_DEPTH] = {-1};
static uint64_t gemmini_runtime[NUM_CORE] = {0}; // to track real runtime without thread create overhead

static bool gemmini_done[NUM_GROUP][SUB_GROUP] = {0};
static bool gemmini_terminate[NUM_SUB_GROUP] = {0};
static bool gemmini_terminate_receive[NUM_SUB_GROUP] = {0};
static uint64_t global_time[NUM_GROUP] = {0};
// dram_bw -1: disable bandwidth modulation (window, target load to 0)
// dram_bw 0: monitor gemmini_bw and priority score 
// dram_bw 0-100: use dram_bw given to compute window, target load 
//static int gemmini_bw[NUM_GROUP] = {0}; // only the cid == 0 updates it
//static int gemmini_score[NUM_GROUP] = {0}; // priority score scaled to 100 (for bw division when it gets over the limit)

int rand_seed(uint32_t seed) {
  static uint32_t x = 777;
  x = x * (1664525 + seed) + 1013904223;
  return x >> 24;
}

int workload_type_assign(bool batch1, bool batch4, bool batch8){
  // currently only batch1
  int rand_mod = 80;
  int rand_base = 0;

  if(batch4){
    rand_base = 0;
    rand_mod = 165;
  }
  else if(batch8){
    rand_base = 0;
    rand_mod = 170;
  }

  static int id = 1;
  uint32_t rand_out = rand();//rand_seed(seed);
  int r = rand_out % rand_mod + rand_base;

 if(batch1){
   if(r < 1){
     id = FCNNET_1;
   }
   else if(r < 1 + 12){
     id = RESNET_1;
   }
   else if (r < 1 + 12 + 17){
     id = ALEXNET_1;
   }
   else if (r < 1 + 12 + 17 + 18){
     id = GOOGLENET_1;
   }
   else if (r < 1 + 12 + 17 + 18 + 32){
     id = SQUEEZENET_1;
   }
 }



/*
  if(r < 8){
    id = 1;
  }
  else if(r < (1+8)){
    id = 0;
  }
  else if(r < (1+8+12)){
    id = 2;
  }
  else if(r < (1+8+12+16)){
    id = 3;
  }
  else if(r < (1+8+12+16+44)){
    id = 4;
  }
  else if(r < (1+8+12+16+44+24)){
    id = 5;
  }
  else if(r < (1+8+12+16+44+24+12)){
    id = 6;
  }
  else{// if(r < (1+8+12+16+44+24+12+43)){
    id = 7;
  }
*/
  if(batch4){
    if(r < 0){
      id = FCNNET_4;
    }
    else if(r < (0+9)){
      id = RESNET_4;
    }
    else if(r < (0+9+20)){
      id = ALEXNET_4;
    }
    else if(r < (0+9+20+17)){
      id = GOOGLENET_4;
    }
    else if(r < (0+9+20+17+47)){
      id = SQUEEZENET_4;
    }
    else if(r < (0+9+20+17+47+22)){
      id = KWSNET_4;
    }
    else if(r < (0+9+20+17+47+22+17)){
      id = YOLONET_4;
    }
    else{// if(r < (0+7+20+17+48+20+9+40)){
      id = YOLOLITENET_4;
    }
  }
  
  if(batch8){
    if(r < 0){
      id = FCNNET_8;
    }
    else if(r < (0+10)){
      id = RESNET_8;
    }
    else if(r < (0+10+25)){
      id = ALEXNET_8;
    }
    else if(r < (0+10+25+17)){
      id = GOOGLENET_8;
    }
    else if(r < (0+10+25+17+48)){
      id = SQUEEZENET_8;
    }
    else if(r < (0+10+25+17+48+22)){
      id = KWSNET_8;
    }
    else if(r < (0+10+25+17+48+22+14)){
      id = YOLONET_8;
    }
    else{// if(r < (1+7+23+17+49+20+12+41)){
      id = YOLOLITENET_8;
    }
  }

  //printf("rand output: %zu, rand output value: %d, workload id: %d \n", rand_out, r, id);
  return id;
}

// mode 1 workload create function (SLA satisfaction)
void workload_create(int num_workload, bool batch1, bool batch4, bool batch8){ 

  int workload_temp[num_workload];
  for (int i = 0; i < num_workload; i++){
    int workload_type = workload_type_assign(batch1, batch4, batch8);
    workload_temp[i] = workload_type;
  }

  int lump = (num_workload / NUM_GROUP);
  for (int group = 0; group < NUM_GROUP; group++){
    for(int i = 0; i < num_workload; i++){
      int index = (i + lump * group) % num_workload;
      total_queue_type[group][index] = workload_temp[i];
//      uint64_t sp_perf = (NUM_GROUP == 4) ? sp_cycles_2[workload_temp[i]] : sp_cycles_4[workload_temp[i]];
//      total_queue_target[group][index] = sp_perf;
    }
  } 

}




#ifndef BAREMETAL
uint64_t workload_function(int workload_id, size_t cid, size_t sub_group_id, int num_gemmini,  pthread_barrier_t *barrier_funct){

  uint64_t* cycles;
  uint64_t total_runtime;
  int dram_util = -1;

  //size_t sub_group_id = group_id * NUM_GROUP + sub_group; // out of total sub-group
  //int group_status = total_queue_status[group_id][queue_id];
  bool part1 = true;//group_status < 1;
  bool part2 = true;//group_status < 2;
  bool part3 = true;//group_status < 3;
  bool part4 = true;//group_status < 4;
//printf("part1: %d, part2: %d, part3: %d, part4: %d\n", part1, part2, part3, part4);
  //uint64_t start = read_cycles();
#if BATCH1 == true
  int orow_divide = num_gemmini;
  int batch_divide = 1; // 1 batch workload
  if(workload_id == 0){
    cycles = fcnnet_function_1(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
    total_runtime = *(cycles+73);
  }
  else if(workload_id == 1){
    if(sub_group_id % 2 == 0) cycles = resnet_function_1(cid, sub_group_id, part1, part2, part3, part4, orow_divide, batch_divide, dram_util, barrier_funct);
    else cycles = resnet_function_11(cid, sub_group_id, part1, part2, part3, part4, orow_divide, batch_divide, dram_util, barrier_funct);
    total_runtime = *(cycles+72);
  }
  else if(workload_id == 2){
    if(sub_group_id % 2 == 0) cycles = alexnet_function_1(cid, sub_group_id, part1, part2, orow_divide, batch_divide, dram_util, barrier_funct);
    else cycles = alexnet_function_11(cid, sub_group_id, part1, part2, orow_divide, batch_divide, dram_util, barrier_funct);
    total_runtime = *(cycles+14);
  }
  else if(workload_id == 3){
    if(sub_group_id % 2 == 0) cycles = googlenet_function_1(cid, sub_group_id, part1, part2, orow_divide, batch_divide, dram_util, barrier_funct);
    else cycles = googlenet_function_11(cid, sub_group_id, part1, part2, orow_divide, batch_divide, dram_util, barrier_funct);
    total_runtime = *(cycles+71);
  }
  else if(workload_id == 4){
dram_util = -1;
    if(sub_group_id % 2 == 0) cycles = squeezenet_function_1(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
    else cycles = squeezenet_function_11(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
    total_runtime = *(cycles+29);
  }
  else if(workload_id == 5){
    if(sub_group_id % 2 == 0) cycles = kwsnet_function_1(cid, sub_group_id, part1, part2, orow_divide, batch_divide, dram_util, barrier_funct);
    else cycles = kwsnet_function_11(cid, sub_group_id, part1, part2, orow_divide, batch_divide, dram_util, barrier_funct);
    total_runtime = *(cycles+40);
  }
  else if(workload_id == 6){
    if(sub_group_id % 2 == 0) cycles = yolonet_function_1(cid, sub_group_id, part1, part2, part3, orow_divide, batch_divide, dram_util, barrier_funct);
    else cycles = yolonet_function_11(cid, sub_group_id, part1, part2, part3, orow_divide, batch_divide, dram_util, barrier_funct);
    total_runtime = *(cycles+26);
  }
  else if(workload_id == 7){
dram_util = -1;
    if(sub_group_id % 2 == 0) cycles = yololitenet_function_1(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
    else cycles = yololitenet_function_11(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
    total_runtime = *(cycles+14);
  }
#endif

#if BATCH4 == true
  if(workload_id >= 8 && workload_id < 16){
    int orow_divide = 1;
    int batch_divide = num_gemmini; // 4 batch workload 
    if(workload_id == 8 + 0){
      cycles = fcnnet_function_4(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+73);
    }
    else if(workload_id == 8 + 1){
      cycles = resnet_function_4(cid, sub_group_id, part1, part2, part3, part4, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+72);
    }
    else if(workload_id == 8 + 2){
      cycles = alexnet_function_4(cid, sub_group_id, part1, part2, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+14);
    }
    else if(workload_id == 8 + 3){
      cycles = googlenet_function_4(cid, sub_group_id, part1, part2, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+71);
    }
    else if(workload_id == 8 + 4){
      cycles = squeezenet_function_4(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+29);
    }
    else if(workload_id == 8 + 5){
      cycles = kwsnet_function_4(cid, sub_group_id, part1, part2, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+40);
    }
    else if(workload_id == 8 + 6){
      cycles = yolonet_function_4(cid, sub_group_id, part1, part2, part3, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+26);
    }
    else if(workload_id == 8 + 7){
      cycles = yololitenet_function_4(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+14);
    }
  }
#endif


#if BATCH8 == true
  if(workload_id >= 16 && workload_id < 24){
    int orow_divide = 1;
    int batch_divide = num_gemmini; // 4 batch workload 
    if(workload_id == 16 + 0){
      cycles = fcnnet_function_8(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+73);
    }
    else if(workload_id == 16 + 1){
      cycles = resnet_function_8(cid, sub_group_id, part1, part2, part3, part4, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+72);
    }
    else if(workload_id == 16 + 2){
      cycles = alexnet_function_8(cid, sub_group_id, part1, part2, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+14);
    }
    else if(workload_id == 16 + 3){
      cycles = googlenet_function_8(cid, sub_group_id, part1, part2, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+71);
    }
    else if(workload_id == 16 + 4){
      cycles = squeezenet_function_8(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+29);
    }
    else if(workload_id == 16 + 5){
      cycles = kwsnet_function_8(cid, sub_group_id, part1, part2, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+40);
    }
    else if(workload_id == 16 + 6){
      cycles = yolonet_function_8(cid, sub_group_id, part1, part2, part3, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+26);
    }
    else if(workload_id == 16 + 7){
      cycles = yololitenet_function_8(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+14);
    }
  }
#endif
  //if(cid == 0) total_queue_status[queue_id] = 100; // just store big value (finished)
  //uint64_t runtime = read_cycles() - start;
  return total_runtime;

}

#endif
