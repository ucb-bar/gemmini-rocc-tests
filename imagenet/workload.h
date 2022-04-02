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
#define RESNET_1 1 // 4 blocks: [12, 25, 44, 54 (mem)] -> with squeezenet(4), yololitenet(7), kwsnet group1(5)
#define ALEXNET_1 2 // 2 blocks: conv, fc -> googlenet(3), resnet group1 (1), kwsnet (5), yolonet group1&2(6)
#define GOOGLENET_1 3 
#define SQUEEZENET_1 4 
#define KWSNET_1 5 // 2 blocks: [13, 25] just divided almost equally based on runtime
#define YOLONET_1 6 // 3 blocks: [4, 13, 19 (mem)] same as ResNet
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

static uint64_t sp2_cycles[NUM_WORKLOAD] =
 {67024601, 16815039, 13401393, 7346737, 2793391, 5284644, 10529885, 3092522,
  129882609, 31787132, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  

static uint64_t sp_prediction_cycles[NUM_CORE][NUM_WORKLOAD] =
{{51998429, 11609760, 9775970, 4233402, 1554138, 3048772, 6483179, 1014052,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // 4 cores
 {78600899, 16342374, 11693769, 6374191, 1975373, 4398102, 9278409, 1361124,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // 2 cores
 {136872531, 26509740, 15546486, 10819934, 2909057, 7122425, 15551184, 2204055,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // 1 core
 {2*136872531, 2*26509740, 2*15546486, 2*10819934, 2*2909057, 2*7122425, 2*15551184, 2*2204055,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // 0.5 core
};

static int workload_group[NUM_WORKLOAD] = {1, 4, 2, 2, 1, 2, 3, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // for now only 1 batch

//#define QUEUE_DEPTH 10
//#define SEED 10 // to randomize workload more
//#define CAP 0.8 // 0 to 1 (smaller number: shorter time between workload dispatch time)

//workload creating capacity: cap * sp_cycles * cap_scale(<1)
//QoS target: cap * (qos+1) *  sp_cycles * target_scale(> 1, < 1)
//QoS 0: 4 cores, 1: 2 cores, 2: 1 core, 3: 0.5 x 1 core

static int total_queue_type[MAX_WORKLOAD] = {-1};
static uint64_t total_queue_dispatch[MAX_WORKLOAD] = {0}; // dispatched time (in order)
static uint64_t total_queue_finish[NUM_CORE][MAX_WORKLOAD] = {0};
static int total_queue_status[MAX_WORKLOAD] = {-1}; // -1: not assigned, 0: in assigned queue, >= 1: part
static int total_queue_priority[MAX_WORKLOAD] = {-1}; // 0 - 11
static int total_queue_qos[MAX_WORKLOAD] = {-1}; // latency sensitivity of workload (target: (qos + 1) * 1.2 * sp_cycles)
static uint64_t total_queue_target[MAX_WORKLOAD] = {0};
static uint64_t total_queue_runtime_thread[NUM_CORE][MAX_WORKLOAD] = {0}; // for checking purpose (end to end runtime)
static uint64_t total_queue_runtime_total[NUM_CORE][MAX_WORKLOAD] = {0}; // for checking purpose (end to end runtime)

#define MAX_ITER (int)(total_workloads / QUEUE_DEPTH)
static int gemmini_workload_assigned[NUM_CORE][MAX_ITER][QUEUE_DEPTH] = {-1};
static int gemmini_runtime[NUM_CORE] = {0}; // to track real runtime without thread create overhead

static int gemmini_workload_grouped[NUM_CORE][MAX_ITER][QUEUE_DEPTH] = {-1};
static bool gemmini_done[NUM_CORE] = {0};

int rand_seed(uint32_t seed) {
  static uint32_t x = 777;
  x = x * (1664525 + seed) + 1013904223;
  return x >> 24;
}

int workload_type_assign(bool batch1, bool batch2, bool batch4, uint32_t seed){
  // currently only batch1
  int rand_mod = 160;
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
  //printf("rand output: %zu, rand output value: %d, workload id: %d \n", rand_out, r, id);
  return id;
}

// mode 1 workload create function (SLA satisfaction)
void workload_mode_1(int qos, int workload, bool batch1, bool batch2, bool batch4, uint32_t seed, int cap, float target_scale, float cap_scale){ 
  // qos < 0 -> mixed
  // qos >= 0 -> workload dispatch qos apart, qos ways at once
  for(int i = 0; i < MAX_WORKLOAD; i++)
    total_queue_status[i]= -1;
  int group = cap * (2+1);
  if (qos == 0){ // mixed QoS
    // extremely high QoS (0) should come really rarely
    // 1: 30%, 2: 40%, 3: 30%
    for (int i = 0; i < workload+2*group; i++){
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
      total_queue_target[i] = (workload_qos + 1) * target_scale * cap * sp_cycles[workload_type];
      for (int j = 0; j < NUM_CORE; j++){
        total_queue_finish[j][i] = 0;
        total_queue_runtime_thread[j][i] = 0;
        total_queue_runtime_total[j][i] = 0;
      }
      if(i < group){
        total_queue_dispatch[i] = 10000*i;
      }
      else{
        total_queue_dispatch[i] = total_queue_dispatch[i - group] + sp_cycles[total_queue_type[i - group]] * group * cap_scale; // is it enough?
      }
    }
/*
    int num_qos_4 = workload / 100; // < 1% of workload 
    for(int i = 0; i < num_qos_4; i++){
      int qos_4_index = 0;
      while (!(qos_4_index > 3 && qos_4_index < 100 * (i + 1))){
        qos_4_index = rand_seed(seed) % workload;
      }
      total_queue_qos[qos_4_index] = 0;
      total_queue_priority[qos_4_index] = 11;
    }
    */
  }
  else{
    group = cap * (qos+1);
    int num_workload_group = ceil_divide_int(workload+2*group, group);
    for(int i = 0; i < num_workload_group; i++){
      for(int j = 0; j < group; j++){
        int index = group * i + j;
        int workload_type = workload_type_assign(batch1, batch2, batch4, seed);
        //int workload_type = rand_base + rand_seed(seed) % rand_mod;
        total_queue_type[index] = workload_type; 
//printf("index: %d, output workload type: %d, stored type: %d\n", index, workload_type, total_queue_type[index]);
        total_queue_priority[index] = 5; // mode 1 -> same priority 
        total_queue_qos[index] = qos;
        total_queue_target[index] = (qos + 1) * target_scale * cap * sp_cycles[workload_type];
        for (int j = 0; j < NUM_CORE; j++){
           total_queue_finish[j][index] = 0;
           total_queue_runtime_thread[j][index] = 0;
           total_queue_runtime_total[j][index] = 0;
      	}
        if(i == 0){
          total_queue_dispatch[index] = 10000*j;
        }
        else{
          total_queue_dispatch[index] = total_queue_dispatch[index - group] + sp_cycles[total_queue_type[index - group]] * group * cap_scale; // is it enough?
        }
      }
    }
  }  

  for(int i = 0; i < workload; i++){
    for(int j = i+1; j < workload+2*group; j++){
      if(total_queue_dispatch[i] > total_queue_dispatch[j]){
        uint64_t a = total_queue_dispatch[i];
        total_queue_dispatch[i] = total_queue_dispatch[j];
        total_queue_dispatch[j] = a;
 
        a = total_queue_target[i];
        total_queue_target[i] = total_queue_target[j];
        total_queue_target[j] = a;
 
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
  for(int i = workload; i < workload+2*group; i++){
    total_queue_dispatch[i] = 0;
    total_queue_priority[i] = -1;
    total_queue_type[i] = -1;
    total_queue_status[i] = -1;
    total_queue_qos[i] = -1;
  }

  //for(int i = 0; i < workload; i++)
    //printf("after mixing entry %d, workload id %d\n", i, total_queue_type[i]);

  for(int i = 0; i < NUM_CORE; i++){
    gemmini_runtime[i] = 0; // initialize time 
  }
  for(int c = 0; c < NUM_CORE; c++)
    for(int i = 0; i < MAX_ITER; i++)
      for(int j = 0; j < QUEUE_DEPTH; j++)
        gemmini_workload_assigned[c][i][j] = -1;
}

void workload_mode_2(int workload, bool batch1, bool batch2, bool batch4, uint32_t seed, int cap, float target_scale, float cap_scale){
  // priority (0: 15, 1: 18 / 2: 10, 4: 15, 6: 15, 8: 15 / 9: 10, 11: 2)
  for(int i = 0; i < MAX_WORKLOAD; i++)
    total_queue_status[i]= -1;
  int qos = 3; // to lowest QoS
  int group = (qos+1)*cap;//8;

  int num_workload_group = ceil_divide_int(workload+group, group);

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
      total_queue_target[index] = (qos+1)*cap*target_scale*sp_cycles[workload_type];
      for (int j = 0; j < NUM_CORE; j++){
        total_queue_finish[j][index] = 0;
   	    total_queue_runtime_thread[j][index] = 0;
   	    total_queue_runtime_total[j][index] = 0;
      }
      if(i == 0){
        total_queue_dispatch[index] = 10000*j;
      }
      else{
        total_queue_dispatch[index] = total_queue_dispatch[index - group] + sp_cycles[total_queue_type[index - group]] * (group) * cap_scale; // is it enough?
      }
    }
  }
 
  for(int i = 0; i < workload; i++){
    for(int j = i+1; j < workload+group; j++){
      if(total_queue_dispatch[i] > total_queue_dispatch[j]){
        uint64_t a = total_queue_dispatch[i];
        total_queue_dispatch[i] = total_queue_dispatch[j];
        total_queue_dispatch[j] = a;
 
        a = total_queue_target[i];
        total_queue_target[i] = total_queue_target[j];
        total_queue_target[j] = a;
 
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

  for(int i = workload; i < workload+group; i++){
    total_queue_dispatch[i] = 0;
    total_queue_priority[i] = -1;
    total_queue_type[i] = -1;
    total_queue_status[i] = -1;
    total_queue_qos[i] = -1;
  }
  for(int i = 0; i < NUM_CORE; i++){
    gemmini_runtime[i] = 0; // initialize time 
  }
  for(int c = 0; c < NUM_CORE; c++)
    for(int i = 0; i < MAX_ITER; i++)
      for(int j = 0; j < QUEUE_DEPTH; j++)
        gemmini_workload_assigned[c][i][j] = -1; 
}

// fcfs static partition of 2 cores each
int workload_fcfs_mp_schedule(int num_group, int num_workload){
  for(int c = 0; c < NUM_CORE; c++)
    for(int i = 0; i < MAX_ITER; i++)
      for(int j = 0; j < QUEUE_DEPTH; j++){
        gemmini_workload_assigned[c][i][j] = -1;
      }

  int index = 0;
  int iter = 0;
  int group[num_group];
  int cycle[num_group];
  for (int i = 0; i < num_group; i++){
    cycle[i] = 0;
    group[i] = 0;
  }

  while(index < num_workload){
//    printf("iter: %d, index: %d, cycle0: %llu, cycle1: %llu, group0: %d, group1: %d\n", iter, index, cycle[0], cycle[1], group[0], group[1]);
    bool full = false;
    for(int i = 0; i < num_group; i++){
      if(group[i] == QUEUE_DEPTH){
        full = true;
        break;
      }
    }

    if(!full){
      for(int k = 0; k < num_group; k++){
        if(group[k] == 0){
          gemmini_workload_assigned[k][iter][group[k]] = index;
          int type = total_queue_type[index];
          group[k] += 1;
          index += 1;
          cycle[k] += sp2_cycles[type];
        }
        else{
          bool smallest = true;
          for(int c = 0; c < num_group; c++){
            if(cycle[k] > cycle[c]){
              smallest = false;
              break;
            }
          }
          if(smallest){
            gemmini_workload_assigned[k][iter][group[k]] = index;
            int type = total_queue_type[index];
            index += 1;
            cycle[k] += sp2_cycles[type];
            group[k] += 1;
          }

        }
      }
    }
    else{
      iter += 1;
      for(int i = 0; i < num_group; i++){
        cycle[i] = 0;
        group[i] = 0;
      }
    }
  }
  return (iter+1); // number of queue group
}

// priority scheduling
int workload_priority_mp(int num_group, int num_workload, int num_iter, uint64_t current_cycle){
  for(int c = 0; c < NUM_CORE; c++)
    for(int i = 0; i < MAX_ITER; i++)
      for(int j = 0; j < QUEUE_DEPTH; j++){
        gemmini_workload_grouped[c][i][j] = -1;
        gemmini_workload_assigned[c][i][j] = -1;
      }

  int group[num_group];
  int cycle[num_group];
  for (int i = 0; i < num_group; i++){
    cycle[i] = current_cycle + 500000;
    group[i] = 0;
  }

  // priority score initialization
  int64_t score[num_workload];
  int max_depth = QUEUE_DEPTH * 1.8;

  int iter = 0;

  // repeat from here
  int pre_assign_queue[max_depth];
  int64_t pre_assign_score[max_depth]; // need this?

  while (iter < num_iter){
    for(int i = 0; i < max_depth; i++){
      pre_assign_queue[i] = -1;
      pre_assign_score[i] = -1;
    }

    uint64_t top_cycle = cycle[0];
    // get max cycle
    for(int i = 0; i < num_group; i++){
      if(top_cycle > cycle[i]){
        top_cycle = cycle[i];
      }
    }

    int pointer = 0;
    for(int i = 0; i < num_workload; i++){
      if(total_queue_dispatch[i] > top_cycle){
        pointer = i;
        break;
      }
      else if(i == num_workload - 1){
        pointer = num_workload;
      }
    }

    bool done = true;
    for (int i = 0; i < pointer; i++){
      if(total_queue_status[i] == -1){ //only take the unassigned ones
        score[i] = total_queue_priority[i];
        done = false;
      }
      else
        score[i] = -1;
    }
    if(done && (pointer == num_workload))
      break;

    //printf("iter: %d, cycle: %llu, cycle0: %llu, cycle1: %llu, dispatch queue pointer: %d\n", iter, top_cycle, cycle[0], cycle[1], pointer);
    
    // ToDo: QoS 0 (extreme priority)

    // assign until num_iter
    // based on expected cycles after num_iter
    for(int i = 0; i < pointer; i++){
      if(score[i] >= 0){
        int qos = total_queue_qos[i];
        int type = total_queue_type[i];
        uint64_t after_dispatch = (top_cycle - total_queue_dispatch[i]);
        //score[i] = score[i]*1000000 + ((1000000*after_dispatch) / sp_prediction_cycles[qos][type]);
        score[i] = score[i]*1000000 + ((1000000*after_dispatch) / (qos*sp_prediction_cycles[1][type]));
      }
    }

    // first, pick candidate
    // next, assign using cycle prediction
    int queue_index = 0;
    int max_index = -1;
    int64_t max_score = -1;
    int pre_assign_length = 0;
    while(queue_index < max_depth){
      for(int i = 0; i < pointer; i++){
        if(total_queue_status[i] == -1){
          if(max_score < score[i]){
            max_score = score[i];
            max_index = i;
          }
        }
      }
      //printf("queue index: %d, max index: %d\n", queue_index, max_index);
      if(max_index == -1){
     //   pre_assign_length = queue_index;
        break;
      }
      pre_assign_queue[queue_index] = max_index;
      pre_assign_score[queue_index] = max_score;
      queue_index ++;
      total_queue_status[max_index] = 0;
      max_index = -1;
      max_score = -1;
    }
    pre_assign_length = queue_index;
    /*
    printf("pre assigned queue length: %d \n", pre_assign_length);
    for(int i = 0; i < pre_assign_length; i++)
      printf("%d, ", pre_assign_queue[i]); 
    printf("\n");
    */

    for (int p = 0; p < pre_assign_length; ){
  //    printf("iter: %d, index: %d, cycle0: %llu, cycle1: %llu, group0: %d, group1: %d\n", iter, index, cycle[0], cycle[1], group[0], group[1]);
      bool full = false;
      for(int i = 0; i < num_group; i++){
        if(group[i] == QUEUE_DEPTH){
          full = true;
          break;
        }
      }

      if(!full){
        for(int k = 0; k < num_group; k++){
          bool smallest = true;
          for(int c = 0; c < num_group; c++){
            if(cycle[k] > cycle[c]){
              smallest = false;
              break;
            }
          }
          if(smallest && p < pre_assign_length){
            int index = pre_assign_queue[p];
            gemmini_workload_assigned[k][iter][group[k]] = index;
            int type = total_queue_type[index];
            cycle[k] += sp2_cycles[type];
            group[k] += 1;
            p++;
          }
        }
      }
      else{
        // release status
        int index = pre_assign_queue[p];
        total_queue_status[index] = -1;
        p++;
      }
    }
    iter += 1;
    for(int i = 0; i < num_group; i++){
      group[i] = 0;
    }
  }

  // if returned value is 0, then it is over
  return (iter); // number of queue group
}

void workload_grouping(int num_iter, int num_group){
  for(int group = 0; group < num_group; group++){
    for(int iter = 0; iter < num_iter; iter++){
      for(int i = 0; i < QUEUE_DEPTH; i++){
        int queue_id = gemmini_workload_assigned[group][iter][i];
            //printf("queue_id: %d\n", queue_id);
        if(queue_id != -1){
          int workload_type = total_queue_type[queue_id];
          if(workload_type == RESNET_1){ // if it is resnet
            bool groupped = false;
            for(int i_next = i; i_next < QUEUE_DEPTH; i_next++){
              int next_queue_id = gemmini_workload_assigned[group][iter][i_next];
              int next_type = total_queue_type[next_queue_id];
              if(next_type == -1) break;
              if(next_type == GOOGLENET_1 ||next_type == SQUEEZENET_1 || next_type == YOLOLITENET_1 || next_type == KWSNET_1){
                if(gemmini_workload_grouped[group][iter][i_next] != -7){
                  gemmini_workload_grouped[group][iter][i_next] = -7; // mark 0
                 
                  gemmini_workload_grouped[group][iter][i] = next_queue_id;
                  groupped = true;
                  break;
                }
              }
            }
            if(!groupped && iter != num_iter - 1){
              //printf("num_iter: %d, iter: %d, queue_id: %d\n", num_iter, iter, queue_id);
              int iter_temp = iter + 1;
              while(iter_temp < num_iter && !groupped){
                for(int i_next = 0; i_next < QUEUE_DEPTH; i_next++){
                  int next_queue_id = gemmini_workload_assigned[group][iter_temp][i_next];
                  int next_type = total_queue_type[next_queue_id];
                  if(next_type == -1) break;
                  if(next_type == GOOGLENET_1 ||next_type == SQUEEZENET_1 || next_type == YOLOLITENET_1 || next_type == KWSNET_1){
                    if(gemmini_workload_grouped[group][iter_temp][i_next] != -7){ 
                      gemmini_workload_grouped[group][iter_temp][i_next] = -7;
                      gemmini_workload_grouped[group][iter][i] = next_queue_id; 
                      groupped = true;
                      break;
                    }
                  }
                }
                iter_temp ++;
              }
            }
          }
          else if(workload_type == ALEXNET_1){ // if it is resnet
            bool groupped = false;
            for(int i_next = i; i_next < QUEUE_DEPTH; i_next++){
              int next_queue_id = gemmini_workload_assigned[group][iter][i_next];
              int next_type = total_queue_type[next_queue_id];
              if(next_type == -1) break;
              if(next_type == GOOGLENET_1 || next_type == YOLONET_1 || next_type == KWSNET_1 || next_type == RESNET_1){
                if(gemmini_workload_grouped[group][iter][i_next] != -7){
                  gemmini_workload_grouped[group][iter][i_next] = -7; // mark 0
                  gemmini_workload_grouped[group][iter][i] = next_queue_id;
                  groupped = true;
                  break;
                }
              }
            }
            if(!groupped && iter != num_iter - 1){ 
              int iter_temp = iter + 1;
              while(iter < num_iter && !groupped){
                for(int i_next = 0; i_next < QUEUE_DEPTH; i_next++){
                  int next_queue_id = gemmini_workload_assigned[group][iter_temp][i_next];
                  int next_type = total_queue_type[next_queue_id];
                  if(next_type == -1) break;
                  if(next_type == GOOGLENET_1 || next_type == YOLONET_1 || next_type == KWSNET_1 || next_type == RESNET_1 ){
                    if(gemmini_workload_grouped[group][iter_temp][i_next] != -7){ 
                      gemmini_workload_grouped[group][iter_temp][i_next] = -7;
                      gemmini_workload_grouped[group][iter][i] = next_queue_id; 
                      groupped = true;
                      break;
                    }
                  }
                }
                iter_temp ++;
              }
            }
          }
          else if(workload_type == YOLONET_1){ // if it is resnet
            bool groupped = false;
            for(int i_next = i; i_next < QUEUE_DEPTH; i_next++){
              int next_queue_id = gemmini_workload_assigned[group][iter][i_next];
              int next_type = total_queue_type[next_queue_id];
              if(next_type == -1) break;
              if(next_type == GOOGLENET_1 ||next_type == SQUEEZENET_1 || next_type == YOLOLITENET_1 || next_type == KWSNET_1){
                if(gemmini_workload_grouped[group][iter][i_next] != -7){
                  gemmini_workload_grouped[group][iter][i_next] = -7;
                  gemmini_workload_grouped[group][iter][i] = next_queue_id;
                  groupped = true;
                  break;
                }
              }
            }
            if(!groupped && iter != num_iter - 1){
              //printf("num_iter: %d, iter: %d, queue_id: %d\n", num_iter, iter, queue_id);
              int iter_temp = iter + 1;
              while(iter_temp < num_iter && !groupped){
                for(int i_next = 0; i_next < QUEUE_DEPTH; i_next++){
                  int next_queue_id = gemmini_workload_assigned[group][iter_temp][i_next];
                  int next_type = total_queue_type[next_queue_id];
                  if(next_type == -1) break;
                  if(next_type == GOOGLENET_1 ||next_type == SQUEEZENET_1 || next_type == YOLOLITENET_1 || next_type == KWSNET_1){
                    if(gemmini_workload_grouped[group][iter_temp][i_next] != -7){ 
                      gemmini_workload_grouped[group][iter_temp][i_next] = -7;
                      gemmini_workload_grouped[group][iter][i] = next_queue_id; 
                      groupped = true;
                      break;
                    }
                  }
                }
                iter_temp ++;
              }
            }
          }
        }
        else break;
      }
    }
  }


}
#ifndef BAREMETAL
uint64_t workload_function(int queue_id, int workload_id, int cid, int num_gemmini, pthread_barrier_t *barrier_funct){
  gemmini_flush(0);
  uint64_t* cycles;
  uint64_t total_runtime;

  int group_status = total_queue_status[queue_id];
  bool part1 = group_status < 1;
  bool part2 = group_status < 2;
  bool part3 = group_status < 3;
  bool part4 = group_status < 4;
//printf("part1: %d, part2: %d, part3: %d, part4: %d\n", part1, part2, part3, part4);
  //uint64_t start = read_cycles();
  if(workload_id < 8){
    int orow_divide = num_gemmini;
    int batch_divide = 1; // 1 batch workload
    if(workload_id == 0){
      cycles = fcnnet_function_1(cid, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+73);
    }
    else if(workload_id == 1){
      cycles = resnet_function_1(cid, part1, part2, part3, part4, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+72);
    }
    else if(workload_id == 2){
      cycles = alexnet_function_1(cid, part1, part2, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+14);
    }
    else if(workload_id == 3){
      cycles = googlenet_function_1(cid, part1, part2, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+71);
    }
    else if(workload_id == 4){
      cycles = squeezenet_function_1(cid, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+29);
    }
    else if(workload_id == 5){
      cycles = kwsnet_function_1(cid, part1, part2, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+40);
    }
    else if(workload_id == 6){
      cycles = yolonet_function_1(cid, part1, part2, part3, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+26);
    }
    else if(workload_id == 7){
      cycles = yololitenet_function_1(cid, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+14);
    }
  }


  if(cid == 0) total_queue_status[queue_id] = 100; // just store big value (finished)
  //uint64_t runtime = read_cycles() - start;
  return total_runtime;

}

uint64_t workload_group_function(int queue_id, int group_queue_id, int original_workload_id, int grouped_workload_id, int cid, int num_gemmini, pthread_barrier_t *barrier_funct){
  gemmini_flush(0);
  uint64_t* cycles;
  uint64_t total_runtime;

  int group_status = total_queue_status[queue_id];
  bool part1 = group_status < 1;
  bool part2 = group_status < 2;
  bool part3 = group_status < 3;
  bool part4 = group_status < 4;

 
  //uint64_t start = read_cycles();
  if(original_workload_id < 8){
    int orow_divide = num_gemmini;
    int batch_divide = 1; // 1 batch workload
    if(original_workload_id == 1){
      cycles = resnet_function_1(cid, part1, part2, part3, false, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+72);
      if(cid == 0){
        cycles = resnet_block_function_1(0, false, false, false, true, 1, 1, 0);
        total_runtime += *(cycles+72);
      }
      else{
        if(grouped_workload_id == SQUEEZENET_1){
          cycles = squeezenet_block_function_1(0, 1, 1, 0); 
          total_runtime = *(cycles+29);
          total_queue_status[group_queue_id] = 50;
        }
        else if(grouped_workload_id == YOLOLITENET_1){
          cycles = yololitenet_block_function_1(0, 1, 1, 0);
          total_runtime = *(cycles + 14);
          total_queue_status[group_queue_id] = 50;
        }
        else if(grouped_workload_id == KWSNET_1){
          cycles = kwsnet_block_function_1(0, true, false, 1, 1, 0);
          total_runtime = *(cycles + 40);
          total_queue_status[group_queue_id] = 1;
        }
        else if(grouped_workload_id == GOOGLENET_1){
          cycles = googlenet_block_function_1(0, true, false, 1, 1, 0); 
          total_runtime = *(cycles+71);
          total_queue_status[group_queue_id] = 1;
        }
      }
    }
    else if(original_workload_id == 2){
      cycles = alexnet_function_1(cid, part1, false, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+14);
      if(cid == 0){
        cycles = alexnet_block_function_1(0, false, true, 1, 1, 0);
        total_runtime += *(cycles+14);
      }
      else{
        if(grouped_workload_id == GOOGLENET_1){
          cycles = googlenet_block_function_1(0, true, true, 1, 1, 0); 
          total_runtime = *(cycles+71);
          total_queue_status[group_queue_id] = 50;
        }
        else if(grouped_workload_id == YOLONET_1){
          cycles = yolonet_block_function_1(0, true, true, false, 1, 1, 0);
          total_runtime = *(cycles + 26);
          total_queue_status[group_queue_id] = 2;
        }
        else if(grouped_workload_id == KWSNET_1){
          cycles = kwsnet_block_function_1(0, true, true, 1, 1, 0);
          total_runtime = *(cycles + 40);
          total_queue_status[group_queue_id] = 50;
        }
        else if(grouped_workload_id == RESNET_1){
          cycles = resnet_block_function_1(0, true, false, false, false, 1, 1, 0);
          total_runtime = *(cycles + 72);
          total_queue_status[group_queue_id] = 1;
        }
      }
    }
    else if(original_workload_id == 6){
      cycles = yolonet_function_1(cid, part1, part2, false, orow_divide, batch_divide, 0, barrier_funct);
      total_runtime = *(cycles+26);
      if(cid == 0){
        cycles = yolonet_block_function_1(0, false, false, true, 1, 1, 0);
        total_runtime += *(cycles+26);
      }
      else{
        if(grouped_workload_id == SQUEEZENET_1){
          cycles = squeezenet_block_function_1(0, 1, 1, 0); 
          total_runtime = *(cycles+29);
          total_queue_status[group_queue_id] = 50;
        }
        else if(grouped_workload_id == YOLOLITENET_1){
          cycles = yololitenet_block_function_1(0, 1, 1, 0);
          total_runtime = *(cycles + 14);
          total_queue_status[group_queue_id] = 50;
        }
        else if(grouped_workload_id == KWSNET_1){
          cycles = kwsnet_block_function_1(0, true, false, 1, 1, 0);
          total_runtime = *(cycles + 40);
          total_queue_status[group_queue_id] = 1;
        }
        else if(grouped_workload_id == GOOGLENET_1){
          cycles = googlenet_block_function_1(0, true, false, 1, 1, 0); 
          total_runtime = *(cycles+71);
          total_queue_status[group_queue_id] = 1;
        }
      }
    }
  }


  if(cid == 0) total_queue_status[queue_id] = 80; // just store big value (finished)
  //uint64_t runtime = read_cycles() - start;
  return total_runtime;

}


#endif
