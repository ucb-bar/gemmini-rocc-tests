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

#define MAX_WORKLOAD 300
#define NUM_WORKLOAD (8*3) // 1, 2, 4 batches

#ifndef total_workloads
#define total_workloads 200
#define QUEUE_DEPTH 10
#endif

#ifndef planaria_scale
#define planaria_scale 1.2
#endif

//[[120778499, 67655896, 40484174], [26048228, 16871159, 13302457], [18153183, 13480880, 9734783], [12059398, 7194660, 6382764], [4325949, 2853555, 2790373], [9193046, 5219157, 4111254], [17391717, 10483108, 8413386], [3859186, 3137930, 3222616]]
//[[485011422, 260614658, 161822568], [103707389, 64801233, 47448663], [36628263, 23877088, 18234202], [46343352, 27089661, 27971820], [16748058, 9558922, 8751179], [38773439, 23131212, 17656532], [67136866, 38408465, 24478347], [15076962, 11593273, 12049913]]
//[[967833971, 519399762, 327567928], [208302990, 129435130, 101819975], [64386337, 39288987, 32103729], [92824678, 52921143, 55384159], [32989741, 18761773, 16752679], [78456021, 46963714, 34225754], [134271000, 75566170, 55725139], [29750804, 22687359, 34942369]]


static uint64_t mem_cycles[NUM_WORKLOAD] = 
{0, 4464256, 9184240, 0, 0, 0, 4231271, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// single program (isolated run) cycles
static uint64_t sp_cycles[NUM_WORKLOAD] = 
 {26452008, 11047170, 6330629, 7483605, 3495019, 4417141, 10807095, 3256638,
// {40484174, 13302457, 9734783, 6382764, 2790373, 4111254, 9382349, 3222616,
  161822568, 47448663, 18234202, 27971820, 8751179, 17656532, 24478347, 12049913,
 //86683306, 27373069, 10167014, 16718710, 6323149, 9366786, 19839420, 1087579
  327567928, 101819975, 32103729, 55384159, 16752679, 34225754, 55725139, 34942369};

static uint64_t sp2_cycles[NUM_WORKLOAD] =
 {67655896, 16871159, 13480880, 7194660, 2853555, 5219157, 10483108, 3137930,
  260614658, 64801233, 23877088, 27089661, 9558922, 23131212, 38408465, 11593273,
  519399762, 129435130, 39288987, 52921143,18761773, 46963714, 75566170, 22687359};  
// {62623637, 15070506, 8382324, 7070440, 2608024, 5132036, 9458914, 1978161,


//update more batches
static uint64_t sp_prediction_cycles[NUM_WORKLOAD] =
 {73285100, 19217091, 12522137, 11047464, 2475758, 4052968, 10189593, 2128228,
  323379193, 70932031, 21652658, 24382388, 7975050, 20436384, 32979430, 7988048,// 1.5*5325365,
  688920519, 187435173, 56257474, 85237598, 20613232, 43801771, 73829109, 22398058};// 1.5*14932039},// 2 cores
/* 
{{51998429, 11609760, 9775970, 4233402, 1554138, 3048772, 6483179, 2*1014052,
 219697314, 53193614, 15264781, 15560770, 6302614, 15107557, 20218408, 2*3894960,
 460570438, 131508421, 32794135, 47788049, 13463166, 31048562, 41018163, 2*8411099},// 4 cores
 {78600899, 16342374, 11693769, 6374191, 1975373, 4398102, 9278409, 2041686,//1.5*1361124,
  323379193, 70932031, 21652658, 24382388, 7975050, 20436384, 32979430, 7988048,// 1.5*5325365,
  688920519, 187435173, 56257474, 85237598, 20613232, 43801771, 73829109, 22398058},// 1.5*14932039},// 2 cores
 {136872531, 26509740, 15546486, 10819934, 2909057, 7122425, 15551184, 2204055,
  552344088, 109148570, 34428412, 42708002, 11587648, 31196813, 59534474, 8744576,
  1189319719, 304024301, 103184153, 161397053, 35484840, 69308189, 139969715, 28259508},// 1 core
 {2*136872531, 2*26509740, 2*15546486, 2*10819934, 2*2909057, 2*7122425, 2*15551184, 2*2204055,
  2*552344088, 2*109148570, 2*34428412, 2*42708002, 2*11587648, 2*31196813, 2*59534474, 2*8744576,
  2378639438, 2*304024301, 2*103184153, 2*161397053, 2*35484840, 2*69308189, 2*139969715, 2*28259508}// 0.5 core
 };
*/
static int workload_group[NUM_WORKLOAD] = {1, 4, 2, 2, 1, 2, 3, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // for now only 1 batch

static int planaria_group[NUM_WORKLOAD] = 
 {1, 10, 4, 5, 2, 4, 4, 2,
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//#define QUEUE_DEPTH 10
//#define SEED 10 // to randomize workload more
//#define CAP 0.8 // 0 to 1 (smaller number: shorter time between workload dispatch time)

//workload creating capacity: cap * sp_cycles * cap_scale(<1)
//QoS target: cap * (qos+1) *  sp_cycles * target_scale(> 1, < 1)
//QoS 0: 4 cores, 1: 2 cores, 2: 1 core, 3: 0.5 x 1 core

static int total_queue_type[NUM_GROUP][MAX_WORKLOAD] = {-1};
static uint64_t total_queue_dispatch[NUM_GROUP][MAX_WORKLOAD] = {0}; // dispatched time (in order)
static uint64_t total_queue_finish[NUM_GROUP][SUB_CORE][MAX_WORKLOAD] = {0};
static int total_queue_status[NUM_GROUP][MAX_WORKLOAD] = {-1}; // -1: not assigned, 0: in assigned queue, >= 1: part
static int total_queue_priority[NUM_GROUP][MAX_WORKLOAD] = {-1}; // 0 - 11
static int total_queue_qos[NUM_GROUP][MAX_WORKLOAD] = {-1}; // latency sensitivity of workload (target: (qos + 1) * 1.2 * sp_cycles)
static uint64_t total_queue_target[NUM_GROUP][MAX_WORKLOAD] = {0};
static uint64_t total_queue_runtime_thread[NUM_GROUP][SUB_CORE][MAX_WORKLOAD] = {0}; // for checking purpose (end to end runtime)
static uint64_t total_queue_runtime_total[NUM_GROUP][SUB_CORE][MAX_WORKLOAD] = {0}; // for checking purpose (end to end runtime)

#define MAX_ITER (int)(total_workloads / QUEUE_DEPTH)
static int gemmini_workload_assigned[NUM_GROUP][SUB_GROUP][MAX_ITER][QUEUE_DEPTH] = {-1};
static uint64_t gemmini_runtime[NUM_CORE] = {0}; // to track real runtime without thread create overhead

static int gemmini_workload_grouped[NUM_GROUP][SUB_GROUP][MAX_ITER][QUEUE_DEPTH] = {-1};
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

int workload_type_assign(bool batch1, bool batch4, bool batch8, uint32_t seed){
  // currently only batch1
  int rand_mod = 160;
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
  uint32_t rand_out = rand_seed(seed);
  int r = rand_out % rand_mod + rand_base;
  if(batch1){
    if(r < 1){
      id = 1;
    }
    else if(r < (1+7)){
      id = 0;
    }
    else if(r < (1+7+14)){
      id = 2;
    }
    else if(r < (1+7+14+16)){
      id = 3;
    }
    else if(r < (1+7+14+16+44)){
      id = 4;
    }
    else if(r < (1+7+14+16+44+21)){
      id = 5;
    }
    else if(r < (1+7+14+16+44+21+13)){
      id = 6;
    }
    else{// if(r < (1+5+11+18+44+24+11+43)){
      id = 7;
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
void workload_mode_1(int qos, int workload, bool batch1, bool batch4, bool batch8, uint32_t seed, int cap, float target_scale, float cap_scale){ 
  // qos < 0 -> mixed
  // qos >= 0 -> workload dispatch qos apart, qos ways at once
  for(int k = 0; k < NUM_GROUP; k++)
    for(int i = 0; i < MAX_WORKLOAD; i++)
      total_queue_status[k][i]= -1;
  
  int first_dispatch_interval = 50000;
  if (batch4) first_dispatch_interval *= 4;
  if (batch8) first_dispatch_interval *= 8;

  int group = cap * (2+1);
  if (qos == 0){ // mixed QoS
    // extremely high QoS (0) should come really rarely
    // 1: 30%, 2: 40%, 3: 30%
    for (int i = 0; i < workload+2*group; i++){
      int select = rand_seed(seed) % 50;
      int workload_qos = 1;
      if(select >= 35)
        workload_qos = 3;
      else if (select >= 15)
        workload_qos = 2;
      else if(select <= 1)
        workload_qos = 0;

      int workload_type = workload_type_assign(batch1, batch4, batch8, seed);//rand_base + rand_seed(seed) % rand_mod;
      total_queue_type[0][i] = workload_type;
      total_queue_priority[0][i] = (workload_qos == 0) ? 10 : 4;
      total_queue_qos[0][i] = workload_qos;
      total_queue_target[0][i] = (workload_qos + 1) * target_scale * cap * sp_cycles[workload_type];

      for(int k = 0; k < NUM_GROUP; k++){
        for (int j = 0; j < SUB_CORE; j++){
          total_queue_finish[k][j][i] = 0;
          total_queue_runtime_thread[k][j][i] = 0;
          total_queue_runtime_total[k][j][i] = 0;
        }
      }
      if(i < group){
        total_queue_dispatch[0][i] = first_dispatch_interval*i;
      }
      else{
        total_queue_dispatch[0][i] = total_queue_dispatch[0][i - group] + sp_cycles[total_queue_type[0][i - group]] * group * cap_scale; // is it enough?
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
        int workload_type = workload_type_assign(batch1, batch4, batch8, seed);
        //int workload_type = rand_base + rand_seed(seed) % rand_mod;
        total_queue_type[0][index] = workload_type; 
//printf("index: %d, output workload type: %d, stored type: %d\n", index, workload_type, total_queue_type[0][index]);
        total_queue_priority[0][index] = 5; // mode 1 -> same priority 
        total_queue_qos[0][index] = qos;
        total_queue_target[0][index] = (qos + 1) * target_scale * cap * sp_cycles[workload_type];

        for(int k = 0; k < NUM_GROUP; k++){
          for (int j = 0; j < SUB_CORE; j++){
            total_queue_finish[k][j][index] = 0;
            total_queue_runtime_thread[k][j][index] = 0;
            total_queue_runtime_total[k][j][index] = 0;
          }
        }
        if(i == 0){
          total_queue_dispatch[0][index] = first_dispatch_interval*j;
        }
        else{
          total_queue_dispatch[0][index] = total_queue_dispatch[0][index - group] + sp_cycles[total_queue_type[0][index - group]] * group * cap_scale; // is it enough?
        }
      }
    }
  }  

  for(int i = 0; i < workload; i++){
    for(int j = i+1; j < workload+2*group; j++){
      if(total_queue_dispatch[0][i] > total_queue_dispatch[0][j]){
        uint64_t a = total_queue_dispatch[0][i];
        total_queue_dispatch[0][i] = total_queue_dispatch[0][j];
        total_queue_dispatch[0][j] = a;
 
        a = total_queue_target[0][i];
        total_queue_target[0][i] = total_queue_target[0][j];
        total_queue_target[0][j] = a;
 
        int b = total_queue_priority[0][i];
        total_queue_priority[0][i] = total_queue_priority[0][j];
        total_queue_priority[0][j] = b;
  
        b = total_queue_type[0][i];
        total_queue_type[0][i] = total_queue_type[0][j];
        total_queue_type[0][j] = b;
 
        b = total_queue_qos[0][i];
        total_queue_qos[0][i] = total_queue_qos[0][j];
        total_queue_qos[0][j] = b;                    
      }
    }
  }

  for (int i = 0; i < workload; i++){
    int workload_qos = total_queue_qos[0][workload - i - 1];

    int workload_type = total_queue_type[0][workload - i - 1];
    total_queue_type[1][i] = workload_type;
    total_queue_priority[1][i] = (workload_qos == 0) ? 10 : 4;
    total_queue_qos[1][i] = workload_qos;
    total_queue_target[1][i] = (workload_qos + 1) * target_scale * cap * sp_cycles[workload_type];

    if(i < group){
      total_queue_dispatch[1][i] = first_dispatch_interval*i;
    }
    else{
      total_queue_dispatch[1][i] = total_queue_dispatch[1][i - group] + sp_cycles[total_queue_type[1][i - group]] * group * cap_scale; // is it enough?
    }
  }

  for(int i = 0; i < workload; i++){
    for(int j = i+1; j < workload; j++){
      if(total_queue_dispatch[1][i] > total_queue_dispatch[1][j]){
        uint64_t a = total_queue_dispatch[1][i];
        total_queue_dispatch[1][i] = total_queue_dispatch[1][j];
        total_queue_dispatch[1][j] = a;
 
        a = total_queue_target[1][i];
        total_queue_target[1][i] = total_queue_target[1][j];
        total_queue_target[1][j] = a;
 
        int b = total_queue_priority[1][i];
        total_queue_priority[1][i] = total_queue_priority[1][j];
        total_queue_priority[1][j] = b;
  
        b = total_queue_type[1][i];
        total_queue_type[1][i] = total_queue_type[1][j];
        total_queue_type[1][j] = b;
 
        b = total_queue_qos[1][i];
        total_queue_qos[1][i] = total_queue_qos[1][j];
        total_queue_qos[1][j] = b;                    
      }
    }
  }
  for(int k = 0; k < NUM_GROUP; k++){
    for(int i = workload; i < workload+2*group; i++){
      total_queue_dispatch[k][i] = 0;
      total_queue_priority[k][i] = -1;
      total_queue_type[k][i] = -1;
      total_queue_status[k][i] = -1;
      total_queue_qos[k][i] = -1;
    }
  }

  //for(int i = 0; i < workload; i++)
    //printf("after mixing entry %d, workload id %d\n", i, total_queue_type[i]);

  for(int i = 0; i < NUM_CORE; i++){
      gemmini_runtime[i] = 0; // initialize time 
  }
  for(int c = 0; c < NUM_GROUP; c++)
    for(int k = 0; k < SUB_GROUP; k++)
      for(int i = 0; i < MAX_ITER; i++)
        for(int j = 0; j < QUEUE_DEPTH; j++)
          gemmini_workload_assigned[c][k][i][j] = -1;
}

void workload_mode_2(int workload, bool batch1, bool batch4, bool batch8, uint32_t seed, int cap, float target_scale, float cap_scale){
  // priority (0: 15, 1: 18 / 2: 10, 4: 15, 6: 15, 8: 15 / 9: 10, 11: 2)
  for(int k = 0; k < NUM_GROUP; k++)
    for(int i = 0; i < MAX_WORKLOAD; i++)
      total_queue_status[k][i]= -1;
  int qos = 1; // to lowest QoS
  int group = (qos+1)*cap;//8;

  int first_dispatch_interval = 50000;
  if (batch4) first_dispatch_interval *= 4;
  if (batch8) first_dispatch_interval *= 8;

  int num_workload_group = ceil_divide_int(workload+group, group);

  for(int i = 0; i < num_workload_group; i++){
    for(int j = 0; j < group; j++){
      int index = group * i + j;
      int workload_type = workload_type_assign(batch1, batch4, batch8, seed);
      //int workload_type = rand_base + rand_seed(seed) % rand_mod;
      total_queue_type[0][index] = workload_type; 
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
      total_queue_priority[0][index] = priority_level; 
      total_queue_qos[0][index] = qos;
      total_queue_target[0][index] = (qos+1)*cap*target_scale*sp_cycles[workload_type];
      for(int k = 0; k < NUM_GROUP; k++){
        for (int j = 0; j < SUB_CORE; j++){
          total_queue_finish[k][j][index] = 0;
          total_queue_runtime_thread[k][j][index] = 0;
          total_queue_runtime_total[k][j][index] = 0;
        }
      }
      if(i == 0){
        total_queue_dispatch[0][index] = first_dispatch_interval*j;
      }
      else{
        total_queue_dispatch[0][index] = total_queue_dispatch[0][index - group] + 2 * sp_cycles[total_queue_type[0][index - group]] * (group) * cap_scale; // divide by 2
      }
    }
  }
 
  for(int i = 0; i < workload; i++){
    for(int j = i+1; j < workload+group; j++){
      if(total_queue_dispatch[0][i] > total_queue_dispatch[0][j]){
        uint64_t a = total_queue_dispatch[0][i];
        total_queue_dispatch[0][i] = total_queue_dispatch[0][j];
        total_queue_dispatch[0][j] = a;
 
        a = total_queue_target[0][i];
        total_queue_target[0][i] = total_queue_target[0][j];
        total_queue_target[0][j] = a;
 
        int b = total_queue_priority[0][i];
        total_queue_priority[0][i] = total_queue_priority[0][j];
        total_queue_priority[0][j] = b;
  
        b = total_queue_type[0][i];
        total_queue_type[0][i] = total_queue_type[0][j];
        total_queue_type[0][j] = b;
 
        b = total_queue_qos[0][i];
        total_queue_qos[0][i] = total_queue_qos[0][j];
        total_queue_qos[0][j] = b;                    
      }
    }
  }

  for (int i = 0; i < workload; i++){
    int workload_qos = total_queue_qos[0][workload - i - 1];
    int workload_priority = total_queue_priority[0][workload - i - 1];
    int workload_type = total_queue_type[0][workload - i - 1];
    total_queue_type[1][i] = workload_type;
    total_queue_priority[1][i] = workload_priority;//(workload_qos == 0) ? 10 : 4;
    total_queue_qos[1][i] = workload_qos;
    total_queue_target[1][i] = (workload_qos + 1) * target_scale * cap * sp_cycles[workload_type];

    if(i < group){
      total_queue_dispatch[1][i] = first_dispatch_interval*i;
    }
    else{
      total_queue_dispatch[1][i] = total_queue_dispatch[1][i - group] + 2 * sp_cycles[total_queue_type[1][i - group]] * group * cap_scale; // is it enough?
    }
  }

  for(int i = 0; i < workload; i++){
    for(int j = i+1; j < workload; j++){
      if(total_queue_dispatch[1][i] > total_queue_dispatch[1][j]){
        uint64_t a = total_queue_dispatch[1][i];
        total_queue_dispatch[1][i] = total_queue_dispatch[1][j];
        total_queue_dispatch[1][j] = a;
 
        a = total_queue_target[1][i];
        total_queue_target[1][i] = total_queue_target[1][j];
        total_queue_target[1][j] = a;
 
        int b = total_queue_priority[1][i];
        total_queue_priority[1][i] = total_queue_priority[1][j];
        total_queue_priority[1][j] = b;
  
        b = total_queue_type[1][i];
        total_queue_type[1][i] = total_queue_type[1][j];
        total_queue_type[1][j] = b;
 
        b = total_queue_qos[1][i];
        total_queue_qos[1][i] = total_queue_qos[1][j];
        total_queue_qos[1][j] = b;                    
      }
    }
  }
  
  for(int k = 0; k < NUM_GROUP; k++){
    for(int i = workload; i < workload+group; i++){
      total_queue_dispatch[k][i] = 0;
      total_queue_priority[k][i] = -1;
      total_queue_type[k][i] = -1;
      total_queue_status[k][i] = -1;
      total_queue_qos[k][i] = -1;
    }
  }


  for(int i = 0; i < NUM_CORE; i++){
      gemmini_runtime[i] = 0; // initialize time 
    
  }
  for(int c = 0; c < NUM_GROUP; c++)
    for(int k = 0; k < SUB_GROUP; k++)
      for(int i = 0; i < MAX_ITER; i++)
        for(int j = 0; j < QUEUE_DEPTH; j++)
          gemmini_workload_assigned[c][k][i][j] = -1;


}

// priority scheduling
// num_sub_group: SUB_CORE (2), or 1 if doing 4+4 partitioning
int workload_priority_mp(int group, int num_sub_group, int num_workload, int num_iter, uint64_t current_cycle){
  for(int c = 0; c < SUB_GROUP; c++)
    for(int i = 0; i < MAX_ITER; i++)
      for(int j = 0; j < QUEUE_DEPTH; j++){
        gemmini_workload_grouped[group][c][i][j] = -1;
        gemmini_workload_assigned[group][c][i][j] = -1;
      }

  int num_batch = 1;
  if(total_queue_type[group][0] >= FCNNET_4)  num_batch = 4;
  if(total_queue_type[group][0] >= FCNNET_8)  num_batch = 8;
//  printf("workload_priority_mp current cycles: %llu\n", current_cycle);
  int group_temp[num_sub_group];
  uint64_t cycle[num_sub_group];
  for (int i = 0; i < num_sub_group; i++){
    cycle[i] = current_cycle + 500000 * num_batch;
    group_temp[i] = 0;
    gemmini_dram_util[group*num_sub_group+i] = 0;
  }

  // priority score initialization
  int64_t score[num_workload];
  int max_depth = QUEUE_DEPTH * 1.8;

  int iter = 0;

  // repeat from here
  int pre_assign_queue[max_depth];
  int64_t pre_assign_score[max_depth]; // need this?
  int pre_assign_length = 0;
  while (iter < num_iter){
    for(int i = 0; i < max_depth; i++){
      pre_assign_queue[i] = -1;
      pre_assign_score[i] = -1;
    }

    uint64_t top_cycle = cycle[0];
    // get max cycle
    for(int i = 0; i < num_sub_group; i++){
      if(top_cycle > cycle[i]){
        top_cycle = cycle[i];
      }
    }

    int pointer = 0;
    for(int i = 0; i < num_workload; i++){
      if(total_queue_dispatch[group][i] > top_cycle){
        pointer = i;
        break;
      }
      else if(i == num_workload - 1){
        pointer = num_workload;
      }
    }
    pointer = (pointer >= num_workload - 5) ? num_workload : pointer; 

    bool done = true;
    for (int i = 0; i < pointer; i++){
      if(total_queue_status[group][i] == -1){ //only take the unassigned ones
        score[i] = total_queue_priority[group][i];
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
        int qos = total_queue_qos[group][i];
        int type = total_queue_type[group][i];
        uint64_t after_dispatch = (top_cycle > total_queue_dispatch[group][i]) ? (top_cycle - total_queue_dispatch[group][i]) : 0;
        score[i] = score[i]*100000 + ((4*100000*after_dispatch) / (CAP*sp_prediction_cycles[type]));
        //score[i] = score[i]*100000 + ((100000*after_dispatch) / (qos*sp_prediction_cycles[1][type]));
      }
    }

    // first, pick candidate
    // next, assign using cycle predictio
    
  
    int queue_index = 0;
    int max_index = -1;
    int64_t max_score = -1;
    while(queue_index < max_depth){
      for(int i = 0; i < pointer; i++){
        if(total_queue_status[group][i] == -1){
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
      /*
      if(queue_index > 0){
        if(iter == 0 && pre_assign_score[queue_index - 1] > max_score + 6 * 100000){
          max_index = -1;
          //printf("max score: %d, pre_assign_score: %d\n", max_score, pre_assign_score[queue_index - 1]);
          max_score = -1;
          break;
        }
      }
      */
      pre_assign_queue[queue_index] = max_index;
      pre_assign_score[queue_index] = max_score;
      queue_index ++;
      total_queue_status[group][max_index] = 0;
      max_index = -1;
      max_score = -1;
    }
    if(queue_index == 0 && ((iter == 0 && pre_assign_length == 0) || (iter == 1 && pre_assign_length == 1))) break;
   // printf("queue_index: %d, pre_assign_length: %d, iter: %d\n", queue_index, pre_assign_length, iter);
   pre_assign_length = queue_index;
   /*
    printf("pre assigned queue length: %d \n", pre_assign_length);
    for(int i = 0; i < pre_assign_length; i++)
      printf("%d, ", pre_assign_queue[i]); 
    printf("\n");
    */

    for (int p = 0; p < pre_assign_length; ){
  //    printf("iter: %d, index: %d, cycle0: %llu, cycle1: %llu, group0: %d, group1: %d\n", iter, index, cycle[0], cycle[1], group_temp[0], group_temp[1]);
      bool full = false;
      for(int i = 0; i < num_sub_group; i++){
        if(group_temp[i] == QUEUE_DEPTH){
          full = true;
          break;
        }
      }

      if(!full){
        // check iter == 1 and first entries are same when executing
        if (pre_assign_length == 1 && iter == 0){
          int index = pre_assign_queue[0];
          int type = total_queue_type[group][index];
          for(int k=0; k < num_sub_group; k++){
            gemmini_workload_assigned[group][k][iter][group_temp[k]] = index;
            cycle[k] += sp2_cycles[type];
            group_temp[k] += 1;
            p++;
          }
          iter = num_iter;
          break;
        }
          
        for(int k = 0; k < num_sub_group; k++){
          bool smallest = true;
          for(int c = 0; c < num_sub_group; c++){
            if(cycle[k] > cycle[c]){
              smallest = false;
              break;
            }
          }
          if(smallest && p < pre_assign_length){
            int index = pre_assign_queue[p];
            gemmini_workload_assigned[group][k][iter][group_temp[k]] = index;
            int type = total_queue_type[group][index];
            cycle[k] += sp2_cycles[type];
            group_temp[k] += 1;
            p++;
          }
        }
      }
      else{
        // release status
        int index = pre_assign_queue[p];
        total_queue_status[group][index] = -1;
        p++;
      }
    }
    iter += 1;
    for(int i = 0; i < num_sub_group; i++){
      group_temp[i] = 0;
    }
  }

  // if returned value is 0, then it is over
  return (iter); // number of queue group
}

void workload_grouping(int num_iter, int group_id){
  for(int group = 0; group < SUB_GROUP; group++){
    for(int iter = 0; iter < num_iter; iter++){
      for(int i = 0; i < QUEUE_DEPTH; i++){
        int queue_id = gemmini_workload_assigned[group_id][group][iter][i];
            //printf("queue_id: %d\n", queue_id);
        if(queue_id != -1){
          int workload_type = total_queue_type[group_id][queue_id];
          if(workload_type == RESNET_1){ // if it is resnet
            bool groupped = false;
            for(int i_next = i; i_next < QUEUE_DEPTH; i_next++){
              int next_queue_id = gemmini_workload_assigned[group_id][group][iter][i_next];
              int next_type = total_queue_type[group_id][next_queue_id];
              if(next_type == -1) break;
              if(next_type == GOOGLENET_1 ||next_type == SQUEEZENET_1 || next_type == YOLOLITENET_1 || next_type == KWSNET_1 || next_type == ALEXNET_1){
                if(gemmini_workload_grouped[group_id][group][iter][i_next] != -7){
                  gemmini_workload_grouped[group_id][group][iter][i_next] = -7; // mark 0
                 
                  gemmini_workload_grouped[group_id][group][iter][i] = next_queue_id;
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
                  int next_queue_id = gemmini_workload_assigned[group_id][group][iter_temp][i_next];
                  int next_type = total_queue_type[group_id][next_queue_id];
                  if(next_type == -1) break;
                  if(next_type == GOOGLENET_1 ||next_type == SQUEEZENET_1 || next_type == YOLOLITENET_1 || next_type == KWSNET_1 || next_type == ALEXNET_1){
                    if(gemmini_workload_grouped[group_id][group][iter_temp][i_next] != -7){ 
                      gemmini_workload_grouped[group_id][group][iter_temp][i_next] = -7;
                      gemmini_workload_grouped[group_id][group][iter][i] = next_queue_id; 
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
              int next_queue_id = gemmini_workload_assigned[group_id][group][iter][i_next];
              int next_type = total_queue_type[group_id][next_queue_id];
              if(next_type == -1) break;
              if(next_type == GOOGLENET_1 || next_type == YOLONET_1 || next_type == KWSNET_1 || next_type == RESNET_1){
                if(gemmini_workload_grouped[group_id][group][iter][i_next] != -7){
                  gemmini_workload_grouped[group_id][group][iter][i_next] = -7; // mark 0
                  gemmini_workload_grouped[group_id][group][iter][i] = next_queue_id;
                  groupped = true;
                  break;
                }
              }
            }
            if(!groupped && iter != num_iter - 1){ 
              int iter_temp = iter + 1;
              while(iter_temp < num_iter && !groupped){
                for(int i_next = 0; i_next < QUEUE_DEPTH; i_next++){
                  int next_queue_id = gemmini_workload_assigned[group_id][group][iter_temp][i_next];
                  int next_type = total_queue_type[group_id][next_queue_id];
                  if(next_type == -1) break;
                  if(next_type == GOOGLENET_1 || next_type == YOLONET_1 || next_type == KWSNET_1 || next_type == RESNET_1 ){
                    if(gemmini_workload_grouped[group_id][group][iter_temp][i_next] != -7){ 
                      gemmini_workload_grouped[group_id][group][iter_temp][i_next] = -7;
                      gemmini_workload_grouped[group_id][group][iter][i] = next_queue_id; 
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
              int next_queue_id = gemmini_workload_assigned[group_id][group][iter][i_next];
              int next_type = total_queue_type[group_id][next_queue_id];
              if(next_type == -1) break;
              if(next_type == GOOGLENET_1 ||next_type == SQUEEZENET_1 || next_type == YOLOLITENET_1 || next_type == KWSNET_1 || next_type == ALEXNET_1){
                if(gemmini_workload_grouped[group_id][group][iter][i_next] != -7){
                  gemmini_workload_grouped[group_id][group][iter][i_next] = -7;
                  gemmini_workload_grouped[group_id][group][iter][i] = next_queue_id;
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
                  int next_queue_id = gemmini_workload_assigned[group_id][group][iter_temp][i_next];
                  int next_type = total_queue_type[group_id][next_queue_id];
                  if(next_type == -1) break;
                  if(next_type == GOOGLENET_1 ||next_type == SQUEEZENET_1 || next_type == YOLOLITENET_1 || next_type == KWSNET_1 || next_type == ALEXNET_1){
                    if(gemmini_workload_grouped[group_id][group][iter_temp][i_next] != -7){ 
                      gemmini_workload_grouped[group_id][group][iter_temp][i_next] = -7;
                      gemmini_workload_grouped[group_id][group][iter][i] = next_queue_id; 
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
uint64_t workload_function(int queue_id, int workload_id, size_t cid, size_t group_id, size_t sub_group, int num_gemmini, int dram_util, pthread_barrier_t *barrier_funct){
  gemmini_flush(0);
  uint64_t* cycles;
  uint64_t total_runtime;

  size_t sub_group_id = sub_group;
  //size_t sub_group_id = group_id * NUM_GROUP + sub_group; // out of total sub-group
  int group_status = total_queue_status[group_id][queue_id];
  bool part1 = group_status < 1;
  bool part2 = group_status < 2;
  bool part3 = group_status < 3;
  bool part4 = group_status < 4;
//printf("part1: %d, part2: %d, part3: %d, part4: %d\n", part1, part2, part3, part4);
  //uint64_t start = read_cycles();
#if BATCH1 == true
  if(workload_id < 8){
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
      if(sub_group_id % 2 == 0) cycles = yololitenet_function_1(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
      else cycles = yololitenet_function_11(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
      total_runtime = *(cycles+14);
    }
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
  if(cid == 0) {
    gemmini_dram_util[sub_group_id] = 0;
    total_queue_status[group_id][queue_id] = 100; // just store big value (finished)
  }

  //if(cid == 0) total_queue_status[queue_id] = 100; // just store big value (finished)
  //uint64_t runtime = read_cycles() - start;
  return total_runtime;

}

// sub_group_id: 0 - 3
uint64_t workload_group_function(int queue_id, int group_queue_id, int original_workload_id, int grouped_workload_id, size_t cid, size_t group_id, size_t sub_group_id, int num_gemmini, int dram_util, pthread_barrier_t *barrier_funct){
  gemmini_flush(0);
  uint64_t* cycles;
  uint64_t total_runtime;

  int group_status = total_queue_status[group_id][queue_id];
  bool part1 = group_status < 1;
  bool part2 = group_status < 2;
  bool part3 = group_status < 3;
  bool part4 = group_status < 4;
#if BATCH1 == true
  //int dram_util_half = (cid == 0) ? dram_util : (dram_util / 2) - 10;
  if(sub_group_id % 2 == 0){
    //uint64_t start = read_cycles();
    if(original_workload_id < 8){
      int orow_divide = num_gemmini;
      int batch_divide = 1; // 1 batch workload
      if(original_workload_id == 1){
        if(part1 || part2 || part3){
          cycles = resnet_function_1(cid, sub_group_id, part1, part2, part3, false, orow_divide, batch_divide, dram_util, barrier_funct);
         // total_runtime = *(cycles+72);
        }

        if(cid == 0){
  //	if(grouped_workload_id == SQUEEZENET_1 || grouped_workload_id == YOLOLITENET_1){
  //	   dram_util += 10;
  //	}
          cycles = resnet_block_function_1(0, sub_group_id, false, false, false, true, 1, 1, dram_util);
          total_runtime = *(cycles+72);
        }
        else{
          if(grouped_workload_id == SQUEEZENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = squeezenet_block_function_1(0, sub_group_id, 1, 1, dram_util); 
            total_runtime = *(cycles+29);
            total_queue_status[group_id][group_queue_id] = 50;
          }
          else if(grouped_workload_id == YOLOLITENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
  //	  dram_util = 10;
            cycles = yololitenet_block_function_1(0, sub_group_id, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_id][group_queue_id] = 50;
          }
          else if(grouped_workload_id == KWSNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = kwsnet_block_function_1(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 40);
            total_queue_status[group_id][group_queue_id] = 1;
          }
          else if(grouped_workload_id == ALEXNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = alexnet_block_function_1(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_id][group_queue_id] = 1;
          }
          else if(grouped_workload_id == GOOGLENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = googlenet_block_function_1(0, sub_group_id, true, false, 1, 1, dram_util); 
            total_runtime = *(cycles+71);
            total_queue_status[group_id][group_queue_id] = 1;
          }
        }
      }
      else if(original_workload_id == 2){
        if(part1){
          cycles = alexnet_function_1(cid, sub_group_id, part1, false, orow_divide, batch_divide, dram_util, barrier_funct);
          //total_runtime = *(cycles+14);
        }

        if(cid == 0){
          cycles = alexnet_block_function_1(0, sub_group_id, false, true, 1, 1, dram_util);
          total_runtime = *(cycles+14);
        }
        else{
          if(grouped_workload_id == GOOGLENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = googlenet_block_function_1(0, sub_group_id, true, true, 1, 1, dram_util); 
            total_runtime = *(cycles+71);
            total_queue_status[group_id][group_queue_id] = 50;
          }
          else if(grouped_workload_id == YOLONET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = yolonet_block_function_1(0, sub_group_id, true, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 26);
            total_queue_status[group_id][group_queue_id] = 2;
          }
          else if(grouped_workload_id == KWSNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = kwsnet_block_function_1(0, sub_group_id, true, true, 1, 1, dram_util);
            total_runtime = *(cycles + 40);
            total_queue_status[group_id][group_queue_id] = 50;
          }
          else if(grouped_workload_id == RESNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = resnet_block_function_1(0, sub_group_id, true, true, false, false, 1, 1, dram_util);
            total_runtime = *(cycles + 72);
            total_queue_status[group_id][group_queue_id] = 2;
          }
        }
      }
      else if(original_workload_id == 6){
        if(part1 || part2){
          cycles = yolonet_function_1(cid, sub_group_id, part1, part2, false, orow_divide, batch_divide, dram_util, barrier_funct);
          //total_runtime = *(cycles+26);
        }

        if(cid == 0){
  //	if(grouped_workload_id == SQUEEZENET_1 || grouped_workload_id == YOLOLITENET_1){
  //	   dram_util += 10;
  //	}
          cycles = yolonet_block_function_1(0, sub_group_id, false, false, true, 1, 1, dram_util);
          total_runtime = *(cycles+26);
        }
        else{
          if(grouped_workload_id == SQUEEZENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
  //	  dram_util = 10;
            cycles = squeezenet_block_function_1(0, sub_group_id, 1, 1, dram_util); 
            total_runtime = *(cycles+29);
            total_queue_status[group_id][group_queue_id] = 50;
          }
          else if(grouped_workload_id == YOLOLITENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
  //	  dram_util = 10;
            cycles = yololitenet_block_function_1(0, sub_group_id, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_id][group_queue_id] = 50;
          }
          else if(grouped_workload_id == KWSNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = kwsnet_block_function_1(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 40);
            total_queue_status[group_id][group_queue_id] = 1;
          }
          else if(grouped_workload_id == ALEXNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = alexnet_block_function_1(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_id][group_queue_id] = 1;
          }
          else if(grouped_workload_id == GOOGLENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = googlenet_block_function_1(0, sub_group_id, true, false, 1, 1, dram_util); 
            total_runtime = *(cycles+71);
            total_queue_status[group_id][group_queue_id] = 1;
          }
        }
      }
    }
  }
  else{
    if(original_workload_id < 8){
      int orow_divide = num_gemmini;
      int batch_divide = 1; // 1 batch workload
      if(original_workload_id == 1){
        if(part1 || part2 || part3){
          cycles = resnet_function_11(cid, sub_group_id, part1, part2, part3, false, orow_divide, batch_divide, dram_util, barrier_funct);
         // total_runtime = *(cycles+72);
        }

        if(cid == 0){
  //	if(grouped_workload_id == SQUEEZENET_1 || grouped_workload_id == YOLOLITENET_1){
  //	   dram_util += 10;
  //	}
          cycles = resnet_block_function_11(0, sub_group_id, false, false, false, true, 1, 1, dram_util);
          total_runtime = *(cycles+72);
        }
        else{
          if(grouped_workload_id == SQUEEZENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = squeezenet_block_function_11(0, sub_group_id, 1, 1, dram_util); 
            total_runtime = *(cycles+29);
            total_queue_status[group_id][group_queue_id] = 50;
          }
          else if(grouped_workload_id == YOLOLITENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
  //	  dram_util = 10;
            cycles = yololitenet_block_function_11(0, sub_group_id, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_id][group_queue_id] = 50;
          }
          else if(grouped_workload_id == KWSNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = kwsnet_block_function_11(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 40);
            total_queue_status[group_id][group_queue_id] = 1;
          }
          else if(grouped_workload_id == ALEXNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = alexnet_block_function_11(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_id][group_queue_id] = 1;
          }
          else if(grouped_workload_id == GOOGLENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = googlenet_block_function_11(0, sub_group_id, true, false, 1, 1, dram_util); 
            total_runtime = *(cycles+71);
            total_queue_status[group_id][group_queue_id] = 1;
          }
        }
      }
      else if(original_workload_id == 2){
        if(part1){
          cycles = alexnet_function_11(cid, sub_group_id, part1, false, orow_divide, batch_divide, dram_util, barrier_funct);
          //total_runtime = *(cycles+14);
        }

        if(cid == 0){
          cycles = alexnet_block_function_11(0, sub_group_id, false, true, 1, 1, dram_util);
          total_runtime = *(cycles+14);
        }
        else{
          if(grouped_workload_id == GOOGLENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = googlenet_block_function_11(0, sub_group_id, true, true, 1, 1, dram_util); 
            total_runtime = *(cycles+71);
            total_queue_status[group_id][group_queue_id] = 50;
          }
          else if(grouped_workload_id == YOLONET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = yolonet_block_function_11(0, sub_group_id, true, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 26);
            total_queue_status[group_id][group_queue_id] = 2;
          }
          else if(grouped_workload_id == KWSNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = kwsnet_block_function_11(0, sub_group_id, true, true, 1, 1, dram_util);
            total_runtime = *(cycles + 40);
            total_queue_status[group_id][group_queue_id] = 50;
          }
          else if(grouped_workload_id == RESNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = resnet_block_function_11(0, sub_group_id, true, true, false, false, 1, 1, dram_util);
            total_runtime = *(cycles + 72);
            total_queue_status[group_id][group_queue_id] = 2;
          }
        }
      }
      else if(original_workload_id == 6){
        if(part1 || part2){
          cycles = yolonet_function_11(cid, sub_group_id, part1, part2, false, orow_divide, batch_divide, dram_util, barrier_funct);
          //total_runtime = *(cycles+26);
        }

        if(cid == 0){
  //	if(grouped_workload_id == SQUEEZENET_1 || grouped_workload_id == YOLOLITENET_1){
  //	   dram_util += 10;
  //	}
          cycles = yolonet_block_function_11(0, sub_group_id, false, false, true, 1, 1, dram_util);
          total_runtime = *(cycles+26);
        }
        else{
          if(grouped_workload_id == SQUEEZENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
  //	  dram_util = 10;
            cycles = squeezenet_block_function_11(0, sub_group_id, 1, 1, dram_util); 
            total_runtime = *(cycles+29);
            total_queue_status[group_id][group_queue_id] = 50;
          }
          else if(grouped_workload_id == YOLOLITENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
  //	  dram_util = 10;
            cycles = yololitenet_block_function_11(0, sub_group_id, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_id][group_queue_id] = 50;
          }
          else if(grouped_workload_id == KWSNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = kwsnet_block_function_11(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 40);
            total_queue_status[group_id][group_queue_id] = 1;
          }
          else if(grouped_workload_id == ALEXNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = alexnet_block_function_11(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_id][group_queue_id] = 1;
          }
          else if(grouped_workload_id == GOOGLENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = googlenet_block_function_11(0, sub_group_id, true, false, 1, 1, dram_util); 
            total_runtime = *(cycles+71);
            total_queue_status[group_id][group_queue_id] = 1;
          }
        }
      }
    }


  }
#endif

  if(cid == 0) {
    gemmini_dram_util[sub_group_id] = 0;
    total_queue_status[group_id][queue_id] = 80; // just store big value (finished)
  }
  //uint64_t runtime = read_cycles() - start;
  return total_runtime;

}


// for planaria
// barrier_funct2: for subgroup (between 2), barrier_funct4: for group (between 4)
uint64_t workload_planaria_function(int queue_id, int workload_id, size_t cid, size_t group_id, size_t sub_group_id, int num_gemmini, uint64_t slack_time,  pthread_barrier_t *barrier_funct){
  gemmini_flush(0);
  uint64_t* cycles;
  uint64_t total_runtime = 0;

  if(cid == 0){
    gemmini_terminate_receive[sub_group_id] = false;
    gemmini_terminate[other_sub_group_id] = false;
  }
  pthread_barrier_wait(barrier_funct);
  int other_sub_group_id = group_id * WORKLOAD_CORE + (sub_group_id % 2 == 0 ? 1 : 0);   
  //size_t sub_group_id = group_id * NUM_GROUP + sub_group; // out of total sub-group
  int group_status = total_queue_status[group_id][queue_id]; // need to update when terminated, check this if it finshed outer loop

  bool part[30];
  for(int i = 0; i < 30; i++){
    part[i] = group_status < (i+1);
  }
#if BATCH1 == true
  if(workload_id < 8){
    int orow_divide = num_gemmini;
    int batch_divide = 1; // 1 batch workload
    if(workload_id == 0){
      cycles = fcnnet_function_1(cid, sub_group_id, orow_divide, batch_divide, -1, barrier_funct);
      total_runtime = *(cycles+73);
    }
    else if(workload_id == 1){
      uint64_t slack = 20000000;
//      int other_sub_group_id = group_id * WORKLOAD_CORE + (sub_group_id % 2 == 0 ? 1 : 0);      
      for(int i = 0; i < 10; i++){
        uint64_t start = read_cycles();
        if(num_gemmini <= 2){
          if(gemmini_score[other_sub_group_id] < gemmini_score[sub_group_id]){    
            if(cid == 0 && slack_time < (slack - 2000000*i) * planaria_scale){
              gemmini_terminate[other_sub_group_id] == true;
            }
          }
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            gemmini_terminate_recieve[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
          if(gemmini_terminate_recieve[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = resnet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = resnet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime += *(cycles+72);
          if(cid == 0) total_queue_status[group_id][queue_id] = (i+1); 
        }
        uint64_t end = read_cycles();
        slack_time = slack_time - (end - start);
      }

    }
    else if(workload_id == 2){
      uint64_t slack = {12522781, 6429112, 2297240, 459726}; 
      for(int i = 0; i < 4; i++){
        uint64_t start = read_cycles(); 
        if(num_gemmini <= 2){ 
          if(gemmini_score[other_sub_group_id] < gemmini_score[sub_group_id]){    
            if(cid == 0 && slack_time < slack[i] * planaria_scale){
              gemmini_terminate[other_sub_group_id] == true;
            }
          }
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            gemmini_terminate_recieve[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
          if(gemmini_terminate_recieve[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = alexnet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = alexnet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime += *(cycles+14);
          if(cid == 0) total_queue_status[group_id][queue_id] = (i+1); 
        }
        uint64_t end = read_cycles();
        slack_time = slack_time - (end - start);
      }
    }
    else if(workload_id == 3){
      uint64_t slack = 10000000;
  //    int other_sub_group_id = group_id * WORKLOAD_CORE + (sub_group_id % 2 == 0 ? 1 : 0);      
      for(int i = 0; i < 5; i++){
        uint64_t start = read_cycles();
        if(num_gemmini <= 2){
          if(gemmini_score[other_sub_group_id] < gemmini_score[sub_group_id]){    
            if(cid == 0 && slack_time < (slack - 2000000*i) * planaria_scale){
              gemmini_terminate[other_sub_group_id] == true;
            }
          }
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            gemmini_terminate_recieve[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
          if(gemmini_terminate_recieve[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = googlenet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = googlenet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime += *(cycles+71);
          if(cid == 0) total_queue_status[group_id][queue_id] = (i+1); 
        }
        uint64_t end = read_cycles();
        slack_time = slack_time - (end - start);
      }

    }
    else if(workload_id == 4){
      uint64_t slack = 2400000;
      for(int i = 0; i < 2; i++){
        uint64_t start = read_cycles();
        if(num_gemmini <= 2){
          if(gemmini_score[other_sub_group_id] < gemmini_score[sub_group_id]){
            if(cid == 0 && slack_time < ((slack - 1200000*i) * planaria_scale)){
              gemmini_terminate[other_sub_group_id] == true;
            }
          }
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            gemmini_terminate_recieve[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
          if(gemmini_terminate_recieve[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = squeezenet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = squeezenet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime = *(cycles+29);
          if(cid == 0) total_queue_status[group_id][queue_id] = i+1;
        }
        uint64_t end = read_cycles();
        slack_time = slack_time - (end - start);
      }
    }
    else if(workload_id == 5){
      uint64_t slack = 4000000;
      for(int i = 0; i < 4; i++){
        uint64_t start = read_cycles();
        if(num_gemmini <= 2){
          if(gemmini_score[other_sub_group_id] < gemmini_score[sub_group_id]){
            if(cid == 0 && slack_time < ((slack - 1000000*i) * planaria_scale)){
              gemmini_terminate[other_sub_group_id] == true;
            }
          }
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            gemmini_terminate_recieve[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
          if(gemmini_terminate_recieve[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = kwsnet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = kwsnet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime = *(cycles+40);
          if(cid == 0) total_queue_status[group_id][queue_id] = i+1;
        }
        uint64_t end = read_cycles();
        slack_time = slack_time - (end - start);
      }
        
    }
    else if(workload_id == 6){
      uint64_t slack = 10000000;
      for(int i = 0; i < 4; i++){
        uint64_t start = read_cycles();
        if(num_gemmini <= 2){
          if(gemmini_score[other_sub_group_id] < gemmini_score[sub_group_id]){
            if(cid == 0 && slack_time < ((slack - 2000000*i) * planaria_scale)){
              gemmini_terminate[other_sub_group_id] == true;
            }
          }
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            gemmini_terminate_recieve[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
          if(gemmini_terminate_recieve[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = yolonet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = yolonet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime = *(cycles+26);
          if(cid == 0) total_queue_status[group_id][queue_id] = i+1;
        }
        uint64_t end = read_cycles();
        slack_time = slack_time - (end - start);
      }
  
    }
    else if(workload_id == 7){
      uint64_t slack = 2000000;
      for(int i = 0; i < 2; i++){
        uint64_t start = read_cycles();
        if(num_gemmini <= 2){
          if(gemmini_score[other_sub_group_id] < gemmini_score[sub_group_id]){
            if(cid == 0 && slack_time < ((slack - 1000000*i) * planaria_scale)){
              gemmini_terminate[other_sub_group_id] == true;
            }
          }
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            gemmini_terminate_recieve[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
          if(gemmini_terminate_recieve[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = yololitenet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = yololitenet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime = *(cycles+14);
          if(cid == 0) total_queue_status[group_id][queue_id] = i+1;
        }
        uint64_t end = read_cycles();
        slack_time = slack_time - (end - start);
      }
    
    }
  }
#endif
/*
  if(cid == 0) {
    gemmini_dram_util[sub_group_id] = 0;
    total_queue_status[group_id][queue_id] = 100; // just store big value (finished)
  }
*/
  //if(cid == 0) total_queue_status[queue_id] = 100; // just store big value (finished)
  //uint64_t runtime = read_cycles() - start;
  return total_runtime;

}



#endif
