// code for each workload
#if BATCH1 == true
#include "imagenet/funct_resnet_1.h"
#include "imagenet/funct_googlenet_1.h"
#include "imagenet/funct_squeezenet_1.h"
#include "imagenet/funct_kwsnet_1.h"
#include "imagenet/funct_alexnet_1.h"
#include "imagenet/funct_yolonet_1.h"
#include "imagenet/funct_yololitenet_1.h"
#endif

#define RESNET_1 1 // 4 blocks: [12, 25, 44, 54 (mem)] -> with squeezenet(4), yololitenet(7), kwsnet group1(5)
#define ALEXNET_1 2 // 2 blocks: conv, fc -> googlenet(3), resnet group1 (1), kwsnet (5), yolonet group1&2(6)
#define GOOGLENET_1 3 
#define SQUEEZENET_1 4 
#define KWSNET_1 5 // 2 blocks: [13, 25] just divided almost equally based on runtime
#define YOLONET_1 6 // 3 blocks: [4, 13, 19 (mem)] same as ResNet
#define YOLOLITENET_1 7 

// some random cycles for dispatch interval
#if WORKLOAD_CORE == 2
static uint64_t sp_cycles[NUM_WORKLOAD] =
{15070506,8382324,7070440,2608024,9458914,5132036,1978161};
#elif WORKLOAD_CORE == 4
static uint64_t sp_cycles[NUM_WORKLOAD] = 
{9829820,5539067,4923351,1538641,6203714,3314037,1998578};
#endif

static uint64_t target_cycles[NUM_WORKLOAD] = 
 {50000000, 33333334,33333334,10000000,33333334,16666667,10000000};

static int workload_group[NUM_WORKLOAD] = {4, 2, 2, 1, 2, 3, 1};
static int planaria_group[NUM_WORKLOAD] = 
{10, 5, 5, 2, 5, 5, 2};

//workload creating capacity: cap * sp_cycles * cap_scale(<1)
//QoS target: cap * (qos+1) *  sp_cycles * target_scale(> 1, < 1)
//QoS 0: 4 cores, 1: 2 cores, 2: 1 core, 3: 0.5 x 1 core
int rand_seed(uint32_t seed) {
  static uint32_t x = 777;
  x = x * (1664525 + seed) + 1013904223;
  return x >> 24;
}

int workload_type_assign(bool batch1, bool batch4, bool batch8, uint32_t seed){
  // currently only batch1
#if WORKLOAD_A == 1
  int rand_mod = 76;
  int rand_base = 0;

  static int id = 1;
  uint32_t rand_out = rand_seed(seed);
  int r = rand_out % rand_mod + rand_base;
  if(batch1){
    //30
      if(r < (30)){
      id = SQUEEZENET_1;
    }
    //21
    else if(r < (30+21)){
      id = KWSNET_1;
    }
    else{//25
      id = YOLOLITENET_1;
    }
  }
#elif WORKLOAD_B == 1
  int rand_mod = 64;
  int rand_base = 0;

  static int id = 1;
  uint32_t rand_out = rand_seed(seed);
  int r = rand_out % rand_mod + rand_base;
  if(batch1){
    if(r < (0+12)){
      id = RESNET_1;
    }
    else if(r < (0+12+18)){
      id = ALEXNET_1;
    }
    else if(r < (0+12+18+16)){
      id = GOOGLENET_1;
    }
    else{
      id = YOLONET_1;
    }
  }
#else
  int rand_mod = 140;//58;//160;
  int rand_base = 0;

  static int id = 1;
  uint32_t rand_out = rand_seed(seed);
  int r = rand_out % rand_mod + rand_base;
  if(batch1){
    if(r < (0+12)){
      id = RESNET_1;
    }
    else if(r < (0+12+18)){
      id = ALEXNET_1;
    }
    else if(r < (0+12+18+16)){
      id = GOOGLENET_1;
    }
    else if(r < (0+12+18+16+30)){
      id = SQUEEZENET_1;
    }
    else if(r < (0+12+18+16+30+21)){
      id = KWSNET_1;
    }
    else if(r < (0+12+18+16+30+21+18)){
      id = YOLONET_1;
    }
    else{
      id = YOLOLITENET_1;
    }
  }
#endif
  return id;
}

// priority scheduling
// num_sub_group: SUB_CORE (2), or 1 if doing 4+4 partitioning
int workload_priority_mp(int num_workload, int num_iter, uint64_t current_cycle){
  for(int group; group < NUM_GROUP; group++)
    for(int c = 0; c < SUB_GROUP; c++)
      for(int i = 0; i < MAX_ITER; i++)
        for(int j = 0; j < QUEUE_DEPTH; j++){
          gemmini_workload_grouped[group][c][i][j] = -1;
          gemmini_workload_assigned[group][c][i][j] = -1;
        }

  int num_batch = 1;
//  printf("workload_priority_mp current cycles: %llu\n", current_cycle);

  // 4 entries if grouping 2 cores
  int group_temp[NUM_SUB_GROUP];
  uint64_t cycle[NUM_SUB_GROUP];
  for (int i = 0; i < NUM_SUB_GROUP; i++){
    cycle[i] = current_cycle + 5000000;
    group_temp[i] = 0;
    gemmini_dram_util[i] = 0;
  }

  // priority score initialization
  int64_t score[num_workload];
  int max_depth = QUEUE_DEPTH * NUM_SUB_GROUP;

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
    for(int i = 0; i < NUM_SUB_GROUP; i++){
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
    pointer = (pointer >= num_workload - 5) ? num_workload : pointer; 

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

   // printf("iter: %d, cycle: %llu, cycle0: %llu, cycle1: %llu, dispatch queue pointer: %d\n", iter, top_cycle, cycle[0], cycle[1], pointer);
    
    // ToDo: QoS 0 (extreme priority)

    // assign until num_iter
    // based on expected cycles after num_iter
    for(int i = 0; i < pointer; i++){
      if(score[i] >= 0){
        int type = total_queue_type[i];
        uint64_t after_dispatch = (top_cycle > total_queue_dispatch[i]) ? (top_cycle - total_queue_dispatch[i]) : 0;
        score[i] = score[i]*100000 + ((100000*after_dispatch) / (sp_prediction_cycles[type-1]));
      }
    }

    // first, pick candidate
    // next, assign using cycle predictio 
    int queue_index = 0;
    int max_index = -1;
    int64_t max_score = -1;
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
      for(int i = 0; i < NUM_SUB_GROUP; i++){
        if(group_temp[i] == QUEUE_DEPTH){
          full = true;
          break;
        }
      }

      if(!full){
/*
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
  */        
        for(int k = 0; k < NUM_SUB_GROUP; k++){
          bool smallest = true;
          for(int c = 0; c < NUM_SUB_GROUP; c++){
            if(cycle[k] > cycle[c]){
              smallest = false;
              break;
            }
          }
          if(smallest && p < pre_assign_length){
            int index = pre_assign_queue[p];
            int group_index = k / NUM_GROUP;
            int sub_group_index = k % NUM_GROUP;
            gemmini_workload_assigned[group_index][sub_group_index][iter][group_temp[k]] = index;
            int type = total_queue_type[index];
            cycle[k] += (tp_prediction_cycles[type-1] * (INTER_SCALE));
            //cycle[k] += (sp_cycles[type] * (CAP_SCALE+0.1));
            group_temp[k] += 1;
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
    for(int i = 0; i < NUM_SUB_GROUP; i++){
      group_temp[i] = 0;
    }
  }

  // if returned value is 0, then it is over
  return (iter); // number of queue group
}


void workload_mode_2(int workload, bool batch1, bool batch4, bool batch8, uint32_t seed, float target_scale, float cap_scale){
  // priority (0: 15, 1: 18 / 2: 10, 4: 15, 6: 15, 8: 15 / 9: 10, 11: 2)
  for(int i = 0; i < MAX_WORKLOAD; i++)
    total_queue_status[i]= -1;
  printf("total workload: %d, num_iter: %d, cap: %d, cap_scale: %d, inter_scale: %d\n", total_workloads, NUM_ITER, CAP, (int)(100*CAP_SCALE), (int)(100*INTER_SCALE));
  
  int first_dispatch_interval = 50000;

  int group = CAP; // set this to 4 for 2 cores
  int num_workload_group = ceil_divide_int(workload+2*group, group);

  for(int i = 0; i < num_workload_group; i++){
    for(int j = 0; j < group; j++){
      int index = group * i + j;
      int workload_type = workload_type_assign(batch1, batch4, batch8, seed);
      //int workload_type = rand_base + rand_seed(seed) % rand_mod;
      total_queue_type[index] = workload_type; 
      int priority_level = rand_seed(seed) % 108; //100;
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
      total_queue_target[index] = target_cycles[workload_type] * target_scale;
      total_queue_togo[index] = tp_prediction_cycles[workload_type-1];
      total_queue_conv[index] = 0;
      for (int j = 0; j < SUB_CORE; j++){
        total_queue_finish[j][index] = 0;
        total_queue_runtime_thread[j][index] = 0;
        total_queue_runtime_total[j][index] = 0;
      }
      if(i == 0){
        total_queue_dispatch[index] = first_dispatch_interval*j;
      }
      else{
        //total_queue_dispatch[index] = total_queue_dispatch[index-group] + tp_prediction_cycle[total_queue_type[index-group]-1]*cap_scale - 45000*(rand()%20);
        total_queue_dispatch[index] = total_queue_dispatch[index - group] + sp_cycles[total_queue_type[index - group]-1] * cap_scale -  45000*(rand()%20); 
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
 
        a = total_queue_togo[i];
        total_queue_togo[i] = total_queue_togo[j];
        total_queue_togo[j] = a;
 
        int b = total_queue_priority[i];
        total_queue_priority[i] = total_queue_priority[j];
        total_queue_priority[j] = b;
  
        b = total_queue_type[i];
        total_queue_type[i] = total_queue_type[j];
        total_queue_type[j] = b;
         
        b = total_queue_conv[i];
        total_queue_conv[i] = total_queue_conv[j];
        total_queue_conv[j] = b;
                     
      }
    }
  }

  
  for(int i = workload; i < workload+2*group; i++){
    total_queue_dispatch[i] = 0;
    total_queue_priority[i] = -1;
    total_queue_type[i] = -1;
    total_queue_status[i] = -1;
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

void workload_grouping(int num_iter){
  for(int group = 0; group < NUM_GROUP; group++){
    for(int sub = 0; sub < SUB_GROUP; sub++){
      for(int iter = 0; iter < num_iter; iter++){
        for(int i = 0; i < QUEUE_DEPTH; i++){
          int queue_id = gemmini_workload_assigned[group][sub][iter][i];
              //printf("queue_id: %d\n", queue_id);
          if(queue_id != -1){
            int workload_type = total_queue_type[queue_id];
            if(workload_type == RESNET_1){ // if it is resnet
              bool groupped = false;
              for(int i_next = i; i_next < QUEUE_DEPTH; i_next++){
                int next_queue_id = gemmini_workload_assigned[group][sub][iter][i_next];
                int next_type = total_queue_type[next_queue_id];
                if(next_type == -1) break;
                if(next_type == GOOGLENET_1 ||next_type == SQUEEZENET_1 || next_type == YOLOLITENET_1 || next_type == KWSNET_1 || next_type == ALEXNET_1){
                  if(gemmini_workload_grouped[group][sub][iter][i_next] != -7){
                    gemmini_workload_grouped[group][sub][iter][i_next] = -7; // mark 0
                   
                    gemmini_workload_grouped[group][sub][iter][i] = next_queue_id;
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
                    int next_queue_id = gemmini_workload_assigned[group][sub][iter_temp][i_next];
                    int next_type = total_queue_type[next_queue_id];
                    if(next_type == -1) break;
                    if(next_type == GOOGLENET_1 ||next_type == SQUEEZENET_1 || next_type == YOLOLITENET_1 || next_type == KWSNET_1 || next_type == ALEXNET_1){
                      if(gemmini_workload_grouped[group][sub][iter_temp][i_next] != -7){ 
                        gemmini_workload_grouped[group][sub][iter_temp][i_next] = -7;
                        gemmini_workload_grouped[group][sub][iter][i] = next_queue_id; 
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
                int next_queue_id = gemmini_workload_assigned[group][sub][iter][i_next];
                int next_type = total_queue_type[next_queue_id];
                if(next_type == -1) break;
                if(next_type == GOOGLENET_1 || next_type == YOLONET_1 || next_type == KWSNET_1 || next_type == RESNET_1){
                  if(gemmini_workload_grouped[group][sub][iter][i_next] != -7){
                    gemmini_workload_grouped[group][sub][iter][i_next] = -7; // mark 0
                    gemmini_workload_grouped[group][sub][iter][i] = next_queue_id;
                    groupped = true;
                    break;
                  }
                }
              }
              if(!groupped && iter != num_iter - 1){ 
                int iter_temp = iter + 1;
                while(iter_temp < num_iter && !groupped){
                  for(int i_next = 0; i_next < QUEUE_DEPTH; i_next++){
                    int next_queue_id = gemmini_workload_assigned[group][sub][iter_temp][i_next];
                    int next_type = total_queue_type[next_queue_id];
                    if(next_type == -1) break;
                    if(next_type == GOOGLENET_1 || next_type == YOLONET_1 || next_type == KWSNET_1 || next_type == RESNET_1 ){
                      if(gemmini_workload_grouped[group][sub][iter_temp][i_next] != -7){ 
                        gemmini_workload_grouped[group][sub][iter_temp][i_next] = -7;
                        gemmini_workload_grouped[group][sub][iter][i] = next_queue_id; 
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
                int next_queue_id = gemmini_workload_assigned[group][sub][iter][i_next];
                int next_type = total_queue_type[next_queue_id];
                if(next_type == -1) break;
                if(next_type == GOOGLENET_1 ||next_type == SQUEEZENET_1 || next_type == YOLOLITENET_1 || next_type == KWSNET_1 || next_type == ALEXNET_1){
                  if(gemmini_workload_grouped[group][sub][iter][i_next] != -7){
                    gemmini_workload_grouped[group][sub][iter][i_next] = -7;
                    gemmini_workload_grouped[group][sub][iter][i] = next_queue_id;
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
                    int next_queue_id = gemmini_workload_assigned[group][sub][iter_temp][i_next];
                    int next_type = total_queue_type[next_queue_id];
                    if(next_type == -1) break;
                    if(next_type == GOOGLENET_1 ||next_type == SQUEEZENET_1 || next_type == YOLOLITENET_1 || next_type == KWSNET_1 || next_type == ALEXNET_1){
                      if(gemmini_workload_grouped[group][sub][iter_temp][i_next] != -7){ 
                        gemmini_workload_grouped[group][sub][iter_temp][i_next] = -7;
                        gemmini_workload_grouped[group][sub][iter][i] = next_queue_id; 
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

}

// prema scheduling
// num_sub_group: SUB_CORE (2), or 1 if doing 4+4 partitioning
int workload_priority_sp(int num_workload, uint64_t current_cycle){
//  int group = 0; // just assume group 0
  int num_sub_group = 1;

  for(int c = 0; c < SUB_GROUP; c++)
    for(int i = 0; i < MAX_ITER; i++)
      for(int j = 0; j < QUEUE_DEPTH; j++){
        gemmini_workload_grouped[0][c][i][j] = -1;
        gemmini_workload_assigned[0][c][i][j] = -1;
      }

  //num_workload = num_workload * NUM_GROUP; // 2x
  int num_batch = 1;
//  printf("workload_priority_mp current cycles: %llu\n", current_cycle);

  
  int group_temp[num_sub_group];
  uint64_t cycle[num_sub_group];
  for (int i = 0; i < num_sub_group; i++){
    cycle[i] = current_cycle + 2000000 * num_batch;
    group_temp[i] = 0;
    //gemmini_dram_util[group*num_sub_group+i] = 0;
  }

  // priority score initialization
  int64_t score[num_workload*NUM_GROUP];
  int max_depth = QUEUE_DEPTH;// * 1.8;

  // repeat from here
  int pre_assign_queue[max_depth];
  int64_t pre_assign_score[max_depth]; // need this?
  int pre_assign_length = 0;

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
  bool finished = true;
  for(int i = 0; i < num_workload; i++){
    if(total_queue_dispatch[i] > top_cycle){
      pointer = i;
      break;
    }
    else if(i == num_workload - 1){
      pointer = num_workload;
    }
  }
  pointer = (pointer >= num_workload - 3) ? num_workload : pointer; 

  bool done = true;
  for (int i = 0; i < pointer; i++){
    int index = i;
    if(total_queue_status[i] == -1){ //only take the unassigned ones
      score[index] = total_queue_priority[i];
      done = false;
    }
    else
      score[index] = -1;
  }
  if(!(done && (pointer == num_workload)))
    finished = false;

  if(finished)
    return -1;
    //break;
//printf("pointer0: %d, pointer1: %d\n", pointer[0], pointer[1]);
  //printf("iter: %d, cycle: %llu, cycle0: %llu, cycle1: %llu, dispatch queue pointer: %d\n", iter, top_cycle, cycle[0], cycle[1], pointer);
     
  for(int i = 0; i < pointer; i++){
    if(score[i] >= 0){
      int type = total_queue_type[i];
      uint64_t after_dispatch = (top_cycle > total_queue_dispatch[i]) ? (top_cycle - total_queue_dispatch[i]) : 0;
      score[i] = score[i]*100000 + ((100000*after_dispatch) / (CAP*sp_prediction_cycles[type-1]));
      //score[index] = score[index]*100000 + ((100000*after_dispatch) / (qos*sp_prediction_cycles[1][type]));
    }
  }
  

  // first, pick candidate
  // next, assign using cycle predictio
  

  int queue_index = 0;
  int max_index = -1;
  int64_t max_score = -1;
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
    total_queue_status[max_index] = 0;
    max_index = -1;
    max_score = -1;
  }
//  if(queue_index == 0 && ((iter == 0 && pre_assign_length == 0) || (iter == 1 && pre_assign_length == 1))) break;
 // printf("queue_index: %d, pre_assign_length: %d, iter: %d\n", queue_index, pre_assign_length, iter);
 pre_assign_length = queue_index;
 /*
  printf("pre assigned queue length: %d \n", pre_assign_length);
  for(int i = 0; i < pre_assign_length; i++)
    printf("%d, ", pre_assign_queue[i]); 
  printf("\n");
  */

  for (int p = 0; p < pre_assign_length; p++){
//    printf("iter: %d, index: %d, cycle0: %llu, cycle1: %llu, group0: %d, group1: %d\n", iter, index, cycle[0], cycle[1], group_temp[0], group_temp[1]);
    bool full = p >= QUEUE_DEPTH; 

    if(!full){
      int index = pre_assign_queue[p];
      gemmini_workload_assigned[0][0][0][p] = index;
      int group = index / num_workload;
      int i = index % num_workload;
      int type = total_queue_type[i];
        
      
    }
    else{
      // release status
      int index = pre_assign_queue[p];
      int group = index / num_workload;
      int i = index % num_workload;
      total_queue_status[i] = -1;
    }
  }

  // if returned value is 0, then it is over
  return pre_assign_length > QUEUE_DEPTH ? QUEUE_DEPTH : pre_assign_length; // number of queue group
}


#ifndef BAREMETAL
uint64_t workload_function(int64_t inner_start, int queue_id, int workload_id, size_t cid, size_t group_id, size_t sub_group, int num_gemmini, int dram_util, pthread_barrier_t *barrier_funct){
  gemmini_flush(0);
  uint64_t* cycles;
  uint64_t total_runtime;

  size_t sub_group_id = sub_group;
  //size_t sub_group_id = group_id * NUM_GROUP + sub_group; // out of total sub-group
  int group_status = total_queue_status[queue_id];
  bool part1 = group_status < 1;
  bool part2 = group_status < 2;
  bool part3 = group_status < 3;
  bool part4 = group_status < 4;
//printf("part1: %d, part2: %d, part3: %d, part4: %d\n", part1, part2, part3, part4);
  //uint64_t start = read_cycles();
  if(cid == 0){
    gemmini_queue_id[sub_group_id] = queue_id;
    gemmini_start_time[sub_group_id] = inner_start;
  }
  //gemmini_dispatch_cycle[sub_group_id] = total_queue_dispatch[queue_id];
  //gemmini_workload_id[sub_group_id] = workload_id - 1;
#if BATCH1 == true
  if(workload_id < 8){
    int orow_divide = num_gemmini;
    int batch_divide = 1; // 1 batch workload
    if(workload_id == 0){
      //cycles = fcnnet_function_1(cid, sub_group_id, orow_divide, batch_divide, dram_util, barrier_funct);
      //total_runtime = *(cycles+73);
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
    total_queue_status[queue_id] = 100; // just store big value (finished)
  }

  //if(cid == 0) total_queue_status[queue_id] = 100; // just store big value (finished)
  //uint64_t runtime = read_cycles() - start;
  return total_runtime;

}

// sub_group_id: 0 - 3
uint64_t workload_group_function(uint64_t inner_start, int queue_id, int group_queue_id, int original_workload_id, int grouped_workload_id, size_t cid, size_t group_id, size_t sub_group_id, int num_gemmini, int dram_util, pthread_barrier_t *barrier_funct){
  gemmini_flush(0);
  uint64_t* cycles;
  uint64_t total_runtime;

  int group_status = total_queue_status[queue_id];
  bool part1 = group_status < 1;
  bool part2 = group_status < 2;
  bool part3 = group_status < 3;
  bool part4 = group_status < 4;
  if(cid == 0){
    gemmini_queue_id[sub_group_id] = queue_id;
    gemmini_start_time[sub_group_id] = inner_start;
  }
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
            total_queue_status[group_queue_id] = 50;
          }
          else if(grouped_workload_id == YOLOLITENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
  //	  dram_util = 10;
            cycles = yololitenet_block_function_1(0, sub_group_id, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_queue_id] = 50;
          }
          else if(grouped_workload_id == KWSNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = kwsnet_block_function_1(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 40);
            total_queue_status[group_queue_id] = 1;
          }
          else if(grouped_workload_id == ALEXNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = alexnet_block_function_1(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_queue_id] = 1;
          }
          else if(grouped_workload_id == GOOGLENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = googlenet_block_function_1(0, sub_group_id, true, false, 1, 1, dram_util); 
            total_runtime = *(cycles+71);
            total_queue_status[group_queue_id] = 1;
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
            total_queue_status[group_queue_id] = 50;
          }
          else if(grouped_workload_id == YOLONET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = yolonet_block_function_1(0, sub_group_id, true, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 26);
            total_queue_status[group_queue_id] = 2;
          }
          else if(grouped_workload_id == KWSNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = kwsnet_block_function_1(0, sub_group_id, true, true, 1, 1, dram_util);
            total_runtime = *(cycles + 40);
            total_queue_status[group_queue_id] = 50;
          }
          else if(grouped_workload_id == RESNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = resnet_block_function_1(0, sub_group_id, true, true, false, false, 1, 1, dram_util);
            total_runtime = *(cycles + 72);
            total_queue_status[group_queue_id] = 2;
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
            total_queue_status[group_queue_id] = 50;
          }
          else if(grouped_workload_id == YOLOLITENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
  //	  dram_util = 10;
            cycles = yololitenet_block_function_1(0, sub_group_id, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_queue_id] = 50;
          }
          else if(grouped_workload_id == KWSNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = kwsnet_block_function_1(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 40);
            total_queue_status[group_queue_id] = 1;
          }
          else if(grouped_workload_id == ALEXNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = alexnet_block_function_1(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_queue_id] = 1;
          }
          else if(grouped_workload_id == GOOGLENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = googlenet_block_function_1(0, sub_group_id, true, false, 1, 1, dram_util); 
            total_runtime = *(cycles+71);
            total_queue_status[group_queue_id] = 1;
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
            total_queue_status[group_queue_id] = 50;
          }
          else if(grouped_workload_id == YOLOLITENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
  //	  dram_util = 10;
            cycles = yololitenet_block_function_11(0, sub_group_id, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_queue_id] = 50;
          }
          else if(grouped_workload_id == KWSNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = kwsnet_block_function_11(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 40);
            total_queue_status[group_queue_id] = 1;
          }
          else if(grouped_workload_id == ALEXNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = alexnet_block_function_11(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_queue_id] = 1;
          }
          else if(grouped_workload_id == GOOGLENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = googlenet_block_function_11(0, sub_group_id, true, false, 1, 1, dram_util); 
            total_runtime = *(cycles+71);
            total_queue_status[group_queue_id] = 1;
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
            total_queue_status[group_queue_id] = 50;
          }
          else if(grouped_workload_id == YOLONET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = yolonet_block_function_11(0, sub_group_id, true, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 26);
            total_queue_status[group_queue_id] = 2;
          }
          else if(grouped_workload_id == KWSNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = kwsnet_block_function_11(0, sub_group_id, true, true, 1, 1, dram_util);
            total_runtime = *(cycles + 40);
            total_queue_status[group_queue_id] = 50;
          }
          else if(grouped_workload_id == RESNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = resnet_block_function_11(0, sub_group_id, true, true, false, false, 1, 1, dram_util);
            total_runtime = *(cycles + 72);
            total_queue_status[group_queue_id] = 2;
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
            total_queue_status[group_queue_id] = 50;
          }
          else if(grouped_workload_id == YOLOLITENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
  //	  dram_util = 10;
            cycles = yololitenet_block_function_11(0, sub_group_id, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_queue_id] = 50;
          }
          else if(grouped_workload_id == KWSNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = kwsnet_block_function_11(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 40);
            total_queue_status[group_queue_id] = 1;
          }
          else if(grouped_workload_id == ALEXNET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = alexnet_block_function_11(0, sub_group_id, true, false, 1, 1, dram_util);
            total_runtime = *(cycles + 14);
            total_queue_status[group_queue_id] = 1;
          }
          else if(grouped_workload_id == GOOGLENET_1){
           dram_util = -1;//(dram_util == 0) ? 30 : dram_util;
            cycles = googlenet_block_function_11(0, sub_group_id, true, false, 1, 1, dram_util); 
            total_runtime = *(cycles+71);
            total_queue_status[group_queue_id] = 1;
          }
        }
      }
    }


  }
#endif

  if(cid == 0) {
    gemmini_dram_util[sub_group_id] = 0;
    total_queue_status[queue_id] = 80; // just store big value (finished)
  }
  //uint64_t runtime = read_cycles() - start;
  return total_runtime;

}


#ifdef planaria_scale
// for planaria
// barrier_funct2: for subgroup (between 2), barrier_funct4: for group (between 4)
uint64_t workload_planaria_function(int queue_id, int workload_id, size_t cid, size_t group_id, size_t sub_group_id, int num_gemmini, uint64_t slack_time,  pthread_barrier_t *barrier_funct){
  gemmini_flush(0);
  uint64_t* cycles;
  uint64_t total_runtime = 0;
  int other_sub_group_id = group_id * WORKLOAD_CORE + (sub_group_id % 2 == 0 ? 1 : 0);   
  if(cid == 0){
//printf("num_gemmini: %d, sub group id: %d, other sub group id: %d  workload id %d queue id %d slack: %llu\n", num_gemmini, sub_group_id, other_sub_group_id, workload_id, queue_id, slack_time);
    gemmini_terminate_receive[sub_group_id] = false;
    gemmini_terminate[other_sub_group_id] = false;
  }
  pthread_barrier_wait(barrier_funct);
  //size_t sub_group_id = group_id * NUM_GROUP + sub_group; // out of total sub-group
  int group_status = total_queue_status[queue_id]; // need to update when terminated, check this if it finshed outer loop

  bool part[30];
  for(int i = 0; i < 30; i++){
    part[i] = group_status < (i+1);
  }
#if BATCH1 == true
  if(workload_id < 8){
    int orow_divide = num_gemmini;
    int batch_divide = 1; // 1 batch workload
    if(workload_id == 1){
      uint64_t slack = 20000000;
//      int other_sub_group_id = group_id * WORKLOAD_CORE + (sub_group_id % 2 == 0 ? 1 : 0);      
      for(int i = 0; i < 10; i++){
        uint64_t start = read_cycles();
        if(num_gemmini <= 2){
          if(gemmini_planaria_score[other_sub_group_id] <= gemmini_planaria_score[sub_group_id]){    
            //if(cid == 0 && slack_time <= (slack - 2000000*i) * planaria_scale){
            if(cid == 0){
              gemmini_terminate[other_sub_group_id] = true;
//printf("group id %d need to terminate others\n", sub_group_id);
            }
          }
      /*    else if(gemmini_planaria_score[other_sub_group_id] <= gemmini_planaria_score[sub_group_id] * 2){    
            if(cid == 0 && slack_time*2 <= (slack - 2000000*i) * planaria_scale){
              gemmini_terminate[other_sub_group_id] = true;
//printf("group id %d need to terminate others\n", sub_group_id);
            }
          }*/
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            if(cid == 0) gemmini_terminate_receive[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
#if debug_print == 1
printf("group id %d workload %d part %d - my score: %d, others score: %d, slack: %llu, slack time: %llu, other terminate me: %d, me terminate other: %d, recieved: %d\n", sub_group_id, workload_id, i, gemmini_planaria_score[sub_group_id], gemmini_planaria_score[other_sub_group_id], slack, slack_time, gemmini_terminate[sub_group_id], gemmini_terminate[other_sub_group_id], gemmini_terminate_receive[sub_group_id]);
#endif 
          if(gemmini_terminate_receive[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = resnet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = resnet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime += *(cycles+72);
          if(cid == 0) total_queue_status[queue_id] = (i+1); 
        }
        uint64_t end = read_cycles();
        slack_time = (slack_time < (end - start)) ? 0 : slack_time - (end - start);
      }

    }
    else if(workload_id == 2){
      uint64_t slack[5] = {12522781, 10500000, 6429112, 2297240, 459726};
      if(num_gemmini == 4){ 
//printf("executing alexnet with all 4cores\n");
      	if(sub_group_id % 2 == 0) cycles = alexnet_function_1(cid, sub_group_id, true, true, orow_divide, batch_divide, -1, barrier_funct);
      	else cycles = alexnet_function_11(cid, sub_group_id, true, true, orow_divide, batch_divide, -1, barrier_funct);
      	total_runtime = *(cycles+14);
        if(cid == 0) total_queue_status[queue_id] = 100;
      }
      else{
      
        for(int i = 0; i < 5; i++){
          uint64_t start = read_cycles(); 
          if(cid == 0) gemmini_terminate[other_sub_group_id] = false;
        /*  else if(gemmini_planaria_score[other_sub_group_id] <= gemmini_planaria_score[sub_group_id] * 2){    
            if(cid == 0 && slack_time*2 <= (slack[i]) * planaria_scale){
              gemmini_terminate[other_sub_group_id] = true;
//printf("group id %d need to terminate others\n", sub_group_id);
            }
          }*/
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            if(cid == 0) gemmini_terminate_receive[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
#if debug_print == 1
printf("group id %d workload %d part %d - my score: %d, others score: %d, slack: %llu, slack time: %llu, other terminate me: %d, me terminate other: %d, recieved: %d\n", sub_group_id, workload_id, i, gemmini_planaria_score[sub_group_id], gemmini_planaria_score[other_sub_group_id], slack, slack_time, gemmini_terminate[sub_group_id], gemmini_terminate[other_sub_group_id], gemmini_terminate_receive[sub_group_id]);
#endif 
          if(gemmini_terminate_receive[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        
          if(part[i]){
            if(sub_group_id % 2 == 0) cycles = alexnet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
            else cycles = alexnet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
            total_runtime += *(cycles+14);
            if(cid == 0) total_queue_status[queue_id] = (i+1); 
          }
          uint64_t end = read_cycles();
          slack_time = (slack_time < (end - start)) ? 0 : slack_time - (end - start);
        }
      }
    }
    else if(workload_id == 3){
      uint64_t slack = 10000000;
  //    int other_sub_group_id = group_id * WORKLOAD_CORE + (sub_group_id % 2 == 0 ? 1 : 0);      
      for(int i = 0; i < 5; i++){
        uint64_t start = read_cycles();
          if(cid == 0) gemmini_terminate[other_sub_group_id] = false;
        if(num_gemmini <= 2){
/*
          if(gemmini_planaria_score[other_sub_group_id] <= gemmini_planaria_score[sub_group_id]){    
            if(cid == 0 && slack_time <= (slack - 2000000*i) * planaria_scale){
              gemmini_terminate[other_sub_group_id] = true;
            }
          }
*/
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            if(cid == 0) gemmini_terminate_receive[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
#if debug_print == 1
printf("group id %d workload %d part %d - my score: %d, others score: %d, slack: %llu, slack time: %llu, other terminate me: %d, me terminate other: %d, recieved: %d\n", sub_group_id, workload_id, i, gemmini_planaria_score[sub_group_id], gemmini_planaria_score[other_sub_group_id], slack, slack_time, gemmini_terminate[sub_group_id], gemmini_terminate[other_sub_group_id], gemmini_terminate_receive[sub_group_id]);
#endif 
          if(gemmini_terminate_receive[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = googlenet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = googlenet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime += *(cycles+71);
          if(cid == 0) total_queue_status[queue_id] = (i+1); 
        }
        uint64_t end = read_cycles();
        slack_time = (slack_time < (end - start)) ? 0 : slack_time - (end - start);
      }

    }
    else if(workload_id == 4){
      uint64_t slack = 2000000;
      for(int i = 0; i < 2; i++){
        uint64_t start = read_cycles();
        if(num_gemmini <= 2){
          if(cid == 0) gemmini_terminate[other_sub_group_id] = false;
         /* else if(gemmini_planaria_score[other_sub_group_id] <= gemmini_planaria_score[sub_group_id] * 2){    
            if(cid == 0 && slack_time*2 <= (slack - 1000000*i) * planaria_scale){
              gemmini_terminate[other_sub_group_id] = true;
//printf("group id %d need to terminate others\n", sub_group_id);
            }
          }*/
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            if(cid == 0) gemmini_terminate_receive[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
#if debug_print == 1
printf("group id %d workload %d part %d - my score: %d, others score: %d, slack: %llu, slack time: %llu, other terminate me: %d, me terminate other: %d, recieved: %d\n", sub_group_id, workload_id, i, gemmini_planaria_score[sub_group_id], gemmini_planaria_score[other_sub_group_id], slack, slack_time, gemmini_terminate[sub_group_id], gemmini_terminate[other_sub_group_id], gemmini_terminate_receive[sub_group_id]);
#endif
         if(gemmini_terminate_receive[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = squeezenet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = squeezenet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime += *(cycles+29);
          if(cid == 0) total_queue_status[queue_id] = i+1;
        }
        uint64_t end = read_cycles();
        slack_time = (slack_time < (end - start)) ? 0 : slack_time - (end - start);
      }
    }
    else if(workload_id == 5){
      uint64_t slack = 4000000;
      for(int i = 0; i < 5; i++){
        uint64_t start = read_cycles();
        if(num_gemmini <= 2){
          if(gemmini_planaria_score[other_sub_group_id] <= gemmini_planaria_score[sub_group_id]){
            //if(cid == 0 && slack_time <= ((slack - 800000*i) * planaria_scale)){
            if(cid == 0){
                gemmini_terminate[other_sub_group_id] = true;
            }
          }
         /* else if(gemmini_planaria_score[other_sub_group_id] <= gemmini_planaria_score[sub_group_id] * 2){    
            if(cid == 0 && slack_time*2 <= (slack - 800000*i) * planaria_scale){
              gemmini_terminate[other_sub_group_id] = true;
//printf("group id %d need to terminate others\n", sub_group_id);
            }
          }*/
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            if(cid == 0) gemmini_terminate_receive[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
#if debug_print == 1
printf("group id %d workload %d part %d - my score: %d, others score: %d, slack: %llu, slack time: %llu, other terminate me: %d, me terminate other: %d, recieved: %d\n", sub_group_id, workload_id, i, gemmini_planaria_score[sub_group_id], gemmini_planaria_score[other_sub_group_id], slack, slack_time, gemmini_terminate[sub_group_id], gemmini_terminate[other_sub_group_id], gemmini_terminate_receive[sub_group_id]);
#endif 
          if(gemmini_terminate_receive[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = kwsnet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = kwsnet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime += *(cycles+40);
          if(cid == 0) total_queue_status[queue_id] = i+1;
        }
        uint64_t end = read_cycles();
        slack_time = (slack_time < (end - start)) ? 0 : slack_time - (end - start);
      }
        
    }
    else if(workload_id == 6){
      uint64_t slack = 10000000;
      for(int i = 0; i < 5; i++){
        uint64_t start = read_cycles();
        if(num_gemmini <= 2){
          if(gemmini_planaria_score[other_sub_group_id] <= gemmini_planaria_score[sub_group_id]){
            if(cid == 0){
            //if(cid == 0 && slack_time <= ((slack - 2000000*i) * planaria_scale)){
              gemmini_terminate[other_sub_group_id] = true;
//printf("group id %d need to terminate others\n", sub_group_id);
            }
          }
         /* else if(gemmini_planaria_score[other_sub_group_id] <= gemmini_planaria_score[sub_group_id] * 2){    
            if(cid == 0 && slack_time*2 <= (slack - 2000000*i) * planaria_scale){
              gemmini_terminate[other_sub_group_id] = true;
//printf("group id %d need to terminate others\n", sub_group_id);
            }
          }*/
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            if(cid == 0) gemmini_terminate_receive[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
#if debug_print == 1
printf("group id %d workload %d part %d - my score: %d, others score: %d, slack: %llu, slack time: %llu, other terminate me: %d, me terminate other: %d, recieved: %d\n", sub_group_id, workload_id, i, gemmini_planaria_score[sub_group_id], gemmini_planaria_score[other_sub_group_id], slack, slack_time, gemmini_terminate[sub_group_id], gemmini_terminate[other_sub_group_id], gemmini_terminate_receive[sub_group_id]);
#endif 
          if(gemmini_terminate_receive[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = yolonet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = yolonet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime += *(cycles+26);
          if(cid == 0) total_queue_status[queue_id] = i+1;
        }
        uint64_t end = read_cycles();
        slack_time = (slack_time < (end - start)) ? 0 : slack_time - (end - start);
      }
  
    }
    else if(workload_id == 7){
      uint64_t slack = 1800000;
      for(int i = 0; i < 2; i++){
        uint64_t start = read_cycles();
        if(num_gemmini <= 2){
          if(cid == 0) gemmini_terminate[other_sub_group_id] = false;
         /* else if(gemmini_planaria_score[other_sub_group_id] <= gemmini_planaria_score[sub_group_id] * 2){    
            if(cid == 0 && slack_time*2 <= (slack - 900000*i) * planaria_scale){
              gemmini_terminate[other_sub_group_id] = true;
//printf("group id %d need to terminate others\n", sub_group_id);
            }
          }*/
          if(gemmini_terminate[other_sub_group_id] || gemmini_terminate[sub_group_id])
            if(cid == 0) gemmini_terminate_receive[sub_group_id] = true;
          pthread_barrier_wait(barrier_funct);
#if debug_print == 1
printf("group id %d workload %d part %d - my score: %d, others score: %d, slack: %llu, slack time: %llu, other terminate me: %d, me terminate other: %d, recieved: %d\n", sub_group_id, workload_id, i, gemmini_planaria_score[sub_group_id], gemmini_planaria_score[other_sub_group_id], slack, slack_time, gemmini_terminate[sub_group_id], gemmini_terminate[other_sub_group_id], gemmini_terminate_receive[sub_group_id]);
#endif 
          if(gemmini_terminate_receive[sub_group_id])
            break;
          pthread_barrier_wait(barrier_funct);
        }
        if(part[i]){
          if(sub_group_id % 2 == 0) cycles = yololitenet_planaria_function_1(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          else cycles = yololitenet_planaria_function_11(cid, sub_group_id, i, orow_divide, batch_divide, -1, barrier_funct);
          total_runtime += *(cycles+14);
          if(cid == 0) total_queue_status[queue_id] = i+1;
        }
        uint64_t end = read_cycles();
        slack_time = (slack_time < (end - start)) ? 0 : slack_time - (end - start);
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

#endif
