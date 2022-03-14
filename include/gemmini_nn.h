#ifndef GEMMINI_NN_H
#define GEMMINI_NN_H

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_testutils.h"

struct ConvParams {
    int batch_size;
    int in_dim, out_dim;
    int kernel_size;
    int in_channels;
    int out_channels;
    int stride;
    int padding;
    bool bias;
    bool depthwise;
    int n_patches;
    int patch_size;
    acc_scale_t output_scale;
    scale_t res_scale;
    int pool_size, pool_stride, pool_padding, out_dim_pooled;
   
    int in_stride, out_stride, weight_stride;
    int dilation;
    int I, J, K;
};

struct FcParams {
    int batch_size;
    int in_features;
    int out_features;
    acc_scale_t output_scale;
    bool bias;
    int out_stride;

    int I, J, K;
};


//enum layer_type_t {CONV, MATMUL, FC, RESADD, POOL}
//size_t priority_score[NUM_CORE] = {0};
//size_t adjust[NUM_CORE] = {0};


#define HIST_IMAGES(IMAGES) \
    for (int num = -128; num <= 127; num++) { \
        int count = 0; \
        for (int i = 0; i < sizeof(IMAGES)/sizeof(IMAGES[0]); i++) { \
            for (int j = 0; j < sizeof(IMAGES[0])/sizeof(IMAGES[0][0]); j++) { \
                for (int k = 0; k < sizeof(IMAGES[0][0])/sizeof(IMAGES[0][0][0]); k++) { \
                    for (int l = 0; l < sizeof(IMAGES[0][0][0])/sizeof(IMAGES[0][0][0][0]); l++) { \
                        if (IMAGES[i][j][k][l] == num) { \
                            count++; \
                        } \
                    } \
                } \
            } \
        } \
        if (count > 0) \
            printf("%d: %d times\n", num, count); \
    }

#define HIST_MATRIX(MATRIX) \
    for (int num = -128; num <= 127; num++) { \
        int count = 0; \
        for (int i = 0; i < sizeof(MATRIX)/sizeof(MATRIX[0]); i++) { \
            for (int j = 0; j < sizeof(MATRIX[0])/sizeof(MATRIX[0][0]); j++) { \
                if (MATRIX[i][j] == num) { \
                    count++; \
                } \
            } \
        } \
        if (count > 0) \
            printf("%d: %d times\n", num, count); \
    }

// This function runs a tiled matrix multiplication, with explicit tiling
// factors
static void tiled_matmul_nn(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const void * D, elem_t C[dim_I][dim_J],
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
    if (check)
        printf("%s: gemmini\n", layer_name);

    tiled_matmul(dim_I, dim_J, dim_K,
        (elem_t*)A, (elem_t*)B, D, (elem_t*)C, 
        dim_K, dim_J, dim_J, dim_J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        act, scale, relu6_shift, repeating_bias,
        tile_I, tile_J, tile_K,
        false, false,
        false, false,
        3,
        tiled_matmul_type, 0, 0);

    if (check) {
        printf("%s: CPU\n", layer_name);
        elem_t gold[dim_I][dim_J];
        tiled_matmul_auto(dim_I, dim_J, dim_K,
            (elem_t*)A, (elem_t*)B, D, (elem_t*)gold, 
            dim_K, dim_J, dim_J, dim_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            act, scale, relu6_shift, repeating_bias,
            false, false,
            false, false,
            3,
            CPU);

        if (!MAT_IS_EQUAL(dim_I, dim_J, C, gold)) {
            printf("Layer calculated incorrectly: %s\n", layer_name);
            exit(1);
        }
    }
}

// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
static void tiled_matmul_nn_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const void * D, elem_t C[dim_I][dim_J],
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
    if (check)
        printf("%s: gemmini\n", layer_name);

    tiled_matmul_auto(dim_I, dim_J, dim_K,
        (elem_t*)A, (elem_t*)B, D, (elem_t*)C, 
        dim_K, dim_J, dim_J, dim_J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        act, scale, relu6_shift, repeating_bias,
        false, false,
        false, false,
        3,
        tiled_matmul_type);

    if (check) {
        printf("%s: CPU\n", layer_name);
        elem_t gold[dim_I][dim_J];
        tiled_matmul_auto(dim_I, dim_J, dim_K,
            (elem_t*)A, (elem_t*)B, D, (elem_t*)gold, 
            dim_K, dim_J, dim_J, dim_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            act, scale, relu6_shift, repeating_bias,
            false, false,
            false, false,
            3,
            CPU);

        if (!MAT_IS_EQUAL(dim_I, dim_J, C, gold)) {
            printf("Layer calculated incorrectly: %s\n", layer_name);
            exit(1);
        }
    }
}

static void tiled_matmul_nn_auto_stride(size_t dim_I, size_t dim_J, size_t dim_K,
  size_t stride_A, size_t stride_B, size_t stride_C,
  elem_t* A, elem_t* B,
  const void * D, elem_t* C,
  int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
  enum tiled_matmul_type_t tiled_matmul_type,
  size_t orow_divide, size_t batch_divide, size_t cid,
  int target_util)
{
  size_t* args_out;
  size_t args[10];
  args_out = tiling_factor_matmul_calculate_auto(dim_I, dim_J, dim_K, orow_divide, batch_divide, cid, args, target_util);
  dim_I = args_out[3];
  dim_J = args_out[4];
  dim_K = args_out[5];
  size_t tile_I = args_out[8];
  size_t tile_J = args_out[9];
  size_t tile_K = args_out[10];

  size_t orow_offset_floor = args_out[6];
  bool row_divisible = (args_out[7] != 0);
  int window = args_out[0];
  int target_load = args_out[1];

  orow_divide = batch_divide * orow_divide;
  batch_divide = 1;
  //size_t total_divide = orow_divide * batch_divide; // number of cores for this workload

  if(!row_divisible) orow_divide = 1;
  int out_offset = (row_divisible) ? 0 : dim_J * cid; // no need to apply offset if we divided row
  int A_orow_offset = (row_divisible && cid != 0) ? stride_A * cid * dim_I + stride_A * orow_offset_floor : 0; // if row is divided, need offset it I dimension
  int C_orow_offset = (row_divisible && cid != 0) ? stride_C * cid * dim_I + stride_C * orow_offset_floor : 0; // if row is divided, need offset it I dimension
//  printf("dim_I: %d, orow_offset_floor: %d, A_row_offset: %d \n", dim_I, orow_offset_floor, A_orow_offset);
  int A_batch_offset = 0;
  int C_batch_offset = 0;
  if (batch_divide > 1){
     A_batch_offset = stride_A * cid * dim_I;
     C_batch_offset = stride_C * cid * dim_I;
  }

  bool no_bias = (D==NULL);
  
  tiled_matmul(dim_I, dim_J, dim_K,
    A + A_orow_offset + A_batch_offset, B + out_offset, no_bias ? NULL : D + out_offset*sizeof(acc_t), C + C_orow_offset + out_offset + C_batch_offset,
    stride_A, stride_B, stride_B, stride_C,
    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
    act, scale, relu6_shift, repeating_bias,
    tile_I, tile_J, tile_K,
    false, false, false, false, 3,
    tiled_matmul_type, 
    window, target_load);

}


static void tiled_matmul_nn_auto_cid(size_t dim_I, size_t dim_J, size_t dim_K,
  size_t stride_C,
  elem_t* A, elem_t* B,
  const void * D, elem_t* C,
  int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
  enum tiled_matmul_type_t tiled_matmul_type,
  size_t orow_divide, size_t batch_divide, size_t cid,
  int target_util){

  size_t stride_A = (dim_K % 128 == 0) ? dim_K + 64 : dim_K;
  size_t stride_B = (dim_J % 128 == 0) ? dim_J + 64 : dim_J;

//  printf("A dram addr: 0x%08lx\n", A);
  tiled_matmul_nn_auto_stride(
      dim_I, dim_J, dim_K,
      stride_A, stride_B, stride_C,
      A, B, D, C,
      act, scale, relu6_shift, repeating_bias,
      WS,
      orow_divide, batch_divide, cid,
      target_util);

}
/*

// two level scheduler:
// 1: periodically get the selected group of workload of top priority updated score
// 2: arrange it into execution order
// store workload_id to global gemmini queue (gemmini_queue) 
int* calm_scheduler_level2(
    int num_workload, // number of workload in the queue
    int candidate_workload[num_workload], // selected workload with top priority
    // uint64_t target[num_workload], // left target cycles for candidate workloads
    int64_t old[num_workload], // how old? (can be negative for SLA test)
    uint64_t expect_time[num_workload], // expect runtime (per unit core - 2 cores)
    float priority_update[num_workload] // initial priority (same for SLA test)
    ){

  for(int i = 0; i < num_workload; i++){
    priority_update[i] += (float)(old[i] / expect_time[i]);
  }

  // priority order
  for(int i = 0; i < num_workload; i++){
    for(int j = i+1; j < num_workload; j++){
      if(priority_update[i] < priority_update[j]){
        float a = priority_update[i];
        priority_update[i] = priority_update[j];
        priority_update[j] = a;

        int b = candidate_workload[i];
        candidate_workload[i] = candidate_workload[j];
        candidate_workload[j] = b;
      }
    }
  }


}




// need pre-compiled data for the pre-emption overhead
// return if there is gemmini core to be killed / preempt (immediate? or for synchronization?)
// invoke when new workload is dispatched
int* calm_dispatch(
    size_t arr[], // copy of current gemmini state (which workload running)
    size_t workload_total_id, // total_id for recording performance
    size_t workload_type, // pre-compiled workload type
    size_t workload_batch,
    size_t user_given_priority,
    size_t user_given_sensitivity
    // uint64_t target_runtime, // target runtime for the new workload
    ){

  uint64_t current_time = read_cycles();

  int new_workload_id = -1;

  // put to the workload queue
  if(workload_type >= 0){
    for(int i = 0; i < MAX_WORKLOAD; i++){
      if(workload_queue[i][ENTRY_TYPE] == -1){
        workload_queue[i][ENTRY_TYPE] = workload_type;
        workload_queue[i][ENTRY_TOTAL_ID] = workload_total_id;
        workload_queue[i][ENTRY_PRIORITY] = user_given_priority;
        workload_queue[i][ENTRY_TARGET] = user_given_sensitivity * prerun_time[workload_type] * Tq;
        workload_queue[i][ENTRY_ARRIVAL_TIME] = current_time;
        workload_queue[i][ENTRY_BATCH_SIZE] = workload_batch;
        workload_queue[i][ENTRY_EXPECT_CYCLE] = compiled_time[workload_type]; // expected isolated runtime, 1 batch cycle
        workload_queue[i][ENTRY_NUM_BLOCK] = compiled_layer_block[workload_type]; // precompiled number of layer block
        new_workload_id = i;

        workload_queue[i][ENTRY_STATE] = 0;
        workload_queue[i][ENTRY_START_TIME] = 0;
        workload_queue[i][ENTRY_BATCH_DONE] = 0; // to track offline batch-processing workload
        workload_queue[i][ENTRY_BATCH_CURRENT] = 0; // currently running number of batches
        workload_queue[i][ENTRY_LAYER] = 0; // which layer has finished executed (to resume once pre-empted) get from gemmini_state tracking
        break;
        //gemmini state tracking array should have real-time layer progress, expected leftover cycles
      }
    }
    if(user_given_priority > PRIORI2 && user_given_sensitivity == QoS3){
      if(Tq == 1){
        // kill others, start

        arr = {1}; // all immediate kill, flush out all the queue (need to launch new thread and schedule run again - add expected cycles)
        return arr;
      }
      else if(Tq == 2){
        // kill two others (latency insesntive ones)

        arr = 
        return arr;
      }
    }
  }

}

// need pre-compiled data when 1/2/4 cores run for a workload
// order by priority -> select few workloads -> re


int* calm_scheduler(
    size_t arr[], // empty entries: -1 
    ){

  uint64_t current_time = read_cycles();

#define SAVE_ENTRIES 5 // number of entries
#define SAVE_ID 0 // to index into workload_queue
#define SAVE_PRIORITY 1 // for initial priority
#define SAVE_TARGET 2 // time left for target
#define SAVE_STATE 3 // 0: not started, > 0: leftover cycle
#define SAVE_NUM_CORE 5 // number of cores to allocate

  int64_t queue_save[4][num_workload][SAVE_ENTRIES] = {-1};
  float priority_update[4][num_workload] = {-1};

  int w[4] = {0};
  uint64_t current_expected_cycle = workload_queue[taken_workload_id][ENTRY_EXPECT_CYCLE] * workload_queue[i][ENTRY_BATCH_CURRENT] / workload_queue[i][ENTRY_NUM_CORE];

  for(int i = 0; i < NUM_CORES; i++){
    // type 0: not started, 1: in the gemmini gueue, 2: running, 3: stopped
    if(workload_queue[i][ENTRY_STATE] == 0 || workload_queue[i][ENTRY_STATE] == 3){
      // add expected cycles -> may need to preempt currently running cores later on
      uint64_t old = workload_queue[i][ENTRY_ARRIVAL_TIME] - current + current_expected_cycle; // add expected cycles of the currently started ones
      uint64_t left_cycles = workload_queue[i][ENTRY_TARGET] - old;
      if(workload_queue[i][ENTRY_SENSITIVITY] == QoS0){ // offline
        queue_save[0][w[0]][SAVE_ID] = i;
        queue_save[0][w[0]][SAVE_TARGET] = 0; // infinite target
        queue_save[0][w[0]][SAVE_EXPECT_TIME] = workload_queue[i][ENTRY_EXPECT_TIME] * QoS1; // 1 core runtime
        queue_save[0][w[0]][SAVE_NUM_CORE] = 1;
        priority_update[0][w[0]] = workload_queue[i][ENTRY_PRIORITY] + (old / queue_save[0][w[0]][SAVE_EXPECT_TIME]); // ToDo: check scaling factor
        w[0] ++;
      }
      else if(workload_queue[i][ENTRY_SENSITIVITY] == QoS1){ // can run with 1 core
        queue_save[1][w[1]][SAVE_ID] = i;
        queue_save[1][w[1]][SAVE_TARGET] = left_cycles;
        queue_save[1][w[1]][SAVE_EXPECT_TIME] = workload_queue[i][ENTRY_EXPECT_TIME] * QoS1; // 1 core runtime
        if(left_cycles < queue_save[1][w[1]][SAVE_EXPECT_TIME] * 0.8) queue_save[1][w[1]][SAVE_NUM_CORE] = 2;
        else queue_save[1][w[1]][SAVE_NUM_CORE] = 1;
        priority_update[1][w[1]] = workload_queue[i][ENTRY_PRIORITY] + (old / queue_save[1][w[1]][SAVE_EXPECT_TIME]);
        w[1] ++;
      }
      else if(workload_queue[i][ENTRY_SENSITIVITY] == QoS2){ // run with 2 core
        queue_save[2][w[2]][SAVE_ID] = i;
        queue_save[2][w[2]][SAVE_TARGET] = left_cycles;
        queue_save[2][w[2]][SAVE_EXPECT_TIME] = workload_queue[i][ENTRY_EXPECT_TIME] * QoS2; // 2 core runtime
        if(left_cycles < queue_save[2][w[2]][SAVE_EXPECT_TIME] * 0.8) queue_save[2][w[2]][SAVE_NUM_CORE] = 4;
        else if(left_cycles > queue_save[2][w[2]][SAVE_EXPECT_TIME] * 2) queue_save[2][w[2]][SAVE_NUM_CORE] = 1; 
        else queue_save[2][w[2]][SAVE_NUM_CORE] = 2;
        priority_update[2][w[2]] = workload_queue[i][ENTRY_PRIORITY] + (old / queue_save[2][w[2]][SAVE_EXPECT_TIME]);
        w[2] ++;
      }
      else if(workload_queue[i][ENTRY_SENSITIVITY] == QoS3){ // when tq > 2
        queue_save[3][w[3]][SAVE_ID] = i;
        queue_save[3][w[3]][SAVE_TARGET] = left_cycles;
        if(left_cycles > queue_save[3][w[3]][SAVE_EXPECT_TIME] * 3) queue_save[3][w[3]][SAVE_NUM_CORE] = 2; 
        else queue_save[3][w[3]][SAVE_NUM_CORE] = 4;
        queue_save[3][w[3]][SAVE_EXPECT_TIME] = workload_queue[i][ENTRY_EXPECT_TIME] * QoS3; // 4 core runtime
        priority_update[3][w[3]] = workload_queue[i][ENTRY_PRIORITY] + (old / queue_save[3][w[3]][SAVE_EXPECT_TIME]);
        w[3] ++;
      }
    }
  }

  // priority order
  for(int wi = 0; wi < 4; wi++){
    int num_workload = w[wi];
    for(int i = 0; i < num_workload; i++){
      for(int j = i+1; j < num_workload; j++){
        if(priority_update[wi][i] < priority_update[wi][j]){
          float a = priority_update[wi][i];
          priority_update[wi][i] = priority_update[wi][j];
          priority_update[wi][j] = a;

          // ToDo: see if pointer swapping works
          int64_t* b = queue_save[wi][i];
          queue_save[wi][i] = queue_save[wi][j];
          queue_save[wi][j] = b;

        }
      }
    }
  }
  
  int num_preempt_need = 0;
  int current_qos = workload_queue[taken_workload_id][ENTRY_SENSITIVITY]; // taken workload's sensitivity (try to match the sensitivity for the next workload
  for(int wi = 3; wi >= 0; wi --){ // in reverse order
    if(w[wi] > 0){ 
      if(current_qos == wi){ // select this
        int num_need_core = queue_save[wi][0][SAVE_NUM_CORE]; 
        num_preempt_need = num_need_core > num_free_core ? num_need_core - num_free_core : 0;
        for (int i = 0; i < NUM_CORES; i++){
          if(num_need_core > 0 && arr[i] == -1){
            arr[i] = queue_save[wi][0][SAVE_ID]; // workload_queue index
            num_need_core --;
            num_free_core --;
          }
        }
        
        if(num_free_core > 0 && w[wi] > 1){ // if there is left cores
          int num_extra_core = queue_save[wi][1][SAVE_NUM_CORE];
          // num_preempt_need = num_extra_core > num_free_core ? num_free_core - num_extra_core : 0; // for now, disable this
          for (int i = 0; i < NUM_CORES; i++){
            if(num_extra_core > 0 && arr[i] == -1){
              arr[i] = queue_save[wi][1][SAVE_ID];
              num_extra_core --;
              num_free_core --;
            }
          }
        }
        break;
      }
      else{
        int num_need_core = queue_save[wi][0][SAVE_NUM_CORE];
        
        
        for(int i = 0; i < NUM_CORES; i++){
          if(num_need_core > 0 && arr[i] == -1){
            arr[i] = queue_save[wi][0][SAVE_ID];
            num_need_core --;
            num_free_core --;
          }
        }
      }
    }
  }

   if(need_terminate){
     immediate_terminate = (queue_save[0][SAVE_PRIORITY] == MAX_PRIORITY); // when max -> immediate terminate
     int expect_new_runtime = queue_save[0][SAVE_EXPECT_TIME] / NUM_UNIT_CORE; // assume perfect scaling
     for(int i = 0; i < num_workload; i++){
       priority_update[i] += expect_new_runtime / workload_queue[(queue_save[w][SAVE_ID])][ENTRY_TARGET] * PRIORITY_SCALE;
     }
    // priority order
     for (int i = 1; i < num_workload; ++i){
        for (int j = i + 1; j < num_workload; ++j){
           if (priority_update[i] < priority_update[j]){
              float a = priority_update[i];
              priority_update[i] = priority_update[j];
              priority_update[j] = a;

              // Todo: see if it works
              int64_t* b = queue_save[i]; 
              queue_save[i] = queue_save[j];
              queue_save[j] = b;
    
              //int64_t c = target[i];
              //target[i] = target[j];
              //target[j] = c;
           }
        }
     }

     // put on the queue (and set the number of cores)

     
   }
   else{
     // put on the queue (and set the number of cores)
   }

   for(int i = 0; i < num_workload; i++){
     gemmini_queue[i][0] = queue_save[i][SAVE_ID]; // save workload ID
     queue_save[i][SAVE_TARGET] 

   }

}

// enum layer_type_t {CONV, MATMUL, FC, RESADD, POOL}
// update cycles
// update remainig macs, next target util
// ideal cycles based on load (FC) or compute macs (CONV)
int64_t* next_target_util(
    int64_t args_out[],
    //int64_t remaining_cycles, int64_t remaining_mem_cycles, int64_t remaining_pool_cycles, int64_t prev_cycles,
    //uint64_t prev_conv_ideal, uint64_t prev_mem_ideal, uint64_t prev_pool_ideal,
    //size_t mem_target, // 1 / num_core
    int compute_target, // conv layer target (from pre-compiled)
    enum layer_type_t prev_layer_type, enum layer_tyepe_t next_layer_type,
    const struct ConvParams * prev_params,
    size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id){ // group_id: valid group id for only 1 core per workload running group 

  // remaining cycles: remain target cycles
  // prev macs: total macs before this layer
  // MEM: update remaining mem cycles
  // conv: update remaining cycle, conv ideal -> subtract expected mem cycle -> compute next layer's target conv
  remaining_cycles = args_out[0];
  remaining_mem_cycles = args_out[1];
  remaining_pool_cycles = args_out[2];
  prev_conv_ideal = args_out[3];
  prev_mem_ideal = args_out[4];
  prev_pool_ideal = args_out[5];
  mem_target = (size_t)(args_out[6]);
  prev_cycles = args_out[7]; // prev_layer_cycles

  remaining_cycles -= prev_cycles;
  uint64_t new_conv_ideal = prev_conv_ideal;
  uint64_t new_mem_ideal = prev_mem_ideal;
  uint64_t new_pool_ideal = prev_pool_ideal;
  if(prev_layer_type == CONV){
    int args_in[10];
    int* args = tiled_conv_A_stride_bubble_calculate(args_in, prev_params -> batch_size, prev_params->in_dim, prev_params->in_channels,
        prev_params->out_channels, prev_params->out_dim, prev_params->stride, prev_params->dilation, prev_params->padding, prev_params->kernel_dim,
        prev_params->pool_size, prev_params->pool_stride, prev_params->pool_padding, prev_params->pool_ceil_dim,
        orow_divide, batch_divide, cid);
    uint64_t prev_layer_ideal = args[9];
    new_conv_ideal -= prev_layer_ideal;
  }
  else if(prev_layer_type == MATMUL){
    size_t args_in[10];
    size_t* args = tiling_factor_matmul_calculate_auto(prev_params->I, prev_params->J, prev_params->K,
        orow_divide, batch_divide, cid, args_in, 0);
    uint64_t prev_layer_ideal = args[2];
    new_conv_ideal -= prev_layer_ideal;
  }
  else if(prev_layer_type == FC){
    size_t args_in[10];
    remaining_mem_cycles -= prev_cycles;
    size_t* args = tiling_factor_matmul_calculate_auto(prev_params->I, prev_params->J, prev_params->K,
        orow_divide, batch_divide, cid, args_in, 0);
    uint64_t prev_layer_ideal = args[2];
    new_mem_ideal -= prev_layer_ideal;
  }
  else if(prev_layer_type == RESADD){
    remaining_mem_cycle -= prev_cycles;
    int args_in[5];
    int* args = tiled_resadd_bubble_calculate(args_in, prev_params->I, prev_params->J,
        orow_divide, batch_divide, 0);
    uint64_t resadd_ideal = args[2];
    new_mem_ideal -= resadd_ideal;
  }
  else if(prev_layer_type == POOL){
    remaining_pool_cycles -= prev_cycles;
    int args_in[5];
    int* args = tiled_pool_bubble_calculate(args_in, prev_params->batch_size, prev_params->in_dim, prev_params->out_channels, prev_params->out_dim,
        prev_params->pool_size, prev_params->pool_stride, prev_params->pool_padding,
        orow_divide, batch_divide, 0);
    uint64_t pool_ideal = args[2];
    new_mem_ideal -= pool_ideal;
  }

  int next_conv_target = 0;
  //if(next_layer_type == CONV || next_layer_type == MATMUL){
  uint64_t remaining_conv_target_cycles = remaining_cycles - (uint64_t)(new_mem_ideal / mem_target) - (uint64_t)(new_pool_ideal / mem_target * 2);
  next_conv_target = (int)(remaining_conv_target_cycles / new_conv_ideal); // Todo: reuse factor
  //}
  if(group_id < NUM_CORE){
    adjust[group_id] = remaining_conv_target_cycles - compute_target;
  }

  int next_mem_target = new_mem_ideal;
  if(new_conv_ideal == 0){ // done conv
    next_mem_target = (int)(remaining_mem_target_cycles / new_mem_ideal);
  }
 
  //ToDo: update priority score
  //ToDo: adjust target

  args_out[0] = remaining_cycles;
  args_out[1] = remaining_mem_cycles;
  args_out[2] = remaining_pool_cycles;
  args_out[3] = new_conv_ideal;
  args_out[4] = new_mem_ideal;
  args_out[5] = new_pool_ideal;
  args_out[6] = new_mem_target;
  args_out[7] = new_conv_target;

  return args_out;

}
*/

static void conv_dw(size_t I, size_t J,
    const size_t batch_size, const size_t channels, const size_t in_dim, const size_t out_dim, const size_t kernel_size,
    const elem_t input[batch_size][in_dim][in_dim][channels],
    const elem_t weight[channels][kernel_size][kernel_size],
    const acc_t * bias,
    // elem_t output [batch_size][out_dim][out_dim][channels],
    elem_t output [I][J],
    const struct ConvParams * params)
{
    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * params->stride - params->padding;

                    acc_t result = 0;
                    if (params->bias) {
                        result = bias[channel];
                    }

                    for (int kernel_row = 0; kernel_row < params->kernel_size; kernel_row++) {
                        int in_col = out_col * params->stride - params->padding;

                        for (int kernel_col = 0; kernel_col < params->kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < params->in_dim && in_col >= 0 && in_col < params->in_dim) {
                                result += input[batch][in_row][in_col][channel] * weight[channel][kernel_row][kernel_col];
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    if (result < 0) {
                        result = 0;
                    }
                    
                    acc_t scaled = ACC_SCALE(result, params->output_scale);

                    if (scaled > elem_t_max) {
                        scaled = elem_t_max;
                    } else if (scaled < elem_t_min) {
                        scaled = elem_t_min;
                    }
                    
                    size_t r = batch * params->out_dim * params->out_dim + out_row * params->out_dim + out_col;
                    output[r][channel] = scaled;
                    // output[batch][out_row][out_col][channel] = scaled;
                }
            }
        }
    }
}

static void conv_dw_with_col2im(size_t prev_I, size_t prev_J, size_t I, size_t J,
    const size_t batch_size, const size_t channels, const size_t out_dim, const size_t kernel_size,
    const elem_t input[prev_I][prev_J],
    const elem_t weight[channels][kernel_size][kernel_size],
    const acc_t * bias,
    // elem_t output [batch_size][out_dim][out_dim][channels],
    elem_t output [I][J],
    const struct ConvParams * params)
{
    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * params->stride - params->padding;

                    acc_t result = 0;
                    if (params->bias) {
                        result = bias[channel];
                    }

                    for (int kernel_row = 0; kernel_row < params->kernel_size; kernel_row++) {
                        int in_col = out_col * params->stride - params->padding;

                        for (int kernel_col = 0; kernel_col < params->kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < params->in_dim && in_col >= 0 && in_col < params->in_dim) {
                                // result += input[batch][in_row][in_col][channel] * weight[channel][kernel_row][kernel_col];

                                size_t r = batch * params->in_dim * params->in_dim + in_row * params->in_dim + in_col;

                                result += input[r][channel] * weight[channel][kernel_row][kernel_col];
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    if (result < 0) {
                        result = 0;
                    }
                    
                    acc_t scaled = ACC_SCALE(result, params->output_scale);

                    if (scaled > elem_t_max) {
                        scaled = elem_t_max;
                    } else if (scaled < elem_t_min) {
                        scaled = elem_t_min;
                    }
                    
                    size_t r = batch * params->out_dim * params->out_dim + out_row * params->out_dim + out_col;
                    output[r][channel] = scaled;
                    // output[batch][out_row][out_col][channel] = scaled;
                }
            }
        }
    }
}

static void im2col(size_t batch_size, size_t channels, size_t im_dim,
    size_t I, size_t K,
    const elem_t input[batch_size][im_dim][im_dim][channels],
    elem_t output[I][K],
    const struct ConvParams * params)
{
    int patch_row = 0;

    for (int n_batch = 0; n_batch < params->batch_size; n_batch++) {
        for (int im_row = -params->padding; im_row < params->in_dim - params->kernel_size + params->padding + 1; im_row += params->stride) {
            for (int im_col = -params->padding; im_col < params->in_dim - params->kernel_size + params->padding + 1; im_col += params->stride) {
                int patch_col = 0;

                for (int filter_row = 0; filter_row < params->kernel_size; filter_row++) {
                    for (int filter_col = 0; filter_col < params->kernel_size; filter_col++) {
                        for (int im_channel = 0; im_channel < params->in_channels; im_channel++) {
                            int pixel_row = im_row + filter_row;
                            int pixel_col = im_col + filter_col;
                            
                            if (pixel_row < 0 || pixel_row >= params->in_dim
                                || pixel_col < 0 || pixel_col >= params->in_dim) {
                                // output[patch_row][patch_col] = 0;
                            } else {
                                output[patch_row][patch_col] = input[n_batch][pixel_row][pixel_col][im_channel];
                            }

                            patch_col++;
                        }
                    }
                }
                
                patch_row++;
            }
        }
    }
}

static void im2col_with_col2im(size_t prev_I, size_t prev_J,
    size_t next_I, size_t next_K,
    const elem_t input[prev_I][prev_J],
    elem_t output[next_I][next_K],
    const struct ConvParams * params)
{
    int out_row = 0;

    for (int n_batch = 0; n_batch < params->batch_size; n_batch++) {
        for (int im_row = -params->padding; im_row < params->in_dim - params->kernel_size + params->padding + 1; im_row += params->stride) {
            for (int im_col = -params->padding; im_col < params->in_dim - params->kernel_size + params->padding + 1; im_col += params->stride) {
                int out_col = 0;

                for (int filter_row = 0; filter_row < params->kernel_size; filter_row++) {
                    for (int filter_col = 0; filter_col < params->kernel_size; filter_col++) {
                        for (int im_channel = 0; im_channel < params->in_channels; im_channel++) {
                            int pixel_row = im_row + filter_row;
                            int pixel_col = im_col + filter_col;

                            if (pixel_row < 0 || pixel_row >= params->in_dim
                                || pixel_col < 0 || pixel_col >= params->in_dim) {
                                // output[out_row][out_col] = 0;
                            } else {
                                int in_row = n_batch * params->in_dim * params->in_dim + pixel_row * params->in_dim + pixel_col;
                                int in_col = im_channel;

                                output[out_row][out_col] = input[in_row][in_col];
                            }

                            out_col++;
                        }
                    }
                }

                out_row++;
            }
        }
    }
}

// Compute C = A + B with saturating add
void vecadd(size_t len, const elem_t * A, const elem_t * B, elem_t * C, scale_t A_shift) {
    for (size_t i = 0; i < len; i++) {
        acc_t result = MVIN_SCALE(A[i], A_shift) + B[i];

        if (result > elem_t_max) {
            result = elem_t_max;
        } else if (result < elem_t_min) {
            result = elem_t_min;
        }

        C[i] = result;
    }
}

void resadd1(const size_t batch_size, const size_t channels, const size_t im_dim,
    const elem_t A[batch_size][im_dim][im_dim][channels],
    const elem_t B[batch_size][im_dim][im_dim][channels],
    elem_t C[batch_size][im_dim][im_dim][channels],
    bool relu,
    const struct ConvParams * params) {

    const int minimum = relu ? 0 : elem_t_min;

    for (size_t batch = 0; batch < params->batch_size; batch++) {
        for (size_t row = 0; row < params->out_dim_pooled; row++) {
            for (size_t col = 0; col < params->out_dim_pooled; col++) {
                for (size_t channel = 0; channel < params->out_channels; channel++) {
                    acc_t result = MVIN_SCALE(A[batch][row][col][channel], params->res_scale) + B[batch][row][col][channel];

                    if (result > elem_t_max) {
                        result = elem_t_max;
                    } else if (result < minimum) {
                        result = minimum;
                    }

                    C[batch][row][col][channel] = result;
                }
            }
        }
    }
}

void resadd2(const size_t I, const size_t J,
    const size_t batch_size, const size_t channels, const size_t im_dim,
    const elem_t A[I][J],
    const elem_t B[batch_size][im_dim][im_dim][channels],
    elem_t C[batch_size][im_dim][im_dim][channels],
    bool relu,
    const struct ConvParams * params) {

    const int minimum = relu ? 0 : elem_t_min;

    for (size_t batch = 0; batch < params->batch_size; batch++) {
        for (size_t row = 0; row < params->out_dim_pooled; row++) {
            for (size_t col = 0; col < params->out_dim_pooled; col++) {
                for (size_t channel = 0; channel < params->out_channels; channel++) {
                    size_t r = batch * params->out_dim_pooled * params->out_dim_pooled + row * params->out_dim_pooled + col;

                    acc_t result = MVIN_SCALE(A[r][channel], params->res_scale) + B[batch][row][col][channel];

                    if (result > elem_t_max) {
                        result = elem_t_max;
                    } else if (result < minimum) {
                        result = minimum;
                    }

                    C[batch][row][col][channel] = result;
                }
            }
        }
    }
}

void resadd3(const size_t I, const size_t J,
    const elem_t A[I][J],
    const elem_t B[I][J],
    elem_t C[I][J],
    bool relu,
    const struct ConvParams * params) {

    const int minimum = relu ? 0 : elem_t_min;

    for (size_t batch = 0; batch < params->batch_size; batch++) {
        for (size_t row = 0; row < params->out_dim_pooled; row++) {
            for (size_t col = 0; col < params->out_dim_pooled; col++) {
                for (size_t channel = 0; channel < params->out_channels; channel++) {
                    size_t r = batch * params->out_dim_pooled * params->out_dim_pooled + row * params->out_dim_pooled + col;

                    acc_t result = MVIN_SCALE(A[r][channel], params->res_scale) + B[r][channel];

                    if (result > elem_t_max) {
                        result = elem_t_max;
                    } else if (result < minimum) {
                        result = minimum;
                    }

                    C[r][channel] = result;
                }
            }
        }
    }
}

// Pooling
void pool(size_t batch_size, size_t channels, size_t in_dim, size_t out_dim,
    elem_t input[batch_size][in_dim][in_dim][channels],
    elem_t output[batch_size][out_dim][out_dim][channels],
    const struct ConvParams * params)
{
    size_t kernel_size = params->pool_size;
    size_t stride = params->pool_stride;
    // size_t in_dim = params->out_dim;
    size_t padding = params->pool_padding;

    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * stride - padding;

                    elem_t result = elem_t_min;

                    for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                        int in_col = out_col * stride - padding;

                        for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < in_dim && in_col >= 0 && in_col < in_dim) {
                                if (input[batch][in_row][in_col][channel] > result) {
                                    result = input[batch][in_row][in_col][channel];
                                }
                            } else if (0 > result) {
                                result = 0;
                            }

                            in_col++;
                        }

                        in_row++;
                    }
                    
                    output[batch][out_row][out_col][channel] = result;
                }
            }
        }
    }
}

void pool_with_col2im(size_t I, size_t J,
    size_t batch_size, size_t channels, size_t out_dim,
    elem_t input[I][J],
    elem_t output[batch_size][out_dim][out_dim][channels],
    const struct ConvParams * params)
{
    size_t kernel_size = params->pool_size;
    size_t stride = params->pool_stride;
    size_t in_dim = params->out_dim;
    size_t padding = params->pool_padding;

    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * stride - padding;

                    elem_t result = elem_t_min;

                    for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                        int in_col = out_col * stride - padding;

                        for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < in_dim && in_col >= 0 && in_col < in_dim) {
                                if (input[batch * in_dim * in_dim + in_row * in_dim + in_col][channel] > result) {
                                    result = input[batch * in_dim * in_dim + in_row * in_dim + in_col][channel];
                                }
                            } else if (0 > result) {
                                result = 0;
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    output[batch][out_row][out_col][channel] = result;
                }
            }
        }
    }
}

#endif // GEMMINI_NN_H

