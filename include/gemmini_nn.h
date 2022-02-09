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
    
    int I, J, K;
};

struct FcParams {
    int batch_size;
    int in_features;
    int out_features;
    acc_scale_t output_scale;
    bool bias;

    int I, J, K;
};

enum layer_type_t {CONV, MATMUL, FC, RESADD, POOL}
size_t priority_score[NUM_CORE] = {0};
size_t adjust[NUM_CORE] = {0};

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

// assign priority score -> get workloads to execute -> assign resources
int* scheduler(
    float priority[], // previous priority (initialization: user given priority)
    uint64_t old[], // how long it has been in the queue
    uint64_t target[]
    ){

  int num_workload = (int)(sizeof(priority) / sizeof(priority[0]));
  uint64_t old_total = 0;
  for (int i = 0; i < num_workload; i += 1)
    old_total += old[i];

  for (int i = 0; i < num_workload; i += 1){
    priority[i] += (old[i] / old_total);
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
    size_t orow_divide, size_t batch_divide, size_t cid, size_t wid){ // wid: workload id

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
  if(wid < NUM_CORE){
    adjust[wid] = remaining_conv_target_cycles - compute_target;
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

