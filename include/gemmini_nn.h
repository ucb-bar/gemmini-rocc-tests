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
/*
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
        0,
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
            0,
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
        0,
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
            0,
            CPU);

        if (!MAT_IS_EQUAL(dim_I, dim_J, C, gold)) {
            printf("Layer calculated incorrectly: %s\n", layer_name);
            exit(1);
        }
    }
}
*/
static void tiled_matmul_nn_auto_multi(size_t dim_I, size_t dim_J, size_t dim_K,
  size_t stride_A, size_t stride_B, size_t stride_C,
  bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram,  
  elem_t* A, elem_t* B,
  const void * D, elem_t* C,
  int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
  enum tiled_matmul_type_t tiled_matmul_type,
  size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id)
{
  size_t* args_out;
  size_t args[10];
  args_out = tiling_factor_matmul_calculate_auto(dim_I, dim_J, dim_K, orow_divide, batch_divide, cid, group_id, args, 0);
  dim_I = args_out[3];
  dim_J = args_out[4];
  dim_K = args_out[5];
  size_t tile_I = args_out[8];
  size_t tile_J = args_out[9];
  size_t tile_K = args_out[10];
  
  //printf("%llu, ", dim_I*dim_J*dim_K);
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
    A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
    act, scale, relu6_shift, repeating_bias,
    tile_I, tile_J, tile_K,
    false, false, false, false, 3,
    tiled_matmul_type); 
    //window, target_load);

}

static void tiled_matmul_nn_auto_cid(size_t dim_I, size_t dim_J, size_t dim_K,
  size_t stride_C,
  elem_t* A, elem_t* B,
  const void * D, elem_t* C,
  int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
  enum tiled_matmul_type_t tiled_matmul_type,
  size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id){

  size_t stride_A = (dim_K % 128 == 0) ? dim_K + 64 : dim_K;
  size_t stride_B = (dim_J % 128 == 0) ? dim_J + 64 : dim_J;

  //printf("A dram addr: 0x%08lx\n", A);
  tiled_matmul_nn_auto_multi(
      dim_I, dim_J, dim_K,
      stride_A, stride_B, stride_C,
      false, false, false, false, // direct dram
      A, B, D, C,
      act, scale, relu6_shift, repeating_bias,
      WS,
      orow_divide, batch_divide, cid, group_id);

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


// division by row dimension
static void tiled_conv_auto_multi( // for sw padding
    int batch_size, int in_dim, int in_channels,
    int out_channels, int out_dim,
    int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,

    int in_stride, int weight_stride, int out_stride,
    bool in_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool out_direct_dram,

    bool wrot180, bool trans_output_1203, bool trans_input_3120,
    bool trans_weight_1203, bool trans_weight_0132,

    const elem_t * input,
    const elem_t * weights,
    const acc_t * bias,
    elem_t * output,

    int act, acc_scale_t scale, size_t relu6_shift,
    int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,

    enum tiled_matmul_type_t tiled_conv_type,
    size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id){

      int target_util = 0;
  const bool no_pool = pool_stride == 0;
   if (no_pool) {
      pool_size = 1;
      pool_stride = 1;
      pool_padding = 0;
   }
#ifdef GEMMINI_ASSERTIONS
   if(batch_size == 1 && batch_divide > 1){
     printf("batch_divide set wrong \n");
     exit(1);
   }
#endif
   //printf("%d, ", batch_size*out_dim*out_dim*in_channels*out_channels*kernel_dim*kernel_dim);

   // tiling, calm configure
   int args_in[10] = {target_util, 0, 0, 0, 0, 0, 0, 0, 0, 0};
   int * args = tiled_conv_bubble_calculate(args_in, batch_size, in_dim, in_channels, out_channels, out_dim, stride, kernel_dilation, padding, kernel_dim, pool_size, pool_stride, pool_padding, pool_ceil_dim, orow_divide, batch_divide, cid, group_id);

  int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;
  if (pool_ceil_dim)
    pool_out_dim += (out_dim + 2*pool_padding - pool_size) % pool_stride != 0;

  //size_t total_divide = orow_divide * batch_divide;
  size_t batch_cid = (size_t)(cid / orow_divide);
  size_t orow_cid = (size_t)(cid % orow_divide);

  // divide in batch dimension
  batch_size = batch_size / batch_divide;
  int batch_in_offset = (batch_divide > 1) ? batch_size*in_dim*in_dim*in_stride*batch_cid : 0;
  int batch_out_offset = (batch_divide > 1) ? batch_size*pool_out_dim*pool_out_dim*out_stride*batch_cid : 0; // not dividing in out_channel dimension
//printf("batch in offset: %d, batch out offset: %d\n", batch_in_offset, batch_out_offset);
// divide in row dimension (single batch)
	bool row_divisible = (orow_divide > 1) && ((pool_out_dim % orow_divide == 0) || (in_channels == 3 && padding == 0)) && (kernel_dilation <= 2);
  //bool row_divisible = (orow_divide > 1) && (pool_out_dim % orow_divide == 0) && (kernel_dilation <= 2);
  if (orow_divide > 1 && padding == 0) row_divisible = true;
  int pool_out_row = (row_divisible) ? (pool_out_dim / orow_divide) : pool_out_dim;
  if(pool_out_dim % orow_divide != 0) {
     if(orow_cid != orow_divide - 1) pool_out_row += 1;
     //else pool_out_row -= 1;
  }
  const size_t och_divide = (row_divisible) ? 1 : orow_divide; //if row isn't divisible, divide channel instead
  out_channels = out_channels / och_divide;
  
  const int out_offset = (och_divide > 1) ? out_channels * orow_cid : 0;

  /*
  int args_in[] = {batch_size, pool_out_row, pool_out_dim, out_channels, kernel_dim, kernel_dim, in_channels};
  int* args;
  args = tiling_factor_calculate(args_in, stride, pool_size, pool_stride, kernel_dilation, padding);
*/
  const int batches = args[0];
  const int orows = args[1];
  const int ocols = args[2];
  const int ochs = args[3];
  const int krows = args[4];
  const int kcols = args[5];
  const int kchs = args[6];

  const int window = args[7];
  const int target_load = args[8];
  const int ideal_cycle = args[9];
  if(row_divisible){
      tiled_conv(
          batch_size, in_dim, in_channels,
          out_channels, out_dim,
          stride, input_dilation, kernel_dilation, padding, kernel_dim,
          in_stride, weight_stride, out_stride,
          in_direct_dram, weight_direct_dram, bias_direct_dram, out_direct_dram,

          wrot180, trans_output_1203, trans_input_3120,
          trans_weight_1203, trans_weight_0132,

          batches,
          orows, ocols, ochs,
          krows, kcols, kchs,

          (elem_t*) input + batch_in_offset,
          (elem_t*) weights,
          (acc_t*) bias,
          output + batch_out_offset,

          act, scale, relu6_shift,
          pool_size, no_pool ? 0 : pool_stride, pool_padding, pool_ceil_dim,

          tiled_conv_type, orow_divide, orow_cid, group_id);
         // window, target_load);

  }else{
    bool no_bias = (bias == NULL);
    tiled_conv(
        batch_size, in_dim, in_channels,
        out_channels, out_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        in_stride, weight_stride, out_stride,
        in_direct_dram, weight_direct_dram, bias_direct_dram, out_direct_dram,

        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,

        batches,
        orows, ocols, ochs,
        krows, kcols, kchs,

        (elem_t*) input + batch_in_offset,
        (elem_t*) weights + out_offset,
        no_bias ? NULL : (acc_t*) bias + out_offset,
        output + out_offset + batch_out_offset,

        act, scale, relu6_shift,
        pool_size, no_pool ? 0 : pool_stride, pool_padding, pool_ceil_dim,

        tiled_conv_type, 1, orow_cid, group_id);
        //window, target_load);

  }
}

// for convert
static void tiled_conv_auto_cid(
    int batch_size, int in_dim, int in_channels,
    int out_channels, int out_dim,
    int stride, int dilation, int padding, int kernel_dim,
    int out_stride, //for concatenation

    const elem_t * input,
    const elem_t * weights,
    const acc_t * bias,
    elem_t * output,

    int act, acc_scale_t scale, size_t relu6_shift,
    int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,

    enum tiled_matmul_type_t tiled_conv_type,
    size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id){

  int in_stride = (in_channels % 128 == 0) ? in_channels + 64 : in_channels;
  int weight_stride = (out_channels % 128 == 0) ? out_channels + 64 : out_channels;
#ifdef GEMMINI_ASSERTIONS
  if(out_stride % 128 == 0){
    printf("need padding\n");
    exit(1);
  }
#endif
  //printf("conv\n");

  tiled_conv_auto_multi(
     batch_size, in_dim, in_channels,
     out_channels, out_dim,
     stride, 1, dilation, padding, kernel_dim,
     in_stride, weight_stride, out_stride,
     false, false, false, false, // dfrom dram

     false, false, false, false, false,

     input, weights, bias, output,

     act, scale, relu6_shift,
     pool_size, pool_stride, pool_padding, pool_ceil_dim,
     tiled_conv_type,
     orow_divide, batch_divide, cid, group_id);
}

static void tiled_resadd_auto_multi(size_t I, size_t J,
    const scale_t A_scale,
    const scale_t B_scale,
    const acc_scale_t C_scale,
    const size_t J_stride,
    bool A_direct_dram, bool B_direct_dram, bool C_direct_dram,
    const elem_t * A,
    const elem_t * B,
    elem_t * C,
    bool relu,
    enum tiled_matmul_type_t matadd_type,
    size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id) {
  if (matadd_type == CPU) {
    resadd_cpu(I, J,
    A_scale, B_scale, C_scale, A, B, C,
    relu);
    return;
  }
  size_t batch_cid = (size_t)(cid / orow_divide);
  size_t orow_cid = (size_t)(cid % orow_divide);

  int args_in[] = {0, 0, 0, 0, 0};
  int target_util = 0;
  int * args = tiled_resadd_bubble_calculate(args_in, I, J, orow_divide, batch_divide, group_id, 0);

  size_t batch_size = I / batch_divide;
  I = batch_size;

  bool row_divisible = orow_divide > 1 && (I % orow_divide == 0);
  I = (row_divisible) ? I / orow_divide : I;
  size_t och_divide = (row_divisible) ? 1 : orow_divide; // if row is divisible, no need to divide channel

  size_t tile_I = I; // divide when it is divisible
  J = J / och_divide;
  size_t tile_J = J;

  int out_offset = (och_divide > 1) ? tile_J * orow_cid : 0; // no offset if divided in row dimension
  int orow_offset = (row_divisible) ? J_stride * orow_cid * I : 0;
  int batch_offset = (batch_divide > 1) ? batch_cid * batch_size * J_stride : 0;

  int window = args[0];
  int target_load = args[1];
  int ideal_cycles = args[2];
  tile_I = args[3];
  tile_J = args[4];

	//printf("ideal cycle: %d, target_cycle: %d, \n", ideal_cycles, target_cycles);
  //printf("priority: %d, window: %d, target_load: %d \n", priority, window, target_load);
 
    // printf("tile_I: %llu\n", tile_I);
    // printf("tile_J: %llu\n", tile_J);

  if (matadd_type == WS) {
    tiled_resadd(I, J, J_stride, A_direct_dram, B_direct_dram, C_direct_dram,
        tile_I, tile_J, 
        A_scale, B_scale, C_scale, A + batch_offset + out_offset + orow_offset, B + batch_offset + orow_offset + out_offset, C + batch_offset + orow_offset + out_offset,
        relu, matadd_type);
  } else if(matadd_type == CPU){
    resadd_cpu(I, J, A_scale, B_scale, C_scale,
        A, B, C, relu);
  }
  else {
    printf("Unsupported type\n");
    exit(1);
  }
}

static void tiled_resadd_auto_cid(size_t I, size_t J,
    const scale_t A_scale,
    const scale_t B_scale,
    const acc_scale_t C_scale,
    const elem_t * A,
    const elem_t * B,
    elem_t * C,
    bool relu,
    enum tiled_matmul_type_t matadd_type,
    size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id) {
 // printf("resadd\n");
  size_t J_stride = (J % 128 == 0) ? J + 64 : J;
  
  tiled_resadd_auto_multi(I, J, A_scale, B_scale, C_scale,
      J_stride,
      false, false, false,
      A, B, C,
      relu, matadd_type,
      orow_divide, batch_divide, cid, group_id);

}


// pooling using Gemmini DMA
static void tiled_pool_auto_multi(int batch_size, int channels, int in_dim,
    int pool_out_dim, int stride,
    int pool_size, int pool_stride, int pool_padding,
    bool input_direct_dram, bool output_direct_dram,
    const elem_t * A,
    elem_t * C,
    size_t och_divide, size_t batch_divide, size_t cid, size_t group_id) {
  
  bool relu = true;
	//int stride = channels;

  bool row_divide = (och_divide > 1 && channels < 64);
  int * args;
  int args_in[] = {0, 0, 0, 0};
  int target_util = 0;
  args = tiled_pool_bubble_calculate(args_in, batch_size, in_dim, channels, pool_out_dim, pool_size, pool_stride, pool_padding, row_divide, och_divide, batch_divide, cid, group_id);
  
  size_t batch_cid = (size_t)(cid / och_divide);
  size_t och_cid = (size_t)(cid % och_divide);


  batch_size = batch_size/batch_divide;
  channels = (row_divide) ? channels : channels / och_divide;
  //int pool_out_dim = (in_dim + 2*pool_padding - pool_size) / pool_stride + 1;
	int batch_in_offset = (batch_divide > 1) ? batch_size*in_dim*in_dim*stride*batch_cid : 0;
	int batch_out_offset = (batch_divide > 1) ? batch_size*pool_out_dim*pool_out_dim*stride*batch_cid : 0; // not dividing in out_channel dimension
 	const int out_offset = (och_divide > 1 && !row_divide) ? channels * och_cid : 0;
  if(!row_divide) och_divide = 1;

  int window = args[0];
  int target_load = args[1]; 
  const int batches = args[3];
  const int porows = args[4];
  const int pocols = args[5];
  const int pochs = args[6];
  //printf("window: %d, target_load: %d \n", window, target_load);

  window = 0;
  target_load = 0; // for now, disable CALM on pooling
  //printf("C dram addr before pool: 0x%08lx\n", C);
  tiled_pool(batch_size, in_dim, channels, pool_out_dim,
				batches, porows, pocols, pochs,
        stride,
        input_direct_dram, output_direct_dram, 
        A + batch_in_offset + out_offset, C + batch_out_offset + out_offset,	
				RELU, MVIN_SCALE_IDENTITY, 0,
				pool_size, pool_stride, pool_padding,
				och_divide, cid, group_id, window, target_load);
  
  //printf("C dram addr after pool: 0x%08lx\n", C);
}

static void tiled_pool_auto_cid(int batch_size, int channels, int in_dim,
    int pool_out_dim, int stride,
    int pool_size, int pool_stride, int pool_padding,
    const elem_t * A,
    elem_t * C,
    size_t och_divide, size_t batch_divide, size_t cid, size_t group_id){
  
  tiled_pool_auto_multi(batch_size, channels, in_dim,
      pool_out_dim, stride,
      pool_size, pool_stride, pool_padding,
      false, false,
      A, C, 
      och_divide, batch_divide, cid, group_id);
}


#endif // GEMMINI_NN_H

