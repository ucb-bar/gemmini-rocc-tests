#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"

#include "mobilenet_params.h"
#include "mobilenet_images.h"

#define PRINT_IMAGES(IMAGES) \
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

#define PRINT_MATRIX(MATRIX) \
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

static void tiled_matmul_compare(size_t DIM_I, size_t DIM_J, size_t DIM_K,
        elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], void * D,
        elem_t C[DIM_I][DIM_J],
        int act, int shift, int relu6_shift, int full_bias_width,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool compare, char * layer_name)
{
    if (compare)
        printf("%s: gemmini\n", layer_name);
        tiled_matmul_option(DIM_I, DIM_J, DIM_K,
            A, B, D, C, act, shift, relu6_shift, full_bias_width,
            tiled_matmul_type);

    if (compare) {
        printf("%s: CPU\n", layer_name);
        elem_t gold[DIM_I][DIM_J];
        tiled_matmul_option(DIM_I, DIM_J, DIM_K,
            A, B, D, gold, act, shift, relu6_shift, full_bias_width,
            CPU);

        if (!MAT_IS_EQUAL(DIM_I, DIM_J, C, gold)) {
            printf("Layer calculated incorrectly: %s\n", layer_name);
            exit(1);
        }
    }
}

static void conv_dw(const size_t batch_size, const size_t channels, const size_t in_dim, const size_t out_dim, const size_t kernel_size,
    const elem_t input[batch_size][channels][in_dim][in_dim],
    const elem_t weight[channels][kernel_size][kernel_size],
    const acc_t * bias,
    elem_t output [batch_size][channels][out_dim][out_dim],
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
                                result += input[batch][channel][in_row][in_col] * weight[channel][kernel_row][kernel_col];
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    /*
                    acc_t abs = result >= 0 ? result : -result;
                    int divisor = 1 << params->output_scale;
                    acc_t shifted = (abs + divisor/2) >> params->output_scale;
                    if (result < 0) {
                        shifted = -shifted;
                    }
                    */

                    if (result < 0) {
                        result = 0;
                    }
                    
                    acc_t shifted = ROUNDING_RIGHT_SHIFT(result, params->output_scale);

                    if (shifted > elem_t_max) {
                        shifted = elem_t_max;
                    }                    

                    output[batch][channel][out_row][out_col] = shifted;
                }
            }
        }
    }
}

static void im2col(size_t batch_size, size_t channels, size_t im_dim,
    size_t I, size_t K,
    const elem_t input[batch_size][channels][im_dim][im_dim],
    elem_t output[I][K],
    const struct ConvParams * params)
{
    int patch_row = 0;
    
    for (int n_batch = 0; n_batch < params->batch_size; n_batch++) {
        for (int im_row = -params->padding; im_row < params->in_dim - params->kernel_size + params->padding + 1; im_row += params->stride) {
            for (int im_col = -params->padding; im_col < params->in_dim - params->kernel_size + params->padding + 1; im_col += params->stride) {
                int patch_col = 0;
                
                for (int im_channel = 0; im_channel < params->in_channels; im_channel++) {
                    for (int filter_row = 0; filter_row < params->kernel_size; filter_row++) {
                        for (int filter_col = 0; filter_col < params->kernel_size; filter_col++) {
                            int pixel_row = im_row + filter_row;
                            int pixel_col = im_col + filter_col;
                            
                            if (pixel_row < 0 || pixel_row >= params->in_dim
                                || pixel_col < 0 || pixel_col >= params->in_dim) {
                                // output[patch_row][patch_col] = 0;
                            } else {
                                output[patch_row][patch_col] = input[n_batch][im_channel][pixel_row][pixel_col];
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

static void col2im(size_t I, size_t J,
    size_t batch_size, size_t channels, size_t im_dim,
    const elem_t input[I][J],
    elem_t output[batch_size][channels][im_dim][im_dim],
    const struct ConvParams * params)
{
    for (int channel = 0; channel < params->out_channels; channel++) {
        int pixel_row = 0;

        for (int batch = 0; batch < params->batch_size; batch++) {
            for (int row = 0; row < params->out_dim; row++) {
                for (int col = 0; col < params->out_dim; col++) {
                    output[batch][channel][row][col] = input[pixel_row][channel];
                    pixel_row++;
                }
            }
        }
    }
}

// Compute C = A + B with saturating add
void vecadd(size_t len, const elem_t * A, const elem_t * B, elem_t * C, int A_shift) {
    for (size_t i = 0; i < len; i++) {
        acc_t result = ROUNDING_RIGHT_SHIFT(A[i], A_shift) + B[i];

        if (result > elem_t_max) {
            result = elem_t_max;
        } else if (result < elem_t_min) {
            result = elem_t_min;
        }

        C[i] = result;
    }
}

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    matmul_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type;
    if (argc < 2) {
        // printf("usage: %s matmul_option\n  matmul_option may be 'os', 'ws', or cpu'\n");
        // exit(0);
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    }

    bool compare;
    if (argc < 3) {
        compare = false;
    } else if (strcmp(argv[2], "compare") == 0) {
        compare = true;
    } else {
        printf("Unknown command-line argument\n");
        exit(1);
    }

    // conv_1
    im2col(conv_1_params.batch_size, conv_1_params.in_channels, conv_1_params.in_dim,
        conv_1_params.I, conv_1_params.K,
        images, conv_1_in, &conv_1_params);

    tiled_matmul_compare(conv_1_params.I, conv_1_params.J, conv_1_params.K,    // dimensions
        conv_1_in, conv_1_w, conv_1_b, conv_1_out,      // addresses
        RELU, conv_1_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_1");
        
    col2im(conv_1_params.I, conv_1_params.J, conv_1_params.batch_size, conv_1_params.out_channels, conv_1_params.out_dim,
        conv_1_out, conv_1_out_reshaped, &conv_1_params);

    // conv_dw_2
    conv_dw(conv_dw_2_params.batch_size, conv_dw_2_params.in_channels, conv_dw_2_params.in_dim, conv_dw_2_params.out_dim, conv_dw_2_params.kernel_size,
        conv_1_out_reshaped, conv_dw_2_w, conv_dw_2_b, conv_dw_2_out, &conv_dw_2_params);

    // conv_3
    im2col(conv_3_params.batch_size, conv_3_params.in_channels, conv_3_params.in_dim,
        conv_3_params.I, conv_3_params.K,
        conv_dw_2_out, conv_3_in, &conv_3_params);

    tiled_matmul_compare(conv_3_params.I, conv_3_params.J, conv_3_params.K,    // dimensions
        conv_3_in, conv_3_w, conv_3_b, conv_3_out,      // addresses
        NO_ACTIVATION, conv_3_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_3");
        
    col2im(conv_3_params.I, conv_3_params.J, conv_3_params.batch_size, conv_3_params.out_channels, conv_3_params.out_dim,
        conv_3_out, conv_3_out_reshaped, &conv_3_params);

    // No need to add residuals here

    // conv_4
    im2col(conv_4_params.batch_size, conv_4_params.in_channels, conv_4_params.in_dim,
        conv_4_params.I, conv_4_params.K,
        conv_3_out_reshaped, conv_4_in, &conv_4_params);

    tiled_matmul_compare(conv_4_params.I, conv_4_params.J, conv_4_params.K,    // dimensions
        conv_4_in, conv_4_w, conv_4_b, conv_4_out,      // addresses
        RELU, conv_4_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_4");
        
    col2im(conv_4_params.I, conv_4_params.J, conv_4_params.batch_size, conv_4_params.out_channels, conv_4_params.out_dim,
        conv_4_out, conv_4_out_reshaped, &conv_4_params);

    // conv_dw_5
    conv_dw(conv_dw_5_params.batch_size, conv_dw_5_params.in_channels, conv_dw_5_params.in_dim, conv_dw_5_params.out_dim, conv_dw_5_params.kernel_size,
        conv_4_out_reshaped, conv_dw_5_w, conv_dw_5_b, conv_dw_5_out, &conv_dw_5_params);

    // conv_6
    im2col(conv_6_params.batch_size, conv_6_params.in_channels, conv_6_params.in_dim,
        conv_6_params.I, conv_6_params.K,
        conv_dw_5_out, conv_6_in, &conv_6_params);

    tiled_matmul_compare(conv_6_params.I, conv_6_params.J, conv_6_params.K,    // dimensions
        conv_6_in, conv_6_w, conv_6_b, conv_6_out,      // addresses
        NO_ACTIVATION, conv_6_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_6");
        
    col2im(conv_6_params.I, conv_6_params.J, conv_6_params.batch_size, conv_6_params.out_channels, conv_6_params.out_dim,
        conv_6_out, conv_6_out_reshaped, &conv_6_params);

    // No need to add residuals here

    // conv_7
    im2col(conv_7_params.batch_size, conv_7_params.in_channels, conv_7_params.in_dim,
        conv_7_params.I, conv_7_params.K,
        conv_6_out_reshaped, conv_7_in, &conv_7_params);

    tiled_matmul_compare(conv_7_params.I, conv_7_params.J, conv_7_params.K,    // dimensions
        conv_7_in, conv_7_w, conv_7_b, conv_7_out,      // addresses
        RELU, conv_7_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_7");
        
    col2im(conv_7_params.I, conv_7_params.J, conv_7_params.batch_size, conv_7_params.out_channels, conv_7_params.out_dim,
        conv_7_out, conv_7_out_reshaped, &conv_7_params);

    // conv_dw_8
    conv_dw(conv_dw_8_params.batch_size, conv_dw_8_params.in_channels, conv_dw_8_params.in_dim, conv_dw_8_params.out_dim, conv_dw_8_params.kernel_size,
        conv_7_out_reshaped, conv_dw_8_w, conv_dw_8_b, conv_dw_8_out, &conv_dw_8_params);

    // conv_9
    im2col(conv_9_params.batch_size, conv_9_params.in_channels, conv_9_params.in_dim,
        conv_9_params.I, conv_9_params.K,
        conv_dw_8_out, conv_9_in, &conv_9_params);

    tiled_matmul_compare(conv_9_params.I, conv_9_params.J, conv_9_params.K,    // dimensions
        conv_9_in, conv_9_w, conv_9_b, conv_9_out,      // addresses
        NO_ACTIVATION, conv_9_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_9");
        
    col2im(conv_9_params.I, conv_9_params.J, conv_9_params.batch_size, conv_9_params.out_channels, conv_9_params.out_dim,
        conv_9_out, conv_9_out_reshaped, &conv_9_params);

    // Add residuals
    vecadd(sizeof(conv_6_out_reshaped) / sizeof(elem_t), (elem_t*)conv_6_out_reshaped, (elem_t*)conv_9_out_reshaped, (elem_t*)conv_9_out_reshaped, conv_9_params.res_scale);
    
    // conv_10
    im2col(conv_10_params.batch_size, conv_10_params.in_channels, conv_10_params.in_dim,
        conv_10_params.I, conv_10_params.K,
        conv_9_out_reshaped, conv_10_in, &conv_10_params);

    tiled_matmul_compare(conv_10_params.I, conv_10_params.J, conv_10_params.K,    // dimensions
        conv_10_in, conv_10_w, conv_10_b, conv_10_out,      // addresses
        RELU, conv_10_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_10");
        
    col2im(conv_10_params.I, conv_10_params.J, conv_10_params.batch_size, conv_10_params.out_channels, conv_10_params.out_dim,
        conv_10_out, conv_10_out_reshaped, &conv_10_params);

    // conv_dw_11
    conv_dw(conv_dw_11_params.batch_size, conv_dw_11_params.in_channels, conv_dw_11_params.in_dim, conv_dw_11_params.out_dim, conv_dw_11_params.kernel_size,
        conv_10_out_reshaped, conv_dw_11_w, conv_dw_11_b, conv_dw_11_out, &conv_dw_11_params);

    // conv_12
    im2col(conv_12_params.batch_size, conv_12_params.in_channels, conv_12_params.in_dim,
        conv_12_params.I, conv_12_params.K,
        conv_dw_11_out, conv_12_in, &conv_12_params);

    tiled_matmul_compare(conv_12_params.I, conv_12_params.J, conv_12_params.K,    // dimensions
        conv_12_in, conv_12_w, conv_12_b, conv_12_out,      // addresses
        NO_ACTIVATION, conv_12_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_12");
        
    col2im(conv_12_params.I, conv_12_params.J, conv_12_params.batch_size, conv_12_params.out_channels, conv_12_params.out_dim,
        conv_12_out, conv_12_out_reshaped, &conv_12_params);

    // No need to add residuals here

    // conv_13
    im2col(conv_13_params.batch_size, conv_13_params.in_channels, conv_13_params.in_dim,
        conv_13_params.I, conv_13_params.K,
        conv_12_out_reshaped, conv_13_in, &conv_13_params);

    tiled_matmul_compare(conv_13_params.I, conv_13_params.J, conv_13_params.K,    // dimensions
        conv_13_in, conv_13_w, conv_13_b, conv_13_out,      // addresses
        RELU, conv_13_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_13");
        
    col2im(conv_13_params.I, conv_13_params.J, conv_13_params.batch_size, conv_13_params.out_channels, conv_13_params.out_dim,
        conv_13_out, conv_13_out_reshaped, &conv_13_params);

    // conv_dw_14
    conv_dw(conv_dw_14_params.batch_size, conv_dw_14_params.in_channels, conv_dw_14_params.in_dim, conv_dw_14_params.out_dim, conv_dw_14_params.kernel_size,
        conv_13_out_reshaped, conv_dw_14_w, conv_dw_14_b, conv_dw_14_out, &conv_dw_14_params);

    // conv_15
    im2col(conv_15_params.batch_size, conv_15_params.in_channels, conv_15_params.in_dim,
        conv_15_params.I, conv_15_params.K,
        conv_dw_14_out, conv_15_in, &conv_15_params);

    tiled_matmul_compare(conv_15_params.I, conv_15_params.J, conv_15_params.K,    // dimensions
        conv_15_in, conv_15_w, conv_15_b, conv_15_out,      // addresses
        NO_ACTIVATION, conv_15_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_15");
        
    col2im(conv_15_params.I, conv_15_params.J, conv_15_params.batch_size, conv_15_params.out_channels, conv_15_params.out_dim,
        conv_15_out, conv_15_out_reshaped, &conv_15_params);

    // Add residuals
    vecadd(sizeof(conv_12_out_reshaped) / sizeof(elem_t), (elem_t*)conv_12_out_reshaped, (elem_t*)conv_15_out_reshaped, (elem_t*)conv_15_out_reshaped, conv_15_params.res_scale);
    
    // conv_16
    im2col(conv_16_params.batch_size, conv_16_params.in_channels, conv_16_params.in_dim,
        conv_16_params.I, conv_16_params.K,
        conv_15_out_reshaped, conv_16_in, &conv_16_params);

    tiled_matmul_compare(conv_16_params.I, conv_16_params.J, conv_16_params.K,    // dimensions
        conv_16_in, conv_16_w, conv_16_b, conv_16_out,      // addresses
        RELU, conv_16_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_16");
        
    col2im(conv_16_params.I, conv_16_params.J, conv_16_params.batch_size, conv_16_params.out_channels, conv_16_params.out_dim,
        conv_16_out, conv_16_out_reshaped, &conv_16_params);

    // conv_dw_17
    conv_dw(conv_dw_17_params.batch_size, conv_dw_17_params.in_channels, conv_dw_17_params.in_dim, conv_dw_17_params.out_dim, conv_dw_17_params.kernel_size,
        conv_16_out_reshaped, conv_dw_17_w, conv_dw_17_b, conv_dw_17_out, &conv_dw_17_params);

    // conv_18
    im2col(conv_18_params.batch_size, conv_18_params.in_channels, conv_18_params.in_dim,
        conv_18_params.I, conv_18_params.K,
        conv_dw_17_out, conv_18_in, &conv_18_params);

    tiled_matmul_compare(conv_18_params.I, conv_18_params.J, conv_18_params.K,    // dimensions
        conv_18_in, conv_18_w, conv_18_b, conv_18_out,      // addresses
        NO_ACTIVATION, conv_18_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_18");
        
    col2im(conv_18_params.I, conv_18_params.J, conv_18_params.batch_size, conv_18_params.out_channels, conv_18_params.out_dim,
        conv_18_out, conv_18_out_reshaped, &conv_18_params);

    // Add residuals
    vecadd(sizeof(conv_15_out_reshaped) / sizeof(elem_t), (elem_t*)conv_15_out_reshaped, (elem_t*)conv_18_out_reshaped, (elem_t*)conv_18_out_reshaped, conv_18_params.res_scale);
    
    // conv_19
    im2col(conv_19_params.batch_size, conv_19_params.in_channels, conv_19_params.in_dim,
        conv_19_params.I, conv_19_params.K,
        conv_18_out_reshaped, conv_19_in, &conv_19_params);

    tiled_matmul_compare(conv_19_params.I, conv_19_params.J, conv_19_params.K,    // dimensions
        conv_19_in, conv_19_w, conv_19_b, conv_19_out,      // addresses
        RELU, conv_19_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_19");
        
    col2im(conv_19_params.I, conv_19_params.J, conv_19_params.batch_size, conv_19_params.out_channels, conv_19_params.out_dim,
        conv_19_out, conv_19_out_reshaped, &conv_19_params);

    // conv_dw_20
    conv_dw(conv_dw_20_params.batch_size, conv_dw_20_params.in_channels, conv_dw_20_params.in_dim, conv_dw_20_params.out_dim, conv_dw_20_params.kernel_size,
        conv_19_out_reshaped, conv_dw_20_w, conv_dw_20_b, conv_dw_20_out, &conv_dw_20_params);

    // conv_21
    im2col(conv_21_params.batch_size, conv_21_params.in_channels, conv_21_params.in_dim,
        conv_21_params.I, conv_21_params.K,
        conv_dw_20_out, conv_21_in, &conv_21_params);

    tiled_matmul_compare(conv_21_params.I, conv_21_params.J, conv_21_params.K,    // dimensions
        conv_21_in, conv_21_w, conv_21_b, conv_21_out,      // addresses
        NO_ACTIVATION, conv_21_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_21");
        
    col2im(conv_21_params.I, conv_21_params.J, conv_21_params.batch_size, conv_21_params.out_channels, conv_21_params.out_dim,
        conv_21_out, conv_21_out_reshaped, &conv_21_params);

    // No need to add residuals here

    // conv_22
    im2col(conv_22_params.batch_size, conv_22_params.in_channels, conv_22_params.in_dim,
        conv_22_params.I, conv_22_params.K,
        conv_21_out_reshaped, conv_22_in, &conv_22_params);

    tiled_matmul_compare(conv_22_params.I, conv_22_params.J, conv_22_params.K,    // dimensions
        conv_22_in, conv_22_w, conv_22_b, conv_22_out,      // addresses
        RELU, conv_22_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_22");
        
    col2im(conv_22_params.I, conv_22_params.J, conv_22_params.batch_size, conv_22_params.out_channels, conv_22_params.out_dim,
        conv_22_out, conv_22_out_reshaped, &conv_22_params);

    // conv_dw_23
    conv_dw(conv_dw_23_params.batch_size, conv_dw_23_params.in_channels, conv_dw_23_params.in_dim, conv_dw_23_params.out_dim, conv_dw_23_params.kernel_size,
        conv_22_out_reshaped, conv_dw_23_w, conv_dw_23_b, conv_dw_23_out, &conv_dw_23_params);

    // conv_24
    im2col(conv_24_params.batch_size, conv_24_params.in_channels, conv_24_params.in_dim,
        conv_24_params.I, conv_24_params.K,
        conv_dw_23_out, conv_24_in, &conv_24_params);

    tiled_matmul_compare(conv_24_params.I, conv_24_params.J, conv_24_params.K,    // dimensions
        conv_24_in, conv_24_w, conv_24_b, conv_24_out,      // addresses
        NO_ACTIVATION, conv_24_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_24");
        
    col2im(conv_24_params.I, conv_24_params.J, conv_24_params.batch_size, conv_24_params.out_channels, conv_24_params.out_dim,
        conv_24_out, conv_24_out_reshaped, &conv_24_params);

    // Add residuals
    vecadd(sizeof(conv_21_out_reshaped) / sizeof(elem_t), (elem_t*)conv_21_out_reshaped, (elem_t*)conv_24_out_reshaped, (elem_t*)conv_24_out_reshaped, conv_24_params.res_scale);
    
    // conv_25
    im2col(conv_25_params.batch_size, conv_25_params.in_channels, conv_25_params.in_dim,
        conv_25_params.I, conv_25_params.K,
        conv_24_out_reshaped, conv_25_in, &conv_25_params);

    tiled_matmul_compare(conv_25_params.I, conv_25_params.J, conv_25_params.K,    // dimensions
        conv_25_in, conv_25_w, conv_25_b, conv_25_out,      // addresses
        RELU, conv_25_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_25");
        
    col2im(conv_25_params.I, conv_25_params.J, conv_25_params.batch_size, conv_25_params.out_channels, conv_25_params.out_dim,
        conv_25_out, conv_25_out_reshaped, &conv_25_params);

    // conv_dw_26
    conv_dw(conv_dw_26_params.batch_size, conv_dw_26_params.in_channels, conv_dw_26_params.in_dim, conv_dw_26_params.out_dim, conv_dw_26_params.kernel_size,
        conv_25_out_reshaped, conv_dw_26_w, conv_dw_26_b, conv_dw_26_out, &conv_dw_26_params);

    // conv_27
    im2col(conv_27_params.batch_size, conv_27_params.in_channels, conv_27_params.in_dim,
        conv_27_params.I, conv_27_params.K,
        conv_dw_26_out, conv_27_in, &conv_27_params);

    tiled_matmul_compare(conv_27_params.I, conv_27_params.J, conv_27_params.K,    // dimensions
        conv_27_in, conv_27_w, conv_27_b, conv_27_out,      // addresses
        NO_ACTIVATION, conv_27_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_27");
        
    col2im(conv_27_params.I, conv_27_params.J, conv_27_params.batch_size, conv_27_params.out_channels, conv_27_params.out_dim,
        conv_27_out, conv_27_out_reshaped, &conv_27_params);

    // Add residuals
    vecadd(sizeof(conv_24_out_reshaped) / sizeof(elem_t), (elem_t*)conv_24_out_reshaped, (elem_t*)conv_27_out_reshaped, (elem_t*)conv_27_out_reshaped, conv_27_params.res_scale);
    
    // conv_28
    im2col(conv_28_params.batch_size, conv_28_params.in_channels, conv_28_params.in_dim,
        conv_28_params.I, conv_28_params.K,
        conv_27_out_reshaped, conv_28_in, &conv_28_params);

    tiled_matmul_compare(conv_28_params.I, conv_28_params.J, conv_28_params.K,    // dimensions
        conv_28_in, conv_28_w, conv_28_b, conv_28_out,      // addresses
        RELU, conv_28_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_28");
        
    col2im(conv_28_params.I, conv_28_params.J, conv_28_params.batch_size, conv_28_params.out_channels, conv_28_params.out_dim,
        conv_28_out, conv_28_out_reshaped, &conv_28_params);

    // conv_dw_29
    conv_dw(conv_dw_29_params.batch_size, conv_dw_29_params.in_channels, conv_dw_29_params.in_dim, conv_dw_29_params.out_dim, conv_dw_29_params.kernel_size,
        conv_28_out_reshaped, conv_dw_29_w, conv_dw_29_b, conv_dw_29_out, &conv_dw_29_params);

    // conv_30
    im2col(conv_30_params.batch_size, conv_30_params.in_channels, conv_30_params.in_dim,
        conv_30_params.I, conv_30_params.K,
        conv_dw_29_out, conv_30_in, &conv_30_params);

    tiled_matmul_compare(conv_30_params.I, conv_30_params.J, conv_30_params.K,    // dimensions
        conv_30_in, conv_30_w, conv_30_b, conv_30_out,      // addresses
        NO_ACTIVATION, conv_30_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_30");
        
    col2im(conv_30_params.I, conv_30_params.J, conv_30_params.batch_size, conv_30_params.out_channels, conv_30_params.out_dim,
        conv_30_out, conv_30_out_reshaped, &conv_30_params);

    // Add residuals
    vecadd(sizeof(conv_27_out_reshaped) / sizeof(elem_t), (elem_t*)conv_27_out_reshaped, (elem_t*)conv_30_out_reshaped, (elem_t*)conv_30_out_reshaped, conv_30_params.res_scale);
    
    // conv_31
    im2col(conv_31_params.batch_size, conv_31_params.in_channels, conv_31_params.in_dim,
        conv_31_params.I, conv_31_params.K,
        conv_30_out_reshaped, conv_31_in, &conv_31_params);

    tiled_matmul_compare(conv_31_params.I, conv_31_params.J, conv_31_params.K,    // dimensions
        conv_31_in, conv_31_w, conv_31_b, conv_31_out,      // addresses
        RELU, conv_31_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_31");
        
    col2im(conv_31_params.I, conv_31_params.J, conv_31_params.batch_size, conv_31_params.out_channels, conv_31_params.out_dim,
        conv_31_out, conv_31_out_reshaped, &conv_31_params);

    // conv_dw_32
    conv_dw(conv_dw_32_params.batch_size, conv_dw_32_params.in_channels, conv_dw_32_params.in_dim, conv_dw_32_params.out_dim, conv_dw_32_params.kernel_size,
        conv_31_out_reshaped, conv_dw_32_w, conv_dw_32_b, conv_dw_32_out, &conv_dw_32_params);

    // conv_33
    im2col(conv_33_params.batch_size, conv_33_params.in_channels, conv_33_params.in_dim,
        conv_33_params.I, conv_33_params.K,
        conv_dw_32_out, conv_33_in, &conv_33_params);

    tiled_matmul_compare(conv_33_params.I, conv_33_params.J, conv_33_params.K,    // dimensions
        conv_33_in, conv_33_w, conv_33_b, conv_33_out,      // addresses
        NO_ACTIVATION, conv_33_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_33");
        
    col2im(conv_33_params.I, conv_33_params.J, conv_33_params.batch_size, conv_33_params.out_channels, conv_33_params.out_dim,
        conv_33_out, conv_33_out_reshaped, &conv_33_params);

    // No need to add residuals here

    // conv_34
    im2col(conv_34_params.batch_size, conv_34_params.in_channels, conv_34_params.in_dim,
        conv_34_params.I, conv_34_params.K,
        conv_33_out_reshaped, conv_34_in, &conv_34_params);

    tiled_matmul_compare(conv_34_params.I, conv_34_params.J, conv_34_params.K,    // dimensions
        conv_34_in, conv_34_w, conv_34_b, conv_34_out,      // addresses
        RELU, conv_34_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_34");
        
    col2im(conv_34_params.I, conv_34_params.J, conv_34_params.batch_size, conv_34_params.out_channels, conv_34_params.out_dim,
        conv_34_out, conv_34_out_reshaped, &conv_34_params);

    // conv_dw_35
    conv_dw(conv_dw_35_params.batch_size, conv_dw_35_params.in_channels, conv_dw_35_params.in_dim, conv_dw_35_params.out_dim, conv_dw_35_params.kernel_size,
        conv_34_out_reshaped, conv_dw_35_w, conv_dw_35_b, conv_dw_35_out, &conv_dw_35_params);

    // conv_36
    im2col(conv_36_params.batch_size, conv_36_params.in_channels, conv_36_params.in_dim,
        conv_36_params.I, conv_36_params.K,
        conv_dw_35_out, conv_36_in, &conv_36_params);

    tiled_matmul_compare(conv_36_params.I, conv_36_params.J, conv_36_params.K,    // dimensions
        conv_36_in, conv_36_w, conv_36_b, conv_36_out,      // addresses
        NO_ACTIVATION, conv_36_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_36");
        
    col2im(conv_36_params.I, conv_36_params.J, conv_36_params.batch_size, conv_36_params.out_channels, conv_36_params.out_dim,
        conv_36_out, conv_36_out_reshaped, &conv_36_params);

    // Add residuals
    vecadd(sizeof(conv_33_out_reshaped) / sizeof(elem_t), (elem_t*)conv_33_out_reshaped, (elem_t*)conv_36_out_reshaped, (elem_t*)conv_36_out_reshaped, conv_36_params.res_scale);
    
    // conv_37
    im2col(conv_37_params.batch_size, conv_37_params.in_channels, conv_37_params.in_dim,
        conv_37_params.I, conv_37_params.K,
        conv_36_out_reshaped, conv_37_in, &conv_37_params);

    tiled_matmul_compare(conv_37_params.I, conv_37_params.J, conv_37_params.K,    // dimensions
        conv_37_in, conv_37_w, conv_37_b, conv_37_out,      // addresses
        RELU, conv_37_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_37");
        
    col2im(conv_37_params.I, conv_37_params.J, conv_37_params.batch_size, conv_37_params.out_channels, conv_37_params.out_dim,
        conv_37_out, conv_37_out_reshaped, &conv_37_params);

    // conv_dw_38
    conv_dw(conv_dw_38_params.batch_size, conv_dw_38_params.in_channels, conv_dw_38_params.in_dim, conv_dw_38_params.out_dim, conv_dw_38_params.kernel_size,
        conv_37_out_reshaped, conv_dw_38_w, conv_dw_38_b, conv_dw_38_out, &conv_dw_38_params);

    // conv_39
    im2col(conv_39_params.batch_size, conv_39_params.in_channels, conv_39_params.in_dim,
        conv_39_params.I, conv_39_params.K,
        conv_dw_38_out, conv_39_in, &conv_39_params);

    tiled_matmul_compare(conv_39_params.I, conv_39_params.J, conv_39_params.K,    // dimensions
        conv_39_in, conv_39_w, conv_39_b, conv_39_out,      // addresses
        NO_ACTIVATION, conv_39_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_39");
        
    col2im(conv_39_params.I, conv_39_params.J, conv_39_params.batch_size, conv_39_params.out_channels, conv_39_params.out_dim,
        conv_39_out, conv_39_out_reshaped, &conv_39_params);

    // Add residuals
    vecadd(sizeof(conv_36_out_reshaped) / sizeof(elem_t), (elem_t*)conv_36_out_reshaped, (elem_t*)conv_39_out_reshaped, (elem_t*)conv_39_out_reshaped, conv_39_params.res_scale);
    
    // conv_40
    im2col(conv_40_params.batch_size, conv_40_params.in_channels, conv_40_params.in_dim,
        conv_40_params.I, conv_40_params.K,
        conv_39_out_reshaped, conv_40_in, &conv_40_params);

    tiled_matmul_compare(conv_40_params.I, conv_40_params.J, conv_40_params.K,    // dimensions
        conv_40_in, conv_40_w, conv_40_b, conv_40_out,      // addresses
        RELU, conv_40_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_40");
        
    col2im(conv_40_params.I, conv_40_params.J, conv_40_params.batch_size, conv_40_params.out_channels, conv_40_params.out_dim,
        conv_40_out, conv_40_out_reshaped, &conv_40_params);

    // conv_dw_41
    conv_dw(conv_dw_41_params.batch_size, conv_dw_41_params.in_channels, conv_dw_41_params.in_dim, conv_dw_41_params.out_dim, conv_dw_41_params.kernel_size,
        conv_40_out_reshaped, conv_dw_41_w, conv_dw_41_b, conv_dw_41_out, &conv_dw_41_params);

    // conv_42
    im2col(conv_42_params.batch_size, conv_42_params.in_channels, conv_42_params.in_dim,
        conv_42_params.I, conv_42_params.K,
        conv_dw_41_out, conv_42_in, &conv_42_params);

    tiled_matmul_compare(conv_42_params.I, conv_42_params.J, conv_42_params.K,    // dimensions
        conv_42_in, conv_42_w, conv_42_b, conv_42_out,      // addresses
        NO_ACTIVATION, conv_42_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_42");
        
    col2im(conv_42_params.I, conv_42_params.J, conv_42_params.batch_size, conv_42_params.out_channels, conv_42_params.out_dim,
        conv_42_out, conv_42_out_reshaped, &conv_42_params);

    // No need to add residuals here

    // conv_43
    im2col(conv_43_params.batch_size, conv_43_params.in_channels, conv_43_params.in_dim,
        conv_43_params.I, conv_43_params.K,
        conv_42_out_reshaped, conv_43_in, &conv_43_params);

    tiled_matmul_compare(conv_43_params.I, conv_43_params.J, conv_43_params.K,    // dimensions
        conv_43_in, conv_43_w, conv_43_b, conv_43_out,      // addresses
        RELU, conv_43_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_43");
        
    col2im(conv_43_params.I, conv_43_params.J, conv_43_params.batch_size, conv_43_params.out_channels, conv_43_params.out_dim,
        conv_43_out, conv_43_out_reshaped, &conv_43_params);

    // conv_dw_44
    conv_dw(conv_dw_44_params.batch_size, conv_dw_44_params.in_channels, conv_dw_44_params.in_dim, conv_dw_44_params.out_dim, conv_dw_44_params.kernel_size,
        conv_43_out_reshaped, conv_dw_44_w, conv_dw_44_b, conv_dw_44_out, &conv_dw_44_params);

    // conv_45
    im2col(conv_45_params.batch_size, conv_45_params.in_channels, conv_45_params.in_dim,
        conv_45_params.I, conv_45_params.K,
        conv_dw_44_out, conv_45_in, &conv_45_params);

    tiled_matmul_compare(conv_45_params.I, conv_45_params.J, conv_45_params.K,    // dimensions
        conv_45_in, conv_45_w, conv_45_b, conv_45_out,      // addresses
        NO_ACTIVATION, conv_45_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_45");
        
    col2im(conv_45_params.I, conv_45_params.J, conv_45_params.batch_size, conv_45_params.out_channels, conv_45_params.out_dim,
        conv_45_out, conv_45_out_reshaped, &conv_45_params);

    // Add residuals
    vecadd(sizeof(conv_42_out_reshaped) / sizeof(elem_t), (elem_t*)conv_42_out_reshaped, (elem_t*)conv_45_out_reshaped, (elem_t*)conv_45_out_reshaped, conv_45_params.res_scale);
    
    // conv_46
    im2col(conv_46_params.batch_size, conv_46_params.in_channels, conv_46_params.in_dim,
        conv_46_params.I, conv_46_params.K,
        conv_45_out_reshaped, conv_46_in, &conv_46_params);

    tiled_matmul_compare(conv_46_params.I, conv_46_params.J, conv_46_params.K,    // dimensions
        conv_46_in, conv_46_w, conv_46_b, conv_46_out,      // addresses
        RELU, conv_46_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_46");
        
    col2im(conv_46_params.I, conv_46_params.J, conv_46_params.batch_size, conv_46_params.out_channels, conv_46_params.out_dim,
        conv_46_out, conv_46_out_reshaped, &conv_46_params);

    // conv_dw_47
    conv_dw(conv_dw_47_params.batch_size, conv_dw_47_params.in_channels, conv_dw_47_params.in_dim, conv_dw_47_params.out_dim, conv_dw_47_params.kernel_size,
        conv_46_out_reshaped, conv_dw_47_w, conv_dw_47_b, conv_dw_47_out, &conv_dw_47_params);

    // conv_48
    im2col(conv_48_params.batch_size, conv_48_params.in_channels, conv_48_params.in_dim,
        conv_48_params.I, conv_48_params.K,
        conv_dw_47_out, conv_48_in, &conv_48_params);

    tiled_matmul_compare(conv_48_params.I, conv_48_params.J, conv_48_params.K,    // dimensions
        conv_48_in, conv_48_w, conv_48_b, conv_48_out,      // addresses
        NO_ACTIVATION, conv_48_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_48");
        
    col2im(conv_48_params.I, conv_48_params.J, conv_48_params.batch_size, conv_48_params.out_channels, conv_48_params.out_dim,
        conv_48_out, conv_48_out_reshaped, &conv_48_params);

    // Add residuals
    vecadd(sizeof(conv_45_out_reshaped) / sizeof(elem_t), (elem_t*)conv_45_out_reshaped, (elem_t*)conv_48_out_reshaped, (elem_t*)conv_48_out_reshaped, conv_48_params.res_scale);
    
    // conv_49
    im2col(conv_49_params.batch_size, conv_49_params.in_channels, conv_49_params.in_dim,
        conv_49_params.I, conv_49_params.K,
        conv_48_out_reshaped, conv_49_in, &conv_49_params);

    tiled_matmul_compare(conv_49_params.I, conv_49_params.J, conv_49_params.K,    // dimensions
        conv_49_in, conv_49_w, conv_49_b, conv_49_out,      // addresses
        RELU, conv_49_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_49");
        
    col2im(conv_49_params.I, conv_49_params.J, conv_49_params.batch_size, conv_49_params.out_channels, conv_49_params.out_dim,
        conv_49_out, conv_49_out_reshaped, &conv_49_params);

    // conv_dw_50
    conv_dw(conv_dw_50_params.batch_size, conv_dw_50_params.in_channels, conv_dw_50_params.in_dim, conv_dw_50_params.out_dim, conv_dw_50_params.kernel_size,
        conv_49_out_reshaped, conv_dw_50_w, conv_dw_50_b, conv_dw_50_out, &conv_dw_50_params);

    // conv_51
    im2col(conv_51_params.batch_size, conv_51_params.in_channels, conv_51_params.in_dim,
        conv_51_params.I, conv_51_params.K,
        conv_dw_50_out, conv_51_in, &conv_51_params);

    tiled_matmul_compare(conv_51_params.I, conv_51_params.J, conv_51_params.K,    // dimensions
        conv_51_in, conv_51_w, conv_51_b, conv_51_out,      // addresses
        NO_ACTIVATION, conv_51_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_51");
        
    col2im(conv_51_params.I, conv_51_params.J, conv_51_params.batch_size, conv_51_params.out_channels, conv_51_params.out_dim,
        conv_51_out, conv_51_out_reshaped, &conv_51_params);

    // No need to add residuals here

    // conv_52
    im2col(conv_52_params.batch_size, conv_52_params.in_channels, conv_52_params.in_dim,
        conv_52_params.I, conv_52_params.K,
        conv_51_out_reshaped, conv_52_in, &conv_52_params);

    tiled_matmul_compare(conv_52_params.I, conv_52_params.J, conv_52_params.K,    // dimensions
        conv_52_in, conv_52_w, conv_52_b, conv_52_out,      // addresses
        RELU, conv_52_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_52");
        
    col2im(conv_52_params.I, conv_52_params.J, conv_52_params.batch_size, conv_52_params.out_channels, conv_52_params.out_dim,
        conv_52_out, conv_52_out_reshaped, &conv_52_params);

    // Global averaging
    // static elem_t average[fc_53_params.K][fc_53_params.J] row_align(1) = {0};
    static elem_t average[1280][16] row_align(1) = {0};

    for (int batch = 0; batch < conv_52_params.batch_size; batch++) {
        for (int channel = 0; channel < conv_52_params.out_channels; channel++) {
            int sum = 0;
            for (int row = 0; row < conv_52_params.out_dim; row++) {
                for (int col = 0; col < conv_52_params.out_dim; col++) {
                    sum += conv_52_out_reshaped[batch][channel][row][col];
                }
            }
            int count = conv_52_params.out_dim * conv_52_params.out_dim;

            average[channel][batch] = (sum + count/2) / count;
        }
    }

    // fc_53
    tiled_matmul_compare(fc_53_params.I, fc_53_params.J, fc_53_params.K,
        fc_53_w, average, fc_53_b, fc_53_out,
        NO_ACTIVATION, fc_53_params.output_scale, 0, 1,
        tiled_matmul_type, compare, "fc_53");

    // Make predictions
    char * ground_truth[] = {"paper_towel", "killer_whale", "hammer", "rock_beauty"};
    char * predictions[] = {"Egyptian_cat", "killer_whale", "maraca", "flatworm"};

    for (int batch = 0; batch < fc_53_params.batch_size; batch++) {
        elem_t max_prob = fc_53_out[0][batch];
        size_t max_idx = 0;
        
        for (int i = 1; i < fc_53_params.out_features; i++) {
            if (fc_53_out[i][batch] > max_prob) {
                max_prob = fc_53_out[i][batch];
                max_idx = i;
            }
        }
        
        printf("Class prediction: %u (%s) (score: %d)\n", max_idx, predictions[batch], max_prob);
    }

    exit(0);
}

