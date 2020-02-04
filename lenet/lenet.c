#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"

#include "lenet_params.h"
#include "lenet_images.h"

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

// Pooling
void pool(size_t batch_size, size_t channels, size_t in_dim, size_t out_dim,
    elem_t input[batch_size][channels][in_dim][in_dim],
    elem_t output[batch_size][channels][out_dim][out_dim],
    size_t kernel_size, size_t stride)
{
    // We assume that the padding is 0 for this function

    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * stride;

                    elem_t result = elem_t_min;

                    for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                        int in_col = out_col * stride;

                        for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < in_dim && in_col >= 0 && in_col < in_dim) {
                                if (input[batch][channel][in_row][in_col] > result) {
                                    result = input[batch][channel][in_row][in_col];
                                }
                            }

                            in_col++;
                        }

                        in_row++;
                    }
                    
                    output[batch][channel][out_row][out_col] = result;
                }
            }
        }
    }
}

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type;
    tiled_matmul_type = WS;

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
        conv_1_in, conv_1_w, NULL, conv_1_out,      // addresses
        RELU, conv_1_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_1");
        
    col2im(conv_1_params.I, conv_1_params.J, conv_1_params.batch_size, conv_1_params.out_channels, conv_1_params.out_dim,
        conv_1_out, conv_1_out_reshaped, &conv_1_params);

    pool(conv_1_params.batch_size, conv_1_params.in_channels, conv_1_params.out_dim, conv_1_params.out_dim_pooled,
        conv_1_out_reshaped, conv_1_out_pooled, conv_1_params.pool_size, conv_1_params.pool_stride);


    // conv_2
    im2col(conv_2_params.batch_size, conv_2_params.in_channels, conv_2_params.in_dim,
        conv_2_params.I, conv_2_params.K,
        conv_1_out_pooled, conv_2_in, &conv_2_params);

    tiled_matmul_compare(conv_2_params.I, conv_2_params.J, conv_2_params.K,    // dimensions
        conv_2_in, conv_2_w, NULL, conv_2_out,      // addresses
        RELU, conv_2_params.output_scale, 0, 1,              // activation, shift, r6_shift, full_width_bias
        tiled_matmul_type, compare, "conv_2");

    col2im(conv_2_params.I, conv_2_params.J, conv_2_params.batch_size, conv_2_params.out_channels, conv_2_params.out_dim,
        conv_2_out, conv_2_out_reshaped, &conv_2_params);

    pool(conv_2_params.batch_size, conv_2_params.in_channels, conv_2_params.out_dim, conv_2_params.out_dim_pooled,
        conv_2_out_reshaped, conv_2_out_pooled, conv_2_params.pool_size, conv_2_params.pool_stride);

    // Convert conv output to fc input
    // static elem_t fc_3_in[fc_3_params.K][fc_3_params.J];
    static elem_t fc_3_in[400][16];
    for (size_t batch = 0; batch < conv_2_params.batch_size; batch++) {
        size_t pixel = 0;
        for (size_t channel = 0; channel < conv_2_params.out_channels; channel++) {
            for (size_t row = 0; row < conv_2_params.out_dim_pooled; row++) {
                for (size_t col = 0; col < conv_2_params.out_dim_pooled; col++) {
                    fc_3_in[pixel][batch] = conv_2_out_pooled[batch][channel][row][col];
                    pixel++;
                }
            }
        }
    }

    // fc_3
    tiled_matmul_compare(fc_3_params.I, fc_3_params.J, fc_3_params.K,
        fc_3_w, fc_3_in, NULL, fc_3_out,
        RELU, fc_3_params.output_scale, 0, 1,
        tiled_matmul_type, compare, "fc_3");



    // fc_4
    tiled_matmul_compare(fc_4_params.I, fc_4_params.J, fc_4_params.K,
        fc_4_w, fc_3_out, NULL, fc_4_out,
        RELU, fc_4_params.output_scale, 0, 1,
        tiled_matmul_type, compare, "fc_4");



    // fc_5
    tiled_matmul_compare(fc_5_params.I, fc_5_params.J, fc_5_params.K,
        fc_5_w, fc_4_out, fc_5_b, fc_5_out,
        NO_ACTIVATION, fc_5_params.output_scale, 0, 1,
        tiled_matmul_type, compare, "fc_5");



    // Find highest probs
    char * classes[] = {"plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

    for (int batch = 0; batch < fc_5_params.batch_size; batch++) {
        elem_t max_prob = fc_5_out[batch][0];
        size_t max_idx = 0;

        for (int i = 1; i < fc_5_params.out_features; i++) {
            if (fc_5_out[batch][i] > max_prob) {
                max_prob = fc_5_out[batch][i];
                max_idx = i;
            }
        }
        
        printf("Class prediction: %u (score: %d) (class: %s)\n", max_idx, max_prob, classes[max_idx]);
    }

    exit(0);
}

