#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "cifar_quant_params.h"
#include "cifar_quant_images.h"

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type;
    if (argc < 2) {
        // printf("usage: %s matmul_option check\n  matmul_option may be 'os', 'ws', or cpu'\n");
        // exit(0);
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    }

    bool check;
    if (argc < 3) {
        check = false;
    } else if (strcmp(argv[2], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        exit(1);
    }

    unsigned long start, end;
    unsigned long im2col_cycles = 0, matmul_cycles = 0, pool_cycles = 0 , other_cycles = 0;

    // conv_1
    start = read_cycles();

    im2col(conv_1_params.batch_size, conv_1_params.in_channels, conv_1_params.in_dim,
        conv_1_params.I, conv_1_params.K,
        images, conv_1_in, &conv_1_params);

    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_1_params.I, conv_1_params.J, conv_1_params.K,
        conv_1_in, conv_1_w, NULL, conv_1_out,
        RELU, conv_1_params.output_scale, false,
        tiled_matmul_type, check, "conv_1");

    end = read_cycles();
    matmul_cycles += end - start;

    start = read_cycles();

    pool_with_col2im(conv_1_params.I, conv_1_params.J,
        conv_1_params.batch_size, conv_1_params.out_channels, conv_1_params.out_dim_pooled,
        conv_1_out, conv_1_out_pooled, &conv_1_params);

    end = read_cycles();
    pool_cycles += end - start;

    // conv_2
    start = read_cycles();

    im2col(conv_2_params.batch_size, conv_2_params.in_channels, conv_2_params.in_dim,
        conv_2_params.I, conv_2_params.K,
        conv_1_out_pooled, conv_2_in, &conv_2_params);

    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_2_params.I, conv_2_params.J, conv_2_params.K,
        conv_2_in, conv_2_w, NULL, conv_2_out,
        RELU, conv_2_params.output_scale, false,
        tiled_matmul_type, check, "conv_2");

    end = read_cycles();
    matmul_cycles += end - start;

    start = read_cycles();

    pool_with_col2im(conv_2_params.I, conv_2_params.J,
        conv_2_params.batch_size, conv_2_params.out_channels, conv_2_params.out_dim_pooled,
        conv_2_out, conv_2_out_pooled, &conv_2_params);

    end = read_cycles();
    pool_cycles += end - start;

    // Convert conv output to fc input
    start = read_cycles();

    static elem_t fc_3_in[400][8];
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

    end = read_cycles();
    other_cycles += end - start;

    // fc_3
    start = read_cycles();

    tiled_matmul_nn_auto(fc_3_params.I, fc_3_params.J, fc_3_params.K,
        fc_3_w, fc_3_in, NULL, fc_3_out,
        RELU, fc_3_params.output_scale, false,
        tiled_matmul_type, check, "fc_3");

    // fc_4
    tiled_matmul_nn_auto(fc_4_params.I, fc_4_params.J, fc_4_params.K,
        fc_4_w, fc_3_out, NULL, fc_4_out,
        RELU, fc_4_params.output_scale, false,
        tiled_matmul_type, check, "fc_4");

    // fc_5
    tiled_matmul_nn_auto(fc_5_params.I, fc_5_params.J, fc_5_params.K,
        fc_5_w, fc_4_out, NULL, fc_5_out,
        NO_ACTIVATION, fc_5_params.output_scale, false,
        tiled_matmul_type, check, "fc_5");

    end = read_cycles();
    matmul_cycles += end - start;

    // Find highest probs
    char * classes[] = {"plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    int preds[fc_5_params.batch_size];

    for (int batch = 0; batch < fc_5_params.batch_size; batch++) {
        elem_t max_prob = fc_5_out[0][batch];
        size_t max_idx = 0;

        for (int i = 1; i < fc_5_params.out_features; i++) {
            if (fc_5_out[i][batch] > max_prob) {
                max_prob = fc_5_out[i][batch];
                max_idx = i;
            }
        }

        printf("Prediction: %u (score: %d) (class: %s)\n", max_idx, max_prob, classes[max_idx]);
        preds[batch] = max_idx;
    }

    unsigned long total_cycles = im2col_cycles + matmul_cycles + pool_cycles + other_cycles;

    printf("\nTotal cycles: %u\nMatmul cycles: %u\nIm2col cycles: %u\nPooling cycles: %u\nOther cycles: %u\n", total_cycles, matmul_cycles, im2col_cycles, pool_cycles, other_cycles);

    int correct[] = {6, 5, 1, 3};
    for (int i = 0; i < fc_5_params.batch_size; i++) {
        if (preds[i] != correct[i]) {
            printf("Prediction %d is incorrect!\nFAIL\n", i);
            exit(1);
        }
    }

    printf("PASS\n");

    exit(0);
}

