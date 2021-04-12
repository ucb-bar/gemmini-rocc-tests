#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "googlenet_params.h"
#include "googlenet_images.h"

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
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool check;
    if (argc < 3) {
        check = false;
    } else if (strcmp(argv[2], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool conv;
    if (argc < 4) {
        // conv = false;
        conv = true;
    } else if (strcmp(argv[3], "conv") == 0) {
        conv = true;
    } else if (strcmp(argv[3], "matmul") == 0) {
        conv = false;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check] [conv]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;

    // HIST_IMAGES(images);
    // exit(1);

    // conv_1
    if (!conv) {
      start = read_cycles();

        im2col(conv_1_params.batch_size, conv_1_params.in_channels, conv_1_params.in_dim,
            conv_1_params.I, conv_1_params.K,
            images, conv_1_in, &conv_1_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_1_params.I, conv_1_params.J, conv_1_params.K,
            conv_1_in, conv_1_w, conv_1_b, conv_1_out,
            // NO_ACTIVATION, conv_1_params.output_scale, 0, true,
            RELU, conv_1_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_1");

        end = read_cycles();
        matmul_cycles += end - start;

        start = read_cycles();

        pool_with_col2im(conv_1_params.I, conv_1_params.J,
            conv_1_params.batch_size, conv_1_params.out_channels, conv_1_params.out_dim_pooled,
            conv_1_out, conv_1_out_pooled, &conv_1_params);

        end = read_cycles();
        pool_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_1_params.batch_size, conv_1_params.in_dim, conv_1_params.in_channels,
            conv_1_params.out_channels, conv_1_params.out_dim,
            conv_1_params.stride, 1, conv_1_params.padding, conv_1_params.kernel_size,
            false,

            (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out_pooled,

            RELU, conv_1_params.output_scale, 0,
            conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // HIST_MATRIX(conv_1_out);
    // exit(1);

    // HIST_IMAGES(conv_1_out_pooled);
    // exit(1);

    // conv_2
    if (!conv) {
      start = read_cycles();

        im2col(conv_2_params.batch_size, conv_2_params.in_channels, conv_2_params.in_dim,
            conv_2_params.I, conv_2_params.K,
            conv_1_out_pooled, conv_2_in, &conv_2_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_2_params.I, conv_2_params.J, conv_2_params.K,
            conv_2_in, conv_2_w, conv_2_b, conv_2_out,
            RELU, conv_2_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_2");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_2_params.I, conv_2_params.J, conv_2_params.K,
            conv_1_out_pooled, conv_2_w, conv_2_b, conv_2_out,
            RELU, conv_2_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_2");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_3
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_2_params.I, conv_2_params.J,
            conv_3_params.I, conv_3_params.K,
            conv_2_out, conv_3_in, &conv_3_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_3_params.I, conv_3_params.J, conv_3_params.K,
            conv_3_in, conv_3_w, conv_3_b, conv_3_out,
            RELU, conv_3_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_3");

        end = read_cycles();
        matmul_cycles += end - start;

        start = read_cycles();

        pool_with_col2im(conv_3_params.I, conv_3_params.J,
            conv_3_params.batch_size, conv_3_params.out_channels, conv_3_params.out_dim_pooled,
            conv_3_out, conv_3_out_pooled, &conv_3_params);

        end = read_cycles();
        pool_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_3_params.batch_size, conv_3_params.in_dim, conv_3_params.in_channels,
            conv_3_params.out_channels, conv_3_params.out_dim,
            conv_3_params.stride, 1, conv_3_params.padding, conv_3_params.kernel_size,
            false,

            (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out_pooled,

            RELU, conv_3_params.output_scale, 0,
            conv_3_params.pool_size, conv_3_params.pool_stride, conv_3_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    HIST_IMAGES(conv_3_out_pooled);
    exit(1);

    // conv_4
    if (!conv) {
      start = read_cycles();

        im2col(conv_4_params.batch_size, conv_4_params.in_channels, conv_4_params.in_dim,
            conv_4_params.I, conv_4_params.K,
            conv_3_out_pooled, conv_4_in, &conv_4_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            conv_4_in, conv_4_w, conv_4_b, conv_4_out,
            RELU, conv_4_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_4");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            conv_3_out_pooled, conv_4_w, conv_4_b, conv_4_out,
            RELU, conv_4_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_4");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_5
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_5_params.I, conv_5_params.J, conv_5_params.K,
            conv_4_out, conv_5_w, conv_5_b, conv_5_out,
            RELU, conv_5_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_5");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_5_params.I, conv_5_params.J, conv_5_params.K,
            conv_4_out, conv_5_w, conv_5_b, conv_5_out,
            RELU, conv_5_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_5");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_6
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_5_params.I, conv_5_params.J,
            conv_6_params.I, conv_6_params.K,
            conv_5_out, conv_6_in, &conv_6_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
            conv_6_in, conv_6_w, conv_6_b, conv_6_out,
            RELU, conv_6_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_6");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_6_params.batch_size, conv_6_params.in_dim, conv_6_params.in_channels,
            conv_6_params.out_channels, conv_6_params.out_dim,
            conv_6_params.stride, 1, conv_6_params.padding, conv_6_params.kernel_size,
            false,

            (elem_t*)conv_5_out, (elem_t*)conv_6_w, (acc_t*)conv_6_b, (elem_t*)conv_6_out,

            RELU, conv_6_params.output_scale, 0,
            conv_6_params.pool_size, 0, conv_6_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_7
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_7_params.I, conv_7_params.J, conv_7_params.K,
            conv_6_out, conv_7_w, conv_7_b, conv_7_out,
            RELU, conv_7_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_7");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_7_params.I, conv_7_params.J, conv_7_params.K,
            conv_6_out, conv_7_w, conv_7_b, conv_7_out,
            RELU, conv_7_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_7");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_8
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_7_params.I, conv_7_params.J,
            conv_8_params.I, conv_8_params.K,
            conv_7_out, conv_8_in, &conv_8_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_8_params.I, conv_8_params.J, conv_8_params.K,
            conv_8_in, conv_8_w, conv_8_b, conv_8_out,
            RELU, conv_8_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_8");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_8_params.batch_size, conv_8_params.in_dim, conv_8_params.in_channels,
            conv_8_params.out_channels, conv_8_params.out_dim,
            conv_8_params.stride, 1, conv_8_params.padding, conv_8_params.kernel_size,
            false,

            (elem_t*)conv_7_out, (elem_t*)conv_8_w, (acc_t*)conv_8_b, (elem_t*)conv_8_out,

            RELU, conv_8_params.output_scale, 0,
            conv_8_params.pool_size, 0, conv_8_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_9
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_9_params.I, conv_9_params.J, conv_9_params.K,
            conv_8_out, conv_9_w, conv_9_b, conv_9_out,
            RELU, conv_9_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_9");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_9_params.I, conv_9_params.J, conv_9_params.K,
            conv_8_out, conv_9_w, conv_9_b, conv_9_out,
            RELU, conv_9_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_9");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_10
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_10_params.I, conv_10_params.J, conv_10_params.K,
            conv_9_out, conv_10_w, conv_10_b, conv_10_out,
            RELU, conv_10_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_10");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_10_params.I, conv_10_params.J, conv_10_params.K,
            conv_9_out, conv_10_w, conv_10_b, conv_10_out,
            RELU, conv_10_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_10");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_11
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_11_params.I, conv_11_params.J, conv_11_params.K,
            conv_10_out, conv_11_w, conv_11_b, conv_11_out,
            RELU, conv_11_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_11");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_11_params.I, conv_11_params.J, conv_11_params.K,
            conv_10_out, conv_11_w, conv_11_b, conv_11_out,
            RELU, conv_11_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_11");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_12
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_11_params.I, conv_11_params.J,
            conv_12_params.I, conv_12_params.K,
            conv_11_out, conv_12_in, &conv_12_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_12_in, conv_12_w, conv_12_b, conv_12_out,
            RELU, conv_12_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_12");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_12_params.batch_size, conv_12_params.in_dim, conv_12_params.in_channels,
            conv_12_params.out_channels, conv_12_params.out_dim,
            conv_12_params.stride, 1, conv_12_params.padding, conv_12_params.kernel_size,
            false,

            (elem_t*)conv_11_out, (elem_t*)conv_12_w, (acc_t*)conv_12_b, (elem_t*)conv_12_out,

            RELU, conv_12_params.output_scale, 0,
            conv_12_params.pool_size, 0, conv_12_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_13
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_13_params.I, conv_13_params.J, conv_13_params.K,
            conv_12_out, conv_13_w, conv_13_b, conv_13_out,
            RELU, conv_13_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_13");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_13_params.I, conv_13_params.J, conv_13_params.K,
            conv_12_out, conv_13_w, conv_13_b, conv_13_out,
            RELU, conv_13_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_13");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_14
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_13_params.I, conv_13_params.J,
            conv_14_params.I, conv_14_params.K,
            conv_13_out, conv_14_in, &conv_14_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_14_params.I, conv_14_params.J, conv_14_params.K,
            conv_14_in, conv_14_w, conv_14_b, conv_14_out,
            RELU, conv_14_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_14");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_14_params.batch_size, conv_14_params.in_dim, conv_14_params.in_channels,
            conv_14_params.out_channels, conv_14_params.out_dim,
            conv_14_params.stride, 1, conv_14_params.padding, conv_14_params.kernel_size,
            false,

            (elem_t*)conv_13_out, (elem_t*)conv_14_w, (acc_t*)conv_14_b, (elem_t*)conv_14_out,

            RELU, conv_14_params.output_scale, 0,
            conv_14_params.pool_size, 0, conv_14_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_15
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_15_params.I, conv_15_params.J, conv_15_params.K,
            conv_14_out, conv_15_w, conv_15_b, conv_15_out,
            RELU, conv_15_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_15");

        end = read_cycles();
        matmul_cycles += end - start;

      start = read_cycles();

        pool_with_col2im(conv_15_params.I, conv_15_params.J,
            conv_15_params.batch_size, conv_15_params.out_channels, conv_15_params.out_dim_pooled,
            conv_15_out, conv_15_out_pooled, &conv_15_params);

        end = read_cycles();
        pool_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_15_params.batch_size, conv_15_params.in_dim, conv_15_params.in_channels,
            conv_15_params.out_channels, conv_15_params.out_dim,
            conv_15_params.stride, 1, conv_15_params.padding, conv_15_params.kernel_size,
            false,

            (elem_t*)conv_14_out, (elem_t*)conv_15_w, (acc_t*)conv_15_b, (elem_t*)conv_15_out_pooled,

            RELU, conv_15_params.output_scale, 0,
            conv_15_params.pool_size, conv_15_params.pool_stride, conv_15_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_16
    if (!conv) {
      start = read_cycles();

        im2col(conv_16_params.batch_size, conv_16_params.in_channels, conv_16_params.in_dim,
            conv_16_params.I, conv_16_params.K,
            conv_15_out_pooled, conv_16_in, &conv_16_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_16_params.I, conv_16_params.J, conv_16_params.K,
            conv_16_in, conv_16_w, conv_16_b, conv_16_out,
            RELU, conv_16_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_16");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_16_params.I, conv_16_params.J, conv_16_params.K,
            conv_15_out_pooled, conv_16_w, conv_16_b, conv_16_out,
            RELU, conv_16_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_16");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_17
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_17_params.I, conv_17_params.J, conv_17_params.K,
            conv_16_out, conv_17_w, conv_17_b, conv_17_out,
            RELU, conv_17_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_17");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_17_params.I, conv_17_params.J, conv_17_params.K,
            conv_16_out, conv_17_w, conv_17_b, conv_17_out,
            RELU, conv_17_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_17");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_18
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_17_params.I, conv_17_params.J,
            conv_18_params.I, conv_18_params.K,
            conv_17_out, conv_18_in, &conv_18_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K,
            conv_18_in, conv_18_w, conv_18_b, conv_18_out,
            RELU, conv_18_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_18");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_18_params.batch_size, conv_18_params.in_dim, conv_18_params.in_channels,
            conv_18_params.out_channels, conv_18_params.out_dim,
            conv_18_params.stride, 1, conv_18_params.padding, conv_18_params.kernel_size,
            false,

            (elem_t*)conv_17_out, (elem_t*)conv_18_w, (acc_t*)conv_18_b, (elem_t*)conv_18_out,

            RELU, conv_18_params.output_scale, 0,
            conv_18_params.pool_size, 0, conv_18_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_19
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            conv_18_out, conv_19_w, conv_19_b, conv_19_out,
            RELU, conv_19_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_19");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            conv_18_out, conv_19_w, conv_19_b, conv_19_out,
            RELU, conv_19_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_19");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_20
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_19_params.I, conv_19_params.J,
            conv_20_params.I, conv_20_params.K,
            conv_19_out, conv_20_in, &conv_20_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_20_params.I, conv_20_params.J, conv_20_params.K,
            conv_20_in, conv_20_w, conv_20_b, conv_20_out,
            RELU, conv_20_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_20");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_20_params.batch_size, conv_20_params.in_dim, conv_20_params.in_channels,
            conv_20_params.out_channels, conv_20_params.out_dim,
            conv_20_params.stride, 1, conv_20_params.padding, conv_20_params.kernel_size,
            false,

            (elem_t*)conv_19_out, (elem_t*)conv_20_w, (acc_t*)conv_20_b, (elem_t*)conv_20_out,

            RELU, conv_20_params.output_scale, 0,
            conv_20_params.pool_size, 0, conv_20_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_21
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_21_params.I, conv_21_params.J, conv_21_params.K,
            conv_20_out, conv_21_w, conv_21_b, conv_21_out,
            RELU, conv_21_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_21");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_21_params.I, conv_21_params.J, conv_21_params.K,
            conv_20_out, conv_21_w, conv_21_b, conv_21_out,
            RELU, conv_21_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_21");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_22
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_21_out, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_22");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_21_out, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_22");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_23
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_23_params.I, conv_23_params.J, conv_23_params.K,
            conv_22_out, conv_23_w, conv_23_b, conv_23_out,
            RELU, conv_23_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_23");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_23_params.I, conv_23_params.J, conv_23_params.K,
            conv_22_out, conv_23_w, conv_23_b, conv_23_out,
            RELU, conv_23_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_23");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_24
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_23_params.I, conv_23_params.J,
            conv_24_params.I, conv_24_params.K,
            conv_23_out, conv_24_in, &conv_24_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K,
            conv_24_in, conv_24_w, conv_24_b, conv_24_out,
            RELU, conv_24_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_24");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_24_params.batch_size, conv_24_params.in_dim, conv_24_params.in_channels,
            conv_24_params.out_channels, conv_24_params.out_dim,
            conv_24_params.stride, 1, conv_24_params.padding, conv_24_params.kernel_size,
            false,

            (elem_t*)conv_23_out, (elem_t*)conv_24_w, (acc_t*)conv_24_b, (elem_t*)conv_24_out,

            RELU, conv_24_params.output_scale, 0,
            conv_24_params.pool_size, 0, conv_24_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_25
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            conv_24_out, conv_25_w, conv_25_b, conv_25_out,
            RELU, conv_25_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_25");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            conv_24_out, conv_25_w, conv_25_b, conv_25_out,
            RELU, conv_25_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_25");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_26
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_25_params.I, conv_25_params.J,
            conv_26_params.I, conv_26_params.K,
            conv_25_out, conv_26_in, &conv_26_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_26_params.I, conv_26_params.J, conv_26_params.K,
            conv_26_in, conv_26_w, conv_26_b, conv_26_out,
            RELU, conv_26_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_26");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_26_params.batch_size, conv_26_params.in_dim, conv_26_params.in_channels,
            conv_26_params.out_channels, conv_26_params.out_dim,
            conv_26_params.stride, 1, conv_26_params.padding, conv_26_params.kernel_size,
            false,

            (elem_t*)conv_25_out, (elem_t*)conv_26_w, (acc_t*)conv_26_b, (elem_t*)conv_26_out,

            RELU, conv_26_params.output_scale, 0,
            conv_26_params.pool_size, 0, conv_26_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_27
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_27_params.I, conv_27_params.J, conv_27_params.K,
            conv_26_out, conv_27_w, conv_27_b, conv_27_out,
            RELU, conv_27_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_27");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_27_params.I, conv_27_params.J, conv_27_params.K,
            conv_26_out, conv_27_w, conv_27_b, conv_27_out,
            RELU, conv_27_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_27");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_28
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_28_params.I, conv_28_params.J, conv_28_params.K,
            conv_27_out, conv_28_w, conv_28_b, conv_28_out,
            RELU, conv_28_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_28");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_28_params.I, conv_28_params.J, conv_28_params.K,
            conv_27_out, conv_28_w, conv_28_b, conv_28_out,
            RELU, conv_28_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_28");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_29
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_29_params.I, conv_29_params.J, conv_29_params.K,
            conv_28_out, conv_29_w, conv_29_b, conv_29_out,
            RELU, conv_29_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_29");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_29_params.I, conv_29_params.J, conv_29_params.K,
            conv_28_out, conv_29_w, conv_29_b, conv_29_out,
            RELU, conv_29_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_29");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_30
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_29_params.I, conv_29_params.J,
            conv_30_params.I, conv_30_params.K,
            conv_29_out, conv_30_in, &conv_30_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_30_params.I, conv_30_params.J, conv_30_params.K,
            conv_30_in, conv_30_w, conv_30_b, conv_30_out,
            RELU, conv_30_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_30");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_30_params.batch_size, conv_30_params.in_dim, conv_30_params.in_channels,
            conv_30_params.out_channels, conv_30_params.out_dim,
            conv_30_params.stride, 1, conv_30_params.padding, conv_30_params.kernel_size,
            false,

            (elem_t*)conv_29_out, (elem_t*)conv_30_w, (acc_t*)conv_30_b, (elem_t*)conv_30_out,

            RELU, conv_30_params.output_scale, 0,
            conv_30_params.pool_size, 0, conv_30_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_31
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_31_params.I, conv_31_params.J, conv_31_params.K,
            conv_30_out, conv_31_w, conv_31_b, conv_31_out,
            RELU, conv_31_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_31");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_31_params.I, conv_31_params.J, conv_31_params.K,
            conv_30_out, conv_31_w, conv_31_b, conv_31_out,
            RELU, conv_31_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_31");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_32
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_31_params.I, conv_31_params.J,
            conv_32_params.I, conv_32_params.K,
            conv_31_out, conv_32_in, &conv_32_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_32_params.I, conv_32_params.J, conv_32_params.K,
            conv_32_in, conv_32_w, conv_32_b, conv_32_out,
            RELU, conv_32_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_32");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_32_params.batch_size, conv_32_params.in_dim, conv_32_params.in_channels,
            conv_32_params.out_channels, conv_32_params.out_dim,
            conv_32_params.stride, 1, conv_32_params.padding, conv_32_params.kernel_size,
            false,

            (elem_t*)conv_31_out, (elem_t*)conv_32_w, (acc_t*)conv_32_b, (elem_t*)conv_32_out,

            RELU, conv_32_params.output_scale, 0,
            conv_32_params.pool_size, 0, conv_32_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_33
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_33_params.I, conv_33_params.J, conv_33_params.K,
            conv_32_out, conv_33_w, conv_33_b, conv_33_out,
            RELU, conv_33_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_33");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_33_params.I, conv_33_params.J, conv_33_params.K,
            conv_32_out, conv_33_w, conv_33_b, conv_33_out,
            RELU, conv_33_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_33");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_34
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_34_params.I, conv_34_params.J, conv_34_params.K,
            conv_33_out, conv_34_w, conv_34_b, conv_34_out,
            RELU, conv_34_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_34");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_34_params.I, conv_34_params.J, conv_34_params.K,
            conv_33_out, conv_34_w, conv_34_b, conv_34_out,
            RELU, conv_34_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_34");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_35
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_35_params.I, conv_35_params.J, conv_35_params.K,
            conv_34_out, conv_35_w, conv_35_b, conv_35_out,
            RELU, conv_35_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_35");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_35_params.I, conv_35_params.J, conv_35_params.K,
            conv_34_out, conv_35_w, conv_35_b, conv_35_out,
            RELU, conv_35_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_35");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_36
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_35_params.I, conv_35_params.J,
            conv_36_params.I, conv_36_params.K,
            conv_35_out, conv_36_in, &conv_36_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_36_params.I, conv_36_params.J, conv_36_params.K,
            conv_36_in, conv_36_w, conv_36_b, conv_36_out,
            RELU, conv_36_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_36");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_36_params.batch_size, conv_36_params.in_dim, conv_36_params.in_channels,
            conv_36_params.out_channels, conv_36_params.out_dim,
            conv_36_params.stride, 1, conv_36_params.padding, conv_36_params.kernel_size,
            false,

            (elem_t*)conv_35_out, (elem_t*)conv_36_w, (acc_t*)conv_36_b, (elem_t*)conv_36_out,

            RELU, conv_36_params.output_scale, 0,
            conv_36_params.pool_size, 0, conv_36_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_37
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_37_params.I, conv_37_params.J, conv_37_params.K,
            conv_36_out, conv_37_w, conv_37_b, conv_37_out,
            RELU, conv_37_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_37");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_37_params.I, conv_37_params.J, conv_37_params.K,
            conv_36_out, conv_37_w, conv_37_b, conv_37_out,
            RELU, conv_37_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_37");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_38
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_37_params.I, conv_37_params.J,
            conv_38_params.I, conv_38_params.K,
            conv_37_out, conv_38_in, &conv_38_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_38_params.I, conv_38_params.J, conv_38_params.K,
            conv_38_in, conv_38_w, conv_38_b, conv_38_out,
            RELU, conv_38_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_38");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_38_params.batch_size, conv_38_params.in_dim, conv_38_params.in_channels,
            conv_38_params.out_channels, conv_38_params.out_dim,
            conv_38_params.stride, 1, conv_38_params.padding, conv_38_params.kernel_size,
            false,

            (elem_t*)conv_37_out, (elem_t*)conv_38_w, (acc_t*)conv_38_b, (elem_t*)conv_38_out,

            RELU, conv_38_params.output_scale, 0,
            conv_38_params.pool_size, 0, conv_38_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_39
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_39_params.I, conv_39_params.J, conv_39_params.K,
            conv_38_out, conv_39_w, conv_39_b, conv_39_out,
            RELU, conv_39_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_39");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_39_params.I, conv_39_params.J, conv_39_params.K,
            conv_38_out, conv_39_w, conv_39_b, conv_39_out,
            RELU, conv_39_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_39");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_40
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_40_params.I, conv_40_params.J, conv_40_params.K,
            conv_39_out, conv_40_w, conv_40_b, conv_40_out,
            RELU, conv_40_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_40");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_40_params.I, conv_40_params.J, conv_40_params.K,
            conv_39_out, conv_40_w, conv_40_b, conv_40_out,
            RELU, conv_40_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_40");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_41
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_41_params.I, conv_41_params.J, conv_41_params.K,
            conv_40_out, conv_41_w, conv_41_b, conv_41_out,
            RELU, conv_41_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_41");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_41_params.I, conv_41_params.J, conv_41_params.K,
            conv_40_out, conv_41_w, conv_41_b, conv_41_out,
            RELU, conv_41_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_41");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_42
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_41_params.I, conv_41_params.J,
            conv_42_params.I, conv_42_params.K,
            conv_41_out, conv_42_in, &conv_42_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_42_params.I, conv_42_params.J, conv_42_params.K,
            conv_42_in, conv_42_w, conv_42_b, conv_42_out,
            RELU, conv_42_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_42");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_42_params.batch_size, conv_42_params.in_dim, conv_42_params.in_channels,
            conv_42_params.out_channels, conv_42_params.out_dim,
            conv_42_params.stride, 1, conv_42_params.padding, conv_42_params.kernel_size,
            false,

            (elem_t*)conv_41_out, (elem_t*)conv_42_w, (acc_t*)conv_42_b, (elem_t*)conv_42_out,

            RELU, conv_42_params.output_scale, 0,
            conv_42_params.pool_size, 0, conv_42_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_43
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_43_params.I, conv_43_params.J, conv_43_params.K,
            conv_42_out, conv_43_w, conv_43_b, conv_43_out,
            RELU, conv_43_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_43");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_43_params.I, conv_43_params.J, conv_43_params.K,
            conv_42_out, conv_43_w, conv_43_b, conv_43_out,
            RELU, conv_43_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_43");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_44
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_43_params.I, conv_43_params.J,
            conv_44_params.I, conv_44_params.K,
            conv_43_out, conv_44_in, &conv_44_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_44_params.I, conv_44_params.J, conv_44_params.K,
            conv_44_in, conv_44_w, conv_44_b, conv_44_out,
            RELU, conv_44_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_44");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_44_params.batch_size, conv_44_params.in_dim, conv_44_params.in_channels,
            conv_44_params.out_channels, conv_44_params.out_dim,
            conv_44_params.stride, 1, conv_44_params.padding, conv_44_params.kernel_size,
            false,

            (elem_t*)conv_43_out, (elem_t*)conv_44_w, (acc_t*)conv_44_b, (elem_t*)conv_44_out,

            RELU, conv_44_params.output_scale, 0,
            conv_44_params.pool_size, 0, conv_44_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_45
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_45_params.I, conv_45_params.J, conv_45_params.K,
            conv_44_out, conv_45_w, conv_45_b, conv_45_out,
            RELU, conv_45_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_45");

        end = read_cycles();
        matmul_cycles += end - start;

      start = read_cycles();

        pool_with_col2im(conv_45_params.I, conv_45_params.J,
            conv_45_params.batch_size, conv_45_params.out_channels, conv_45_params.out_dim_pooled,
            conv_45_out, conv_45_out_pooled, &conv_45_params);

        end = read_cycles();
        pool_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_45_params.batch_size, conv_45_params.in_dim, conv_45_params.in_channels,
            conv_45_params.out_channels, conv_45_params.out_dim,
            conv_45_params.stride, 1, conv_45_params.padding, conv_45_params.kernel_size,
            false,

            (elem_t*)conv_44_out, (elem_t*)conv_45_w, (acc_t*)conv_45_b, (elem_t*)conv_45_out_pooled,

            RELU, conv_45_params.output_scale, 0,
            conv_45_params.pool_size, conv_45_params.pool_stride, conv_45_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_46
    if (!conv) {
      start = read_cycles();

        im2col(conv_46_params.batch_size, conv_46_params.in_channels, conv_46_params.in_dim,
            conv_46_params.I, conv_46_params.K,
            conv_45_out_pooled, conv_46_in, &conv_46_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_46_params.I, conv_46_params.J, conv_46_params.K,
            conv_46_in, conv_46_w, conv_46_b, conv_46_out,
            RELU, conv_46_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_46");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_46_params.I, conv_46_params.J, conv_46_params.K,
            conv_45_out_pooled, conv_46_w, conv_46_b, conv_46_out,
            RELU, conv_46_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_46");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_47
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_47_params.I, conv_47_params.J, conv_47_params.K,
            conv_46_out, conv_47_w, conv_47_b, conv_47_out,
            RELU, conv_47_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_47");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_47_params.I, conv_47_params.J, conv_47_params.K,
            conv_46_out, conv_47_w, conv_47_b, conv_47_out,
            RELU, conv_47_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_47");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_48
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_47_params.I, conv_47_params.J,
            conv_48_params.I, conv_48_params.K,
            conv_47_out, conv_48_in, &conv_48_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_48_params.I, conv_48_params.J, conv_48_params.K,
            conv_48_in, conv_48_w, conv_48_b, conv_48_out,
            RELU, conv_48_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_48");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_48_params.batch_size, conv_48_params.in_dim, conv_48_params.in_channels,
            conv_48_params.out_channels, conv_48_params.out_dim,
            conv_48_params.stride, 1, conv_48_params.padding, conv_48_params.kernel_size,
            false,

            (elem_t*)conv_47_out, (elem_t*)conv_48_w, (acc_t*)conv_48_b, (elem_t*)conv_48_out,

            RELU, conv_48_params.output_scale, 0,
            conv_48_params.pool_size, 0, conv_48_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_49
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_49_params.I, conv_49_params.J, conv_49_params.K,
            conv_48_out, conv_49_w, conv_49_b, conv_49_out,
            RELU, conv_49_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_49");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_49_params.I, conv_49_params.J, conv_49_params.K,
            conv_48_out, conv_49_w, conv_49_b, conv_49_out,
            RELU, conv_49_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_49");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_50
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_49_params.I, conv_49_params.J,
            conv_50_params.I, conv_50_params.K,
            conv_49_out, conv_50_in, &conv_50_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_50_params.I, conv_50_params.J, conv_50_params.K,
            conv_50_in, conv_50_w, conv_50_b, conv_50_out,
            RELU, conv_50_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_50");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_50_params.batch_size, conv_50_params.in_dim, conv_50_params.in_channels,
            conv_50_params.out_channels, conv_50_params.out_dim,
            conv_50_params.stride, 1, conv_50_params.padding, conv_50_params.kernel_size,
            false,

            (elem_t*)conv_49_out, (elem_t*)conv_50_w, (acc_t*)conv_50_b, (elem_t*)conv_50_out,

            RELU, conv_50_params.output_scale, 0,
            conv_50_params.pool_size, 0, conv_50_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_51
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_51_params.I, conv_51_params.J, conv_51_params.K,
            conv_50_out, conv_51_w, conv_51_b, conv_51_out,
            RELU, conv_51_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_51");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_51_params.I, conv_51_params.J, conv_51_params.K,
            conv_50_out, conv_51_w, conv_51_b, conv_51_out,
            RELU, conv_51_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_51");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_52
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_52_params.I, conv_52_params.J, conv_52_params.K,
            conv_51_out, conv_52_w, conv_52_b, conv_52_out,
            RELU, conv_52_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_52");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_52_params.I, conv_52_params.J, conv_52_params.K,
            conv_51_out, conv_52_w, conv_52_b, conv_52_out,
            RELU, conv_52_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_52");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_53
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_53_params.I, conv_53_params.J, conv_53_params.K,
            conv_52_out, conv_53_w, conv_53_b, conv_53_out,
            RELU, conv_53_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_53");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_53_params.I, conv_53_params.J, conv_53_params.K,
            conv_52_out, conv_53_w, conv_53_b, conv_53_out,
            RELU, conv_53_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_53");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_54
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_53_params.I, conv_53_params.J,
            conv_54_params.I, conv_54_params.K,
            conv_53_out, conv_54_in, &conv_54_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_54_params.I, conv_54_params.J, conv_54_params.K,
            conv_54_in, conv_54_w, conv_54_b, conv_54_out,
            RELU, conv_54_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_54");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_54_params.batch_size, conv_54_params.in_dim, conv_54_params.in_channels,
            conv_54_params.out_channels, conv_54_params.out_dim,
            conv_54_params.stride, 1, conv_54_params.padding, conv_54_params.kernel_size,
            false,

            (elem_t*)conv_53_out, (elem_t*)conv_54_w, (acc_t*)conv_54_b, (elem_t*)conv_54_out,

            RELU, conv_54_params.output_scale, 0,
            conv_54_params.pool_size, 0, conv_54_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_55
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_55_params.I, conv_55_params.J, conv_55_params.K,
            conv_54_out, conv_55_w, conv_55_b, conv_55_out,
            RELU, conv_55_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_55");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_55_params.I, conv_55_params.J, conv_55_params.K,
            conv_54_out, conv_55_w, conv_55_b, conv_55_out,
            RELU, conv_55_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_55");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_56
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_55_params.I, conv_55_params.J,
            conv_56_params.I, conv_56_params.K,
            conv_55_out, conv_56_in, &conv_56_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_56_params.I, conv_56_params.J, conv_56_params.K,
            conv_56_in, conv_56_w, conv_56_b, conv_56_out,
            RELU, conv_56_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_56");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_A_stride_auto(
            conv_56_params.batch_size, conv_56_params.in_dim, conv_56_params.in_channels,
            conv_56_params.out_channels, conv_56_params.out_dim,
            conv_56_params.stride, 1, conv_56_params.padding, conv_56_params.kernel_size,
            false,

            (elem_t*)conv_55_out, (elem_t*)conv_56_w, (acc_t*)conv_56_b, (elem_t*)conv_56_out,

            RELU, conv_56_params.output_scale, 0,
            conv_56_params.pool_size, 0, conv_56_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_57
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_57_params.I, conv_57_params.J, conv_57_params.K,
            conv_56_out, conv_57_w, conv_57_b, conv_57_out,
            RELU, conv_57_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_57");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_57_params.I, conv_57_params.J, conv_57_params.K,
            conv_56_out, conv_57_w, conv_57_b, conv_57_out,
            RELU, conv_57_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_57");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // fc_58
    start = read_cycles();

    tiled_matmul_nn_auto(fc_58_params.I, fc_58_params.J, fc_58_params.K,
        fc_58_w, conv_57_out, fc_58_b, fc_58_out,
        NO_ACTIVATION, fc_58_params.output_scale, 0, false,
        tiled_matmul_type, check, "fc_58");

    end = read_cycles();
    matmul_cycles += end - start;

    // Find highest probs
    for (int batch = 0; batch < fc_58_params.batch_size; batch++) {
        elem_t max_prob = fc_58_out[0][batch];
        size_t max_idx = 0;

        for (int i = 1; i < fc_58_params.out_features; i++) {
            if (fc_58_out[i][batch] > max_prob) {
                max_prob = fc_58_out[i][batch];
                max_idx = i;
            }
        }

        printf("Prediction: %u (score: %d)\n", max_idx, max_prob);
    }

    uint64_t total_cycles = im2col_cycles + matmul_cycles + pool_cycles + conv_dw_cycles + res_add_cycles + other_cycles;

    printf("\nTotal cycles: %llu (100%%)\n", total_cycles);
    printf("Matmul cycles: %llu (%d%%)\n", matmul_cycles, (matmul_cycles * 100) / total_cycles);
    printf("Im2col cycles: %llu (%d%%)\n", im2col_cycles, (im2col_cycles * 100) / total_cycles);
    printf("Conv cycles: %llu (%d%%)\n", conv_cycles, (conv_cycles * 100) / total_cycles);
    printf("Pooling cycles: %llu (%d%%)\n", pool_cycles, (pool_cycles * 100) / total_cycles);
    printf("Depthwise convolution cycles: %llu (%d%%)\n", conv_dw_cycles, (conv_dw_cycles * 100) / total_cycles);
    printf("Res add cycles: %llu (%d%%)\n", res_add_cycles, (res_add_cycles * 100) / total_cycles);
    printf("Other cycles: %llu (%d%%)\n", other_cycles, (other_cycles * 100) / total_cycles);

    exit(0);
}

