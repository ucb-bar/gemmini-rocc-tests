#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "squeezenet1_params.h"
#include "squeezenet1_images.h"

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
        conv = false;
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

        tiled_conv_auto(
            conv_1_params.batch_size, conv_1_params.in_dim, conv_1_params.in_channels,
            conv_1_params.out_channels, conv_1_params.out_dim,
            conv_1_params.stride, conv_1_params.padding, conv_1_params.kernel_size,

            (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out_pooled,

            RELU, conv_1_params.output_scale, 0,
            conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

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

        tiled_matmul_nn_auto(conv_3_params.I, conv_3_params.J, conv_3_params.K,
            conv_2_out, conv_3_w, conv_3_b, conv_3_out,
            RELU, conv_3_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_3");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_3_params.I, conv_3_params.J, conv_3_params.K,
            conv_2_out, conv_3_w, conv_3_b, conv_3_out,
            RELU, conv_3_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_3");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_4
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_3_params.I, conv_3_params.J,
            conv_4_params.I, conv_4_params.K,
            conv_3_out, conv_4_in, &conv_4_params);

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

        tiled_conv_auto(
            conv_4_params.batch_size, conv_4_params.in_dim, conv_4_params.in_channels,
            conv_4_params.out_channels, conv_4_params.out_dim,
            conv_4_params.stride, conv_4_params.padding, conv_4_params.kernel_size,

            (elem_t*)conv_3_out, (elem_t*)conv_4_w, (acc_t*)conv_4_b, (elem_t*)conv_4_out,

            RELU, conv_4_params.output_scale, 0,
            conv_4_params.pool_size, 0, conv_4_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
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

        tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
            conv_5_out, conv_6_w, conv_6_b, conv_6_out,
            RELU, conv_6_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_6");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
            conv_5_out, conv_6_w, conv_6_b, conv_6_out,
            RELU, conv_6_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_6");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_7
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_6_params.I, conv_6_params.J,
            conv_7_params.I, conv_7_params.K,
            conv_6_out, conv_7_in, &conv_7_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_7_params.I, conv_7_params.J, conv_7_params.K,
            conv_7_in, conv_7_w, conv_7_b, conv_7_out,
            RELU, conv_7_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_7");

        end = read_cycles();
        matmul_cycles += end - start;

      start = read_cycles();

        pool_with_col2im(conv_7_params.I, conv_7_params.J,
            conv_7_params.batch_size, conv_7_params.out_channels, conv_7_params.out_dim_pooled,
            conv_7_out, conv_7_out_pooled, &conv_7_params);

        end = read_cycles();
        pool_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_7_params.batch_size, conv_7_params.in_dim, conv_7_params.in_channels,
            conv_7_params.out_channels, conv_7_params.out_dim,
            conv_7_params.stride, conv_7_params.padding, conv_7_params.kernel_size,

            (elem_t*)conv_6_out, (elem_t*)conv_7_w, (acc_t*)conv_7_b, (elem_t*)conv_7_out_pooled,

            RELU, conv_7_params.output_scale, 0,
            conv_7_params.pool_size, conv_7_params.pool_stride, conv_7_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_8
    if (!conv) {
      start = read_cycles();

        im2col(conv_8_params.batch_size, conv_8_params.in_channels, conv_8_params.in_dim,
            conv_8_params.I, conv_8_params.K,
            conv_7_out_pooled, conv_8_in, &conv_8_params);

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

        tiled_matmul_nn_auto(conv_8_params.I, conv_8_params.J, conv_8_params.K,
            conv_7_out_pooled, conv_8_w, conv_8_b, conv_8_out,
            RELU, conv_8_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_8");

        end = read_cycles();
        matmul_cycles += end - start;
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

        im2col_with_col2im(conv_9_params.I, conv_9_params.J,
            conv_10_params.I, conv_10_params.K,
            conv_9_out, conv_10_in, &conv_10_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_10_params.I, conv_10_params.J, conv_10_params.K,
            conv_10_in, conv_10_w, conv_10_b, conv_10_out,
            RELU, conv_10_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_10");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_10_params.batch_size, conv_10_params.in_dim, conv_10_params.in_channels,
            conv_10_params.out_channels, conv_10_params.out_dim,
            conv_10_params.stride, conv_10_params.padding, conv_10_params.kernel_size,

            (elem_t*)conv_9_out, (elem_t*)conv_10_w, (acc_t*)conv_10_b, (elem_t*)conv_10_out,

            RELU, conv_10_params.output_scale, 0,
            conv_10_params.pool_size, 0, conv_10_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
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

        tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_11_out, conv_12_w, conv_12_b, conv_12_out,
            RELU, conv_12_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_12");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_11_out, conv_12_w, conv_12_b, conv_12_out,
            RELU, conv_12_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_12");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_13
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_12_params.I, conv_12_params.J,
            conv_13_params.I, conv_13_params.K,
            conv_12_out, conv_13_in, &conv_13_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_13_params.I, conv_13_params.J, conv_13_params.K,
            conv_13_in, conv_13_w, conv_13_b, conv_13_out,
            RELU, conv_13_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_13");

        end = read_cycles();
        matmul_cycles += end - start;

      start = read_cycles();

        pool_with_col2im(conv_13_params.I, conv_13_params.J,
            conv_13_params.batch_size, conv_13_params.out_channels, conv_13_params.out_dim_pooled,
            conv_13_out, conv_13_out_pooled, &conv_13_params);

        end = read_cycles();
        pool_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_13_params.batch_size, conv_13_params.in_dim, conv_13_params.in_channels,
            conv_13_params.out_channels, conv_13_params.out_dim,
            conv_13_params.stride, conv_13_params.padding, conv_13_params.kernel_size,

            (elem_t*)conv_12_out, (elem_t*)conv_13_w, (acc_t*)conv_13_b, (elem_t*)conv_13_out_pooled,

            RELU, conv_13_params.output_scale, 0,
            conv_13_params.pool_size, conv_13_params.pool_stride, conv_13_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_14
    if (!conv) {
      start = read_cycles();

        im2col(conv_14_params.batch_size, conv_14_params.in_channels, conv_14_params.in_dim,
            conv_14_params.I, conv_14_params.K,
            conv_13_out_pooled, conv_14_in, &conv_14_params);

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

        tiled_matmul_nn_auto(conv_14_params.I, conv_14_params.J, conv_14_params.K,
            conv_13_out_pooled, conv_14_w, conv_14_b, conv_14_out,
            RELU, conv_14_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_14");

        end = read_cycles();
        matmul_cycles += end - start;
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

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_15_params.I, conv_15_params.J, conv_15_params.K,
            conv_14_out, conv_15_w, conv_15_b, conv_15_out,
            RELU, conv_15_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_15");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_16
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_15_params.I, conv_15_params.J,
            conv_16_params.I, conv_16_params.K,
            conv_15_out, conv_16_in, &conv_16_params);

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

        tiled_conv_auto(
            conv_16_params.batch_size, conv_16_params.in_dim, conv_16_params.in_channels,
            conv_16_params.out_channels, conv_16_params.out_dim,
            conv_16_params.stride, conv_16_params.padding, conv_16_params.kernel_size,

            (elem_t*)conv_15_out, (elem_t*)conv_16_w, (acc_t*)conv_16_b, (elem_t*)conv_16_out,

            RELU, conv_16_params.output_scale, 0,
            conv_16_params.pool_size, 0, conv_16_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
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

        tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K,
            conv_17_out, conv_18_w, conv_18_b, conv_18_out,
            RELU, conv_18_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_18");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K,
            conv_17_out, conv_18_w, conv_18_b, conv_18_out,
            RELU, conv_18_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_18");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_19
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_18_params.I, conv_18_params.J,
            conv_19_params.I, conv_19_params.K,
            conv_18_out, conv_19_in, &conv_19_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            conv_19_in, conv_19_w, conv_19_b, conv_19_out,
            RELU, conv_19_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_19");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_19_params.batch_size, conv_19_params.in_dim, conv_19_params.in_channels,
            conv_19_params.out_channels, conv_19_params.out_dim,
            conv_19_params.stride, conv_19_params.padding, conv_19_params.kernel_size,

            (elem_t*)conv_18_out, (elem_t*)conv_19_w, (acc_t*)conv_19_b, (elem_t*)conv_19_out,

            RELU, conv_19_params.output_scale, 0,
            conv_19_params.pool_size, 0, conv_19_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_20
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_20_params.I, conv_20_params.J, conv_20_params.K,
            conv_19_out, conv_20_w, conv_20_b, conv_20_out,
            RELU, conv_20_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_20");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_20_params.I, conv_20_params.J, conv_20_params.K,
            conv_19_out, conv_20_w, conv_20_b, conv_20_out,
            RELU, conv_20_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_20");

        end = read_cycles();
        matmul_cycles += end - start;
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

        im2col_with_col2im(conv_21_params.I, conv_21_params.J,
            conv_22_params.I, conv_22_params.K,
            conv_21_out, conv_22_in, &conv_22_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_22_in, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_22");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_22_params.batch_size, conv_22_params.in_dim, conv_22_params.in_channels,
            conv_22_params.out_channels, conv_22_params.out_dim,
            conv_22_params.stride, conv_22_params.padding, conv_22_params.kernel_size,

            (elem_t*)conv_21_out, (elem_t*)conv_22_w, (acc_t*)conv_22_b, (elem_t*)conv_22_out,

            RELU, conv_22_params.output_scale, 0,
            conv_22_params.pool_size, 0, conv_22_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
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

        tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K,
            conv_23_out, conv_24_w, conv_24_b, conv_24_out,
            RELU, conv_24_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_24");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K,
            conv_23_out, conv_24_w, conv_24_b, conv_24_out,
            RELU, conv_24_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_24");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_25
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_24_params.I, conv_24_params.J,
            conv_25_params.I, conv_25_params.K,
            conv_24_out, conv_25_in, &conv_25_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            conv_25_in, conv_25_w, conv_25_b, conv_25_out,
            RELU, conv_25_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_25");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_25_params.batch_size, conv_25_params.in_dim, conv_25_params.in_channels,
            conv_25_params.out_channels, conv_25_params.out_dim,
            conv_25_params.stride, conv_25_params.padding, conv_25_params.kernel_size,

            (elem_t*)conv_24_out, (elem_t*)conv_25_w, (acc_t*)conv_25_b, (elem_t*)conv_25_out,

            RELU, conv_25_params.output_scale, 0,
            conv_25_params.pool_size, 0, conv_25_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_26
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_26_params.I, conv_26_params.J, conv_26_params.K,
            conv_25_out, conv_26_w, conv_26_b, conv_26_out,
            RELU, conv_26_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_26");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_26_params.I, conv_26_params.J, conv_26_params.K,
            conv_25_out, conv_26_w, conv_26_b, conv_26_out,
            RELU, conv_26_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_26");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // Global averaging
    static elem_t average[1000][4] row_align(1);

    start = read_cycles();

    for (int batch = 0; batch < conv_26_params.batch_size; batch++) {
        for (int channel = 0; channel < conv_26_params.out_channels; channel++) {
            int sum = 0;
            for (int row = 0; row < conv_26_params.out_dim; row++) {
                for (int col = 0; col < conv_26_params.out_dim; col++) {
                    size_t r = batch * conv_26_params.out_dim * conv_26_params.out_dim + row * conv_26_params.out_dim + col;

                    sum += conv_26_out[r][channel];
                }
            }
            const int count = conv_26_params.out_dim * conv_26_params.out_dim;

            average[channel][batch] = (sum + count/2) / count;
        }
    }

    end = read_cycles();
    other_cycles += end - start;



    // Find highest probs
    for (int batch = 0; batch < conv_26_params.batch_size; batch++) {
        elem_t max_prob = average[0][batch];
        size_t max_idx = 0;

        for (int i = 1; i < conv_26_params.out_channels; i++) {
            if (average[i][batch] > max_prob) {
                max_prob = average[i][batch];
                max_idx = i;
            }
        }

        printf("Prediction: %u (score: %d)\n", max_idx, max_prob);
    }

    uint64_t total_cycles = im2col_cycles + matmul_cycles + pool_cycles + conv_dw_cycles + res_add_cycles + other_cycles + conv_cycles;

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

