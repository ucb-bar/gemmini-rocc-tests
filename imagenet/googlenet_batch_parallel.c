#include "include/gemmini_mt.h"
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

void * thread_main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    const enum tiled_matmul_type_t tiled_matmul_type = WS;
    const bool conv = true;
    const bool check = false;
    bool quiet = false;

    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;

    uint64_t conv_1_cycles, matmul_2_cycles, conv_3_cycles, matmul_4_cycles,
             matmul_5_cycles, conv_6_cycles, matmul_7_cycles, conv_8_cycles,
             pool_9_cycles, matmul_10_cycles, matmul_11_cycles,
             matmul_12_cycles, conv_13_cycles, matmul_14_cycles, conv_15_cycles,
             pool_16_cyles, matmul_17_cycles, pool_18_cycles, matmul_19_cycles,
             matmul_20_cycles, conv_21_cycles, matmul_22_cycles, conv_23_cycles,
             pool_24_cycles, matmul_25_cycles, matmul_26_cycles,
             matmul_27_cycles, conv_28_cycles, matmul_29_cycles, conv_30_cycles,
             pool_31_cycles, matmul_32_cycles, matmul_33_cycles,
             matmul_34_cycles, conv_35_cycles, matmul_36_cycles, conv_37_cycles,
             pool_38_cycles, matmul_39_cycles, matmul_40_cycles,
             matmul_41_cycles, conv_42_cycles, matmul_43_cycles, conv_44_cycles,
             pool_45_cycles, matmul_46_cycles, matmul_47_cycles,
             matmul_48_cycles, conv_49_cycles, matmul_50_cycles, conv_51_cycles,
             pool_52_cycles, matmul_53_cycles, pool_54_cycles, matmul_55_cycles,
             matmul_56_cycles, conv_57_cycles, matmul_58_cycles, conv_59_cycles,
             pool_60_cycles, matmul_60_cycles, matmul_62_cycles,
             matmul_63_cycles, conv_64_cycles, matmul_65_cycles, conv_66_cycles,
             pool_67_cycles, matmul_67_cycles, fc_69_cycles;

    // conv_1
    if (!conv) {
        start = read_cycles();

        im2col(conv_1_params.batch_size, conv_1_params.in_channels, conv_1_params.in_dim,
            conv_1_params.I, conv_1_params.K,
            images, conv_1_in, &conv_1_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_1_params.I, conv_1_params.J, conv_1_params.K,
            conv_1_params.J,
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

        tiled_conv_batch_parallel_auto(
            conv_1_params.batch_size, conv_1_params.in_dim, conv_1_params.in_channels,
            conv_1_params.out_channels, conv_1_params.out_dim,
            conv_1_params.stride, 1, conv_1_params.padding, conv_1_params.kernel_size,
            false,
            
            conv_1_params.out_channels,

            (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out_pooled,

            RELU, conv_1_params.output_scale, 0,
            conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;

        conv_1_cycles = end - start;
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

        tiled_matmul_nn_auto_extended(conv_2_params.I, conv_2_params.J, conv_2_params.K,
            conv_2_params.J,
            conv_2_in, conv_2_w, conv_2_b, conv_2_out,
            RELU, conv_2_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_2");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_2_params.I, conv_2_params.J, conv_2_params.K,
            conv_2_params.J,
            conv_1_out_pooled, conv_2_w, conv_2_b, conv_2_out,
            RELU, conv_2_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_2");

        end = read_cycles();
        matmul_cycles += end - start;

        matmul_2_cycles = end - start;
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

        tiled_matmul_nn_auto_extended(conv_3_params.I, conv_3_params.J, conv_3_params.K,
            conv_3_params.J,
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

        tiled_conv_batch_parallel_auto(
            conv_3_params.batch_size, conv_3_params.in_dim, conv_3_params.in_channels,
            conv_3_params.out_channels, conv_3_params.out_dim,
            conv_3_params.stride, 1, conv_3_params.padding, conv_3_params.kernel_size,
            false,
            
            conv_3_params.out_channels,

            (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out_pooled,

            RELU, conv_3_params.output_scale, 0,
            conv_3_params.pool_size, conv_3_params.pool_stride, conv_3_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;

        conv_3_cycles = end - start;
    }

    // Inception 3a
    // Branch 1
    // conv_4
    if (!conv) {
        start = read_cycles();

        im2col(conv_4_params.batch_size, conv_4_params.in_channels, conv_4_params.in_dim,
            conv_4_params.I, conv_4_params.K,
            conv_3_out_pooled, conv_4_in, &conv_4_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            256,
            conv_4_in, conv_4_w, conv_4_b, ((elem_t*)inception3a_out + 0),
            RELU, conv_4_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_4");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            256,
            conv_3_out_pooled, conv_4_w, conv_4_b, ((elem_t*)inception3a_out + 0),
            RELU, conv_4_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_4");

        end = read_cycles();
        matmul_cycles += end - start;

        matmul_4_cycles = end - start;
    }

    // Branch 2
    // conv_5
    if (!conv) {
        start = read_cycles();

        im2col(conv_5_params.batch_size, conv_5_params.in_channels, conv_5_params.in_dim,
            conv_5_params.I, conv_5_params.K,
            conv_3_out_pooled, conv_5_in, &conv_5_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_5_params.I, conv_5_params.J, conv_5_params.K,
            conv_5_params.J,
            conv_5_in, conv_5_w, conv_5_b, conv_5_out,
            RELU, conv_5_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_5");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_5_params.I, conv_5_params.J, conv_5_params.K,
            conv_5_params.J,
            conv_3_out_pooled, conv_5_w, conv_5_b, conv_5_out,
            RELU, conv_5_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_5");

        end = read_cycles();
        matmul_cycles += end - start;

        matmul_5_cycles = end - start;
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

        tiled_matmul_nn_auto_extended(conv_6_params.I, conv_6_params.J, conv_6_params.K,
            256,
            conv_6_in, conv_6_w, conv_6_b, ((elem_t*)inception3a_out + 64),
            RELU, conv_6_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_6");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_6_params.batch_size, conv_6_params.in_dim, conv_6_params.in_channels,
            conv_6_params.out_channels, conv_6_params.out_dim,
            conv_6_params.stride, 1, conv_6_params.padding, conv_6_params.kernel_size,
            false,
            
            256,

            (elem_t*)conv_5_out, (elem_t*)conv_6_w, (acc_t*)conv_6_b, (elem_t*)((elem_t*)inception3a_out + 64),

            RELU, conv_6_params.output_scale, 0,
            conv_6_params.pool_size, 0, conv_6_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;

        conv_6_cycles = end - start;
    }

    // Branch 3
    // conv_7
    if (!conv) {
        start = read_cycles();

        im2col(conv_7_params.batch_size, conv_7_params.in_channels, conv_7_params.in_dim,
            conv_7_params.I, conv_7_params.K,
            conv_3_out_pooled, conv_7_in, &conv_7_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_7_params.I, conv_7_params.J, conv_7_params.K,
            conv_7_params.J,
            conv_7_in, conv_7_w, conv_7_b, conv_7_out,
            RELU, conv_7_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_7");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_7_params.I, conv_7_params.J, conv_7_params.K,
            conv_7_params.J,
            conv_3_out_pooled, conv_7_w, conv_7_b, conv_7_out,
            RELU, conv_7_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_7");

        end = read_cycles();
        matmul_cycles += end - start;

        matmul_7_cycles = end - start;
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

        tiled_matmul_nn_auto_extended(conv_8_params.I, conv_8_params.J, conv_8_params.K,
            256,
            conv_8_in, conv_8_w, conv_8_b, ((elem_t*)inception3a_out + 192),
            RELU, conv_8_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_8");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_8_params.batch_size, conv_8_params.in_dim, conv_8_params.in_channels,
            conv_8_params.out_channels, conv_8_params.out_dim,
            conv_8_params.stride, 1, conv_8_params.padding, conv_8_params.kernel_size,
            false,
            
            256,

            (elem_t*)conv_7_out, (elem_t*)conv_8_w, (acc_t*)conv_8_b, (elem_t*)((elem_t*)inception3a_out + 192),

            RELU, conv_8_params.output_scale, 0,
            conv_8_params.pool_size, 0, conv_8_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;

        conv_8_cycles = end - start;
    }

    // pool_9
    start = read_cycles();
    tiled_pool_auto(pool_9_params.batch_size, pool_9_params.out_dim, pool_9_params.out_channels,
        pool_9_params.pool_size, pool_9_params.pool_stride, pool_9_params.pool_padding,
        conv_3_out_pooled, pool_9_out,
        true,
        tiled_matmul_type);
    end = read_cycles();
    pool_cycles += end - start;
    pool_9_cycles = end - start;

    // Branch 4
    // conv_10
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_10_params.I, conv_10_params.J, conv_10_params.K,
            256,
            pool_9_out, conv_10_w, conv_10_b, ((elem_t*)inception3a_out + 224),
            RELU, conv_10_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_10");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_10_params.I, conv_10_params.J, conv_10_params.K,
            256,
            pool_9_out, conv_10_w, conv_10_b, ((elem_t*)inception3a_out + 224),
            RELU, conv_10_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_10");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_10_cycles = end - start;
    }

    // Inception 3b
    // Branch 1
    // conv_11
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_11_params.I, conv_11_params.J, conv_11_params.K,
            480,
            ((elem_t*)inception3a_out), conv_11_w, conv_11_b, ((elem_t*)inception3b_out + 0),
            RELU, conv_11_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_11");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_11_params.I, conv_11_params.J, conv_11_params.K,
            480,
            ((elem_t*)inception3a_out), conv_11_w, conv_11_b, ((elem_t*)inception3b_out + 0),
            RELU, conv_11_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_11");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_11_cycles = end - start;
    }

    // Branch 2
    // conv_12
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_12_params.J,
            ((elem_t*)inception3a_out), conv_12_w, conv_12_b, conv_12_out,
            RELU, conv_12_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_12");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_12_params.J,
            ((elem_t*)inception3a_out), conv_12_w, conv_12_b, conv_12_out,
            RELU, conv_12_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_12");

        end = read_cycles();
        matmul_cycles += end - start;

        matmul_12_cycles = end - start;
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

        tiled_matmul_nn_auto_extended(conv_13_params.I, conv_13_params.J, conv_13_params.K,
            480,
            conv_13_in, conv_13_w, conv_13_b, ((elem_t*)inception3b_out + 128),
            RELU, conv_13_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_13");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_13_params.batch_size, conv_13_params.in_dim, conv_13_params.in_channels,
            conv_13_params.out_channels, conv_13_params.out_dim,
            conv_13_params.stride, 1, conv_13_params.padding, conv_13_params.kernel_size,
            false,
            
            480,

            (elem_t*)conv_12_out, (elem_t*)conv_13_w, (acc_t*)conv_13_b, (elem_t*)((elem_t*)inception3b_out + 128),

            RELU, conv_13_params.output_scale, 0,
            conv_13_params.pool_size, 0, conv_13_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_13_cycles = end - start;
    }

    // Branch 3
    // conv_14
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_14_params.I, conv_14_params.J, conv_14_params.K,
            conv_14_params.J,
            ((elem_t*)inception3a_out), conv_14_w, conv_14_b, conv_14_out,
            RELU, conv_14_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_14");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_14_params.I, conv_14_params.J, conv_14_params.K,
            conv_14_params.J,
            ((elem_t*)inception3a_out), conv_14_w, conv_14_b, conv_14_out,
            RELU, conv_14_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_14");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_14_cycles = end - start;
    }

    // conv_15
    if (!conv) {
        start = read_cycles();

        im2col_with_col2im(conv_14_params.I, conv_14_params.J,
            conv_15_params.I, conv_15_params.K,
            conv_14_out, conv_15_in, &conv_15_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_15_params.I, conv_15_params.J, conv_15_params.K,
            480,
            conv_15_in, conv_15_w, conv_15_b, ((elem_t*)inception3b_out + 320),
            RELU, conv_15_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_15");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_15_params.batch_size, conv_15_params.in_dim, conv_15_params.in_channels,
            conv_15_params.out_channels, conv_15_params.out_dim,
            conv_15_params.stride, 1, conv_15_params.padding, conv_15_params.kernel_size,
            false,
            
            480,

            (elem_t*)conv_14_out, (elem_t*)conv_15_w, (acc_t*)conv_15_b, (elem_t*)((elem_t*)inception3b_out + 320),

            RELU, conv_15_params.output_scale, 0,
            conv_15_params.pool_size, 0, conv_15_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_15_cycles = end - start;
    }

    // pool_16
    start = read_cycles();
    tiled_pool_auto(pool_16_params.batch_size, pool_16_params.out_dim, pool_16_params.out_channels,
        pool_16_params.pool_size, pool_16_params.pool_stride, pool_16_params.pool_padding,
        ((elem_t*)inception3a_out), pool_16_out,
        true,
        tiled_matmul_type);

    end = read_cycles();
    pool_cycles += end - start;
    pool_16_cyles = end - start;

    // Branch 4
    // conv_17
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_17_params.I, conv_17_params.J, conv_17_params.K,
            480,
            pool_16_out, conv_17_w, conv_17_b, ((elem_t*)inception3b_out + 416),
            RELU, conv_17_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_17");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_17_params.I, conv_17_params.J, conv_17_params.K,
            480,
            pool_16_out, conv_17_w, conv_17_b, ((elem_t*)inception3b_out + 416),
            RELU, conv_17_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_17");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_17_cycles = end - start;
    }

    // pool_18
    start = read_cycles();
    tiled_pool_auto(pool_18_params.batch_size, pool_18_params.out_dim, pool_18_params.out_channels,
        pool_18_params.pool_size, pool_18_params.pool_stride, pool_18_params.pool_padding,
        ((elem_t*)inception3b_out), pool_18_out,
        true,
        tiled_matmul_type);
    end = read_cycles();
    pool_cycles += end - start;
    pool_18_cycles = end - start;

    // Inception 4a
    // Branch 1
    // conv_19
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            512,
            pool_18_out, conv_19_w, conv_19_b, ((elem_t*)inception4a_out + 0),
            RELU, conv_19_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_19");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            512,
            pool_18_out, conv_19_w, conv_19_b, ((elem_t*)inception4a_out + 0),
            RELU, conv_19_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_19");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_19_cycles = end - start;
    }

    // Branch 2
    // conv_20
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_20_params.I, conv_20_params.J, conv_20_params.K,
            conv_20_params.J,
            pool_18_out, conv_20_w, conv_20_b, conv_20_out,
            RELU, conv_20_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_20");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_20_params.I, conv_20_params.J, conv_20_params.K,
            conv_20_params.J,
            pool_18_out, conv_20_w, conv_20_b, conv_20_out,
            RELU, conv_20_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_20");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_20_cycles = end - start;
    }

    // conv_21
    if (!conv) {
        start = read_cycles();

        im2col_with_col2im(conv_20_params.I, conv_20_params.J,
            conv_21_params.I, conv_21_params.K,
            conv_20_out, conv_21_in, &conv_21_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_21_params.I, conv_21_params.J, conv_21_params.K,
            512,
            conv_21_in, conv_21_w, conv_21_b, ((elem_t*)inception4a_out + 192),
            RELU, conv_21_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_21");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_21_params.batch_size, conv_21_params.in_dim, conv_21_params.in_channels,
            conv_21_params.out_channels, conv_21_params.out_dim,
            conv_21_params.stride, 1, conv_21_params.padding, conv_21_params.kernel_size,
            false,
            
            512,

            (elem_t*)conv_20_out, (elem_t*)conv_21_w, (acc_t*)conv_21_b, (elem_t*)((elem_t*)inception4a_out + 192),

            RELU, conv_21_params.output_scale, 0,
            conv_21_params.pool_size, 0, conv_21_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_21_cycles = end - start;
    }

    // Branch 3
    // conv_22
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_22_params.J,
            pool_18_out, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_22");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_22_params.J,
            pool_18_out, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_22");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_22_cycles = end - start;
    }

    // conv_23
    if (!conv) {
        start = read_cycles();

        im2col_with_col2im(conv_22_params.I, conv_22_params.J,
            conv_23_params.I, conv_23_params.K,
            conv_22_out, conv_23_in, &conv_23_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_23_params.I, conv_23_params.J, conv_23_params.K,
            512,
            conv_23_in, conv_23_w, conv_23_b, ((elem_t*)inception4a_out + 400),
            RELU, conv_23_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_23");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_23_params.batch_size, conv_23_params.in_dim, conv_23_params.in_channels,
            conv_23_params.out_channels, conv_23_params.out_dim,
            conv_23_params.stride, 1, conv_23_params.padding, conv_23_params.kernel_size,
            false,
            
            512,

            (elem_t*)conv_22_out, (elem_t*)conv_23_w, (acc_t*)conv_23_b, (elem_t*)((elem_t*)inception4a_out + 400),

            RELU, conv_23_params.output_scale, 0,
            conv_23_params.pool_size, 0, conv_23_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_23_cycles = end - start;
    }

    // pool_24
    start = read_cycles();
    tiled_pool_auto(pool_24_params.batch_size, pool_24_params.out_dim, pool_24_params.out_channels,
        pool_24_params.pool_size, pool_24_params.pool_stride, pool_24_params.pool_padding,
        pool_18_out, pool_24_out,
        true,
        tiled_matmul_type);
    end = read_cycles();
    pool_cycles += end - start;
    pool_24_cycles = end - start;

    // Branch 4
    // conv_25
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            512,
            pool_24_out, conv_25_w, conv_25_b, ((elem_t*)inception4a_out + 448),
            RELU, conv_25_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_25");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            512,
            pool_24_out, conv_25_w, conv_25_b, ((elem_t*)inception4a_out + 448),
            RELU, conv_25_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_25");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_25_cycles = end - start;
    }

    // Inception 4b
    // Branch 1
    // conv_26
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_26_params.I, conv_26_params.J, conv_26_params.K,
            512,
            ((elem_t*)inception4a_out), conv_26_w, conv_26_b, ((elem_t*)inception4b_out + 0),
            RELU, conv_26_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_26");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_26_params.I, conv_26_params.J, conv_26_params.K,
            512,
            ((elem_t*)inception4a_out), conv_26_w, conv_26_b, ((elem_t*)inception4b_out + 0),
            RELU, conv_26_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_26");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_26_cycles = end - start;
    }

    // Branch 2
    // conv_27
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_27_params.I, conv_27_params.J, conv_27_params.K,
            conv_27_params.J,
            ((elem_t*)inception4a_out), conv_27_w, conv_27_b, conv_27_out,
            RELU, conv_27_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_27");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_27_params.I, conv_27_params.J, conv_27_params.K,
            conv_27_params.J,
            ((elem_t*)inception4a_out), conv_27_w, conv_27_b, conv_27_out,
            RELU, conv_27_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_27");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_27_cycles = end - start;
    }

    // conv_28
    if (!conv) {
        start = read_cycles();

        im2col_with_col2im(conv_27_params.I, conv_27_params.J,
            conv_28_params.I, conv_28_params.K,
            conv_27_out, conv_28_in, &conv_28_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_28_params.I, conv_28_params.J, conv_28_params.K,
            512,
            conv_28_in, conv_28_w, conv_28_b, ((elem_t*)inception4b_out + 160),
            RELU, conv_28_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_28");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_28_params.batch_size, conv_28_params.in_dim, conv_28_params.in_channels,
            conv_28_params.out_channels, conv_28_params.out_dim,
            conv_28_params.stride, 1, conv_28_params.padding, conv_28_params.kernel_size,
            false,
            
            512,

            (elem_t*)conv_27_out, (elem_t*)conv_28_w, (acc_t*)conv_28_b, (elem_t*)((elem_t*)inception4b_out + 160),

            RELU, conv_28_params.output_scale, 0,
            conv_28_params.pool_size, 0, conv_28_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_28_cycles = end - start;
    }

    // Branch 3
    // conv_29
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_29_params.I, conv_29_params.J, conv_29_params.K,
            conv_29_params.J,
            ((elem_t*)inception4a_out), conv_29_w, conv_29_b, conv_29_out,
            RELU, conv_29_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_29");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_29_params.I, conv_29_params.J, conv_29_params.K,
            conv_29_params.J,
            ((elem_t*)inception4a_out), conv_29_w, conv_29_b, conv_29_out,
            RELU, conv_29_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_29");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_29_cycles = end - start;
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

        tiled_matmul_nn_auto_extended(conv_30_params.I, conv_30_params.J, conv_30_params.K,
            512,
            conv_30_in, conv_30_w, conv_30_b, ((elem_t*)inception4b_out + 384),
            RELU, conv_30_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_30");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_30_params.batch_size, conv_30_params.in_dim, conv_30_params.in_channels,
            conv_30_params.out_channels, conv_30_params.out_dim,
            conv_30_params.stride, 1, conv_30_params.padding, conv_30_params.kernel_size,
            false,
            
            512,

            (elem_t*)conv_29_out, (elem_t*)conv_30_w, (acc_t*)conv_30_b, (elem_t*)((elem_t*)inception4b_out + 384),

            RELU, conv_30_params.output_scale, 0,
            conv_30_params.pool_size, 0, conv_30_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_30_cycles = end - start;
    }

    // pool_31
    start = read_cycles();
    tiled_pool_auto(pool_31_params.batch_size, pool_31_params.out_dim, pool_31_params.out_channels,
        pool_31_params.pool_size, pool_31_params.pool_stride, pool_31_params.pool_padding,
        ((elem_t*)inception4a_out), pool_31_out,
        true,
        tiled_matmul_type);
    end = read_cycles();
    pool_cycles += end - start;
    pool_31_cycles = end - start;

    // Branch 4
    // conv_32
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_32_params.I, conv_32_params.J, conv_32_params.K,
            512,
            pool_31_out, conv_32_w, conv_32_b, ((elem_t*)inception4b_out + 448),
            RELU, conv_32_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_32");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_32_params.I, conv_32_params.J, conv_32_params.K,
            512,
            pool_31_out, conv_32_w, conv_32_b, ((elem_t*)inception4b_out + 448),
            RELU, conv_32_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_32");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_32_cycles = end - start;
    }

    // Inception 4c
    // Branch 1
    // conv_33
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_33_params.I, conv_33_params.J, conv_33_params.K,
            512,
            ((elem_t*)inception4b_out), conv_33_w, conv_33_b, ((elem_t*)inception4c_out + 0),
            RELU, conv_33_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_33");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_33_params.I, conv_33_params.J, conv_33_params.K,
            512,
            ((elem_t*)inception4b_out), conv_33_w, conv_33_b, ((elem_t*)inception4c_out + 0),
            RELU, conv_33_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_33");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_33_cycles = end - start;
    }

    // Branch 2
    // conv_34
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_34_params.I, conv_34_params.J, conv_34_params.K,
            conv_34_params.J,
            ((elem_t*)inception4b_out), conv_34_w, conv_34_b, conv_34_out,
            RELU, conv_34_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_34");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_34_params.I, conv_34_params.J, conv_34_params.K,
            conv_34_params.J,
            ((elem_t*)inception4b_out), conv_34_w, conv_34_b, conv_34_out,
            RELU, conv_34_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_34");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_34_cycles = end - start;
    }

    // conv_35
    if (!conv) {
        start = read_cycles();

        im2col_with_col2im(conv_34_params.I, conv_34_params.J,
            conv_35_params.I, conv_35_params.K,
            conv_34_out, conv_35_in, &conv_35_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_35_params.I, conv_35_params.J, conv_35_params.K,
            512,
            conv_35_in, conv_35_w, conv_35_b, ((elem_t*)inception4c_out + 128),
            RELU, conv_35_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_35");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_35_params.batch_size, conv_35_params.in_dim, conv_35_params.in_channels,
            conv_35_params.out_channels, conv_35_params.out_dim,
            conv_35_params.stride, 1, conv_35_params.padding, conv_35_params.kernel_size,
            false,
            
            512,

            (elem_t*)conv_34_out, (elem_t*)conv_35_w, (acc_t*)conv_35_b, (elem_t*)((elem_t*)inception4c_out + 128),

            RELU, conv_35_params.output_scale, 0,
            conv_35_params.pool_size, 0, conv_35_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_35_cycles = end - start;
    }

    // Branch 3
    // conv_36
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_36_params.I, conv_36_params.J, conv_36_params.K,
            conv_36_params.J,
            ((elem_t*)inception4b_out), conv_36_w, conv_36_b, conv_36_out,
            RELU, conv_36_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_36");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_36_params.I, conv_36_params.J, conv_36_params.K,
            conv_36_params.J,
            ((elem_t*)inception4b_out), conv_36_w, conv_36_b, conv_36_out,
            RELU, conv_36_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_36");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_36_cycles = end - start;
    }

    // conv_37
    if (!conv) {
        start = read_cycles();

        im2col_with_col2im(conv_36_params.I, conv_36_params.J,
            conv_37_params.I, conv_37_params.K,
            conv_36_out, conv_37_in, &conv_37_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_37_params.I, conv_37_params.J, conv_37_params.K,
            512,
            conv_37_in, conv_37_w, conv_37_b, ((elem_t*)inception4c_out + 384),
            RELU, conv_37_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_37");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_37_params.batch_size, conv_37_params.in_dim, conv_37_params.in_channels,
            conv_37_params.out_channels, conv_37_params.out_dim,
            conv_37_params.stride, 1, conv_37_params.padding, conv_37_params.kernel_size,
            false,
            
            512,

            (elem_t*)conv_36_out, (elem_t*)conv_37_w, (acc_t*)conv_37_b, (elem_t*)((elem_t*)inception4c_out + 384),

            RELU, conv_37_params.output_scale, 0,
            conv_37_params.pool_size, 0, conv_37_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_37_cycles = end - start;
    }

    // pool_38
    start = read_cycles();
    tiled_pool_auto(pool_38_params.batch_size, pool_38_params.out_dim, pool_38_params.out_channels,
        pool_38_params.pool_size, pool_38_params.pool_stride, pool_38_params.pool_padding,
        ((elem_t*)inception4b_out), pool_38_out,
        true,
        tiled_matmul_type);
    end = read_cycles();
    pool_cycles += end - start;
    pool_38_cycles = end - start;

    // Branch 4
    // conv_39
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_39_params.I, conv_39_params.J, conv_39_params.K,
            512,
            pool_38_out, conv_39_w, conv_39_b, ((elem_t*)inception4c_out + 448),
            RELU, conv_39_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_39");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_39_params.I, conv_39_params.J, conv_39_params.K,
            512,
            pool_38_out, conv_39_w, conv_39_b, ((elem_t*)inception4c_out + 448),
            RELU, conv_39_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_39");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_39_cycles = end - start;
    }

    // Inception 4d
    // Branch 1
    // conv_40
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_40_params.I, conv_40_params.J, conv_40_params.K,
            528,
            ((elem_t*)inception4c_out), conv_40_w, conv_40_b, ((elem_t*)inception4d_out + 0),
            RELU, conv_40_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_40");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_40_params.I, conv_40_params.J, conv_40_params.K,
            528,
            ((elem_t*)inception4c_out), conv_40_w, conv_40_b, ((elem_t*)inception4d_out + 0),
            RELU, conv_40_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_40");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_40_cycles = end - start;
    }

    // Branch 2
    // conv_41
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_41_params.I, conv_41_params.J, conv_41_params.K,
            conv_41_params.J,
            ((elem_t*)inception4c_out), conv_41_w, conv_41_b, conv_41_out,
            RELU, conv_41_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_41");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_41_params.I, conv_41_params.J, conv_41_params.K,
            conv_41_params.J,
            ((elem_t*)inception4c_out), conv_41_w, conv_41_b, conv_41_out,
            RELU, conv_41_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_41");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_41_cycles = end - start;
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

        tiled_matmul_nn_auto_extended(conv_42_params.I, conv_42_params.J, conv_42_params.K,
            528,
            conv_42_in, conv_42_w, conv_42_b, ((elem_t*)inception4d_out + 112),
            RELU, conv_42_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_42");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_42_params.batch_size, conv_42_params.in_dim, conv_42_params.in_channels,
            conv_42_params.out_channels, conv_42_params.out_dim,
            conv_42_params.stride, 1, conv_42_params.padding, conv_42_params.kernel_size,
            false,
            
            528,

            (elem_t*)conv_41_out, (elem_t*)conv_42_w, (acc_t*)conv_42_b, (elem_t*)((elem_t*)inception4d_out + 112),

            RELU, conv_42_params.output_scale, 0,
            conv_42_params.pool_size, 0, conv_42_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_42_cycles = end - start;
    }

    // Branch 3
    // conv_43
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_43_params.I, conv_43_params.J, conv_43_params.K,
            conv_43_params.J,
            ((elem_t*)inception4c_out), conv_43_w, conv_43_b, conv_43_out,
            RELU, conv_43_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_43");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_43_params.I, conv_43_params.J, conv_43_params.K,
            conv_43_params.J,
            ((elem_t*)inception4c_out), conv_43_w, conv_43_b, conv_43_out,
            RELU, conv_43_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_43");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_43_cycles = end - start;
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

        tiled_matmul_nn_auto_extended(conv_44_params.I, conv_44_params.J, conv_44_params.K,
            528,
            conv_44_in, conv_44_w, conv_44_b, ((elem_t*)inception4d_out + 400),
            RELU, conv_44_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_44");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_44_params.batch_size, conv_44_params.in_dim, conv_44_params.in_channels,
            conv_44_params.out_channels, conv_44_params.out_dim,
            conv_44_params.stride, 1, conv_44_params.padding, conv_44_params.kernel_size,
            false,
            
            528,

            (elem_t*)conv_43_out, (elem_t*)conv_44_w, (acc_t*)conv_44_b, (elem_t*)((elem_t*)inception4d_out + 400),

            RELU, conv_44_params.output_scale, 0,
            conv_44_params.pool_size, 0, conv_44_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_44_cycles = end - start;
    }

    // pool_45
    start = read_cycles();
    tiled_pool_auto(pool_45_params.batch_size, pool_45_params.out_dim, pool_45_params.out_channels,
        pool_45_params.pool_size, pool_45_params.pool_stride, pool_45_params.pool_padding,
        ((elem_t*)inception4c_out), pool_45_out,
        true,
        tiled_matmul_type);
    end = read_cycles();
    pool_cycles += end - start;
    pool_45_cycles = end - start;

    // Branch 4
    // conv_46
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_46_params.I, conv_46_params.J, conv_46_params.K,
            528,
            pool_45_out, conv_46_w, conv_46_b, ((elem_t*)inception4d_out + 464),
            RELU, conv_46_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_46");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_46_params.I, conv_46_params.J, conv_46_params.K,
            528,
            pool_45_out, conv_46_w, conv_46_b, ((elem_t*)inception4d_out + 464),
            RELU, conv_46_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_46");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_46_cycles = end - start;
    }

    // Inception 4e
    // Branch 1
    // conv_47
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_47_params.I, conv_47_params.J, conv_47_params.K,
            832,
            ((elem_t*)inception4d_out), conv_47_w, conv_47_b, ((elem_t*)inception4e_out + 0),
            RELU, conv_47_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_47");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_47_params.I, conv_47_params.J, conv_47_params.K,
            832,
            ((elem_t*)inception4d_out), conv_47_w, conv_47_b, ((elem_t*)inception4e_out + 0),
            RELU, conv_47_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_47");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_47_cycles = end - start;
    }

    // Branch 2
    // conv_48
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_48_params.I, conv_48_params.J, conv_48_params.K,
            conv_48_params.J,
            ((elem_t*)inception4d_out), conv_48_w, conv_48_b, conv_48_out,
            RELU, conv_48_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_48");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_48_params.I, conv_48_params.J, conv_48_params.K,
            conv_48_params.J,
            ((elem_t*)inception4d_out), conv_48_w, conv_48_b, conv_48_out,
            RELU, conv_48_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_48");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_48_cycles = end - start;
    }

    // conv_49
    if (!conv) {
        start = read_cycles();

        im2col_with_col2im(conv_48_params.I, conv_48_params.J,
            conv_49_params.I, conv_49_params.K,
            conv_48_out, conv_49_in, &conv_49_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_49_params.I, conv_49_params.J, conv_49_params.K,
            832,
            conv_49_in, conv_49_w, conv_49_b, ((elem_t*)inception4e_out + 256),
            RELU, conv_49_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_49");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_49_params.batch_size, conv_49_params.in_dim, conv_49_params.in_channels,
            conv_49_params.out_channels, conv_49_params.out_dim,
            conv_49_params.stride, 1, conv_49_params.padding, conv_49_params.kernel_size,
            false,
            
            832,

            (elem_t*)conv_48_out, (elem_t*)conv_49_w, (acc_t*)conv_49_b, (elem_t*)((elem_t*)inception4e_out + 256),

            RELU, conv_49_params.output_scale, 0,
            conv_49_params.pool_size, 0, conv_49_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_49_cycles = end - start;
    }

    // Branch 3
    // conv_50
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_50_params.I, conv_50_params.J, conv_50_params.K,
            conv_50_params.J,
            ((elem_t*)inception4d_out), conv_50_w, conv_50_b, conv_50_out,
            RELU, conv_50_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_50");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_50_params.I, conv_50_params.J, conv_50_params.K,
            conv_50_params.J,
            ((elem_t*)inception4d_out), conv_50_w, conv_50_b, conv_50_out,
            RELU, conv_50_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_50");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_50_cycles = end - start;
    }

    // conv_51
    if (!conv) {
        start = read_cycles();

        im2col_with_col2im(conv_50_params.I, conv_50_params.J,
            conv_51_params.I, conv_51_params.K,
            conv_50_out, conv_51_in, &conv_51_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_51_params.I, conv_51_params.J, conv_51_params.K,
            832,
            conv_51_in, conv_51_w, conv_51_b, ((elem_t*)inception4e_out + 576),
            RELU, conv_51_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_51");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_51_params.batch_size, conv_51_params.in_dim, conv_51_params.in_channels,
            conv_51_params.out_channels, conv_51_params.out_dim,
            conv_51_params.stride, 1, conv_51_params.padding, conv_51_params.kernel_size,
            false,
            
            832,

            (elem_t*)conv_50_out, (elem_t*)conv_51_w, (acc_t*)conv_51_b, (elem_t*)((elem_t*)inception4e_out + 576),

            RELU, conv_51_params.output_scale, 0,
            conv_51_params.pool_size, 0, conv_51_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_51_cycles = end - start;
    }

    // pool_52
    start = read_cycles();
    tiled_pool_auto(pool_52_params.batch_size, pool_52_params.out_dim, pool_52_params.out_channels,
        pool_52_params.pool_size, pool_52_params.pool_stride, pool_52_params.pool_padding,
        ((elem_t*)inception4d_out), pool_52_out,
        true,
        tiled_matmul_type);
    end = read_cycles();
    pool_cycles += end - start;
    pool_52_cycles = end - start;

    // Branch 4
    // conv_53
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_53_params.I, conv_53_params.J, conv_53_params.K,
            832,
            pool_52_out, conv_53_w, conv_53_b, ((elem_t*)inception4e_out + 704),
            RELU, conv_53_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_53");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_53_params.I, conv_53_params.J, conv_53_params.K,
            832,
            pool_52_out, conv_53_w, conv_53_b, ((elem_t*)inception4e_out + 704),
            RELU, conv_53_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_53");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_53_cycles = end - start;
    }

    // pool_54
    start = read_cycles();
    tiled_pool_auto(pool_54_params.batch_size, pool_54_params.out_dim, pool_54_params.out_channels,
        pool_54_params.pool_size, pool_54_params.pool_stride, pool_54_params.pool_padding,
        ((elem_t*)inception4e_out), pool_54_out,
        true,
        tiled_matmul_type);
    end = read_cycles();
    pool_cycles += end - start;
    pool_54_cycles = end - start;

    // Inception 5a
    // Branch 1
    // conv_55
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_55_params.I, conv_55_params.J, conv_55_params.K,
            832,
            pool_54_out, conv_55_w, conv_55_b, ((elem_t*)inception5a_out + 0),
            RELU, conv_55_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_55");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_55_params.I, conv_55_params.J, conv_55_params.K,
            832,
            pool_54_out, conv_55_w, conv_55_b, ((elem_t*)inception5a_out + 0),
            RELU, conv_55_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_55");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_55_cycles = end - start;
    }

    // Branch 2
    // conv_56
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_56_params.I, conv_56_params.J, conv_56_params.K,
            conv_56_params.J,
            pool_54_out, conv_56_w, conv_56_b, conv_56_out,
            RELU, conv_56_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_56");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_56_params.I, conv_56_params.J, conv_56_params.K,
            conv_56_params.J,
            pool_54_out, conv_56_w, conv_56_b, conv_56_out,
            RELU, conv_56_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_56");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_56_cycles = end - start;
    }

    // conv_57
    if (!conv) {
        start = read_cycles();

        im2col_with_col2im(conv_56_params.I, conv_56_params.J,
            conv_57_params.I, conv_57_params.K,
            conv_56_out, conv_57_in, &conv_57_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_57_params.I, conv_57_params.J, conv_57_params.K,
            832,
            conv_57_in, conv_57_w, conv_57_b, ((elem_t*)inception5a_out + 256),
            RELU, conv_57_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_57");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_57_params.batch_size, conv_57_params.in_dim, conv_57_params.in_channels,
            conv_57_params.out_channels, conv_57_params.out_dim,
            conv_57_params.stride, 1, conv_57_params.padding, conv_57_params.kernel_size,
            false,
            
            832,

            (elem_t*)conv_56_out, (elem_t*)conv_57_w, (acc_t*)conv_57_b, (elem_t*)((elem_t*)inception5a_out + 256),

            RELU, conv_57_params.output_scale, 0,
            conv_57_params.pool_size, 0, conv_57_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_57_cycles = end - start;
    }

    // Branch 3
    // conv_58
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_58_params.I, conv_58_params.J, conv_58_params.K,
            conv_58_params.J,
            pool_54_out, conv_58_w, conv_58_b, conv_58_out,
            RELU, conv_58_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_58");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_58_params.I, conv_58_params.J, conv_58_params.K,
            conv_58_params.J,
            pool_54_out, conv_58_w, conv_58_b, conv_58_out,
            RELU, conv_58_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_58");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_58_cycles = end - start;
    }

    // conv_59
    if (!conv) {
        start = read_cycles();

        im2col_with_col2im(conv_58_params.I, conv_58_params.J,
            conv_59_params.I, conv_59_params.K,
            conv_58_out, conv_59_in, &conv_59_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_59_params.I, conv_59_params.J, conv_59_params.K,
            832,
            conv_59_in, conv_59_w, conv_59_b, ((elem_t*)inception5a_out + 576),
            RELU, conv_59_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_59");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_59_params.batch_size, conv_59_params.in_dim, conv_59_params.in_channels,
            conv_59_params.out_channels, conv_59_params.out_dim,
            conv_59_params.stride, 1, conv_59_params.padding, conv_59_params.kernel_size,
            false,
            
            832,

            (elem_t*)conv_58_out, (elem_t*)conv_59_w, (acc_t*)conv_59_b, (elem_t*)((elem_t*)inception5a_out + 576),

            RELU, conv_59_params.output_scale, 0,
            conv_59_params.pool_size, 0, conv_59_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_59_cycles = end - start;
    }

    // pool_60
    start = read_cycles();
    tiled_pool_auto(pool_60_params.batch_size, pool_60_params.out_dim, pool_60_params.out_channels,
        pool_60_params.pool_size, pool_60_params.pool_stride, pool_60_params.pool_padding,
        pool_54_out, pool_60_out,
        true,
        tiled_matmul_type);
    end = read_cycles();
    pool_cycles += end - start;
    pool_60_cycles = end - start;

    // Branch 4
    // conv_61
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_61_params.I, conv_61_params.J, conv_61_params.K,
            832,
            pool_60_out, conv_61_w, conv_61_b, ((elem_t*)inception5a_out + 704),
            RELU, conv_61_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_61");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_61_params.I, conv_61_params.J, conv_61_params.K,
            832,
            pool_60_out, conv_61_w, conv_61_b, ((elem_t*)inception5a_out + 704),
            RELU, conv_61_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_61");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_60_cycles = end - start;
    }

    // Inception 5b
    // Branch 1
    // conv_62
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_62_params.I, conv_62_params.J, conv_62_params.K,
            1024,
            ((elem_t*)inception5a_out), conv_62_w, conv_62_b, ((elem_t*)inception5b_out + 0),
            RELU, conv_62_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_62");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_62_params.I, conv_62_params.J, conv_62_params.K,
            1024,
            ((elem_t*)inception5a_out), conv_62_w, conv_62_b, ((elem_t*)inception5b_out + 0),
            RELU, conv_62_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_62");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_62_cycles = end - start;
    }

    // Branch 2
    // conv_63
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_63_params.I, conv_63_params.J, conv_63_params.K,
            conv_63_params.J,
            ((elem_t*)inception5a_out), conv_63_w, conv_63_b, conv_63_out,
            RELU, conv_63_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_63");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_63_params.I, conv_63_params.J, conv_63_params.K,
            conv_63_params.J,
            ((elem_t*)inception5a_out), conv_63_w, conv_63_b, conv_63_out,
            RELU, conv_63_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_63");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_63_cycles = end - start;
    }

    // conv_64
    if (!conv) {
        start = read_cycles();

        im2col_with_col2im(conv_63_params.I, conv_63_params.J,
            conv_64_params.I, conv_64_params.K,
            conv_63_out, conv_64_in, &conv_64_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_64_params.I, conv_64_params.J, conv_64_params.K,
            1024,
            conv_64_in, conv_64_w, conv_64_b, ((elem_t*)inception5b_out + 384),
            RELU, conv_64_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_64");

        end = read_cycles();
        matmul_cycles += end - start;
    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_64_params.batch_size, conv_64_params.in_dim, conv_64_params.in_channels,
            conv_64_params.out_channels, conv_64_params.out_dim,
            conv_64_params.stride, 1, conv_64_params.padding, conv_64_params.kernel_size,
            false,
            
            1024,

            (elem_t*)conv_63_out, (elem_t*)conv_64_w, (acc_t*)conv_64_b, (elem_t*)((elem_t*)inception5b_out + 384),

            RELU, conv_64_params.output_scale, 0,
            conv_64_params.pool_size, 0, conv_64_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_64_cycles = end - start;
    }

    // Branch 3
    // conv_65
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_65_params.I, conv_65_params.J, conv_65_params.K,
            conv_65_params.J,
            ((elem_t*)inception5a_out), conv_65_w, conv_65_b, conv_65_out,
            RELU, conv_65_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_65");

        end = read_cycles();
        matmul_cycles += end - start;
    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_65_params.I, conv_65_params.J, conv_65_params.K,
            conv_65_params.J,
            ((elem_t*)inception5a_out), conv_65_w, conv_65_b, conv_65_out,
            RELU, conv_65_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_65");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_65_cycles = end - start;
    }

    // conv_66
    if (!conv) {
        start = read_cycles();

        im2col_with_col2im(conv_65_params.I, conv_65_params.J,
            conv_66_params.I, conv_66_params.K,
            conv_65_out, conv_66_in, &conv_66_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_66_params.I, conv_66_params.J, conv_66_params.K,
            1024,
            conv_66_in, conv_66_w, conv_66_b, ((elem_t*)inception5b_out + 768),
            RELU, conv_66_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_66");

        end = read_cycles();
        matmul_cycles += end - start;
    } else {
        start = read_cycles();

        tiled_conv_batch_parallel_auto(
            conv_66_params.batch_size, conv_66_params.in_dim, conv_66_params.in_channels,
            conv_66_params.out_channels, conv_66_params.out_dim,
            conv_66_params.stride, 1, conv_66_params.padding, conv_66_params.kernel_size,
            false,
            
            1024,

            (elem_t*)conv_65_out, (elem_t*)conv_66_w, (acc_t*)conv_66_b, (elem_t*)((elem_t*)inception5b_out + 768),

            RELU, conv_66_params.output_scale, 0,
            conv_66_params.pool_size, 0, conv_66_params.pool_padding, true,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        conv_66_cycles = end - start;
    }

    // pool_67
    start = read_cycles();
    tiled_pool_auto(pool_67_params.batch_size, pool_67_params.out_dim, pool_67_params.out_channels,
        pool_67_params.pool_size, pool_67_params.pool_stride, pool_67_params.pool_padding,
        ((elem_t*)inception5a_out), pool_67_out,
        true,
        tiled_matmul_type);
    end = read_cycles();
    pool_cycles += end - start;
    pool_67_cycles = end - start;

    // Branch 4
    // conv_68
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_68_params.I, conv_68_params.J, conv_68_params.K,
            1024,
            pool_67_out, conv_68_w, conv_68_b, ((elem_t*)inception5b_out + 896),
            RELU, conv_68_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_68");

        end = read_cycles();
        matmul_cycles += end - start;
    } else {
        start = read_cycles();

        tiled_matmul_nn_auto_extended(conv_68_params.I, conv_68_params.J, conv_68_params.K,
            1024,
            pool_67_out, conv_68_w, conv_68_b, ((elem_t*)inception5b_out + 896),
            RELU, conv_68_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_68");

        end = read_cycles();
        matmul_cycles += end - start;
        matmul_67_cycles = end - start;
    }

    // Global averaging
    static elem_t average[4][1024] row_align(1);

    start = read_cycles();

    tiled_global_average_auto(inception5b_out, average, 4, 1024, 7, WS);

    end = read_cycles();
    other_cycles += end - start;

    // HIST_MATRIX(average);
    // exit(1);

    // fc_69
    start = read_cycles();

    tiled_matmul_nn_auto(fc_69_params.I, fc_69_params.J, fc_69_params.K,
        average, fc_69_w, fc_69_b, fc_69_out,
        NO_ACTIVATION, fc_69_params.output_scale, 0, false,
        tiled_matmul_type, check, "fc_69");

    end = read_cycles();
    matmul_cycles += end - start;
    fc_69_cycles = end - start;

    if (!quiet) {
      printf("conv_1_cycles: %llu\n", conv_1_cycles);
      printf("matmul_2_cycles: %llu\n", matmul_2_cycles);
      printf("conv_3_cycles: %llu\n", conv_3_cycles);
      printf("matmul_4_cycles: %llu\n", matmul_4_cycles);
      printf("matmul_5_cycles: %llu\n", matmul_5_cycles);
      printf("conv_6_cycles: %llu\n", conv_6_cycles);
      printf("matmul_7_cycles: %llu\n", matmul_7_cycles);
      printf("conv_8_cycles: %llu\n", conv_8_cycles);
      printf("pool_9_cycles: %llu\n", pool_9_cycles);
      printf("matmul_10_cycles: %llu\n", matmul_10_cycles);
      printf("matmul_11_cycles: %llu\n", matmul_11_cycles);
      printf("matmul_12_cycles: %llu\n", matmul_12_cycles);
      printf("conv_13_cycles: %llu\n", conv_13_cycles);
      printf("matmul_14_cycles: %llu\n", matmul_14_cycles);
      printf("conv_15_cycles: %llu\n", conv_15_cycles);
      printf("pool_16_cyles: %llu\n", pool_16_cyles);
      printf("matmul_17_cycles: %llu\n", matmul_17_cycles);
      printf("pool_18_cycles: %llu\n", pool_18_cycles);
      printf("matmul_19_cycles: %llu\n", matmul_19_cycles);
      printf("matmul_20_cycles: %llu\n", matmul_20_cycles);
      printf("conv_21_cycles: %llu\n", conv_21_cycles);
      printf("matmul_22_cycles: %llu\n", matmul_22_cycles);
      printf("conv_23_cycles: %llu\n", conv_23_cycles);
      printf("pool_24_cycles: %llu\n", pool_24_cycles);
      printf("matmul_25_cycles: %llu\n", matmul_25_cycles);
      printf("matmul_26_cycles: %llu\n", matmul_26_cycles);
      printf("matmul_27_cycles: %llu\n", matmul_27_cycles);
      printf("conv_28_cycles: %llu\n", conv_28_cycles);
      printf("matmul_29_cycles: %llu\n", matmul_29_cycles);
      printf("conv_30_cycles: %llu\n", conv_30_cycles);
      printf("pool_31_cycles: %llu\n", pool_31_cycles);
      printf("matmul_32_cycles: %llu\n", matmul_32_cycles);
      printf("matmul_33_cycles: %llu\n", matmul_33_cycles);
      printf("matmul_34_cycles: %llu\n", matmul_34_cycles);
      printf("conv_35_cycles: %llu\n", conv_35_cycles);
      printf("matmul_36_cycles: %llu\n", matmul_36_cycles);
      printf("conv_37_cycles: %llu\n", conv_37_cycles);
      printf("pool_38_cycles: %llu\n", pool_38_cycles);
      printf("matmul_39_cycles: %llu\n", matmul_39_cycles);
      printf("matmul_40_cycles: %llu\n", matmul_40_cycles);
      printf("matmul_41_cycles: %llu\n", matmul_41_cycles);
      printf("conv_42_cycles: %llu\n", conv_42_cycles);
      printf("matmul_43_cycles: %llu\n", matmul_43_cycles);
      printf("conv_44_cycles: %llu\n", conv_44_cycles);
      printf("pool_45_cycles: %llu\n", pool_45_cycles);
      printf("matmul_46_cycles: %llu\n", matmul_46_cycles);
      printf("matmul_47_cycles: %llu\n", matmul_47_cycles);
      printf("matmul_48_cycles: %llu\n", matmul_48_cycles);
      printf("conv_49_cycles: %llu\n", conv_49_cycles);
      printf("matmul_50_cycles: %llu\n", matmul_50_cycles);
      printf("conv_51_cycles: %llu\n", conv_51_cycles);
      printf("pool_52_cycles: %llu\n", pool_52_cycles);
      printf("matmul_53_cycles: %llu\n", matmul_53_cycles);
      printf("pool_54_cycles: %llu\n", pool_54_cycles);
      printf("matmul_55_cycles: %llu\n", matmul_55_cycles);
      printf("matmul_56_cycles: %llu\n", matmul_56_cycles);
      printf("conv_57_cycles: %llu\n", conv_57_cycles);
      printf("matmul_58_cycles: %llu\n", matmul_58_cycles);
      printf("conv_59_cycles: %llu\n", conv_59_cycles);
      printf("pool_60_cycles: %llu\n", pool_60_cycles);
      printf("matmul_60_cycles: %llu\n", matmul_60_cycles);
      printf("matmul_62_cycles: %llu\n", matmul_62_cycles);
      printf("matmul_63_cycles: %llu\n", matmul_63_cycles);
      printf("conv_64_cycles: %llu\n", conv_64_cycles);
      printf("matmul_65_cycles: %llu\n", matmul_65_cycles);
      printf("conv_66_cycles: %llu\n", conv_66_cycles);
      printf("pool_67_cycles: %llu\n", pool_67_cycles);
      printf("matmul_67_cycles: %llu\n", matmul_67_cycles);
      printf("fc_69_cycles: %llu\n\n", fc_69_cycles);
    }

    // Find highest probs
    int preds[fc_69_params.batch_size];
    for (int batch = 0; batch < fc_69_params.batch_size; batch++) {
        elem_t max_prob = fc_69_out[batch][0];
        size_t max_idx = 0;

        for (int i = 1; i < fc_69_params.out_features; i++) {
            if (fc_69_out[batch][i] > max_prob) {
                max_prob = fc_69_out[batch][i];
                max_idx = i;
            }
        }

        preds[batch] = max_idx;

        if (!quiet) {
          printf("Prediction: %u (score: %d)\n", max_idx, max_prob);
        }
    }

    uint64_t total_cycles = im2col_cycles + matmul_cycles + conv_cycles + pool_cycles + conv_dw_cycles + res_add_cycles + other_cycles;

    if (!quiet) {
      printf("\nTotal cycles: %llu (100%%)\n", total_cycles);
      printf("Matmul cycles: %llu (%d%%)\n", matmul_cycles, (matmul_cycles * 100) / total_cycles);
      printf("Im2col cycles: %llu (%d%%)\n", im2col_cycles, (im2col_cycles * 100) / total_cycles);
      printf("Conv cycles: %llu (%d%%)\n", conv_cycles, (conv_cycles * 100) / total_cycles);
      printf("Pooling cycles: %llu (%d%%)\n", pool_cycles, (pool_cycles * 100) / total_cycles);
      printf("Depthwise convolution cycles: %llu (%d%%)\n", conv_dw_cycles, (conv_dw_cycles * 100) / total_cycles);
      printf("Res add cycles: %llu (%d%%)\n", res_add_cycles, (res_add_cycles * 100) / total_cycles);
      printf("Other cycles: %llu (%d%%)\n", other_cycles, (other_cycles * 100) / total_cycles);
    }

    int correct[] = {375, 770, 249, 891};
    for (int i = 0; i < fc_69_params.batch_size; i++) {
        if (preds[i] != correct[i] && fc_69_out[i][preds[i]] != fc_69_out[i][correct[i]]) {
            printf("Prediction %d is incorrect!\nFAIL\n", i+1);
            exit(1);
        }
    }

    if (!quiet) {
      printf("PASS\n");
    }

    END_THREADS();
    exit(0);
}

START_THREADPOOL()

