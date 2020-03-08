#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "resnet50_params.h"
#include "images.h"

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

    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;

    // conv_1
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

    // conv_2
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

    // conv_3
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

    // conv_4
    start = read_cycles();

    tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K,
        conv_3_out, conv_4_w, conv_4_b, conv_4_out,
        NO_ACTIVATION, conv_4_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_4");

    end = read_cycles();
    matmul_cycles += end - start;

    // Downsampling conv_1_out_pooled
    // conv_5
    start = read_cycles();
    
    im2col(conv_5_params.batch_size, conv_5_params.in_channels, conv_5_params.in_dim,
        conv_5_params.I, conv_5_params.K,
        conv_1_out_pooled, conv_5_in, &conv_5_params);
    
    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_5_params.I, conv_5_params.J, conv_5_params.K,
        conv_5_in, conv_5_w, conv_5_b, conv_5_out,
        NO_ACTIVATION, conv_5_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_5");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_4_params.I, conv_4_params.J,
        conv_5_out,
        conv_4_out,
        conv_4_out,
        true,
        &conv_4_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_6
    start = read_cycles();

    tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
        conv_4_out, conv_6_w, conv_6_b, conv_6_out,
        RELU, conv_6_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_6");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_7
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

    // conv_8
    start = read_cycles();

    tiled_matmul_nn_auto(conv_8_params.I, conv_8_params.J, conv_8_params.K,
        conv_7_out, conv_8_w, conv_8_b, conv_8_out,
        NO_ACTIVATION, conv_8_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_8");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_8_params.I, conv_8_params.J,
        conv_4_out,
        conv_8_out,
        conv_8_out,
        true,
        &conv_8_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_9
    start = read_cycles();

    tiled_matmul_nn_auto(conv_9_params.I, conv_9_params.J, conv_9_params.K,
        conv_8_out, conv_9_w, conv_9_b, conv_9_out,
        RELU, conv_9_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_9");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_10
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

    // conv_11
    start = read_cycles();

    tiled_matmul_nn_auto(conv_11_params.I, conv_11_params.J, conv_11_params.K,
        conv_10_out, conv_11_w, conv_11_b, conv_11_out,
        NO_ACTIVATION, conv_11_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_11");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_11_params.I, conv_11_params.J,
        conv_8_out,
        conv_11_out,
        conv_11_out,
        true,
        &conv_11_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_12
    start = read_cycles();

    tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K,
        conv_11_out, conv_12_w, conv_12_b, conv_12_out,
        RELU, conv_12_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_12");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_13
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

    // conv_14
    start = read_cycles();

    tiled_matmul_nn_auto(conv_14_params.I, conv_14_params.J, conv_14_params.K,
        conv_13_out, conv_14_w, conv_14_b, conv_14_out,
        NO_ACTIVATION, conv_14_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_14");

    end = read_cycles();
    matmul_cycles += end - start;

    // Downsampling conv_11_out
    // conv_15
    start = read_cycles();

    im2col_with_col2im(conv_11_params.I, conv_11_params.J,
        conv_15_params.I, conv_15_params.K,
        conv_11_out, conv_15_in, &conv_15_params);

    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_15_params.I, conv_15_params.J, conv_15_params.K,
        conv_15_in, conv_15_w, conv_15_b, conv_15_out,
        NO_ACTIVATION, conv_15_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_15");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_14_params.I, conv_14_params.J,
        conv_15_out,
        conv_14_out,
        conv_14_out,
        true,
        &conv_14_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_16
    start = read_cycles();

    tiled_matmul_nn_auto(conv_16_params.I, conv_16_params.J, conv_16_params.K,
        conv_14_out, conv_16_w, conv_16_b, conv_16_out,
        RELU, conv_16_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_16");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_17
    start = read_cycles();

    im2col_with_col2im(conv_16_params.I, conv_16_params.J,
        conv_17_params.I, conv_17_params.K,
        conv_16_out, conv_17_in, &conv_17_params);

    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_17_params.I, conv_17_params.J, conv_17_params.K,
        conv_17_in, conv_17_w, conv_17_b, conv_17_out,
        RELU, conv_17_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_17");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_18
    start = read_cycles();

    tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K,
        conv_17_out, conv_18_w, conv_18_b, conv_18_out,
        NO_ACTIVATION, conv_18_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_18");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_18_params.I, conv_18_params.J,
        conv_14_out,
        conv_18_out,
        conv_18_out,
        true,
        &conv_18_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_19
    start = read_cycles();

    tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K,
        conv_18_out, conv_19_w, conv_19_b, conv_19_out,
        RELU, conv_19_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_19");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_20
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

    // conv_21
    start = read_cycles();

    tiled_matmul_nn_auto(conv_21_params.I, conv_21_params.J, conv_21_params.K,
        conv_20_out, conv_21_w, conv_21_b, conv_21_out,
        NO_ACTIVATION, conv_21_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_21");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_21_params.I, conv_21_params.J,
        conv_18_out,
        conv_21_out,
        conv_21_out,
        true,
        &conv_21_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_22
    start = read_cycles();

    tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K,
        conv_21_out, conv_22_w, conv_22_b, conv_22_out,
        RELU, conv_22_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_22");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_23
    start = read_cycles();

    im2col_with_col2im(conv_22_params.I, conv_22_params.J,
        conv_23_params.I, conv_23_params.K,
        conv_22_out, conv_23_in, &conv_23_params);

    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_23_params.I, conv_23_params.J, conv_23_params.K,
        conv_23_in, conv_23_w, conv_23_b, conv_23_out,
        RELU, conv_23_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_23");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_24
    start = read_cycles();

    tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K,
        conv_23_out, conv_24_w, conv_24_b, conv_24_out,
        NO_ACTIVATION, conv_24_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_24");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_24_params.I, conv_24_params.J,
        conv_21_out,
        conv_24_out,
        conv_24_out,
        true,
        &conv_24_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_25
    start = read_cycles();

    tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K,
        conv_24_out, conv_25_w, conv_25_b, conv_25_out,
        RELU, conv_25_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_25");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_26
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

    // conv_27
    start = read_cycles();

    tiled_matmul_nn_auto(conv_27_params.I, conv_27_params.J, conv_27_params.K,
        conv_26_out, conv_27_w, conv_27_b, conv_27_out,
        NO_ACTIVATION, conv_27_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_27");

    end = read_cycles();
    matmul_cycles += end - start;

    // Downsampling conv_24_out
    // conv_28
    start = read_cycles();

    im2col_with_col2im(conv_24_params.I, conv_24_params.J,
        conv_28_params.I, conv_28_params.K,
        conv_24_out, conv_28_in, &conv_28_params);

    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_28_params.I, conv_28_params.J, conv_28_params.K,
        conv_28_in, conv_28_w, conv_28_b, conv_28_out,
        NO_ACTIVATION, conv_28_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_28");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_27_params.I, conv_27_params.J,
        conv_28_out,
        conv_27_out,
        conv_27_out,
        true,
        &conv_27_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_29
    start = read_cycles();

    tiled_matmul_nn_auto(conv_29_params.I, conv_29_params.J, conv_29_params.K,
        conv_27_out, conv_29_w, conv_29_b, conv_29_out,
        RELU, conv_29_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_29");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_30
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

    // conv_31
    start = read_cycles();

    tiled_matmul_nn_auto(conv_31_params.I, conv_31_params.J, conv_31_params.K,
        conv_30_out, conv_31_w, conv_31_b, conv_31_out,
        NO_ACTIVATION, conv_31_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_31");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_31_params.I, conv_31_params.J,
        conv_27_out,
        conv_31_out,
        conv_31_out,
        true,
        &conv_31_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_32
    start = read_cycles();

    tiled_matmul_nn_auto(conv_32_params.I, conv_32_params.J, conv_32_params.K,
        conv_31_out, conv_32_w, conv_32_b, conv_32_out,
        RELU, conv_32_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_32");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_33
    start = read_cycles();

    im2col_with_col2im(conv_32_params.I, conv_32_params.J,
        conv_33_params.I, conv_33_params.K,
        conv_32_out, conv_33_in, &conv_33_params);

    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_33_params.I, conv_33_params.J, conv_33_params.K,
        conv_33_in, conv_33_w, conv_33_b, conv_33_out,
        RELU, conv_33_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_33");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_34
    start = read_cycles();

    tiled_matmul_nn_auto(conv_34_params.I, conv_34_params.J, conv_34_params.K,
        conv_33_out, conv_34_w, conv_34_b, conv_34_out,
        NO_ACTIVATION, conv_34_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_34");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_34_params.I, conv_34_params.J,
        conv_31_out,
        conv_34_out,
        conv_34_out,
        true,
        &conv_34_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_35
    start = read_cycles();

    tiled_matmul_nn_auto(conv_35_params.I, conv_35_params.J, conv_35_params.K,
        conv_34_out, conv_35_w, conv_35_b, conv_35_out,
        RELU, conv_35_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_35");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_36
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

    // conv_37
    start = read_cycles();

    tiled_matmul_nn_auto(conv_37_params.I, conv_37_params.J, conv_37_params.K,
        conv_36_out, conv_37_w, conv_37_b, conv_37_out,
        NO_ACTIVATION, conv_37_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_37");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_37_params.I, conv_37_params.J,
        conv_34_out,
        conv_37_out,
        conv_37_out,
        true,
        &conv_37_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_38
    start = read_cycles();

    tiled_matmul_nn_auto(conv_38_params.I, conv_38_params.J, conv_38_params.K,
        conv_37_out, conv_38_w, conv_38_b, conv_38_out,
        RELU, conv_38_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_38");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_39
    start = read_cycles();

    im2col_with_col2im(conv_38_params.I, conv_38_params.J,
        conv_39_params.I, conv_39_params.K,
        conv_38_out, conv_39_in, &conv_39_params);

    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_39_params.I, conv_39_params.J, conv_39_params.K,
        conv_39_in, conv_39_w, conv_39_b, conv_39_out,
        RELU, conv_39_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_39");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_40
    start = read_cycles();

    tiled_matmul_nn_auto(conv_40_params.I, conv_40_params.J, conv_40_params.K,
        conv_39_out, conv_40_w, conv_40_b, conv_40_out,
        NO_ACTIVATION, conv_40_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_40");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_40_params.I, conv_40_params.J,
        conv_37_out,
        conv_40_out,
        conv_40_out,
        true,
        &conv_40_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_41
    start = read_cycles();

    tiled_matmul_nn_auto(conv_41_params.I, conv_41_params.J, conv_41_params.K,
        conv_40_out, conv_41_w, conv_41_b, conv_41_out,
        RELU, conv_41_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_41");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_42
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

    // conv_43
    start = read_cycles();

    tiled_matmul_nn_auto(conv_43_params.I, conv_43_params.J, conv_43_params.K,
        conv_42_out, conv_43_w, conv_43_b, conv_43_out,
        NO_ACTIVATION, conv_43_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_43");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_43_params.I, conv_43_params.J,
        conv_40_out,
        conv_43_out,
        conv_43_out,
        true,
        &conv_43_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_44
    start = read_cycles();

    tiled_matmul_nn_auto(conv_44_params.I, conv_44_params.J, conv_44_params.K,
        conv_43_out, conv_44_w, conv_44_b, conv_44_out,
        RELU, conv_44_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_44");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_45
    start = read_cycles();

    im2col_with_col2im(conv_44_params.I, conv_44_params.J,
        conv_45_params.I, conv_45_params.K,
        conv_44_out, conv_45_in, &conv_45_params);

    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_45_params.I, conv_45_params.J, conv_45_params.K,
        conv_45_in, conv_45_w, conv_45_b, conv_45_out,
        RELU, conv_45_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_45");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_46
    start = read_cycles();

    tiled_matmul_nn_auto(conv_46_params.I, conv_46_params.J, conv_46_params.K,
        conv_45_out, conv_46_w, conv_46_b, conv_46_out,
        NO_ACTIVATION, conv_46_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_46");

    end = read_cycles();
    matmul_cycles += end - start;

    // Downsampling conv_43_out
    // conv_47
    start = read_cycles();

    im2col_with_col2im(conv_43_params.I, conv_43_params.J,
        conv_47_params.I, conv_47_params.K,
        conv_43_out, conv_47_in, &conv_47_params);

    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_47_params.I, conv_47_params.J, conv_47_params.K,
        conv_47_in, conv_47_w, conv_47_b, conv_47_out,
        NO_ACTIVATION, conv_47_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_47");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_46_params.I, conv_46_params.J,
        conv_47_out,
        conv_46_out,
        conv_46_out,
        true,
        &conv_46_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_48
    start = read_cycles();

    tiled_matmul_nn_auto(conv_48_params.I, conv_48_params.J, conv_48_params.K,
        conv_46_out, conv_48_w, conv_48_b, conv_48_out,
        RELU, conv_48_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_48");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_49
    start = read_cycles();

    im2col_with_col2im(conv_48_params.I, conv_48_params.J,
        conv_49_params.I, conv_49_params.K,
        conv_48_out, conv_49_in, &conv_49_params);

    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_49_params.I, conv_49_params.J, conv_49_params.K,
        conv_49_in, conv_49_w, conv_49_b, conv_49_out,
        RELU, conv_49_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_49");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_50
    start = read_cycles();

    tiled_matmul_nn_auto(conv_50_params.I, conv_50_params.J, conv_50_params.K,
        conv_49_out, conv_50_w, conv_50_b, conv_50_out,
        NO_ACTIVATION, conv_50_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_50");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_50_params.I, conv_50_params.J,
        conv_46_out,
        conv_50_out,
        conv_50_out,
        true,
        &conv_50_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_51
    start = read_cycles();

    tiled_matmul_nn_auto(conv_51_params.I, conv_51_params.J, conv_51_params.K,
        conv_50_out, conv_51_w, conv_51_b, conv_51_out,
        RELU, conv_51_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_51");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_52
    start = read_cycles();

    im2col_with_col2im(conv_51_params.I, conv_51_params.J,
        conv_52_params.I, conv_52_params.K,
        conv_51_out, conv_52_in, &conv_52_params);

    end = read_cycles();
    im2col_cycles += end - start;

    start = read_cycles();

    tiled_matmul_nn_auto(conv_52_params.I, conv_52_params.J, conv_52_params.K,
        conv_52_in, conv_52_w, conv_52_b, conv_52_out,
        RELU, conv_52_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_52");

    end = read_cycles();
    matmul_cycles += end - start;

    // conv_53
    start = read_cycles();

    tiled_matmul_nn_auto(conv_53_params.I, conv_53_params.J, conv_53_params.K,
        conv_52_out, conv_53_w, conv_53_b, conv_53_out,
        NO_ACTIVATION, conv_53_params.output_scale, 0, true,
        tiled_matmul_type, check, "conv_53");

    end = read_cycles();
    matmul_cycles += end - start;

    // Add residuals
    start = read_cycles();

    resadd3(conv_53_params.I, conv_53_params.J,
        conv_50_out,
        conv_53_out,
        conv_53_out,
        true,
        &conv_53_params
    );

    end = read_cycles();
    res_add_cycles += end - start;
    
    // Global averaging
    static elem_t average[2048][64] row_align(1);

    start = read_cycles();

    for (int batch = 0; batch < conv_53_params.batch_size; batch++) {
        for (int channel = 0; channel < conv_53_params.out_channels; channel++) {
            int sum = 0;
            for (int row = 0; row < conv_53_params.out_dim; row++) {
                for (int col = 0; col < conv_53_params.out_dim; col++) {
                    size_t r = batch * conv_53_params.out_dim * conv_53_params.out_dim + row * conv_53_params.out_dim + col;

                    sum += conv_53_out[r][channel];
                }
            }
            const int count = conv_53_params.out_dim * conv_53_params.out_dim;

            average[channel][batch] = (sum + count/2) / count;
        }
    }

    end = read_cycles();
    other_cycles += end - start;

    // fc_54
    start = read_cycles();

    tiled_matmul_nn_auto(fc_54_params.I, fc_54_params.J, fc_54_params.K,
        fc_54_w, average, fc_54_b, fc_54_out,
        NO_ACTIVATION, fc_54_params.output_scale, 0, false,
        tiled_matmul_type, check, "fc_54");

    end = read_cycles();
    matmul_cycles += end - start;

    // Find highest probs
    int preds[fc_54_params.batch_size];
    for (int batch = 0; batch < fc_54_params.batch_size; batch++) {
        elem_t max_prob = fc_54_out[0][batch];
        size_t max_idx = 0;

        for (int i = 1; i < fc_54_params.out_features; i++) {
            if (fc_54_out[i][batch] > max_prob) {
                max_prob = fc_54_out[i][batch];
                max_idx = i;
            }
        }
        
        printf("Prediction: %u (score: %d)\n", max_idx, max_prob);
        preds[batch] = max_idx;
    }

    uint64_t total_cycles = im2col_cycles + matmul_cycles + pool_cycles + conv_dw_cycles + res_add_cycles + other_cycles;

    printf("\nTotal cycles: %llu\n", total_cycles);
    printf("Matmul cycles: %llu\n", matmul_cycles);
    printf("Im2col cycles: %llu\n", im2col_cycles);
    printf("Pooling cycles: %llu\n", pool_cycles);
    printf("Depthwise convolution cycles: %llu\n", conv_dw_cycles);
    printf("Other cycles: %llu\n", other_cycles);

    int correct[] = {553, 233, 43, 617};
    for (int i = 0; i < fc_54_params.batch_size; i++) {
        if (preds[i] != correct[i]) {
            printf("Prediction %d is incorrect!\nFAIL\n", i+1);
            exit(1);
        }
    }

    printf("PASS\n");

    exit(0);
}

