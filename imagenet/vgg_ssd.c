
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "vgg_ssd_params.h"
// #include "resnet50_params_1batch.h"
#include "images.h"


int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type = WS;

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

    bool conv = true;
    
    if (argc < 3) {
        conv = true;
    } else if (strcmp(argv[2], "conv") == 0) {
        conv = true;
    } else if (strcmp(argv[2], "matmul") == 0) {
        conv = false;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check] [conv]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool check = false;

    if (argc < 4) {
        check = false;
    } else if (strcmp(argv[3], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;



    // Conv1
    printf("Starting Conv 1...\n");
    tiled_conv_auto(
        conv_1_params.batch_size, 
        conv_1_params.in_dim, conv_1_params.in_dim,
        conv_1_params.in_channels, conv_1_params.out_channels, 
        conv_1_params.out_dim, conv_1_params.out_dim,
        conv_1_params.stride, 
        1, 1, conv_1_params.padding, 
        conv_1_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_1_in, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out,

        RELU, conv_1_params.output_scale,
        conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding,

        tiled_matmul_type);


    // Conv2
    printf("Starting Conv 2...\n");
    tiled_conv_auto(
        conv_2_params.batch_size, 
        conv_2_params.in_dim, conv_2_params.in_dim, 
        conv_2_params.in_channels, conv_2_params.out_channels, 
        conv_2_params.out_dim, conv_2_params.out_dim,
        conv_2_params.stride, 1, 1, conv_2_params.padding, 
        conv_2_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_1_out, (elem_t*)conv_2_w, (acc_t*)conv_2_b, (elem_t*)conv_1_out,

        RELU, conv_2_params.output_scale,
        conv_2_params.pool_size, conv_2_params.pool_stride, conv_2_params.pool_padding,

        tiled_matmul_type);

    // Conv3
    printf("Starting Conv 3...\n");
    tiled_conv_auto(
        conv_3_params.batch_size,   
        conv_3_params.in_dim, conv_3_params.in_dim, 
        conv_3_params.in_channels, conv_3_params.out_channels, 
        conv_3_params.out_dim, conv_3_params.out_dim,
        conv_3_params.stride, 1, 1, conv_3_params.padding, 
        conv_3_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out,

        RELU, conv_3_params.output_scale,
        conv_3_params.pool_size, conv_3_params.pool_stride, conv_3_params.pool_padding,

        tiled_matmul_type);

    // Conv4
    printf("Starting Conv 4...\n");
    tiled_conv_auto(
        conv_4_params.batch_size,   
        conv_4_params.in_dim, conv_4_params.in_dim, 
        conv_4_params.in_channels, conv_4_params.out_channels, 
        conv_4_params.out_dim, conv_4_params.out_dim,
        conv_4_params.stride, 1, 1, conv_4_params.padding, 
        conv_4_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_3_out, (elem_t*)conv_4_w, (acc_t*)conv_4_b, (elem_t*)conv_4_out,

        RELU, conv_4_params.output_scale,
        conv_4_params.pool_size, conv_4_params.pool_stride, conv_4_params.pool_padding,

        tiled_matmul_type);

    // Conv5
    printf("Starting Conv 5...\n");
    tiled_conv_auto(
        conv_5_params.batch_size,   
        conv_5_params.in_dim, conv_5_params.in_dim, 
        conv_5_params.in_channels, conv_5_params.out_channels, 
        conv_5_params.out_dim, conv_5_params.out_dim,
        conv_5_params.stride, 1, 1, conv_5_params.padding, 
        conv_5_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_4_out, (elem_t*)conv_5_w, (acc_t*)conv_5_b, (elem_t*)conv_5_out,

        RELU, conv_5_params.output_scale,
        conv_5_params.pool_size, conv_5_params.pool_stride, conv_5_params.pool_padding,

        tiled_matmul_type);

    // Conv6
    printf("Starting Conv 6...\n");
    tiled_conv_auto(
        conv_6_params.batch_size,   
        conv_6_params.in_dim, conv_6_params.in_dim, 
        conv_6_params.in_channels, conv_6_params.out_channels, 
        conv_6_params.out_dim, conv_6_params.out_dim,
        conv_6_params.stride, 1, 1, conv_6_params.padding, 
        conv_6_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_5_out, (elem_t*)conv_6_w, (acc_t*)conv_6_b, (elem_t*)conv_6_out,

        RELU, conv_6_params.output_scale,
        conv_6_params.pool_size, conv_6_params.pool_stride, conv_6_params.pool_padding,

        tiled_matmul_type);


    // Conv7
    printf("Starting Conv 7...\n");
    tiled_conv_auto(
        conv_7_params.batch_size,   
        conv_7_params.in_dim, conv_7_params.in_dim, 
        conv_7_params.in_channels, conv_7_params.out_channels, 
        conv_7_params.out_dim, conv_7_params.out_dim, 
        conv_7_params.stride, 1, 1, conv_7_params.padding, 
        conv_7_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_6_out, (elem_t*)conv_7_w, (acc_t*)conv_7_b, (elem_t*)conv_7_out,

        RELU, conv_8_params.output_scale,
        conv_7_params.pool_size, conv_7_params.pool_stride, conv_7_params.pool_padding,

        tiled_matmul_type);

    // Conv8
    printf("Starting Conv 8...\n");
    tiled_conv_auto(
        conv_8_params.batch_size,   
        conv_8_params.in_dim, conv_8_params.in_dim, 
        conv_8_params.in_channels, conv_8_params.out_channels, 
        conv_8_params.out_dim, conv_8_params.out_dim, 
        conv_8_params.stride, 1, 1, conv_8_params.padding, 
        conv_8_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_7_out, (elem_t*)conv_8_w, (acc_t*)conv_8_b, (elem_t*)conv_8_out,

        RELU, conv_8_params.output_scale,
        conv_8_params.pool_size, conv_8_params.pool_stride, conv_8_params.pool_padding,

        tiled_matmul_type);

    // Conv9
    printf("Starting Conv 9...\n");
    tiled_conv_auto(
        conv_9_params.batch_size,   
        conv_9_params.in_dim, conv_9_params.in_dim, 
        conv_9_params.in_channels, conv_9_params.out_channels, 
        conv_9_params.out_dim, conv_9_params.out_dim, 
        conv_9_params.stride, 1, 1, conv_9_params.padding, 
        conv_9_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_8_out, (elem_t*)conv_9_w, (acc_t*)conv_9_b, (elem_t*)conv_9_out,

        RELU, conv_9_params.output_scale,
        conv_9_params.pool_size, conv_9_params.pool_stride, conv_9_params.pool_padding,

        tiled_matmul_type);

    // Conv10
    printf("Starting Conv 10...\n");
    tiled_conv_auto(
        conv_10_params.batch_size,   
        conv_10_params.in_dim, conv_10_params.in_dim, 
        conv_10_params.in_channels, conv_10_params.out_channels, 
        conv_10_params.out_dim, conv_10_params.out_dim,
        conv_10_params.stride, 1, 1, conv_10_params.padding, 
        conv_10_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_9_out, (elem_t*)conv_10_w, (acc_t*)conv_10_b, (elem_t*)conv_10_out,

        RELU, conv_10_params.output_scale,
        conv_10_params.pool_size, conv_10_params.pool_stride, conv_10_params.pool_padding,

        tiled_matmul_type);

    //Conv11
    printf("Starting Conv 11...\n");
    tiled_conv_auto(
        conv_11_params.batch_size,   
        conv_11_params.in_dim, conv_11_params.in_dim, 
        conv_11_params.in_channels, conv_11_params.out_channels, 
        conv_11_params.out_dim, conv_11_params.out_dim, 
        conv_11_params.stride, 1, 1, conv_11_params.padding, 
        conv_11_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_10_out, (elem_t*)conv_11_w, (acc_t*)conv_11_b, (elem_t*)conv_11_out,

        RELU, conv_11_params.output_scale,
        conv_11_params.pool_size, conv_11_params.pool_stride, conv_11_params.pool_padding,

        tiled_matmul_type);

    //Conv12
    printf("Starting Conv 12...\n");
    tiled_conv_auto(
        conv_12_params.batch_size,   
        conv_12_params.in_dim, conv_12_params.in_dim, 
        conv_12_params.in_channels, conv_12_params.out_channels, 
        conv_12_params.out_dim, conv_12_params.out_dim, 
        conv_12_params.stride, 1, 1, conv_12_params.padding, 
        conv_12_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_11_out, (elem_t*)conv_12_w, (acc_t*)conv_12_b, (elem_t*)conv_12_out,

        RELU, conv_12_params.output_scale,
        conv_12_params.pool_size, conv_12_params.pool_stride, conv_12_params.pool_padding,

        tiled_matmul_type);

    //Conv13
    printf("Starting Conv 13...\n");
    tiled_conv_auto(
        conv_13_params.batch_size,   
        conv_13_params.in_dim, conv_13_params.in_dim, 
        conv_13_params.in_channels, conv_13_params.out_channels, 
        conv_13_params.out_dim, conv_13_params.out_dim, 
        conv_13_params.stride, 1, 1, conv_13_params.padding, 
        conv_13_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_12_out, (elem_t*)conv_13_w, (acc_t*)conv_13_b, (elem_t*)conv_13_out,

        RELU, conv_13_params.output_scale,
        conv_13_params.pool_size, conv_13_params.pool_stride, conv_13_params.pool_padding,

        tiled_matmul_type);



    //Auxilliary Convs for SSD
    printf("Starting Conv Aux 6_1...\n");
    tiled_conv_auto(
        conv_aux_6_1_params.batch_size,   
        conv_aux_6_1_params.in_dim, conv_aux_6_1_params.in_dim,  
        conv_aux_6_1_params.in_channels, conv_aux_6_1_params.out_channels, 
        conv_aux_6_1_params.out_dim, conv_aux_6_1_params.out_dim, 
        conv_aux_6_1_params.stride, 1, 1, conv_aux_6_1_params.padding, 
        conv_aux_6_1_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_12_out, (elem_t*)conv_aux_6_1_w, (acc_t*)conv_aux_6_1_b, (elem_t*)conv_aux_6_1_out,

        RELU, conv_aux_6_1_params.output_scale,
        conv_aux_6_1_params.pool_size, conv_aux_6_1_params.pool_stride, conv_aux_6_1_params.pool_padding,

        tiled_matmul_type);
    
    //printf("Starting Conv Aux 6_2...\n");
    //tiled_conv_auto(
    //    conv_aux_6_2_params.batch_size,   conv_aux_6_2_params.in_dim,  conv_aux_6_2_params.in_channels,
    //    conv_aux_6_2_params.out_channels, conv_aux_6_2_params.out_dim,
    //    conv_aux_6_2_params.stride, 1, 1, conv_aux_6_2_params.padding, conv_aux_6_2_params.kernel_size,
    //    false, false, false, false, false,

    //    (elem_t*)conv_aux_6_1_out, (elem_t*)conv_aux_6_2_w, (acc_t*)conv_aux_6_2_b, (elem_t*)conv_aux_6_2_out,

    //    RELU, conv_aux_6_2_params.output_scale,
    //    conv_aux_6_2_params.pool_size, conv_aux_6_2_params.pool_stride, conv_aux_6_2_params.pool_padding,

    //    tiled_matmul_type);

    printf("Starting Conv Aux 7_1...\n");
    tiled_conv_auto(
        conv_aux_7_1_params.batch_size,   
        conv_aux_7_1_params.in_dim,  conv_aux_7_1_params.in_dim,  
        conv_aux_7_1_params.in_channels, conv_aux_7_1_params.out_channels, 
        conv_aux_7_1_params.out_dim, conv_aux_7_1_params.out_dim, 
        conv_aux_7_1_params.stride, 1, 1, conv_aux_7_1_params.padding, 
        conv_aux_7_1_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_6_1_out, (elem_t*)conv_aux_7_1_w, (acc_t*)conv_aux_7_1_b, (elem_t*)conv_aux_7_1_out,

        RELU, conv_aux_7_1_params.output_scale,
        conv_aux_7_1_params.pool_size, conv_aux_7_1_params.pool_stride, conv_aux_7_1_params.pool_padding,

        tiled_matmul_type);
    
    //printf("Starting Conv Aux 7_2...\n");
    //tiled_conv_auto(
    //    conv_aux_7_2_params.batch_size,   conv_aux_7_2_params.in_dim,  conv_aux_7_2_params.in_channels,
    //    conv_aux_7_2_params.out_channels, conv_aux_7_2_params.out_dim,
    //    conv_aux_7_2_params.stride, 1, 1, conv_aux_7_2_params.padding, conv_aux_7_2_params.kernel_size,
    //    false, false, false, false, false,

    //    (elem_t*)conv_aux_7_1_out, (elem_t*)conv_aux_7_2_w, (acc_t*)conv_aux_7_2_b, (elem_t*)conv_aux_7_2_out,

    //    RELU, conv_aux_7_2_params.output_scale,
    //    conv_aux_7_2_params.pool_size, conv_aux_7_2_params.pool_stride, conv_aux_7_2_params.pool_padding,

    //    tiled_matmul_type);


    printf("Starting Conv Aux 8_1...\n");
    tiled_conv_auto(
        conv_aux_8_1_params.batch_size,   
        conv_aux_8_1_params.in_dim,  conv_aux_8_1_params.in_dim,  
        conv_aux_8_1_params.in_channels, conv_aux_8_1_params.out_channels, 
        conv_aux_8_1_params.out_dim, conv_aux_8_1_params.out_dim, 
        conv_aux_8_1_params.stride, 1, 1, conv_aux_8_1_params.padding, 
        conv_aux_8_1_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_7_1_out, (elem_t*)conv_aux_8_1_w, (acc_t*)conv_aux_8_1_b, (elem_t*)conv_aux_8_1_out,

        RELU, conv_aux_8_1_params.output_scale,
        conv_aux_8_1_params.pool_size, conv_aux_8_1_params.pool_stride, conv_aux_8_1_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Aux 8_2...\n");
    tiled_conv_auto(
        conv_aux_8_2_params.batch_size,   
        conv_aux_8_2_params.in_dim,  conv_aux_8_2_params.in_dim,  
        conv_aux_8_2_params.in_channels, conv_aux_8_2_params.out_channels, 
        conv_aux_8_2_params.out_dim, conv_aux_8_2_params.out_dim, 
        conv_aux_8_2_params.stride, 1, 1, conv_aux_8_2_params.padding, 
        conv_aux_8_2_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_8_1_out, (elem_t*)conv_aux_8_2_w, (acc_t*)conv_aux_8_2_b, (elem_t*)conv_aux_8_2_out,

        RELU, conv_aux_8_2_params.output_scale,
        conv_aux_8_2_params.pool_size, conv_aux_8_2_params.pool_stride, conv_aux_8_2_params.pool_padding,

        tiled_matmul_type);


    printf("Starting Conv Aux 9_1...\n");
    tiled_conv_auto(
        conv_aux_9_1_params.batch_size,   
        conv_aux_9_1_params.in_dim, conv_aux_9_1_params.in_dim, 
        conv_aux_9_1_params.in_channels, conv_aux_9_1_params.out_channels, 
        conv_aux_9_1_params.out_dim, conv_aux_9_1_params.out_dim, 
        conv_aux_9_1_params.stride, 1, 1, conv_aux_9_1_params.padding, 
        conv_aux_9_1_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_8_2_out, (elem_t*)conv_aux_9_1_w, (acc_t*)conv_aux_9_1_b, (elem_t*)conv_aux_9_1_out,

        RELU, conv_aux_9_1_params.output_scale,
        conv_aux_9_1_params.pool_size, conv_aux_9_1_params.pool_stride, conv_aux_9_1_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Aux 9_2...\n");
    tiled_conv_auto(
        conv_aux_9_2_params.batch_size,   
        conv_aux_9_2_params.in_dim,  conv_aux_9_2_params.in_dim,  
        conv_aux_9_2_params.in_channels, conv_aux_9_2_params.out_channels, 
        conv_aux_9_2_params.out_dim, conv_aux_9_2_params.out_dim, 
        conv_aux_9_2_params.stride, 1, 1, conv_aux_9_2_params.padding, 
        conv_aux_9_2_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_9_1_out, (elem_t*)conv_aux_9_2_w, (acc_t*)conv_aux_9_2_b, (elem_t*)conv_aux_9_2_out,

        RELU, conv_aux_9_2_params.output_scale,
        conv_aux_9_2_params.pool_size, conv_aux_9_2_params.pool_stride, conv_aux_9_2_params.pool_padding,

        tiled_matmul_type);


    printf("Starting Conv Aux 10_1...\n");
    tiled_conv_auto(
        conv_aux_10_1_params.batch_size,   
        conv_aux_10_1_params.in_dim,  conv_aux_10_1_params.in_dim,  
        conv_aux_10_1_params.in_channels, conv_aux_10_1_params.out_channels, 
        conv_aux_10_1_params.out_dim, conv_aux_10_1_params.out_dim, 
        conv_aux_10_1_params.stride, 1, 1, conv_aux_10_1_params.padding, 
        conv_aux_10_1_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_9_2_out, (elem_t*)conv_aux_10_1_w, (acc_t*)conv_aux_10_1_b, (elem_t*)conv_aux_10_1_out,

        RELU, conv_aux_10_1_params.output_scale,
        conv_aux_10_1_params.pool_size, conv_aux_10_1_params.pool_stride, conv_aux_10_1_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Aux 10_2...\n");
    tiled_conv_auto(
        conv_aux_10_2_params.batch_size,   
        conv_aux_10_2_params.in_dim,  conv_aux_10_2_params.in_dim,  
        conv_aux_10_2_params.in_channels, conv_aux_10_2_params.out_channels, 
        conv_aux_10_2_params.out_dim, conv_aux_10_2_params.out_dim, 
        conv_aux_10_2_params.stride, 1, 1, conv_aux_10_2_params.padding, 
        conv_aux_10_2_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_10_1_out, (elem_t*)conv_aux_10_2_w, (acc_t*)conv_aux_10_2_b, (elem_t*)conv_aux_10_2_out,

        RELU, conv_aux_10_2_params.output_scale,
        conv_aux_10_2_params.pool_size, conv_aux_10_2_params.pool_stride, conv_aux_10_2_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Aux 11_1...\n");
    tiled_conv_auto(
        conv_aux_11_1_params.batch_size,   
        conv_aux_11_1_params.in_dim,  conv_aux_11_1_params.in_dim,  
        conv_aux_11_1_params.in_channels, conv_aux_11_1_params.out_channels, 
        conv_aux_11_1_params.out_dim, conv_aux_11_1_params.out_dim, 
        conv_aux_11_1_params.stride, 1, 1, conv_aux_11_1_params.padding, 
        conv_aux_11_1_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_10_2_out, (elem_t*)conv_aux_11_1_w, (acc_t*)conv_aux_11_1_b, (elem_t*)conv_aux_11_1_out,

        RELU, conv_aux_11_1_params.output_scale,
        conv_aux_11_1_params.pool_size, conv_aux_11_1_params.pool_stride, conv_aux_11_1_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Aux 11_2...\n");
    tiled_conv_auto(
        conv_aux_11_2_params.batch_size,   
        conv_aux_11_2_params.in_dim,  conv_aux_11_2_params.in_dim,  
        conv_aux_11_2_params.in_channels, conv_aux_11_2_params.out_channels, 
        conv_aux_11_2_params.out_dim, conv_aux_11_2_params.out_dim, 
        conv_aux_11_2_params.stride, 1, 1, conv_aux_11_2_params.padding, 
        conv_aux_11_2_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_11_1_out, (elem_t*)conv_aux_11_2_w, (acc_t*)conv_aux_11_2_b, (elem_t*)conv_aux_11_2_out,

        RELU, conv_aux_11_2_params.output_scale,
        conv_aux_11_2_params.pool_size, conv_aux_11_2_params.pool_stride, conv_aux_11_2_params.pool_padding,

        tiled_matmul_type);

    /*
    *   Localization Prediction
    */
    printf("Starting Conv Loc Pred 1...\n");
    tiled_conv_auto(
        conv_loc_pred_1_params.batch_size,   
        conv_loc_pred_1_params.in_dim,  conv_loc_pred_1_params.in_dim,  
        conv_loc_pred_1_params.in_channels, conv_loc_pred_1_params.out_channels, 
        conv_loc_pred_1_params.out_dim, conv_loc_pred_1_params.out_dim, 
        conv_loc_pred_1_params.stride, 1, 1, conv_loc_pred_1_params.padding, 
        conv_loc_pred_1_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_10_out, (elem_t*)conv_loc_pred_1_w, (acc_t*)conv_loc_pred_1_b, (elem_t*)conv_loc_pred_1_out,

        RELU, conv_loc_pred_1_params.output_scale,
        conv_loc_pred_1_params.pool_size, conv_loc_pred_1_params.pool_stride, conv_loc_pred_1_params.pool_padding,

        tiled_matmul_type);


    printf("Starting Conv Loc Pred 2...\n");
    tiled_conv_auto(
        conv_loc_pred_2_params.batch_size,   
        conv_loc_pred_2_params.in_dim,  conv_loc_pred_2_params.in_dim,  
        conv_loc_pred_2_params.in_channels, conv_loc_pred_2_params.out_channels, 
        conv_loc_pred_2_params.out_dim, conv_loc_pred_2_params.out_dim, 
        conv_loc_pred_2_params.stride, 1, 1, conv_loc_pred_2_params.padding, 
        conv_loc_pred_2_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_7_1_out, (elem_t*)conv_loc_pred_2_w, (acc_t*)conv_loc_pred_2_b, (elem_t*)conv_loc_pred_2_out,

        RELU, conv_loc_pred_2_params.output_scale,
        conv_loc_pred_2_params.pool_size, conv_loc_pred_2_params.pool_stride, conv_loc_pred_2_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Loc Pred 3...\n");
    tiled_conv_auto(
        conv_loc_pred_3_params.batch_size,   
        conv_loc_pred_3_params.in_dim,  conv_loc_pred_3_params.in_dim,  
        conv_loc_pred_3_params.in_channels, conv_loc_pred_3_params.out_channels, 
        conv_loc_pred_3_params.out_dim, conv_loc_pred_3_params.out_dim, 
        conv_loc_pred_3_params.stride, 1, 1, conv_loc_pred_3_params.padding, 
        conv_loc_pred_3_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_8_2_out, (elem_t*)conv_loc_pred_3_w, (acc_t*)conv_loc_pred_3_b, (elem_t*)conv_loc_pred_3_out,

        RELU, conv_loc_pred_3_params.output_scale,
        conv_loc_pred_3_params.pool_size, conv_loc_pred_3_params.pool_stride, conv_loc_pred_3_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Loc Pred 4...\n");
    tiled_conv_auto(
        conv_loc_pred_4_params.batch_size,   
        conv_loc_pred_4_params.in_dim,  conv_loc_pred_4_params.in_dim,  
        conv_loc_pred_4_params.in_channels, conv_loc_pred_4_params.out_channels, 
        conv_loc_pred_4_params.out_dim, conv_loc_pred_4_params.out_dim, 
        conv_loc_pred_4_params.stride, 1, 1, conv_loc_pred_4_params.padding, 
        conv_loc_pred_4_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_9_2_out, (elem_t*)conv_loc_pred_4_w, (acc_t*)conv_loc_pred_4_b, (elem_t*)conv_loc_pred_4_out,

        RELU, conv_loc_pred_4_params.output_scale,
        conv_loc_pred_4_params.pool_size, conv_loc_pred_4_params.pool_stride, conv_loc_pred_4_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Loc Pred 5...\n");
    tiled_conv_auto(
        conv_loc_pred_5_params.batch_size,   
        conv_loc_pred_5_params.in_dim,  conv_loc_pred_5_params.in_dim,  
        conv_loc_pred_5_params.in_channels, conv_loc_pred_5_params.out_channels, 
        conv_loc_pred_5_params.out_dim, conv_loc_pred_5_params.out_dim, 
        conv_loc_pred_5_params.stride, 1, 1, conv_loc_pred_5_params.padding, 
        conv_loc_pred_5_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_10_2_out, (elem_t*)conv_loc_pred_5_w, (acc_t*)conv_loc_pred_5_b, (elem_t*)conv_loc_pred_5_out,

        RELU, conv_loc_pred_5_params.output_scale,
        conv_loc_pred_5_params.pool_size, conv_loc_pred_5_params.pool_stride, conv_loc_pred_5_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Loc Pred 6...\n");
    tiled_conv_auto(
        conv_loc_pred_6_params.batch_size,   
        conv_loc_pred_6_params.in_dim,  conv_loc_pred_6_params.in_dim,  
        conv_loc_pred_6_params.in_channels, conv_loc_pred_6_params.out_channels, 
        conv_loc_pred_6_params.out_dim, conv_loc_pred_6_params.out_dim, 
        conv_loc_pred_6_params.stride, 1, 1, conv_loc_pred_6_params.padding, 
        conv_loc_pred_6_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_11_2_out, (elem_t*)conv_loc_pred_6_w, (acc_t*)conv_loc_pred_6_b, (elem_t*)conv_loc_pred_6_out,

        RELU, conv_loc_pred_6_params.output_scale,
        conv_loc_pred_6_params.pool_size, conv_loc_pred_6_params.pool_stride, conv_loc_pred_6_params.pool_padding,

        tiled_matmul_type);

   /*
   *    Class Prediction
   */
    printf("Starting Conv Class Pred 1...\n");
    tiled_conv_auto(
        conv_class_pred_1_params.batch_size,   
        conv_class_pred_1_params.in_dim,  conv_class_pred_1_params.in_dim,  
        conv_class_pred_1_params.in_channels, conv_class_pred_1_params.out_channels, 
        conv_class_pred_1_params.out_dim, conv_class_pred_1_params.out_dim, 
        conv_class_pred_1_params.stride, 1, 1, conv_class_pred_1_params.padding, 
        conv_class_pred_1_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_10_out, (elem_t*)conv_class_pred_1_w, (acc_t*)conv_class_pred_1_b, (elem_t*)conv_class_pred_1_out,

        RELU, conv_class_pred_1_params.output_scale,
        conv_class_pred_1_params.pool_size, conv_class_pred_1_params.pool_stride, conv_class_pred_1_params.pool_padding,

        tiled_matmul_type);


    printf("Starting Conv Class Pred 2...\n");
    tiled_conv_auto(
        conv_class_pred_2_params.batch_size,   
        conv_class_pred_2_params.in_dim,  conv_class_pred_2_params.in_dim,  
        conv_class_pred_2_params.in_channels, conv_class_pred_2_params.out_channels, 
        conv_class_pred_2_params.out_dim, conv_class_pred_2_params.out_dim, 
        conv_class_pred_2_params.stride, 1, 1, conv_class_pred_2_params.padding, 
        conv_class_pred_2_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_7_1_out, (elem_t*)conv_class_pred_2_w, (acc_t*)conv_class_pred_2_b, (elem_t*)conv_class_pred_2_out,

        RELU, conv_class_pred_2_params.output_scale,
        conv_class_pred_2_params.pool_size, conv_class_pred_2_params.pool_stride, conv_class_pred_2_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Class Pred 3...\n");
    tiled_conv_auto(
        conv_class_pred_3_params.batch_size,   
        conv_class_pred_3_params.in_dim,  conv_class_pred_3_params.in_dim,  
        conv_class_pred_3_params.in_channels, conv_class_pred_3_params.out_channels, 
        conv_class_pred_3_params.out_dim, conv_class_pred_3_params.out_dim, 
        conv_class_pred_3_params.stride, 1, 1, conv_class_pred_3_params.padding, 
        conv_class_pred_3_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_8_2_out, (elem_t*)conv_class_pred_3_w, (acc_t*)conv_class_pred_3_b, (elem_t*)conv_class_pred_3_out,

        RELU, conv_class_pred_3_params.output_scale,
        conv_class_pred_3_params.pool_size, conv_class_pred_3_params.pool_stride, conv_class_pred_3_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Class Pred 4...\n");
    tiled_conv_auto(
        conv_class_pred_4_params.batch_size,   
        conv_class_pred_4_params.in_dim,  conv_class_pred_4_params.in_dim,  
        conv_class_pred_4_params.in_channels, conv_class_pred_4_params.out_channels, 
        conv_class_pred_4_params.out_dim, conv_class_pred_4_params.out_dim, 
        conv_class_pred_4_params.stride, 1, 1, conv_class_pred_4_params.padding, 
        conv_class_pred_4_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_9_2_out, (elem_t*)conv_class_pred_4_w, (acc_t*)conv_class_pred_4_b, (elem_t*)conv_class_pred_4_out,

        RELU, conv_class_pred_4_params.output_scale,
        conv_class_pred_4_params.pool_size, conv_class_pred_4_params.pool_stride, conv_class_pred_4_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Class Pred 5...\n");
    tiled_conv_auto(
        conv_class_pred_5_params.batch_size,   
        conv_class_pred_5_params.in_dim,  conv_class_pred_5_params.in_dim,  
        conv_class_pred_5_params.in_channels, conv_class_pred_5_params.out_channels, 
        conv_class_pred_5_params.out_dim, conv_class_pred_5_params.out_dim, 
        conv_class_pred_5_params.stride, 1, 1, conv_class_pred_5_params.padding, 
        conv_class_pred_5_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_10_2_out, (elem_t*)conv_class_pred_5_w, (acc_t*)conv_class_pred_5_b, (elem_t*)conv_class_pred_5_out,

        RELU, conv_class_pred_5_params.output_scale,
        conv_class_pred_5_params.pool_size, conv_class_pred_5_params.pool_stride, conv_class_pred_5_params.pool_padding,

        tiled_matmul_type);

    printf("Starting Conv Class Pred 6...\n");
    tiled_conv_auto(
        conv_class_pred_6_params.batch_size,   
        conv_class_pred_6_params.in_dim,  conv_class_pred_6_params.in_dim,  
        conv_class_pred_6_params.in_channels, conv_class_pred_6_params.out_channels, 
        conv_class_pred_6_params.out_dim, conv_class_pred_6_params.out_dim, 
        conv_class_pred_6_params.stride, 1, 1, conv_class_pred_6_params.padding, 
        conv_class_pred_6_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_aux_11_2_out, (elem_t*)conv_class_pred_6_w, (acc_t*)conv_class_pred_6_b, (elem_t*)conv_class_pred_6_out,

        RELU, conv_class_pred_6_params.output_scale,
        conv_class_pred_6_params.pool_size, conv_class_pred_6_params.pool_stride, conv_class_pred_6_params.pool_padding,

        tiled_matmul_type);

}