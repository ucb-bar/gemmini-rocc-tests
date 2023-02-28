#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "gemmini_fftradnet_dh_params.h"

// The NN model is FFTRadNet
// Below is only a part of it - Detection_Header (DH)

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type=WS;

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

    bool check=false;
    
    if (argc < 3) {
        check = false;
    } else if (strcmp(argv[2], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool conv=true;
    
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
    uint64_t conv_cycles = 0;

    // conv_1
    start = read_cycles();

    tiled_conv_auto(
        conv_1_params.batch_size, conv_1_params.in_row_dim, conv_1_params.in_col_dim, conv_1_params.in_channels,
        conv_1_params.out_channels, conv_1_params.out_row_dim, conv_1_params.out_col_dim,
        conv_1_params.stride, 1, 1, conv_1_params.padding, conv_1_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_1_in, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out,

        NO_ACTIVATION, conv_1_params.output_scale,
        conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding,

        tiled_matmul_type);


    end = read_cycles();
    conv_cycles += end - start;
    
    printf("conv_1 cycles: %llu \n", end - start);

    // conv_2
    start = read_cycles();

    tiled_conv_auto(
        conv_2_params.batch_size, conv_2_params.in_row_dim, conv_2_params.in_col_dim, conv_2_params.in_channels,
        conv_2_params.out_channels, conv_2_params.out_row_dim, conv_2_params.out_col_dim,
        conv_2_params.stride, 1, 1, conv_2_params.padding, conv_2_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_1_out, (elem_t*)conv_2_w, (acc_t*)conv_2_b, (elem_t*)conv_2_out,

        NO_ACTIVATION, conv_2_params.output_scale,
        conv_2_params.pool_size, conv_2_params.pool_stride, conv_2_params.pool_padding,

        tiled_matmul_type);


    end = read_cycles();
    conv_cycles += end - start;
    
    printf("conv_2 cycles: %llu \n", end - start);

    // conv_3
    start = read_cycles();

    tiled_conv_auto(
        conv_3_params.batch_size, conv_3_params.in_row_dim, conv_3_params.in_col_dim, conv_3_params.in_channels,
        conv_3_params.out_channels, conv_3_params.out_row_dim, conv_3_params.out_col_dim, 
        conv_3_params.stride, 1, 1, conv_3_params.padding, conv_3_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out,

        NO_ACTIVATION, conv_3_params.output_scale,
        conv_3_params.pool_size, conv_3_params.pool_stride, conv_3_params.pool_padding,

        tiled_matmul_type);


    end = read_cycles();
    conv_cycles += end - start;
    
    printf("conv_3 cycles: %llu \n", end - start);

    // conv_4
    start = read_cycles();

    tiled_conv_auto(
        conv_4_params.batch_size, conv_4_params.in_row_dim, conv_4_params.in_col_dim, conv_4_params.in_channels,
        conv_4_params.out_channels, conv_4_params.out_row_dim, conv_4_params.out_col_dim,
        conv_4_params.stride, 1, 1, conv_4_params.padding, conv_4_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_3_out, (elem_t*)conv_4_w, (acc_t*)conv_4_b, (elem_t*)conv_4_out,

        NO_ACTIVATION, conv_4_params.output_scale,
        conv_4_params.pool_size, conv_4_params.pool_stride, conv_4_params.pool_padding,

        tiled_matmul_type);


    end = read_cycles();
    conv_cycles += end - start;
    
    printf("conv_4 cycles: %llu \n", end - start);

    // conv_5
    start = read_cycles();

    tiled_conv_auto(
        conv_5_params.batch_size, conv_5_params.in_row_dim, conv_5_params.in_col_dim, conv_5_params.in_channels,
        conv_5_params.out_channels, conv_5_params.out_row_dim, conv_5_params.out_col_dim,
        conv_5_params.stride, 1, 1, conv_5_params.padding, conv_5_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_4_out, (elem_t*)conv_5_w, (acc_t*)conv_5_b, (elem_t*)conv_5_out,

        NO_ACTIVATION, conv_5_params.output_scale,
        conv_5_params.pool_size, conv_5_params.pool_stride, conv_5_params.pool_padding,

        tiled_matmul_type);


    end = read_cycles();
    conv_cycles += end - start;
    
    printf("conv_5 cycles: %llu \n", end - start);

    // conv_6
    start = read_cycles();

    tiled_conv_auto(
        conv_6_params.batch_size, conv_6_params.in_row_dim, conv_6_params.in_col_dim, conv_6_params.in_channels,
        conv_6_params.out_channels, conv_6_params.out_row_dim, conv_6_params.out_col_dim,
        conv_6_params.stride, 1, 1, conv_6_params.padding, conv_6_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_5_out, (elem_t*)conv_6_w, (acc_t*)conv_6_b, (elem_t*)conv_6_out,

        NO_ACTIVATION, conv_6_params.output_scale,
        conv_6_params.pool_size, conv_6_params.pool_stride, conv_6_params.pool_padding,

        tiled_matmul_type);


    end = read_cycles();
    conv_cycles += end - start;
    
    printf("conv_6 cycles: %llu \n", end - start);

    /* Estimation: */
    uint64_t total_cycles = conv_cycles;
    printf("\nTotal cycles: %llu (100%%)\n", total_cycles);
    printf("Conv cycles: %llu (%d%%)\n", conv_cycles, (conv_cycles * 100) / total_cycles);

    return 0;
}

