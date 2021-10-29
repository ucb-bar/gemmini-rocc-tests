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

    enum tiled_matmul_type_t tiled_matmul_type = WS;
    bool conv = true;

    uint64_t start, end, total_cycles;

    // conv_6
    tiled_matmul_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
        conv_5_out, conv_6_w, conv_6_b, conv_6_out,
        conv_6_params.K, conv_6_params.J, conv_6_params.J, conv_6_params.J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        RELU, conv_6_params.output_scale, 0, true,
        false, false,
        false, false,
        0, WS);

    // conv_7
    tiled_conv_A_stride_auto(
        conv_7_params.batch_size, conv_7_params.in_dim, conv_7_params.in_channels,
        conv_7_params.out_channels, conv_7_params.out_dim,
        conv_7_params.stride, 1, 1, conv_7_params.padding, conv_7_params.kernel_size,
        false, false, false, false, false,

        (elem_t*)conv_6_out, (elem_t*)conv_7_w, (acc_t*)conv_7_b, (elem_t*)conv_7_out,

        RELU, conv_7_params.output_scale, 0,
        conv_7_params.pool_size, 0, conv_7_params.pool_padding,

        WS);

    // conv_8
    tiled_matmul_auto(conv_8_params.I, conv_8_params.J, conv_8_params.K,
        conv_7_out, conv_8_w, conv_8_b, conv_8_out,
        conv_8_params.K, conv_8_params.J, conv_8_params.J, conv_8_params.J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, conv_8_params.output_scale, 0, true,
        false, false,
        false, false,
        0, WS);

    // Add residuals
    start = read_cycles();

    /*
     * Add code here to configure "total_bytes_read" and "total_read_latency"
     * counters.
     */
    uint32_t total_bytes_read, total_read_latency;

    counter_configure(0, RDMA_BYTES_REC);
    counter_configure(1, RDMA_BYTES_REC);

    tiled_resadd_auto(conv_8_params.I, conv_8_params.J,
        conv_8_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_5_out,
        conv_8_out,
        conv_8_out,
        true,
        WS);

    end = read_cycles();
    total_cycles = end - start;

    /*
     * Add code here to read "total_bytes_read" and "total_read_latency"
     * counters.
     */
    total_bytes_read = counter_read(0);
    total_read_latency = counter_read(1);

    printf("Total cycles taken for resadd: %llu\n", total_cycles);
    printf("Read DMA bandwidth (bytes/cycle): %llu\n", total_bytes_read / total_cycles);
    printf("DMA request latency: %llu\n", total_read_latency);

    exit(0);
}
