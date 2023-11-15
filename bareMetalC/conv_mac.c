#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#ifndef BAREMETAL

#define BATCH_SIZE 4
#define IN_ROW_DIM 224
#define IN_COL_DIM 224
#define IN_CHANNELS 3
#define OUT_CHANNELS 32
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 2

#else

#define BATCH_SIZE 1
#define IN_ROW_DIM 14
#define IN_COL_DIM 14
#define IN_CHANNELS 128
#define OUT_CHANNELS 128
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 1
#endif

#define NO_BIAS false

#define OUT_ROW_DIM ((IN_ROW_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
#define OUT_COL_DIM ((IN_COL_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
#define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)
#define N_PATCHES (BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM)

#define MAC 1

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    int cfgid = 0;
    int i = 0;
    //for(int i = 0; i < 2; i++){
        bool acquired = rr_acquire_single(cfgid, i);
        if(acquired){
            printf("gemmini %d acquired to cfgid %d\n", i, cfgid);
            //break;
        }
    //}
    rr_set_opc(XCUSTOM_ACC, cfgid);
    gemmini_flush(0);

    // assert((in_dim + 2*padding - kernel_dim) % stride == 0);

    printf("Input dimensions (rows by columns): %u by %u\n", IN_ROW_DIM, IN_COL_DIM);
    printf("Output dimensions (rows by columns): %u by %u\n\n", OUT_ROW_DIM, OUT_COL_DIM);

    static elem_t input[BATCH_SIZE][IN_ROW_DIM][IN_COL_DIM][IN_CHANNELS];
    //static elem_t weights_mat[OUT_CHANNELS][KERNEL_DIM][KERNEL_DIM][IN_CHANNELS];
    static acc_t bias[OUT_CHANNELS];
    static elem_t weights_mat[PATCH_SIZE][OUT_CHANNELS];
    static elem_t output_mat[N_PATCHES][OUT_CHANNELS];
    //static elem_t output[BATCH_SIZE][OUT_ROW_DIM][OUT_COL_DIM][OUT_CHANNELS];

    printf("Gemmini conv...\n");
    uint64_t start_gemmini = read_cycles();
    tiled_conv_auto(
        BATCH_SIZE, IN_ROW_DIM, IN_COL_DIM, IN_CHANNELS,
        OUT_CHANNELS, OUT_ROW_DIM, OUT_COL_DIM,
        STRIDE, 1, 1, PADDING, KERNEL_DIM,
        false, false, false, false, false,

        MAC == 1 ? NULL: (elem_t*)input,
        MAC == 1 ? NULL: (elem_t*)weights_mat,
        (NO_BIAS || MAC==1) ? NULL : (acc_t*)bias,
        MAC == 1 ? NULL : (elem_t*)output_mat,

        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0,

        WS);
    rr_fence(cfgid);
    uint64_t end_gemmini = read_cycles();
    rr_release(cfgid);
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);
    const uint64_t total_macs = BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM * KERNEL_DIM * KERNEL_DIM * IN_CHANNELS * OUT_CHANNELS;
    const uint64_t ideal_cycles = total_macs / (DIM * DIM);
    const uint64_t utilization = 100 * ideal_cycles / (end_gemmini-start_gemmini);
    printf("Total macs: %llu\n", total_macs);
    printf("Ideal cycles: %llu\n", ideal_cycles);
    printf("Utilization: %llu%%\n", utilization);


    return 0;
}
