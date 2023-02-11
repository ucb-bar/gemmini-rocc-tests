#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define HEAP_SIZE (8*1024*1024)

int str2int(char * str)
{
    int res = 0;
    for (int i = 0; str[i] != '\0'; ++i)
        res = res * 10 + str[i] - '0';
    return res;
}

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    int BATCH_SIZE = 3;
    int IN_DIM = 112;
    int CHANNELS = 17;
    int KERNEL_DIM = 3;
    int PADDING = 1;
    int STRIDE = 2;
    bool NO_BIAS = false;

    if (argc == 8) {
      BATCH_SIZE = str2int(argv[1]);
      IN_DIM = str2int(argv[2]);
      CHANNELS = str2int(argv[3]);
      KERNEL_DIM = str2int(argv[4]);
      PADDING = str2int(argv[5]);
      STRIDE = str2int(argv[6]);
      NO_BIAS = str2int(argv[7]);
    } else if (argc > 1) {
      printf("BATCH_SIZE IN_DIM CHANNELS KERNEL_DIM PADDING STRIDE NO_BIAS\n");
      exit(1);
    }

    int OUT_DIM = ((IN_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1);

    printf("BATCH_SIZE = %d\n", BATCH_SIZE);
    printf("IN_DIM = %d\n", IN_DIM);
    printf("CHANNELS = %d\n", CHANNELS);
    printf("KERNEL_DIM = %d\n", KERNEL_DIM);
    printf("PADDING = %d\n", PADDING);
    printf("STRIDE = %d\n", STRIDE);
    printf("NO_BIAS = %d\n", NO_BIAS);

    gemmini_flush(0);

    printf("Output dimension: %u\n\n", OUT_DIM);

    static uint8_t heap[HEAP_SIZE];

    // static elem_t input[BATCH_SIZE][IN_DIM][IN_DIM][CHANNELS];
    // static elem_t weights[CHANNELS][KERNEL_DIM][KERNEL_DIM];
    // static acc_t bias[CHANNELS];
    // static elem_t output[BATCH_SIZE][OUT_DIM][OUT_DIM][CHANNELS];

    elem_t * input = (elem_t*)(&heap[0]);
    elem_t * weights = (elem_t*)((elem_t*)input + BATCH_SIZE*IN_DIM*IN_DIM*CHANNELS);
    acc_t * bias = (acc_t*)((elem_t*)weights + CHANNELS*KERNEL_DIM*KERNEL_DIM);
    elem_t * output = (elem_t*)((acc_t*)bias + CHANNELS);

    {
      uint8_t * end = (uint8_t*)((elem_t*)output + BATCH_SIZE*OUT_DIM*OUT_DIM*CHANNELS);
      if (end >= &heap[HEAP_SIZE]) {
          printf("problem size is too large to fit in memory");
          exit(1);
      }
    }

    printf("Gemmini conv...\n");
    uint64_t start_gemmini = read_cycles();

    tiled_conv_dw_auto(BATCH_SIZE, IN_DIM, IN_DIM, CHANNELS, OUT_DIM, OUT_DIM,
        STRIDE, PADDING, KERNEL_DIM,

        (elem_t*)input,
        (elem_t*)weights,
        (acc_t*)bias,
        (elem_t*)output,

        NO_ACTIVATION, ACC_SCALE_IDENTITY, 1, 0, 0,

        WS);

    uint64_t end_gemmini = read_cycles();
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);

    return 0;
}

