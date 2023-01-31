#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define HEAP_SIZE (4*1024*1024)

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

    int BATCH_SIZE = 4;
    int IN_DIM = 224;
    int IN_CHANNELS = 3;
    int OUT_CHANNELS = 32;
    int KERNEL_DIM = 3;
    int PADDING = 1;
    int STRIDE = 2;
    bool NO_BIAS = false;

    if (argc == 9) {
      BATCH_SIZE = str2int(argv[1]);
      IN_DIM = str2int(argv[2]);
      IN_CHANNELS = str2int(argv[3]);
      OUT_CHANNELS = str2int(argv[4]);
      KERNEL_DIM = str2int(argv[5]);
      PADDING = str2int(argv[6]);
      STRIDE = str2int(argv[7]);
      NO_BIAS = str2int(argv[8]);
    } else if (argc > 1) {
      printf("BATCH_SIZE IN_DIM IN_CHANNELS OUT_CHANNELS KERNEL_DIM PADDING STRIDE NO_BIAS\n");
      exit(1);
    }

    int OUT_DIM = ((IN_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1);

    printf("BATCH_SIZE = %d\n", BATCH_SIZE);
    printf("IN_DIM = %d\n", IN_DIM);
    printf("IN_CHANNELS = %d\n", IN_CHANNELS);
    printf("OUT_CHANNELS = %d\n", OUT_CHANNELS);
    printf("KERNEL_DIM = %d\n", KERNEL_DIM);
    printf("PADDING = %d\n", PADDING);
    printf("STRIDE = %d\n", STRIDE);
    printf("NO_BIAS = %d\n", NO_BIAS);
    printf("Output dimension: %u\n\n", OUT_DIM);

    bool map_to_matmul = KERNEL_DIM == 1 && PADDING == 0 && STRIDE == 1;
    int I = BATCH_SIZE * OUT_DIM * OUT_DIM;
    int J = OUT_CHANNELS;
    int K = KERNEL_DIM * KERNEL_DIM * IN_CHANNELS;

    if (map_to_matmul) {
      printf("I = %d\n", I);
      printf("J = %d\n", J);
      printf("K = %d\n", K);
    }

    gemmini_flush(0);

    static uint8_t heap[HEAP_SIZE];

    // static elem_t input[BATCH_SIZE][IN_DIM][IN_DIM][IN_CHANNELS];
    // static elem_t weights[OUT_CHANNELS][KERNEL_DIM][KERNEL_DIM][IN_CHANNELS];
    // static acc_t bias[OUT_CHANNELS];
    // static elem_t output[BATCH_SIZE][OUT_DIM][OUT_DIM][OUT_CHANNELS];

    elem_t * input = (elem_t*)(&heap[0]);
    elem_t * weights = (elem_t*)((elem_t*)input + BATCH_SIZE*IN_DIM*IN_DIM*IN_CHANNELS);
    acc_t * bias = (acc_t*)((elem_t*)weights + OUT_CHANNELS*KERNEL_DIM*KERNEL_DIM*IN_CHANNELS);
    elem_t * output = (elem_t*)((acc_t*)bias + OUT_CHANNELS);

    {
      uint8_t * end = (uint8_t*)((elem_t*)output + BATCH_SIZE*OUT_DIM*OUT_DIM*OUT_CHANNELS);
      if (end >= &heap[HEAP_SIZE]) {
          printf("problem size is too large to fit in memory");
          exit(1);
      }
    }

    printf("Gemmini conv...\n");
    uint64_t start_gemmini = read_cycles();

    if (map_to_matmul) {

      tiled_matmul_auto(I, J, K,
          input, weights, NO_BIAS ? NULL : bias, output,
          K, J, J, J,
          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, true,
          false, false,
          false, false,
          0,
          WS);

    } else {

      tiled_conv_auto(
          BATCH_SIZE, IN_DIM, IN_DIM, IN_CHANNELS,
          OUT_CHANNELS, OUT_DIM, OUT_DIM,
          STRIDE, 1, 1, PADDING, KERNEL_DIM,
          false, false, false, false, false,

          (elem_t*)input,
          (elem_t*)weights,
          NO_BIAS ? NULL : (acc_t*)bias,
          (elem_t*)output,

          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0,

          WS);
    }

    uint64_t end_gemmini = read_cycles();
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);

    return 0;
}

