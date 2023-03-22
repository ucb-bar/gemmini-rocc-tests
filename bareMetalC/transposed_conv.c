#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define BATCH_SIZE 1
#define IN_DIM 4
#define PADDING 0
#define CHANNELS 1
#define KERNEL_DIM 2
#define STRIDE 2
#define OUTPUT_PADDING 0

#define A ((IN_DIM + 2*PADDING - KERNEL_DIM) % STRIDE)
#define ZERO_PAD_TRANS (KERNEL_DIM-PADDING-1)
#define STRETCHED_IN_DIM (IN_DIM + (STRIDE-1)*(IN_DIM-1))
#define A_PADDED_IN_DIM (STRETCHED_IN_DIM + A)
#define OUT_DIM  ((IN_DIM-1)*STRIDE - 2*PADDING + (KERNEL_DIM-1) + 1 + OUTPUT_PADDING)

#define FAST

//void transposed_conv_cpu(elem_t In[IN_DIM][IN_DIM][CHANNELS],
//                        elem_t Wght[CHANNELS][KERNEL_DIM][KERNEL_DIM][CHANNELS],
//                        elem_t Out[OUT_DIM][OUT_DIM][CHANNELS])
//{
//
//}

void init_random(elem_t * buf, int len, elem_t init) {
    elem_t i = 0;
    for (elem_t * ptr = buf; ptr < buf + len; ptr++) {
        // *ptr = (rand() % 32) - 16;
#ifdef FAST
      *ptr = init;
#else
      *ptr = (rand() % 5) - 2;
#endif
    }
}

void dump_matrix(elem_t *  buf, int len, const char * filename){
    for (elem_t * ptr = buf; ptr < buf + len; ptr++) {
        printf("%d\n", *ptr);
    }
}

int main (int argc, char * argv[]) {
    #ifndef BAREMETAL
        if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
        }
    #endif

    printf("IN_DIM: %d\n", IN_DIM);
    //printf("STRETCH_IN: %d\n", STRETCHED_IN_DIM);
    //printf("A: %d\n", A);
    //printf("PADDED_IN_DIM: %d\n", A_PADDED_IN_DIM);
    printf("KERNEL_DIM: %d\n", KERNEL_DIM);
    printf("OUT_DIM: %d\n", OUT_DIM);

    elem_t input[BATCH_SIZE][IN_DIM][IN_DIM][CHANNELS];
    elem_t weights[CHANNELS][KERNEL_DIM][KERNEL_DIM][CHANNELS];
    elem_t output[BATCH_SIZE][OUT_DIM][OUT_DIM][CHANNELS];

    init_random(&input[0][0][0][0], sizeof(input)/sizeof(elem_t), 1);
    init_random(&weights[0][0][0][0], sizeof(weights)/sizeof(elem_t), 1);
    init_random(&output[0][0][0][0], sizeof(output)/sizeof(elem_t), -1);
    
    dump_matrix(&output[0][0][0][0], sizeof(output)/sizeof(elem_t), "test_data.txt");
    dump_matrix(&input[0][0][0][0], sizeof(input)/sizeof(elem_t), "test_data.txt");
    dump_matrix(&weights[0][0][0][0], sizeof(weights)/sizeof(elem_t), "test_data.txt");

    //for(int bs = 0; bs < BATCH_SIZE; bs++){
    //    for(int row = 0; row < IN_DIM; row++){
    //        if(row == IN_DIM-1){
    //            for(int col = 0; col < IN_DIM; col++){
    //                for(int chan = 0; chan < CHANNELS; chan++){
    //                    input[bs][row][col][chan] = 0;
    //                }
    //            }
    //        }
    //        else{
    //            for(int chan = 0; chan < CHANNELS; chan++){
    //                input[bs][row][IN_DIM-1][chan] = 0;
    //            }
    //            continue;
    //        }
    //    }
    //}

    gemmini_flush(0);

    printf("Gemmini transposed conv...\n");
    uint64_t start_gemmini = read_cycles();

    tiled_transposed_conv_auto(
        BATCH_SIZE, IN_DIM, CHANNELS, OUT_DIM,
        STRIDE, PADDING, KERNEL_DIM,

        (elem_t*)input,
        (elem_t*)weights,
        (elem_t*)output,

        ACC_SCALE_IDENTITY
    );

    uint64_t end_gemmini = read_cycles();
    printf("Gemmini transposed conv took %llu cycles\n", end_gemmini - start_gemmini);

    dump_matrix(&output[0][0][0][0], sizeof(output)/sizeof(elem_t), "test_data.txt");

    //printf("Slow CPU transposed conv...\n");

    return 0;
}

