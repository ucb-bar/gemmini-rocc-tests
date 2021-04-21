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
#define IN_DIM 224
#define CHANNELS 3

#define POOL_SIZE 3
#define POOL_STRIDE 2
#define POOL_PADDING 1

#else

#ifdef FAST
#define IN_DIM 9
#define CHANNELS 5
#else
#define IN_DIM 17
#define CHANNELS 18
#endif

#define BATCH_SIZE 2

#define POOL_SIZE 3
#define POOL_STRIDE 2
#define POOL_PADDING 1

#endif

#define OUT_DIM ((IN_DIM + 2*POOL_PADDING - POOL_SIZE) / POOL_STRIDE + 1)

void pool(int batch_size, int channels, int in_dim, int out_dim,
        int window_dim, int stride, int padding,
        elem_t input[batch_size][in_dim][in_dim][channels],
        elem_t output[batch_size][out_dim][out_dim][channels]) {

    for (int b = 0; b < batch_size; b++) {
        for (int orow = 0; orow < out_dim; orow++) {
            for (int ocol = 0; ocol < out_dim; ocol++) {
                for (int ch = 0; ch < channels; ch++) {
                    output[b][orow][ocol][ch] = elem_t_min;

                    for (int wrow = 0; wrow < window_dim; wrow++) {
                        for (int wcol = 0; wcol < window_dim; wcol++) {
                            int irow = orow * stride + wrow - padding;
                            int icol = ocol * stride + wcol - padding;

                            elem_t pixel = irow < 0 || irow >= in_dim ||
                                icol < 0 || icol >= in_dim ?
                                0 : input[b][irow][icol][ch];

                            if (pixel > output[b][orow][ocol][ch]) {
                                output[b][orow][ocol][ch] = pixel;
                            }
                        }
                    }
                }
            }
        }
    }
}

bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i])
            return false;
    return true;
}

void init_random(elem_t * buf, int len) {
    elem_t i = 0;
    for (elem_t * ptr = buf; ptr < buf + len; ptr++) {
        *ptr = (rand() % 5) - 2;
    }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    printf("Output dimension: %u\n", OUT_DIM);

    static elem_t input[BATCH_SIZE][IN_DIM][IN_DIM][CHANNELS];
    static elem_t output[BATCH_SIZE][OUT_DIM][OUT_DIM][CHANNELS];

    printf("Randomize inputs...\n");
    init_random(&input[0][0][0][0], sizeof(input) / sizeof(elem_t));

    printf("CPU pool...\n");
    uint64_t start_cpu_pool = read_cycles();
    pool(BATCH_SIZE, CHANNELS, IN_DIM, OUT_DIM,
            POOL_SIZE, POOL_STRIDE, POOL_PADDING,
            input,
            output);
    uint64_t end_cpu_pool = read_cycles();
    printf("CPU pool took %llu cycles\n", end_cpu_pool - start_cpu_pool);

    static elem_t output_mat[BATCH_SIZE*OUT_DIM*OUT_DIM][CHANNELS];

    printf("Gemmini pool...\n");
    uint64_t start_gemmini = read_cycles();

    tiled_pool_auto(BATCH_SIZE, IN_DIM, CHANNELS,
        POOL_SIZE, POOL_STRIDE, POOL_PADDING,
        (elem_t*)input, (elem_t*)output_mat, false,
        WS);
        // CPU);

    uint64_t end_gemmini = read_cycles();
    printf("Gemmini pool took %llu cycles\n", end_gemmini - start_gemmini);

    assert(sizeof(output_mat) == sizeof(output));

    bool success = vec_is_equal(&output[0][0][0][0], &output_mat[0][0], sizeof(output) / sizeof(elem_t));

    if (!success) {
        // return 1;

        printf("input:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("[");
            for (int irow = 0; irow < IN_DIM; irow++) {
                printf("[");
                for (int icol = 0; icol < IN_DIM; icol++) {
                    printf("[");
                    for (int ich = 0; ich < CHANNELS; ich++) {
                        printf("%d,", input[batch][irow][icol][ich]);
                    }
                    printf("\b],");
                }
                printf("\b],\n");
            }
            printf("\b],");
        }
        printf("\b\n\n");

        printf("output:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("[");
            for (int orow = 0; orow < OUT_DIM; orow++) {
                printf("[");
                for (int ocol = 0; ocol < OUT_DIM; ocol++) {
                    printf("[");
                    for (int och = 0; och < CHANNELS; och++) {
                        printf("%d,", output[batch][orow][ocol][och]);
                    }
                    printf("\b],");
                }
                printf("\b],\n");
            }
            printf("\b],");
        }
        printf("\b\n\n");

        printf("output_mat:\n");
        for (int orow = 0; orow < BATCH_SIZE * OUT_DIM * OUT_DIM; orow++) {
            printf("[");
            for (int ocol = 0; ocol < CHANNELS; ocol++) {
                printf("%d,", output_mat[orow][ocol]);
            }
            printf("\b],\n");
        }
        printf("\b\n\n");

        printf("Output dimension: %u\n", OUT_DIM);

        return 1;
    }

    return 0;
}

