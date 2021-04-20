#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#define CHECK_RESULT

//#define FAST

#define BATCH_SIZE 2
#define OUT_DIM 17
#define OUT_CHANNELS 32
#define POOL_SIZE 3
#define POOL_STRIDE 2
#define POOL_PADDING 1

#define OCH_DIVIDE 1
#define BATCH_DIVIDE 2
#define OCH_STRIDE OUT_CHANNELS
#define N_PATCHES (BATCH_SIZE * OUT_DIM * OUT_DIM)

#define POOL_OUT_DIM ((OUT_DIM + 2*POOL_PADDING - POOL_SIZE) / POOL_STRIDE + 1)

#define NO_POOL false

#if NO_POOL == true && !(POOL_SIZE == 1 && POOL_STRIDE == 1 && POOL_PADDING == 0)
#error NO_POOL is not set correctly
#endif

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

void flatten_weights(int out_channels, int kernel_dim, int in_channels,
        int patch_size,
        elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
        elem_t weights_mat[patch_size][OCH_STRIDE]) {

    assert(patch_size == kernel_dim * kernel_dim * in_channels);

    for (int outc = 0; outc < out_channels; outc++) {
        for (int krow = 0; krow < kernel_dim; krow++) {
            for (int kcol = 0; kcol < kernel_dim; kcol++) {
                for (int inc = 0; inc < in_channels; inc++) {
                    int wmatrow = krow * kernel_dim * in_channels +
                        kcol * in_channels +
                        inc;

                    weights_mat[wmatrow][outc] =
                        weights[outc][krow][kcol][inc];
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
        // *ptr = (rand() % 32) - 16;
#ifdef FAST
      *ptr = 1;
#else
        *ptr = (rand() % 5) - 2;
#endif
    }
}

void init_random_acc(acc_t * buf, int len) {
    elem_t i = 0;
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        // *ptr = (rand() % 32) - 16;
#ifdef FAST
      *ptr = 1;
#else
        *ptr = (rand() % 5) - 2;
#endif
    }
}

void init_zeros_acc(acc_t * buf, int len) {
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        *ptr = 0;
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

    // assert((IN_DIM + 2*PADDING - KERNEL_DIM) % STRIDE == 0);
    // assert((OUT_DIM + 2*PADDING - POOL_SIZE) % POOL_STRIDE == 0);

    printf("Output dimension: %u\n", OUT_DIM);
    printf("Pooling output dimension: %u\n\n", POOL_OUT_DIM);

    static elem_t output[BATCH_SIZE][OUT_DIM][OUT_DIM][OUT_CHANNELS];
    static elem_t pool_output[BATCH_SIZE][POOL_OUT_DIM][POOL_OUT_DIM][OUT_CHANNELS];
#ifdef CHECK_RESULT
    printf("Randomize inputs...\n");
    init_random(&output[0][0][0][0], sizeof(output) / sizeof(elem_t));

#ifndef FAST
    printf("CPU pool...\n");
    uint64_t start_cpu_pool = read_cycles();
    pool(BATCH_SIZE, OUT_CHANNELS, OUT_DIM, POOL_OUT_DIM,
            POOL_SIZE, POOL_STRIDE, POOL_PADDING,
            output,
            pool_output);
    uint64_t end_cpu_pool = read_cycles();
    printf("CPU pool took %llu cycles\n", end_cpu_pool - start_cpu_pool);

//    printf("CPU conv+pool took %llu cycles\n", end_cpu_pool - start_cpu_pool + end_cpu - start_cpu);
#endif

#endif
    static elem_t pool_output_mat[BATCH_SIZE*POOL_OUT_DIM*POOL_OUT_DIM][OCH_STRIDE];
    printf("Gemmini conv...\n");

	 for(int cid = 0; cid < BATCH_DIVIDE; cid ++){
     	uint64_t start_gemmini = read_cycles();
		 tiled_pool_auto_cid(
			  BATCH_SIZE, OUT_CHANNELS,
			  OUT_DIM,
			  POOL_SIZE, POOL_STRIDE, POOL_PADDING,
			  (elem_t*) output,
			  (elem_t*) pool_output_mat,
			  OCH_DIVIDE, BATCH_DIVIDE, cid);
		 uint64_t end_gemmini = read_cycles();
		 printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);
	 }
    assert(sizeof(pool_output_mat) == sizeof(pool_output));

#ifdef CHECK_RESULT

#ifdef FAST
    bool success = true;
    for (int orow = 0; orow < BATCH_SIZE * POOL_OUT_DIM * POOL_OUT_DIM; orow++) {
      for (int ocol = 0; ocol < OUT_CHANNELS; ocol++) {
	if (pool_output_mat[orow][ocol] != 46) {
	  success = false;
	  break;
	}
      }
    }
#else
    bool success = vec_is_equal(&pool_output[0][0][0][0], &pool_output_mat[0][0], sizeof(pool_output) / sizeof(elem_t));
#endif

    if (!success) {
        // return 1;
/*
        printf("bias:\n");
        for (int och = 0; och < OUT_CHANNELS; och++) {
            printf("%d,", bias[och]);
        }
        printf("\b\n\n");

        printf("weights:\n");
        for (int och = 0; och < OUT_CHANNELS; och++) {
            printf("[");
            for (int wrow = 0; wrow < KERNEL_DIM; wrow++) {
                printf("[");
                for (int wcol = 0; wcol < KERNEL_DIM; wcol++) {
                    printf("[");
                    for (int ich = 0; ich < IN_CHANNELS; ich++) {
                        printf("%d,", weights[och][wrow][wcol][ich]);
                    }
                    printf("\b],");
                }
                printf("\b],\n");
            }
            printf("\b],");
        }
        printf("\b\n\n");

        printf("weights_mat:\n");
        for (int wrow = 0; wrow < KERNEL_DIM * KERNEL_DIM * IN_CHANNELS; wrow++) {
            printf("[");
            for (int wcol = 0; wcol < OUT_CHANNELS; wcol++) {
                printf("%d,", weights_mat[wrow][wcol]);
            }
            printf("\b],\n");
        }
        printf("\b\n\n");

        printf("input:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("[");
            for (int irow = 0; irow < IN_DIM; irow++) {
                printf("[");
                for (int icol = 0; icol < IN_DIM; icol++) {
                    printf("[");
                    for (int ich = 0; ich < IN_CHANNELS; ich++) {
                        printf("%d,", input[batch][irow][icol][ich]);
                    }
                    printf("\b],");
                }
                printf("\b],\n");
            }
            printf("\b],");
        }
        printf("\b\n\n");
*/
        printf("output:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("[");
            for (int orow = 0; orow < OUT_DIM; orow++) {
                printf("[");
                for (int ocol = 0; ocol < OUT_DIM; ocol++) {
                    printf("[");
                    for (int och = 0; och < OUT_CHANNELS; och++) {
                        printf("%d,", output[batch][orow][ocol][och]);
                    }
                    printf("\b],\n");
                }
                printf("\b],\n");
            }
            printf("\b],");
        }
        printf("\b\n\n");

        printf("pool_output:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("[");
            for (int orow = 0; orow < POOL_OUT_DIM; orow++) {
                printf("[");
                for (int ocol = 0; ocol < POOL_OUT_DIM; ocol++) {
                    printf("[");
                    for (int och = 0; och < OUT_CHANNELS; och++) {
                        printf("%d,", pool_output[batch][orow][ocol][och]);
                    }
                    printf("\b],\n");
                }
                printf("\b],\n");
            }
            printf("\b],");
        }
        printf("\b\n\n");

        printf("pool_output_mat:\n");
        for (int orow = 0; orow < BATCH_SIZE * POOL_OUT_DIM * POOL_OUT_DIM; orow++) {
            printf("[");
            for (int ocol = 0; ocol < OUT_CHANNELS; ocol++) {
                printf("%d,", pool_output_mat[orow][ocol]);
            }
            printf("\b],\n");
        }
        printf("\b\n\n");

        printf("Output dimension: %u\n", OUT_DIM);
        printf("Pooling output dimension: %u\n\n", POOL_OUT_DIM);

        return 1;
    }
#endif
    return 0;
}
