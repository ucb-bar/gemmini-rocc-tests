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
#define IN_CHANNELS 3
#define OUT_CHANNELS 32
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 2

#define POOL_SIZE 3
#define POOL_STRIDE 2
#define POOL_PADDING 1

#else
#define BATCH_SIZE 3
#define IN_DIM 23
#define IN_CHANNELS 17
#define OUT_CHANNELS 31
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 2

#define POOL_SIZE 1
#define POOL_STRIDE 1
#define POOL_PADDING 0

#endif

#define OUT_DIM ((IN_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
#define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)
#define N_PATCHES (BATCH_SIZE * OUT_DIM * OUT_DIM)

#define POOL_OUT_DIM ((OUT_DIM + 2*POOL_PADDING - POOL_SIZE) / POOL_STRIDE + 1)

#define NO_1D true
#define NO_POOL true
/*
#if NO_POOL == true && !(POOL_SIZE == 1 && POOL_STRIDE == 1 && POOL_PADDING == 0)
#error NO_POOL is not set correctly
#endif
*/
#define NO_BIAS true


// TODO add bias
void conv(int batch_size, int in_channels, int in_dim,
        int out_channels, int kernel_dim,
        int out_dim,
        int stride, int padding,
        elem_t input[batch_size][in_dim][in_dim][in_channels],
        elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
        acc_t bias[out_channels],
        elem_t output[batch_size][out_dim][out_dim][out_channels]) {

#ifdef GEMMINI_ASSERTIONS
    if (out_dim != (in_dim + 2*padding - kernel_dim) / stride + 1) {
        printf("conv out_dim is not correct\n");
        exit(1);
    }
#endif

    for (int b = 0; b < batch_size; b++) {
        for (int orow = 0; orow < out_dim; orow++) {
            for (int ocol = 0; ocol < out_dim; ocol++) {
                for (int och = 0; och < out_channels; och++) {
    //                output[b][orow][ocol][och] = bias[och];
			acc_t result = bias[och];

                    for (int krow = 0; krow < kernel_dim; krow++) {
                        for (int kcol = 0; kcol < kernel_dim; kcol++) {
                            for (int kch = 0; kch < in_channels; kch++) {
                                int irow = orow * stride + krow - padding;
                                int icol = ocol * stride + kcol - padding;

                                elem_t pixel = irow < 0 || irow >= in_dim ||
                                    icol < 0 || icol >= in_dim ?
                                    0 : input[b][irow][icol][kch];

     //                           output[b][orow][ocol][och] +=
				result +=
                                    weights[och][krow][kcol][kch] *
                                    pixel;
                            }
                        }
                    }
                   // Shift while rounding to nearest integer (ties round to negative infinity)
                    result = ROUNDING_RIGHT_SHIFT(result, 0);

                    // Clip result
                    result = result > elem_t_max ? elem_t_max : (result < elem_t_min ? elem_t_min : result);

                    output[b][orow][ocol][och] = result;

                }
            }
        }
    }
}
/*
void flatten_weights(int out_channels, int kernel_dim, int in_channels,
        int patch_size,
        elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
        elem_t weights_mat[patch_size][out_channels]) {

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
*/

void flatten_weights(int out_channels, int kernel_dim, int in_channels,
        int patch_size,
        elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
        elem_t weights_mat[patch_size][out_channels]) {

    assert(patch_size == kernel_dim * kernel_dim * in_channels);

    for (int outc = 0; outc < out_channels; outc++) {
	for(int inc = 0; inc < in_channels; inc+=DIM){
	      const int K = in_channels - inc > DIM ? DIM : in_channels - inc;
	      for (int krow = 0; krow < kernel_dim; krow++) {
        	    for (int kcol = 0; kcol < kernel_dim; kcol++) {
			 for(int ic = 0; ic < K; ic++){
				int wmatrow = ic + inc * kernel_dim * kernel_dim + K * (krow * kernel_dim + kcol);
        	           	 weights_mat[wmatrow][outc] =
                        	weights[outc][krow][kcol][inc+ic];
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
    for (elem_t * ptr = buf; ptr < buf + len; ptr++) {
        // *ptr = (rand() % 32) - 16;
        *ptr = (rand() % 5) - 2;
    }
}

void init_random_acc(acc_t * buf, int len) {
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        // *ptr = (rand() % 32) - 16;
        *ptr = (rand() % 5) - 2;
    }
}

void init_zeros_acc(acc_t * buf, int len) {
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        *ptr = 0;
    }
}

int main() {
    // assert((in_dim + 2*padding - kernel_dim) % stride == 0);

    printf("Output dimension: %u\n", OUT_DIM);

    static elem_t input[BATCH_SIZE][IN_DIM][IN_DIM][IN_CHANNELS];
    static elem_t weights[OUT_CHANNELS][KERNEL_DIM][KERNEL_DIM][IN_CHANNELS];
    static acc_t bias[OUT_CHANNELS];
    static elem_t output[BATCH_SIZE][OUT_DIM][OUT_DIM][OUT_CHANNELS];

    static elem_t w1[OUT_CHANNELS][KERNEL_DIM][KERNEL_DIM][IN_CHANNELS];
 	for(int i = 0; i < OUT_CHANNELS; i++)
		for(int j = 0; j < KERNEL_DIM; j++)
			for(int k = 0; k<KERNEL_DIM; k++)
				for (int c = 0; c<IN_CHANNELS; c++)
					w1[i][j][k][c] = 1;

    static elem_t i1[BATCH_SIZE][IN_DIM][IN_DIM][IN_CHANNELS];
 	for(int i = 0; i < BATCH_SIZE; i++)
		for(int j = 0; j < IN_DIM; j++)
			for(int k = 0; k< IN_DIM; k++)
				for (int c = 0; c<IN_CHANNELS; c++)
					i1[i][j][k][c] = 1;


    printf("Randomize inputs...\n");
    init_random(&input[0][0][0][0], sizeof(input) / sizeof(elem_t));

    printf("Randomize weights...\n");
    init_random(&weights[0][0][0][0], sizeof(weights) / sizeof(elem_t));

    printf("Randomize bias...\n");
    if (NO_BIAS)
        init_zeros_acc(&bias[0], sizeof(bias) / sizeof(acc_t));
    else
        init_random_acc(&bias[0], sizeof(bias) / sizeof(acc_t));

    printf("CPU conv...\n");
    uint64_t start_cpu = read_cycles();
    conv(BATCH_SIZE, IN_CHANNELS, IN_DIM,
            OUT_CHANNELS, KERNEL_DIM,
            OUT_DIM,
            STRIDE, PADDING,
            input,
//	    i1,
//            w1,
	    weights,
            bias,
            output);
    uint64_t end_cpu = read_cycles();
    printf("CPU conv took %llu cycles\n", end_cpu - start_cpu);

    static elem_t weights_mat[PATCH_SIZE][OUT_CHANNELS];
    static elem_t output_mat[N_PATCHES][OUT_CHANNELS];

    printf("Flatten weights...\n");
    flatten_weights(OUT_CHANNELS, KERNEL_DIM, IN_CHANNELS,
            PATCH_SIZE,
            weights,
            weights_mat);

    printf("Gemmini conv...\n");
    uint64_t start_gemmini = read_cycles();
    tiled_conv_auto(
        BATCH_SIZE, IN_DIM, IN_CHANNELS,
        OUT_CHANNELS, OUT_DIM,
        STRIDE, PADDING, KERNEL_DIM,

        (elem_t*)input,
        (elem_t*)weights_mat,
        NO_BIAS ? NULL : (acc_t*)bias,
        (elem_t*)output_mat,

        NO_ACTIVATION, 0, 0, 
	NO_1D ? 0 : POOL_SIZE, NO_POOL ? 0 : POOL_STRIDE, POOL_PADDING);

    uint64_t end_gemmini = read_cycles();
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);

    assert(sizeof(output_mat) == sizeof(output));

//    bool success = vec_is_equal(&output[0][0][0][0], &output_mat[0][0], sizeof(output) / sizeof(elem_t));
	bool success = false;
    if (!success) {
        // return 1;
/*
        printf("bias:\n");
        for (int och = 0; och < OUT_CHANNELS; och++) {
            printf("%d,", bias[och]);
        }
        printf("\b\n\n");
*/
        printf("weights:\n");
        for (int och = 0; och < OUT_CHANNELS; och++) {
            printf("och: %d \n[", och);
            for (int wrow = 0; wrow < KERNEL_DIM; wrow++) {
                printf("wrow: %d \n[", wrow);
                for (int wcol = 0; wcol < KERNEL_DIM; wcol++) {
                    printf("wcol: %d [", wcol);
                    for (int ich = 0; ich < IN_CHANNELS; ich++) {
                        printf("%d,", weights[och][wrow][wcol][ich]);
                    }
                    printf("\b], \n");
                }
                printf("\b],\n");
            }
            printf("\b],\n\n");
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

/*
        printf("input:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("[");
            for (int irow = 0; irow < IN_DIM; irow++) {
                printf("row %d \n[", irow);
                for (int icol = 0; icol < IN_DIM; icol++) {
                    printf("[");
                    for (int ich = 0; ich < IN_CHANNELS; ich++) {
                        printf("%d,", input[batch][irow][icol][ich]);
                    }
                    printf("\b], \n");
                }
                printf("\b],\n");
            }
            printf("\b],\n");
        }
        printf("\b\n\n");
*/
        printf("output:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("batch: %d \n[", batch);
            for (int orow = 0; orow < OUT_DIM; orow++) {
                printf("orow: %d \n[", orow);
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
/*
        printf("output_mat:\n");
        for (int orow = 0; orow < BATCH_SIZE * OUT_DIM * OUT_DIM; orow++) {
            printf("[");
            for (int ocol = 0; ocol < OUT_CHANNELS; ocol++) {
                printf("%d,", output_mat[orow][ocol]);
            }
            printf("\b],\n");
        }
        printf("\b\n\n");

        return 1;
  */  }

    return 0;
}

