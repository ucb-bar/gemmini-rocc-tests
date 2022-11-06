#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "data_conv.h"

#define BATCH_SIZE 1
#define NO_BIAS false
#define N_PATCHES (BATCH_SIZE * OUT_DIM * OUT_DIM)
#define NUM_ARRAY 2
#define OP 3

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
                    acc_t result = bias[och];

                    for (int krow = 0; krow < kernel_dim; krow++) {
                        for (int kcol = 0; kcol < kernel_dim; kcol++) {
                            for (int kch = 0; kch < in_channels; kch++) {
                                int irow = orow * stride + krow - padding;
                                int icol = ocol * stride + kcol - padding;

                                elem_t pixel = irow < 0 || irow >= in_dim ||
                                    icol < 0 || icol >= in_dim ?
                                    0 : input[b][irow][icol][kch];

                                result +=
                                    weights[och][krow][kcol][kch] *
                                    pixel;
                            }
                        }
                    }

                    // Clip result
                    result = result > elem_t_max ? elem_t_max : (result < elem_t_min ? elem_t_min : result);

                    output[b][orow][ocol][och] = result;
                }
            }
        }
    }
}

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

bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++){
        if (a[i] != b[i]){
            printf("channel: %d, col: %d, row: %d, gold: %d, out: %d\n", i%(OUT_CHANNELS), (i/OUT_CHANNELS)%OUT_DIM, ((int)(i/OUT_CHANNELS)/OUT_DIM), a[i], b[i]);
            //return false;
        }
    }
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

    // assert((in_dim + 2*padding - kernel_dim) % stride == 0);

  printf("attempting rerocc_acquire\n");
  for(int i = 0; i < NUM_ARRAY; i++)
    while(!rerocc_acquire(i, 0xf)){}

  printf("rerocc acquired \n");

   
  for (int i = 0; i < NUM_ARRAY; i++) {
    rerocc_assign(OP, i);
    gemmini_flush(0);
  }
    printf("Output dimension: %u\n\n", OUT_DIM);

    static elem_t output_mat[N_PATCHES][OUT_CHANNELS];

    printf("Gemmini conv...\n");
    uint64_t start_gemmini = read_cycles();
    tiled_opcode_conv_auto(
        BATCH_SIZE, IN_DIM, IN_CHANNELS,
        OUT_CHANNELS, OUT_DIM,
        STRIDE, 1, 1, PADDING, KERNEL_DIM,
        IN_CHANNELS, OUT_CHANNELS, OUT_CHANNELS,

        false, false, false, false, 

        false, false, false, false, false,

        (elem_t*)input,
        (elem_t*)weight_mat,
        NO_BIAS ? NULL : (acc_t*)bias,
        (elem_t*)output_mat,

        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0, 0,
        NUM_ARRAY, 0,
        WS);
    uint64_t end_gemmini = read_cycles();
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);

    // Release all the trackers
    for (int i = 0; i < NUM_ARRAY; i++) {
      printf("released array %d\n", i);
      rerocc_release(i);
    }
    assert(sizeof(output_mat) == sizeof(output));

    bool success = vec_is_equal(&output[0][0][0], &output_mat[0][0], sizeof(output) / sizeof(elem_t));
    if (!success) {
        // return 1;
        printf("output:\n");
//        for (int batch = 0; batch < BATCH_SIZE; batch++) {
//            printf("[");
            for (int orow = 0; orow < OUT_DIM; orow++) {
                printf("[");
                for (int ocol = 0; ocol < OUT_DIM; ocol++) {
                    printf("[");
                    for (int och = 0; och < OUT_CHANNELS; och++) {
                        printf("%d,", output[orow][ocol][och]);
                    }
                    printf("\b],");
                }
                printf("\b],\n");
            }
//            printf("\b],");
//        }
        printf("\b\n\n");

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
    }

    return 0;
}
