#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_nn.h"
#include "include/gemmini_testutils.h"
#include "mobilenet_params.h"

#define out_tile2 1
#define in_tile2 0
#define bank2 1

#ifndef BAREMETAL
#define BATCH_SIZE 4
#define IN_DIM 56
#define IN_CHANNELS 64
#define OUT_CHANNELS 64
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 1
#define out_scale 4
#else
#define BATCH_SIZE 4
#define IN_DIM 112
#define IN_CHANNELS 32
#define OUT_CHANNELS 32
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 1
#define out_scale 6
#endif
//parameter of dw_8

#define NO_BIAS false

#define OUT_DIM ((IN_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
#define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * 1)
#define N_PATCHES (BATCH_SIZE * OUT_DIM * OUT_DIM)

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
    printf("len: %d \n", len);
    for (int i = 0; i < len; i++)
        if (a[i] != b[i])
            return false;
    return true;
}

void init_random(elem_t * buf, int len) {
    for (elem_t * ptr = buf; ptr < buf + len; ptr++) {
         *ptr = (rand() % 32) - 16;
        //*ptr = (rand() % 16) - 8;
    }
}

void init_random_acc(acc_t * buf, int len) {
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
         *ptr = (rand() % 32) - 16;
        //*ptr = (rand() % 16) - 8;
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

    // assert((in_dim + 2*padding - kernel_dim) % stride == 0);


    static elem_t weights_mat2[9][32];

//    printf("Flatten weights...\n");
    flatten_weights(conv_dw_2_params.out_channels, conv_dw_2_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_2_w,
            weights_mat2);


        printf("static const elem_t conv_dw_2_w_flat[%d][%d] row_align(1) = {", conv_dw_2_params.patch_size, conv_dw_2_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_2_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_2_params.out_channels; wcol++) {
                printf("%d,", weights_mat2[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat5[9][96];

    	flatten_weights(conv_dw_5_params.out_channels, conv_dw_5_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_5_w,
            weights_mat5);


        printf("static const elem_t conv_dw_5_w_flat[%d][%d] row_align(1) = {", conv_dw_5_params.patch_size, conv_dw_5_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_5_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_5_params.out_channels; wcol++) {
                printf("%d,", weights_mat5[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat8[9][144];

    	flatten_weights(conv_dw_8_params.out_channels, conv_dw_8_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_8_w,
            weights_mat8);


        printf("static const elem_t conv_dw_8_w_flat[%d][%d] row_align(1) = {", conv_dw_8_params.patch_size, conv_dw_8_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_8_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_8_params.out_channels; wcol++) {
                printf("%d,", weights_mat8[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat11[9][144];

    	flatten_weights(conv_dw_11_params.out_channels, conv_dw_11_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_11_w,
            weights_mat11);


        printf("static const elem_t conv_dw_11_w_flat[%d][%d] row_align(1) = {", conv_dw_11_params.patch_size, conv_dw_11_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_11_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_11_params.out_channels; wcol++) {
                printf("%d,", weights_mat11[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat14[9][192];

    	flatten_weights(conv_dw_14_params.out_channels, conv_dw_14_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_14_w,
            weights_mat14);


        printf("static const elem_t conv_dw_14_w_flat[%d][%d] row_align(1) = {", conv_dw_14_params.patch_size, conv_dw_14_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_14_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_14_params.out_channels; wcol++) {
                printf("%d,", weights_mat14[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat17[9][192];

    	flatten_weights(conv_dw_17_params.out_channels, conv_dw_17_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_17_w,
            weights_mat17);


        printf("static const elem_t conv_dw_17_w_flat[%d][%d] row_align(1) = {", conv_dw_17_params.patch_size, conv_dw_17_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_17_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_17_params.out_channels; wcol++) {
                printf("%d,", weights_mat17[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat20[9][192];

    	flatten_weights(conv_dw_20_params.out_channels, conv_dw_20_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_20_w,
            weights_mat20);


        printf("static const elem_t conv_dw_20_w_flat[%d][%d] row_align(1) = {", conv_dw_20_params.patch_size, conv_dw_20_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_20_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_20_params.out_channels; wcol++) {
                printf("%d,", weights_mat20[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat23[9][384];

    	flatten_weights(conv_dw_23_params.out_channels, conv_dw_23_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_23_w,
            weights_mat23);


        printf("static const elem_t conv_dw_23_w_flat[%d][%d] row_align(1) = {", conv_dw_23_params.patch_size, conv_dw_23_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_23_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_23_params.out_channels; wcol++) {
                printf("%d,", weights_mat23[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");


    	static elem_t weights_mat26[9][384];

    	flatten_weights(conv_dw_26_params.out_channels, conv_dw_26_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_26_w,
            weights_mat26);


        printf("static const elem_t conv_dw_26_w_flat[%d][%d] row_align(1) = {", conv_dw_26_params.patch_size, conv_dw_26_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_26_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_26_params.out_channels; wcol++) {
                printf("%d,", weights_mat26[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat29[9][384];

    	flatten_weights(conv_dw_29_params.out_channels, conv_dw_29_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_29_w,
            weights_mat29);


        printf("static const elem_t conv_dw_29_w_flat[%d][%d] row_align(1) = {", conv_dw_29_params.patch_size, conv_dw_29_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_29_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_29_params.out_channels; wcol++) {
                printf("%d,", weights_mat29[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat32[9][384];

    	flatten_weights(conv_dw_32_params.out_channels, conv_dw_32_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_32_w,
            weights_mat32);


        printf("static const elem_t conv_dw_32_w_flat[%d][%d] row_align(1) = {", conv_dw_32_params.patch_size, conv_dw_32_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_32_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_32_params.out_channels; wcol++) {
                printf("%d,", weights_mat32[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");


    	static elem_t weights_mat35[9][576];

    	flatten_weights(conv_dw_35_params.out_channels, conv_dw_35_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_35_w,
            weights_mat35);


        printf("static const elem_t conv_dw_35_w_flat[%d][%d] row_align(1) = {", conv_dw_35_params.patch_size, conv_dw_35_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_35_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_35_params.out_channels; wcol++) {
                printf("%d,", weights_mat35[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat38[9][576];

    	flatten_weights(conv_dw_38_params.out_channels, conv_dw_38_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_38_w,
            weights_mat38);


        printf("static const elem_t conv_dw_38_w_flat[%d][%d] row_align(1) = {", conv_dw_38_params.patch_size, conv_dw_38_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_38_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_38_params.out_channels; wcol++) {
                printf("%d,", weights_mat38[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat41[9][576];

    	flatten_weights(conv_dw_41_params.out_channels, conv_dw_41_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_41_w,
            weights_mat41);


        printf("static const elem_t conv_dw_41_w_flat[%d][%d] row_align(1) = {", conv_dw_41_params.patch_size, conv_dw_41_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_41_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_41_params.out_channels; wcol++) {
                printf("%d,", weights_mat41[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat44[9][960];

    	flatten_weights(conv_dw_44_params.out_channels, conv_dw_44_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_44_w,
            weights_mat44);


        printf("static const elem_t conv_dw_44_w_flat[%d][%d] row_align(1) = {", conv_dw_44_params.patch_size, conv_dw_44_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_44_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_44_params.out_channels; wcol++) {
                printf("%d,", weights_mat44[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat47[9][960];

    	flatten_weights(conv_dw_47_params.out_channels, conv_dw_47_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_47_w,
            weights_mat47);


        printf("static const elem_t conv_dw_47_w_flat[%d][%d] row_align(1) = {", conv_dw_47_params.patch_size, conv_dw_47_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_47_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_47_params.out_channels; wcol++) {
                printf("%d,", weights_mat47[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");

    	static elem_t weights_mat50[9][960];

    	flatten_weights(conv_dw_50_params.out_channels, conv_dw_50_params.kernel_size, 1,
            PATCH_SIZE,
            conv_dw_50_w,
            weights_mat50);


        printf("static const elem_t conv_dw_50_w_flat[%d][%d] row_align(1) = {", conv_dw_50_params.patch_size, conv_dw_50_params.out_channels);
	for (int wrow = 0; wrow < conv_dw_50_params.patch_size; wrow++) {
            printf("{");
            for (int wcol = 0; wcol < conv_dw_50_params.out_channels; wcol++) {
                printf("%d,", weights_mat50[wrow][wcol]);
            }
            printf("},");
        }
        printf("};\n\n");


    return 0;
}

