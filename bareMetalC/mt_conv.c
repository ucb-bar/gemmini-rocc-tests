#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "util.h"

//#define FAST
#define SKIP_WEIGHT false

#ifndef BAREMETAL

#define BATCH_SIZE 4
#define IN_DIM 224
#define IN_CHANNELS 3
#define OUT_CHANNELS 32
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 2
#define DILATION 1

#else

#define BATCH_SIZE 1
#define IN_DIM 28
#define IN_CHANNELS 128
#define OUT_CHANNELS 128
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 1
#define DILATION 1

#endif

#define NO_BIAS false

#define OCH_PADDING false
#define ICH_PADDING false

#if OCH_PADDING == 1
#define OCH_STRIDE OUT_CHANNELS + 64
#else
#define OCH_STRIDE OUT_CHANNELS
#endif

#if ICH_PADDING == 1
#define ICH_STRIDE IN_CHANNELS + 64
#else
#define ICH_STRIDE IN_CHANNELS
#endif

#define OUT_DIM ((IN_DIM + 2*PADDING - DILATION * (KERNEL_DIM - 1) - 1) / STRIDE + 1)
#define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)
#define N_PATCHES (BATCH_SIZE * OUT_DIM * OUT_DIM)

void conv(int batch_size, int in_channels, int in_dim,
        int out_channels, int kernel_dim,
        int out_dim,
        int stride, int dilation, int padding,
        elem_t input[batch_size][in_dim][in_dim][in_channels],
        elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
        acc_t bias[out_channels],
        elem_t output[batch_size][out_dim][out_dim][out_channels]) {

#ifdef GEMMINI_ASSERTIONS
    if (out_dim != (in_dim + 2*padding - dilation * (kernel_dim - 1) - 1) / stride + 1) {
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
                                int irow = orow * stride + krow * dilation - padding;
                                int icol = ocol * stride + kcol * dilation - padding;

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

static elem_t in0[BATCH_SIZE][IN_DIM][IN_DIM][ICH_STRIDE] row_align(MAX_BLOCK_LEN);
static acc_t bias0[OCH_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t weights0[PATCH_SIZE][OCH_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t out0[N_PATCHES][OCH_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t in1[BATCH_SIZE][IN_DIM][IN_DIM][ICH_STRIDE] row_align(MAX_BLOCK_LEN);
static acc_t bias1[OCH_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t weights1[PATCH_SIZE][OCH_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t out1[N_PATCHES][OCH_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t in2[BATCH_SIZE][IN_DIM][IN_DIM][ICH_STRIDE] row_align(MAX_BLOCK_LEN);
static acc_t bias2[OCH_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t weights2[PATCH_SIZE][OCH_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t out2[N_PATCHES][OCH_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t in3[BATCH_SIZE][IN_DIM][IN_DIM][ICH_STRIDE] row_align(MAX_BLOCK_LEN);
static acc_t bias3[OCH_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t weights3[PATCH_SIZE][OCH_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t out3[N_PATCHES][OCH_STRIDE] row_align(MAX_BLOCK_LEN);

void thread_entry(int cid, int nc)
{
  for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
    barrier(nc);
  }
  	 elem_t* A = (cid == 0)? (elem_t*) in0:(elem_t*) in1;
	 elem_t* B = (cid == 0)? (elem_t*) weights0:(elem_t*) weights1;
	 elem_t* C = (cid == 0)? (elem_t*) out0:(elem_t*) out1;
	 acc_t * D = (cid == 0) ? (acc_t*) bias0:(acc_t*) bias1;

	 if(cid == 2){
		 A = (elem_t*) in2;
		 B = (elem_t*) weights2;
		 C = (elem_t*) out2;
		 D = (acc_t*) bias2;
	 }
	 if(cid == 3){
		 A = (elem_t*) in3;
		 B = (elem_t*) weights3;
		 C = (elem_t*) out3;
		 D = (acc_t*) bias3;
	 }
//ToDo: warm-up

	 for (int i = 0; i < nc; i++) {
    	if (i == cid) printf("Starting gemmini tiled_matmul\n");
    	barrier(nc);
  	 }
  	 gemmini_flush(0);
	 for(int i = 0; i < nc; i++){
		 if(i == cid && cid == 0){
			 printf("Gemmini conv...\n");
    		 printf("kernel dim: %d, input dim: %d, output dim: %d, batch size: %d, input channel: %d, output channel: %d \n", KERNEL_DIM, IN_DIM, OUT_DIM, BATCH_SIZE, IN_CHANNELS, OUT_CHANNELS);
    		 printf("NO_BIAS: %d\n", NO_BIAS); 
		 }
		 barrier(nc);
	 }
	 barrier(nc);
    uint64_t start_gemmini = read_cycles();
	 for(int j = 0; j < nc; j++){
		 if(j == cid){	
			tiled_conv_A_stride_auto_loopld(
				  BATCH_SIZE, IN_DIM, IN_CHANNELS,
				  OUT_CHANNELS, OUT_DIM,
				  STRIDE, DILATION, PADDING, KERNEL_DIM,
				  A,
				  B,
				  NO_BIAS ? NULL : D,
				  C,
				  NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0, 0,
				  WS, ICH_PADDING, OCH_PADDING, (cid < 2) ? false : SKIP_WEIGHT);
		 }
	 }
    uint64_t end_gemmini = read_cycles();

  for(int i = 0; i < nc; i++){
	  if (i == cid) {
		 printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);
		 const int total_macs = KERNEL_DIM * KERNEL_DIM * OUT_DIM * OUT_DIM * IN_CHANNELS * OUT_CHANNELS * BATCH_SIZE;
		 const int ideal_cycles = total_macs / (DIM * DIM);
		 const int utilization = 100 * ideal_cycles / (end_gemmini-start_gemmini);
		 printf("Utilization: %d%%\n", utilization);
	  }
	  barrier(nc);
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
/*
    printf("Output dimension: %u\n\n", OUT_DIM);

    static elem_t input[BATCH_SIZE][IN_DIM][IN_DIM][IN_CHANNELS] row_align(MAX_BLOCK_LEN);
    static elem_t weights[OUT_CHANNELS][KERNEL_DIM][KERNEL_DIM][IN_CHANNELS] row_align(MAX_BLOCK_LEN);
    static acc_t bias[OUT_CHANNELS] row_align(MAX_BLOCK_LEN);
    static elem_t output[BATCH_SIZE][OUT_DIM][OUT_DIM][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);

    static elem_t weights_mat[PATCH_SIZE][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);
    static elem_t output_mat[N_PATCHES][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);

    printf("Flatten weights...\n");
    flatten_weights(OUT_CHANNELS, KERNEL_DIM, IN_CHANNELS,
            PATCH_SIZE,
            weights,
            weights_mat);

    printf("Gemmini conv...\n");
    printf("kernel dim: %d, input dim: %d, output dim: %d, batch size: %d, input channel: %d, output channel: %d \n", KERNEL_DIM, IN_DIM, OUT_DIM, BATCH_SIZE, IN_CHANNELS, OUT_CHANNELS);
    printf("NO_BIAS: %d\n", NO_BIAS); 
	 uint64_t start_gemmini = read_cycles();
    tiled_conv_A_stride_auto_loopld(
        BATCH_SIZE, IN_DIM, IN_CHANNELS,
        OUT_CHANNELS, OUT_DIM,
        STRIDE, DILATION, PADDING, KERNEL_DIM,

        (elem_t*)input,
        (elem_t*)weights_mat,
        NO_BIAS ? NULL : (acc_t*)bias,
        (elem_t*)output_mat,

        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0, 0,

        WS, SKIP_WEIGHT);
    uint64_t end_gemmini = read_cycles();

	 printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);
    const int total_macs = KERNEL_DIM * KERNEL_DIM * OUT_DIM * OUT_DIM * IN_CHANNELS * OUT_CHANNELS * BATCH_SIZE;
    const int ideal_cycles = total_macs / (DIM * DIM);
    const int utilization = 100 * ideal_cycles / (end-start);
    printf("Utilization: %d%%\n", utilization);


    assert(sizeof(output_mat) == sizeof(output));
*/

    return 0;
}
