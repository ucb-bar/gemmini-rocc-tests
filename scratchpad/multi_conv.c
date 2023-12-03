#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#define FLOAT false
#include "include/gemmini_testutils.h"
#define DATA 1

#include "data_conv.h"
#define IN_ROW_DIM IN_DIM
#define IN_COL_DIM IN_DIM
#define BASE_ADDR 0x70000000L

#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1
#define PageSize 4096

#define CHECK_RESULT 1
//#define ACC_ID 3
#define ACC_T acc_t

#define OUT_ROW_DIM ((IN_ROW_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
#define OUT_COL_DIM ((IN_COL_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
//#define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)
#define N_PATCHES (BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM)
#define NUM_INT 8
#define NUM_FP 4

#define NUM_ARRAY 4

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

    uint64_t A_copy_addr = (BASE_ADDR & ~(PageSize-1));
    printf("A copy addr: 0x%08lx\n", A_copy_addr);
    uint64_t B_copy_addr = A_copy_addr + (BATCH_SIZE * IN_ROW_DIM * IN_COL_DIM * IN_CHANNELS) * sizeof(elem_t);// + 64*3;
    uint64_t C_copy_addr = B_copy_addr + (OUT_CHANNELS * KERNEL_DIM * KERNEL_DIM * IN_CHANNELS) * sizeof(elem_t);// + 64*3;
    uint64_t D_copy_addr = C_copy_addr + (BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM * OUT_CHANNELS) * sizeof(elem_t);

    printf("B copy addr: 0x%08lx\n", B_copy_addr);
    printf("C copy addr: 0x%08lx\n", C_copy_addr);

    printf("perform memcpy\n");
    bool granted = false;
    int index = 0;
    printf("copy A\n");
    memcpy((elem_t*) A_copy_addr, (elem_t*) input, sizeof(elem_t)*BATCH_SIZE*IN_ROW_DIM*IN_COL_DIM*IN_CHANNELS);
    //for(int i = 0; i < KERNEL_DIM*KERNEL_DIM; i++){
    //printf("copy B\n");
    //memcpy((elem_t*) (B_copy_addr+i*OUT_CHANNELS*IN_CHANNELS), (elem_t*) (weights_mat+i*OUT_CHANNELS*IN_CHANNELS), sizeof(elem_t)*OUT_CHANNELS*IN_CHANNELS);
    //}
    printf("copy B\n");
    memcpy((elem_t*) B_copy_addr, (elem_t*) weights_mat, sizeof(elem_t)*OUT_CHANNELS*KERNEL_DIM*KERNEL_DIM*IN_CHANNELS);
 
    printf("copy D\n");
    if(!NO_BIAS) memcpy((acc_t*) D_copy_addr, (acc_t*) bias, sizeof(acc_t)*OUT_CHANNELS);
    int cfgid = 0;
    for(int i = 0; i < NUM_INT + NUM_FP; i++){   
#if FLOAT
        if(i < NUM_INT)
            continue;
#else
        if(i >= NUM_INT)
            continue;
#endif
        bool acquired = rr_acquire_single(cfgid, i);
        if(acquired){
            printf("gemmini %d acquired to cfgid %d\n", i, cfgid);
            cfgid ++;
            if(cfgid == NUM_ARRAY)
                break;
        }
    }
    for(int i = 0; i < NUM_ARRAY; i++){
      rr_set_opc(XCUSTOM_ACC, i);
      gemmini_flush(0);
    }
    // assert((in_dim + 2*padding - kernel_dim) % stride == 0);

    printf("Input dimensions (rows by columns): %u by %u\n", IN_ROW_DIM, IN_COL_DIM);
    printf("Output dimensions (rows by columns): %u by %u\n\n", OUT_ROW_DIM, OUT_COL_DIM);

    static elem_t output_mat[N_PATCHES][OUT_CHANNELS];
    printf("Gemmini conv...\n");
    uint64_t start_gemmini = read_cycles();
    multi_tiled_conv_auto(
        BATCH_SIZE, IN_ROW_DIM, IN_COL_DIM, IN_CHANNELS,
        OUT_CHANNELS, OUT_ROW_DIM, OUT_COL_DIM,
        STRIDE, 1, 1, PADDING, KERNEL_DIM,
        IN_CHANNELS, OUT_CHANNELS, OUT_CHANNELS,
        false, false, false, false, false,

        (elem_t*)A_copy_addr,
        (elem_t*)B_copy_addr,
        NO_BIAS ? NULL : (acc_t*)D_copy_addr,
        (elem_t*)C_copy_addr,

        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0,

        NUM_ARRAY);
    uint64_t end_gemmini = read_cycles();
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);

    for(int i = 0; i < NUM_ARRAY; i++)
      rr_release(i);
    assert(sizeof(output_mat) == sizeof(output));

    printf("copy C\n");
    memcpy((elem_t*) output_mat, (elem_t*) C_copy_addr, sizeof(elem_t)*N_PATCHES*OUT_CHANNELS);
    bool success = vec_is_equal(&output[0][0][0][0], &output_mat[0][0], sizeof(output) / sizeof(elem_t));
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
            for (int irow = 0; irow < IN_ROW_DIM; irow++) {
                printf("[");
                for (int icol = 0; icol < IN_COL_DIM; icol++) {
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
        printf("output_mat:\n");
        for (int orow = 0; orow < BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM; orow++) {
            printf("[");
            for (int ocol = 0; ocol < OUT_CHANNELS; ocol++) {
                printf("%d,", output_mat[orow][ocol]);
            }
            printf("\b],\n");
        }
        printf("\b\n\n");
        printf("output:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("[");
            for (int orow = 0; orow < OUT_ROW_DIM; orow++) {
                printf("[");
                for (int ocol = 0; ocol < OUT_COL_DIM; ocol++) {
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


        return 1;
    }

    return 0;
}

