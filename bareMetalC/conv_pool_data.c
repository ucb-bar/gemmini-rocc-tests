#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#define FLOAT false
#include "include/gemmini_testutils.h"
#define DATA 1

#include "data_pool.h"
#define IN_ROW_DIM IN_DIM
#define IN_COL_DIM IN_DIM
#define NO_BIAS false
#define NO_BIAS false

#define OUT_ROW_DIM ((IN_ROW_DIM + 2 * PADDING - KERNEL_DIM) / STRIDE + 1)
#define OUT_COL_DIM ((IN_COL_DIM + 2 * PADDING - KERNEL_DIM) / STRIDE + 1)
//#define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)
#define N_PATCHES (BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM)

#define POOL_OUT_ROW_DIM ((OUT_ROW_DIM + 2 * POOL_PADDING - POOL_SIZE) / POOL_STRIDE + 1)
#define POOL_OUT_COL_DIM ((OUT_COL_DIM + 2 * POOL_PADDING - POOL_SIZE) / POOL_STRIDE + 1)

#define NO_POOL false

#if NO_POOL == true && !(POOL_SIZE == 1 && POOL_STRIDE == 1 && POOL_PADDING == 0)
#error NO_POOL is not set correctly
#endif

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

    int cfgid = 0;
    int i = 0;
    //for(int i = 0; i < 2; i++){
        bool acquired = rr_acquire_single(cfgid, i);
        if(acquired){
            printf("gemmini %d acquired to cfgid %d\n", i, cfgid);
            //break;
        }
    //}
    rr_set_opc(XCUSTOM_ACC, cfgid);
    gemmini_flush(0);

    // assert((IN_DIM + 2*PADDING - KERNEL_DIM) % STRIDE == 0);
    // assert((OUT_DIM + 2*PADDING - POOL_SIZE) % POOL_STRIDE == 0);

    printf("Output dimensions (rows by columns): %u by %u\n", OUT_ROW_DIM, OUT_COL_DIM);
    printf("Pooling output dimensions (rows by columns): %u by %u\n\n", POOL_OUT_ROW_DIM, POOL_OUT_COL_DIM);

    static elem_t output_mat[N_PATCHES][OUT_CHANNELS];
    static elem_t pool_output_mat[BATCH_SIZE * POOL_OUT_ROW_DIM * POOL_OUT_COL_DIM][OUT_CHANNELS];

    printf("Gemmini conv...\n");
    uint64_t start_gemmini = read_cycles();

    tiled_conv_auto(
        BATCH_SIZE, IN_ROW_DIM, IN_COL_DIM, IN_CHANNELS,
        OUT_CHANNELS, OUT_ROW_DIM, OUT_COL_DIM,
        STRIDE, 1, 1, PADDING, KERNEL_DIM,
        false, false, false, false, false,

        // 1,
        // 1, 1, 1,
        // 1, 1, 1,

        (elem_t*)input,
        (elem_t*)weights_mat,
        NO_BIAS ? NULL : (acc_t*)bias,
        (elem_t*)pool_output_mat,

        NO_ACTIVATION, ACC_SCALE_IDENTITY,
        POOL_SIZE, NO_POOL ? 0 : POOL_STRIDE, POOL_PADDING,

        WS);
    rr_fence(cfgid);
    uint64_t end_gemmini = read_cycles();
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);

    rr_release(cfgid);
    assert(sizeof(pool_output_mat) == sizeof(pool_output));

    bool success = vec_is_equal(&pool_output[0][0][0][0], &pool_output_mat[0][0], sizeof(pool_output) / sizeof(elem_t));
    if (!success) {
        // return 1;
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
                    printf("\b],");
                }
                printf("\b],\n");
            }
            printf("\b],");
        }
        printf("\b\n\n");

        printf("pool_output:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("[");
            for (int orow = 0; orow < POOL_OUT_ROW_DIM; orow++) {
                printf("[");
                for (int ocol = 0; ocol < POOL_OUT_COL_DIM; ocol++) {
                    printf("[");
                    for (int och = 0; och < OUT_CHANNELS; och++) {
                        printf("%d,", pool_output[batch][orow][ocol][och]);
                    }
                    printf("\b],");
                }
                printf("\b],\n");
            }
            printf("\b],");
        }
        printf("\b\n\n");


        printf("pool_output_mat:\n");
        for (int orow = 0; orow < BATCH_SIZE * POOL_OUT_ROW_DIM * POOL_OUT_COL_DIM; orow++) {
            printf("[");
            for (int ocol = 0; ocol < OUT_CHANNELS; ocol++) {
                printf("%d,", pool_output_mat[orow][ocol]);
            }
            printf("\b],\n");
        }
        printf("\b\n\n");

        printf("Output dimensions (rows by columns): %u by %u\n", OUT_ROW_DIM, OUT_COL_DIM);
        printf("Pooling output dimensions: %u by %u\n\n", POOL_OUT_ROW_DIM, POOL_OUT_COL_DIM);

        return 1;
    }

    return 0;
}
