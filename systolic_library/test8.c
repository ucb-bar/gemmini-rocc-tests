

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <sys/mman.h>
#include "include/gemmini.h"
#include "parameters8.h"

#define verbose(layer_num,old_C,filter,C) printf("layer %d: operand %d %d filter %d %d result %d %d\n", layer_num, LEN(old_C),LEN(old_C[0]),LEN(filter),LEN(filter[0]),LEN(C),LEN(C[0]));

static void tiled_matmul_compare(size_t DIM_I, size_t DIM_J, size_t DIM_K,
        elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], acc_t D[DIM_I][DIM_J],
        elem_t C[DIM_I][DIM_J], int no_bias, int act, int shift, int relu6_shift,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool compare, char * layer_name)
{
    if (compare)
        printf("%s: gemmini\n", layer_name);
    tiled_matmul_option(DIM_I, DIM_J, DIM_K,
        A, B, D, C, no_bias, act, shift, relu6_shift,
        tiled_matmul_type);

    if (compare) {
        printf("%s: CPU\n", layer_name);
        elem_t gold[DIM_I][DIM_J];
        tiled_matmul_option(DIM_I, DIM_J, DIM_K,
            A, B, D, gold, no_bias, act, shift, relu6_shift,
            CPU);

        printf("%s: comparing\n", layer_name);
        for (size_t i = 0; i < DIM_I; i++) {
            for (size_t j = 0; j < DIM_J; j++) {
                if (C[i][j] != gold[i][j]) {
                    printf("Layer calculated incorrectly: %s\n", layer_name);
                    exit(1);
                }
            }
        }
    }
}

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    matmul_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type;
    if (argc < 2) {
        // printf("usage: %s matmul_option\n  matmul_option may be 'os', 'ws', or cpu'\n");
        // exit(0);
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    }

    bool compare;
    if (argc < 3) {
        compare = false;
    } else if (strcmp(argv[2], "compare") == 0) {
        compare = true;
    } else {
        printf("Unknown command-line argument\n");
        exit(1);
    }


    unsigned long cycles[2]={0};
    unsigned long start,end;
    start = read_cycles();

    /* matmul number: 0 */

    tiled_matmul_compare(64, 4608, 3072,    // dimensions
    input_mat, weights0, NULL, inter_results0,      // addresses
    1, RELU, 0, 0,              // no_bias, act, shift, r6_shift
    tiled_matmul_type, compare, "layer_0");
    // verbose(0,input_mat,weights0,inter_results0)
    /* end of matmul number: 0 */

    end = read_cycles();
    cycles[0] = end-start;
    start = end;


    /* matmul number: 1 */

    tiled_matmul_compare(64, 3072, 4608,    // dimensions
    inter_results0, weights1, NULL, inter_results1,      // addresses
    1, RELU, 0, 0,              // no_bias, act, shift, r6_shift
    tiled_matmul_type, compare, "layer_1");
    // verbose(1,inter_results0,weights1,inter_results1)
    /* end of matmul number: 1 */

    end = read_cycles();
    cycles[1] = end-start;
    start = end;

    unsigned long overall_cycles = 0;
    for(int cyc = 0; cyc < 2 ; cyc++){
        overall_cycles += cycles[cyc];
        printf("Cycles taken in layer %d: %lu\n", cyc,cycles[cyc]);
    }
    printf("Overall cycles taken: %lu\n",overall_cycles);


    return 0;
}

