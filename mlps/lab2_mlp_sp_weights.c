// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "parameters_290.h"
#include "include/gemmini_290_lab2.h"

#define CHECK_RESULT 1

#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif


void full_matmul(elem_t* A, elem_t* B, acc_t * D, full_t* C_full, int I, int J, int K) {
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      full_t sum = D != NULL ? D[i*J + j] : 0; // Initialize C_full[i][j] with D[i][j] if D is not NULL, else 0
      for (int k = 0; k < K; k++) {
        sum += A[i*K + k] * B[k*J + j];
      }
      C_full[i*J + j] = sum;
    }
  }
}

void full_printMatrix(elem_t* m, int I, int J) {
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++)
      printf("%d ", m[i*J + j]);
    printf("\n");
  }
}

int full_is_equal(elem_t* x, elem_t* y, int I, int J) {
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      if (x[i*J + j] != y[i*J + j])
        return 0;
  return 1;
}

void full_matscale(full_t* full, elem_t* out, acc_scale_t scale, int I, int J) {
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      int index = i*J + j;
      full_t scaled = ACC_SCALE(full[index], scale);
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[index] = elem;
#else
      out[index] = scaled; // In case of floating point, direct assignment (optional saturation could be added here)
#endif
    }
  }
}

void full_relu(elem_t* full, int I, int J) {
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      int index = i*J + j;
      full[index] = full[index] > 0 ? full[index] : 0;
    }
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

    static full_t inter_results0_gold_full[BATCH_SIZE][HIDDEN_SIZE];
    static elem_t inter_results0_gold[BATCH_SIZE][HIDDEN_SIZE];

    static full_t inter_results1_gold_full[BATCH_SIZE][HIDDEN_SIZE];
    static elem_t inter_results1_gold[BATCH_SIZE][HIDDEN_SIZE];

    static full_t inter_results2_gold_full[BATCH_SIZE][HIDDEN_SIZE];
    static elem_t inter_results2_gold[BATCH_SIZE][HIDDEN_SIZE];


#if CHECK_RESULT == 1
#ifdef FAST
#define RAND 1
#else
#define RAND rand()
#endif

    elem_t identity[BATCH_SIZE][BATCH_SIZE];
    for (size_t i = 0; i < BATCH_SIZE; ++i) {
      for (size_t j = 0; j < BATCH_SIZE; ++j) {
        identity[i][j] = i == j;
      }
    }

    for (size_t i = 0; i < INPUT_SIZE; ++i) {
      for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
        //weights0[i][j] = RAND % 3 - 1;
        weights0[i][j] = (i*INPUT_SIZE + j)/4;
      }
    }

    for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
      for (size_t j = 0; j < HIDDEN_SIZE; ++j) {
        weights1[i][j] = RAND % 3 - 1;
      }
    }

    for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
      for (size_t j = 0; j < OUTPUT_SIZE; ++j) {
        weights2[i][j] = RAND % 3 - 1;
      }
    }

    for (size_t i = 0; i < BATCH_SIZE; ++i) {
      for (size_t j = 0; j < INPUT_SIZE; ++j) {
        input_mat[i][j] = RAND % 3 - 1;
      }
    }

    uint32_t weights0_sp, weights1_sp, weights2_sp;
    uint32_t static_alloc_start;

    // TODO LAB2: Calculate the sizes of each of the weight matrices
    size_t weights0_size = 00000;
    size_t weights1_size = 00000;
    size_t weights2_size = 00000;

    // TODO LAB2: Set the starting address for loading values from
    // DRAM into the scratchpad for static allocation. This should
    // alllocate enough space for activations.
    static_alloc_start = 00000;

    // TODO LAB2: Allocate the scratchpad pointers for the weights.
    // Remember to use row addressing.
    weights0_sp = 00000;
    weights1_sp = 00000;
    weights2_sp = 00000;

    // TODO LAB2: Copy the weights to the scratchpad
    mvin_matrix(00000, 00000, weights0, weights0_sp);
    mvin_matrix(00000, 00000, weights1, weights1_sp);
    mvin_matrix(00000, 00000, weights2, weights2_sp);

    unsigned long start, end;
    unsigned long layer0_cycles, layer1_cycles, layer2_cycles;
    unsigned long cpu_layer0_cycles, cpu_layer1_cycles, cpu_layer2_cycles;

    printf("==================================\n");
    printf("EE290-2 Lab2 Scratchpad Weights\n");
    printf("==================================\n");
    printf("Executing Gemmini MLP\n");
    printf("----------------------------------\n");
    printf("Starting gemmini matmul\n");
    start = read_cycles();

    sp_tiled_matmul_auto_dram_spad_ws(BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE,
            (elem_t*)input_mat, weights0_sp, NULL, (elem_t*) inter_results0, RELU);

    end = read_cycles();
    layer0_cycles = end-start;
    printf("Cycles taken: %u\n", layer0_cycles);

    printf("Starting gemmini matmul\n");
    start = read_cycles();

    sp_tiled_matmul_auto_dram_spad_ws(BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE,
            (elem_t*)inter_results0, weights1_sp, NULL, (elem_t*) inter_results1, RELU);

    end = read_cycles();
    layer1_cycles = end-start;
    printf("Cycles taken: %u\n", layer1_cycles);

    printf("Starting gemmini matmul\n");
    start = read_cycles();

    sp_tiled_matmul_auto_dram_spad_ws(BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE,
            (elem_t*)inter_results1, weights2_sp, NULL, (elem_t*) inter_results2, RELU);

    end = read_cycles();
    layer2_cycles = end-start;
    printf("Cycles taken: %u\n", layer2_cycles);


    printf("==================================\n");
    printf("Executing CPU MLP\n");
    printf("----------------------------------\n");

    unsigned long cpu_start, cpu_end;
    printf("Starting slow CPU matmul\n");
    cpu_start = read_cycles();
    full_matmul(input_mat, weights0, NULL, inter_results0_gold_full, BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE);
    full_matscale(inter_results0_gold_full, inter_results0_gold, ACC_SCALE_IDENTITY, BATCH_SIZE, HIDDEN_SIZE);
    full_relu(inter_results0_gold,  BATCH_SIZE, HIDDEN_SIZE);
    cpu_end = read_cycles();
    cpu_layer0_cycles = cpu_end-cpu_start;
    printf("Cycles taken (CPU): %lu\n", cpu_layer0_cycles);

    printf("Starting slow CPU matmul\n");
    cpu_start = read_cycles();
    full_matmul(inter_results0_gold, weights1, NULL, inter_results1_gold_full, BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE);
    full_matscale(inter_results1_gold_full, inter_results1_gold, ACC_SCALE_IDENTITY, BATCH_SIZE, HIDDEN_SIZE);
    full_relu(inter_results1_gold,  BATCH_SIZE, HIDDEN_SIZE);
    cpu_end = read_cycles();
    cpu_layer1_cycles = cpu_end-cpu_start;
    printf("Cycles taken (CPU): %lu\n", cpu_layer1_cycles);

    printf("Starting slow CPU matmul\n");
    cpu_start = read_cycles();
    full_matmul(inter_results1_gold, weights2, NULL, inter_results2_gold_full, BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
    full_matscale(inter_results2_gold_full, inter_results2_gold, ACC_SCALE_IDENTITY, BATCH_SIZE, OUTPUT_SIZE);
    full_relu(inter_results2_gold,  BATCH_SIZE, HIDDEN_SIZE);
    cpu_end = read_cycles();
    cpu_layer2_cycles = cpu_end-cpu_start;
    printf("Cycles taken (CPU): %lu\n", cpu_layer2_cycles);

    printf("==================================\n");
    printf("Cycle Breakdown (Gemmini):\n");
    printf("\tLayer 0: %lu\n", layer0_cycles);
    printf("\tLayer 1: %lu\n", layer1_cycles);
    printf("\tLayer 2: %lu\n", layer2_cycles);
    printf("\tTotal:   %lu\n", layer0_cycles + layer1_cycles + layer2_cycles);
    printf("----------------------------------\n");
    printf("Cycle Breakdown (CPU):\n");
    printf("\tLayer 0: %lu\n", cpu_layer0_cycles);
    printf("\tLayer 1: %lu\n", cpu_layer1_cycles);
    printf("\tLayer 2: %lu\n", cpu_layer2_cycles);
    printf("\tTotal:   %lu\n", cpu_layer0_cycles + cpu_layer1_cycles + cpu_layer2_cycles);
    printf("==================================\n");
    printf("\n");


#endif

#if CHECK_RESULT == 1
    printf("==================================\n");
    printf("Checking Correctness\n");
    printf("----------------------------------\n");
    printf("Checking result layer 0\n");
    if (!full_is_equal(inter_results0, inter_results0_gold, BATCH_SIZE, HIDDEN_SIZE)) {
      printf("C:\n");
      full_printMatrix(inter_results0, BATCH_SIZE, HIDDEN_SIZE);
      printf("inter_results0_gold:\n");
      full_printMatrix(inter_results0_gold, BATCH_SIZE, HIDDEN_SIZE);
      printf("\n");

      exit(1);
    }
    printf("Checking result layer 1\n");
    if (!full_is_equal(inter_results1, inter_results1_gold, BATCH_SIZE, HIDDEN_SIZE)) {
      printf("C:\n");
      full_printMatrix(inter_results1, BATCH_SIZE, HIDDEN_SIZE);
      printf("inter_results0_gold:\n");
      full_printMatrix(inter_results1_gold, BATCH_SIZE, HIDDEN_SIZE);
      printf("\n");

      exit(1);
    }
    printf("Checking result layer 2\n");
    if (!full_is_equal(inter_results2, inter_results2_gold, BATCH_SIZE, OUTPUT_SIZE)) {
      printf("C:\n");
      full_printMatrix(inter_results2, BATCH_SIZE, OUTPUT_SIZE);
      printf("inter_results0_gold:\n");
      full_printMatrix(inter_results2_gold, BATCH_SIZE, OUTPUT_SIZE);
      printf("\n");

      exit(1);
    }
    printf("==================================\n");
#endif

  exit(0);
}

 