
#include <stdio.h>
#include "include/gemmini.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

#define BATCH_SIZE  16
#define INPUT_SIZE  64
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 16

static elem_t input_mat[BATCH_SIZE][INPUT_SIZE] row_align(1)= {0};
static elem_t weights0[INPUT_SIZE][HIDDEN_SIZE] row_align(1)= {0};
static elem_t inter_results0[BATCH_SIZE][HIDDEN_SIZE] row_align(1)= {0};
static elem_t weights1[HIDDEN_SIZE][HIDDEN_SIZE] row_align(1)= {0};
static elem_t inter_results1[BATCH_SIZE][HIDDEN_SIZE] row_align(1)= {0};
static elem_t weights2[HIDDEN_SIZE][OUTPUT_SIZE] row_align(1)= {0};
static elem_t inter_results2[BATCH_SIZE][OUTPUT_SIZE] row_align(1)= {0};
