
#include <stdio.h>
#include "include/systolic.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// batch size: 16
// before zeropad: 440x500x440
// after zeropad: 448x512x448
elem_t input_mat[16][448] row_align(1)= {0};
elem_t weights0[448][512] row_align(1)= {0};
elem_t inter_results0[16][512] row_align(1)= {0};
elem_t weights1[512][448] row_align(1)= {0};
elem_t inter_results1[16][448] row_align(1)= {0};
