
#include <stdio.h>
#include "include/systolic.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// batch size: 16
// before zeropad: 784x800x10
// after zeropad: 784x800x16
elem_t input_mat[16][784] row_align(1)= {0};
elem_t weights0[784][800] row_align(1)= {0};
elem_t inter_results0[16][800] row_align(1)= {0};
elem_t weights1[800][16] row_align(1)= {0};
elem_t inter_results1[16][16] row_align(1)= {0};
