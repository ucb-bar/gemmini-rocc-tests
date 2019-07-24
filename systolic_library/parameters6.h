
#include <stdio.h>
#include "include/systolic.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// batch size: 32
// before zeropad: 784x800x10
// after zeropad: 800x800x32
elem_t input_mat[32][800] row_align(1)= {0};
elem_t weights0[800][800] row_align(1)= {0};
elem_t inter_results0[32][800] row_align(1)= {0};
elem_t weights1[800][32] row_align(1)= {0};
elem_t inter_results1[32][32] row_align(1)= {0};
