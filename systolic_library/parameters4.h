
#include <stdio.h>
#include "include/systolic.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// batch size: 16
// before zeropad: 3036x4554x3036
// after zeropad: 3040x4560x3040
elem_t input_mat[16][3040] row_align(1)= {0};
elem_t weights0[3040][4560] row_align(1)= {0};
elem_t inter_results0[16][4560] row_align(1)= {0};
elem_t weights1[4560][3040] row_align(1)= {0};
elem_t inter_results1[16][3040] row_align(1)= {0};
