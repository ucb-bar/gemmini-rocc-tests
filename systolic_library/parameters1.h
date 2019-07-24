
#include <stdio.h>
#include "include/systolic.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// batch size: 16
// before zeropad: 784x2500x2000x1500x1000x500x10
// after zeropad: 784x2512x2000x1504x1008x512x16
elem_t input_mat[16][784] row_align(1)= {0};
elem_t weights0[784][2512] row_align(1)= {0};
elem_t inter_results0[16][2512] row_align(1)= {0};
elem_t weights1[2512][2000] row_align(1)= {0};
elem_t inter_results1[16][2000] row_align(1)= {0};
elem_t weights2[2000][1504] row_align(1)= {0};
elem_t inter_results2[16][1504] row_align(1)= {0};
elem_t weights3[1504][1008] row_align(1)= {0};
elem_t inter_results3[16][1008] row_align(1)= {0};
elem_t weights4[1008][512] row_align(1)= {0};
elem_t inter_results4[16][512] row_align(1)= {0};
elem_t weights5[512][16] row_align(1)= {0};
elem_t inter_results5[16][16] row_align(1)= {0};
