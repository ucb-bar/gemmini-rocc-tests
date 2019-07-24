
#include <stdio.h>
#include "include/systolic.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// batch size: 32
// before zeropad: 784x2500x2000x1500x1000x500x10
// after zeropad: 800x2528x2016x1504x1024x512x32
elem_t input_mat[32][800] row_align(1)= {0};
elem_t weights0[800][2528] row_align(1)= {0};
elem_t inter_results0[32][2528] row_align(1)= {0};
elem_t weights1[2528][2016] row_align(1)= {0};
elem_t inter_results1[32][2016] row_align(1)= {0};
elem_t weights2[2016][1504] row_align(1)= {0};
elem_t inter_results2[32][1504] row_align(1)= {0};
elem_t weights3[1504][1024] row_align(1)= {0};
elem_t inter_results3[32][1024] row_align(1)= {0};
elem_t weights4[1024][512] row_align(1)= {0};
elem_t inter_results4[32][512] row_align(1)= {0};
elem_t weights5[512][32] row_align(1)= {0};
elem_t inter_results5[32][32] row_align(1)= {0};
