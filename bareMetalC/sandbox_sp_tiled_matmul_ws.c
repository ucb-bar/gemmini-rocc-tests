
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"



#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

#ifndef BAREMETAL
#define MAT_DIM_I 512
#define MAT_DIM_K 512
#define MAT_DIM_J 512
#else
#define MAT_DIM_I 64
#define MAT_DIM_K 64
#define MAT_DIM_J 64
#endif


int main() {
#ifndef BAREMETAL
	if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
		perror("mlockall failed");
		exit(1);
	}
#endif


	gemmini_flush(0);

	static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
	static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
	static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
	static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);


}
