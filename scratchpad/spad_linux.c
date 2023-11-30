// See LICENSE for license details.
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#endif
#include <unistd.h>
#include <fcntl.h>
#include "include/gemmini_testutils.h"

#define CHECK_RESULT 0 // 1
#define BASE 0x4000000

#define MAT_DIM_I 35
#define MAT_DIM_J 27

#define A_SCALE 2
#define B_SCALE MVIN_SCALE_IDENTITY
#define C_SCALE ACC_SCALE_IDENTITY
#define USE_RELU true

void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i])
            printf("a: %d, b: %d, index: %d\n", a[i], b[i], i);
            //return false;
    return true;
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    printf("I: %d, J: %d\n", MAT_DIM_I, MAT_DIM_J);
    int len = MAT_DIM_I * MAT_DIM_J;

#ifndef BAREMETAL
    size_t pagesize = sysconf(_SC_PAGE_SIZE);
    printf("page size: %d\n", pagesize);
	int mem_fd;
	mem_fd = open("/dev/mem", O_RDWR | O_SYNC);	  
	printf("set mem_fd: %d\n", mem_fd);

    elem_t* ptr = mmap(0, pagesize, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, BASE);  	
	if(ptr == MAP_FAILED){
		printf("Mapping Failed\n");
		return 1;
	}
    else
        printf("done mapping\n");
	printf("Ptr: %x\n", ptr);
    
    gemmini_flush(0);

    static elem_t A[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static elem_t B[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];
    printf("Init A and B\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        A[i][j] = (rand() % 64) - 32;
        B[i][j] = (rand() % 8) - 4;
        gold[i][j] = A[i][j]+B[i][j];
      }
    }
    
    for(size_t i = 0; i < 10; i++){
        printf("i%d \n", i);
        *(ptr + i) = 1;
    }
    uint64_t A_copy_addr = (uint64_t) ptr;//(uint64_t*) ptr;
    printf("A_copy_addr: %x\n", A_copy_addr);
    //elem_t* A_copy_addr = BASE + OFFSET;

    printf("start memcpy\n");
    memcpy((elem_t*) A_copy_addr, (elem_t*) A, sizeof(elem_t)*len);
    printf("done memcpy\n");
    vec_is_equal(&A[0][0], (elem_t*) A_copy_addr, sizeof(A)/sizeof(elem_t));
    printf("checked A and copied A are same\n");
    tiled_resadd_auto(MAT_DIM_I, MAT_DIM_J, 1, 1, 1, (elem_t*) A_copy_addr, (elem_t*) B, (elem_t*) C, false, WS);
    vec_is_equal(&gold[0][0], &C[0][0], sizeof(gold)/sizeof(elem_t));

    printf("unmmaping\n");
	int err = munmap(ptr, pagesize*sizeof(elem_t));
	if(err != 0){
		printf("UnMapping Failed\n");
		return 1;
	}
#endif
	exit(0);
}

