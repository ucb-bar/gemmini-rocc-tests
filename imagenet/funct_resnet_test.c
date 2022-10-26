// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#define NUM_LAYER (20+34+16+4)
#define NUM_CORE 8
#define NUM_GROUP 2


#include "include/gemmini_testutils.h"
#include "include/gemmini_nn.h"
#include "funct_resnet_1.h"
#include "util.h"

#include "workload.h"

void thread_entry(int cid, int nc)
{
  gemmini_flush(0);
	for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
    barrier(nc);
  }
  int gemmini_cid = cid; 
 
  gemmini_dram_util[1] = 60;
  gemmini_score[1] = 4;
  gemmini_score[0] = 6;
  barrier(nc);

  uint64_t dummy[NUM_LAYER] = {0};
  //uint64_t * cycles[NUM_LAYER];
  uint64_t * cycles;


  for(int j = 0; j < nc; j++){
    if(j == cid && j==0){//j < NUM_CORE){
      cycles = resnet_function_1(j, 0, true, true, true, true, 1, 1, -1);
    }
  }

	for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d ending\n", cid, nc);
    barrier(nc);
  }
	for (int i = 0; i < nc; i++) {
    if (i == cid) {
      printf("conv0 cycles: %llu\n", *(cycles+0));
      printf("conv1 cycles: %llu\n", *(cycles+1));
    }
    barrier(nc);
  }
  exit(0);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);
  exit(0);
}

