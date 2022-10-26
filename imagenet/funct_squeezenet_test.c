// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#define NUM_CORE 8
#define NUM_GROUP 2


#include "include/gemmini_testutils.h"
#include "include/gemmini_nn.h"
#include "util.h"

#define NUM_LAYER (12+15+1+4)

#include "funct_squeezenet_1.h"
#include "workload.h"
void thread_entry(int cid, int nc)
{
  gemmini_flush(0);
	for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
    barrier(nc);
  }
  int gemmini_cid = cid; 
  
  barrier(nc);
 gemmini_dram_util[1] = 60;
  gemmini_score[1] = 5;
  gemmini_score[0] = 5;
  barrier(nc);


  uint64_t cycles[NUM_LAYER] = {0};

  for(int j = 0; j < nc; j++){
    if(j == cid && j == 0){
#ifndef BAREMETAL
      *cycles = squeezenet_function_1(j, 1, NUM_CORE, 10, &barrier);
#else
      *cycles = squeezenet_function_1(j, 0, 8, 1, -1);
#endif
    }
  }
barrier(nc);
	for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d ending\n", cid, nc);
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

