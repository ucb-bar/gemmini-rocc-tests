// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#define num_proc 8
#define NUM_LAYER (19+36+16+4)
#define NUM_CORE num_proc
#include "include/gemmini_testutils.h"
#include "include/gemmini_nn.h"
#include "funct_fcnnet_1.h"
#include "util.h"


void thread_entry(int cid, int nc)
{
  gemmini_flush(0);
	for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
    barrier(nc);
  }
  int gemmini_cid = cid; 
  
  barrier(nc);

  uint64_t dummy[NUM_LAYER] = {0};
  //uint64_t * cycles[NUM_LAYER];
  uint64_t * cycles;


  for(int j = 0; j < nc; j++){
    if(j == cid && j == 0){// < NUM_CORE){
#ifndef BAREMETAL
      *cycles = fcnnet_function(j, NUM_LAYER, cycles, 4, 1, 0, &barrier);
#else
      cycles = fcnnet_function_1(j, 0, num_proc, 1, -1);
#endif
    }
  }

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

