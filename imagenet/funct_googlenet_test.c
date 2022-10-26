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

#include "funct_googlenet_1.h"
#include "util.h"

#define NUM_LAYER (36+22+9+4)
#define NUM_CORE num_proc
#include "include/gemmini_testutils.h"
#include "include/gemmini_nn.h"


void thread_entry(int cid, int nc)
{
  gemmini_flush(0);
	for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
    barrier(nc);
  }
  int gemmini_cid = cid; 
  
  barrier(nc);

  uint64_t cycles[NUM_LAYER] = {0};

  for(int j = 0; j < nc; j++){
    if(j == cid && j == 0){
#ifndef BAREMETAL
      *cycles = googlenet_function_1(j, NUM_CORE, 1, 10, &barrier);
#else
      *cycles = googlenet_function_1(j, 0, true, true, NUM_CORE, 1, -1);
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

