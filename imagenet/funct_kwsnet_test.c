// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/gemmini_nn.h"
#include "funct_kwsnet.h"
#include "util.h"

#define NUM_LAYER (25+1+11+4)
#define NUM_CORE 1

void thread_entry(int cid, int nc)
{
  gemmini_flush(0);
	for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
    barrier(nc);
  }
  int gemmini_cid = cid; 
  
  barrier(nc);

  uint64_t * cycles;


  for(int j = 0; j < nc; j++){
    if(j == cid && j < NUM_CORE){
#ifndef BAREMETAL
      *cycles = kwsnet_function(j, NUM_LAYER, cycles, 4, 1, 0, &barrier);
#else
      cycles = kwsnet_function(j, NUM_CORE, 1, 10);
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

