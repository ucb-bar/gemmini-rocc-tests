// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#define FLOAT false
#include "include/gemmini_testutils.h"

#define NUM_INT 4
#define NUM_FP 0

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    for(int i = 0; i < NUM_INT; i++){
        int cfgid = i;
        bool acquired = rr_acquire_single(cfgid, i);
        if(acquired){
            printf("int gemmini %d acquired to cfgid %d\n", i, cfgid);
            //break;
        }
    }
    for(int i = 0; i < NUM_INT; i++){
      rr_set_opc(XCUSTOM_ACC, i);
      gemmini_flush(0);
    }
    for(int i = 0; i < NUM_INT; i++)
      rr_release(i);
    printf("all int gemmini flushed\n");


    for(int i = 0; i < NUM_FP; i++){
        int cfgid = i;
        bool acquired = rr_acquire_single(cfgid, i+NUM_INT);
        if(acquired){
            printf("fp gemmini %d acquired to cfgid %d\n", i+NUM_INT, cfgid);
            //break;
        }
    }
    for(int i = 0; i < NUM_FP; i++){
      rr_set_opc(XCUSTOM_ACC, i);
      gemmini_flush(0);
    }
    for(int i = 0; i < NUM_FP; i++)
      rr_release(i);
    printf("all fp gemmini flushed\n");


  exit(0);
}

