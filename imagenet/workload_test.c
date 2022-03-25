// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#define NUM_CORE 4
#define num_proc NUM_CORE 
#include "include/gemmini_testutils.h"
#include "include/gemmini_nn.h"
#include "workload.h"
#include "util.h"

#define workloads 100
#define SEED 100
#define CAP 0.8 // 0 to 1 (smaller number: shorter time between workload dispatch time)

void thread_entry(int cid, int nc)
{
  gemmini_flush(0);
	for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
    barrier(nc);
  }
  int gemmini_cid = cid; 
  
  barrier(nc);
  int qos = 0; 

  for(int j = 0; j < nc; j++){
    if(j == cid && j == 0){
      workload_mode_1(qos, workloads, true, false, false, SEED, CAP);
      //workload_mode_2(workloads, true, false, false, SEED, CAP);
    }
  }
  barrier(nc);

	for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("cid %d workload type: ", i);
      for(int j = 0; j < workloads; j++){
        printf(" %d,", total_queue_type[j]);
      }
      printf("\n");
    }
    barrier(nc);
  }

  for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("cid %d workload priority: ", i);
      for(int j = 0; j < workloads; j++){
        printf(" %d,", total_queue_priority[j]);
      }
      printf("\n");
    }
    barrier(nc);
  }

  for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("cid %d workload qos: ", i);
      for(int j = 0; j < workloads; j++){
        printf(" %d,", total_queue_qos[j]);
      }
      printf("\n");
    }
    barrier(nc);
  }

  for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("cid %d workload dispatch time: ", i);
      for(int j = 0; j < workloads; j++){
        printf(" %llu,", total_queue_dispatch[j]);
      }
      printf("\n");
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

