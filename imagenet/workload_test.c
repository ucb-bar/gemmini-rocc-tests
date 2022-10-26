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
#include "util.h"
#define SEED 0
#define total_workloads 200
#define QoS 0
#define WORKLOAD_CORE 2
#define NUM_GROUP (int)(NUM_CORE / WORKLOAD_CORE)
#define QUEUE_DEPTH 5
#define NUM_ITER 4
#define CAP 4 // 0 to 1 (smaller number: shorter time between workload dispatch time)
#define CAP_SCALE 0.7
#define TARGET_SCALE 1

#include "include/gemmini_testutils.h"
#include "include/gemmini_nn.h"
#include "workload.h"

static int workload_groups = 0;
bool done = false;

void thread_entry(int cid, int nc)
{
  gemmini_flush(0);
	for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
    barrier(nc);
  }
  int gemmini_cid = cid; 
  
  barrier(nc);
//  int qos = 0; 

  for(int j = 0; j < nc; j++){
    if(j == cid && j == 0){
      //workload_mode_1(QoS, total_workloads, true, false, false, SEED, CAP, TARGET_SCALE, CAP_SCALE);
      workload_mode_2(total_workloads, true, false, false, SEED, CAP, TARGET_SCALE, CAP_SCALE);
    }
  }
  barrier(nc);
	for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("cid %d workload type: ", i);
      for(int j = 0; j < total_workloads; j++){
        printf(" %d,", total_queue_type[j]);
      }
      printf("\n");
    }
    barrier(nc);
  }

  uint64_t cycle[7] = {0, 84364462, 218804975, 279434942, 305350463, 354981134, 372031312};
  int c = 0;
  while(!done && c < 7){
    for(int j = 0; j < nc; j++){
      if(j == cid && j == 0){
        workload_groups = workload_priority_mp(NUM_GROUP, total_workloads, NUM_ITER, cycle[c]); 
        printf("number of groups: %d\n", workload_groups);
        c += 1;
        done = (workload_groups == 0);
        workload_grouping(workload_groups, NUM_GROUP);
        printf("finished grouping\n");
      }
    }
    barrier(nc);
    if(done)
      break;

    for (int i = 0; i < nc; i++) {
      if (i == cid && i == 1) {
        for(int x = 0; x < workload_groups; x++){
          for(int y = 0; y < NUM_GROUP; y++){
            printf("queue %d, group %d: ", x, y);
            for(int j = 0; j < QUEUE_DEPTH; j++)
              printf("%d, ", gemmini_workload_assigned[y][x][j]);
            printf("\n");
          }
        }
        printf("\n");
      }
    }
    barrier(nc);



    for (int i = 0; i < nc; i++) {
      if (i == cid && i == 1) {
        for(int x = 0; x < workload_groups; x++){
          for(int y = 0; y < NUM_GROUP; y++){
            printf("queue %d, group %d: ", x, y);
            for(int j = 0; j < QUEUE_DEPTH; j++){
              int index = gemmini_workload_assigned[y][x][j]; 
              if (index != -1) printf("%d, ", total_queue_type[index]);
            }
            printf("\n");
          }
        }
        printf("\n");
      }
    }
    barrier(nc);

    for (int i = 0; i < nc; i++) {
      if (i == cid && i == 1) {
        printf("grouped \n");
        for(int x = 0; x < workload_groups; x++){
          for(int y = 0; y < NUM_GROUP; y++){
            printf("queue %d, group %d: ", x, y);
            for(int j = 0; j < QUEUE_DEPTH; j++)
              printf("%d, ", gemmini_workload_grouped[y][x][j]);
            printf("\n");
          }
        }
        printf("\n");
      }
    }
    barrier(nc);
  }


  for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("cid %d workload target: ", i);
      for(int j = 0; j < total_workloads; j++){
        printf(" %llu,", total_queue_target[j]);
      }
      printf("\n");
    }
    barrier(nc);
  }

  for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("cid %d workload priority: ", i);
      for(int j = 0; j < total_workloads; j++){
        printf(" %d,", total_queue_priority[j]);
      }
      printf("\n");
    }
    barrier(nc);
  }

  for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("cid %d workload qos: ", i);
      for(int j = 0; j < total_workloads; j++){
        printf(" %d,", total_queue_qos[j]);
      }
      printf("\n");
    }
    barrier(nc);
  }

  for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("cid %d workload dispatch time: ", i);
      for(int j = 0; j < total_workloads; j++){
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

