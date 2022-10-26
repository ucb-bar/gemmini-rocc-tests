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
#define total_workloads 100
#define QoS 0
#define WORKLOAD_CORE 2
#define NUM_GROUP (int)(NUM_CORE / WORKLOAD_CORE)
#define QUEUE_DEPTH 5
#define NUM_ITER 5
#define CAP 5 // 0 to 1 (smaller number: shorter time between workload dispatch time)
#define CAP_SCALE 0.8
#define TARGET_SCALE 1

#include "include/gemmini_testutils.h"
#include "include/gemmini_nn.h"
#include "workload_8.h"

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
      workload_mode_2(total_workloads, true, false, false, SEED, TARGET_SCALE, CAP_SCALE);
    }
  }
  barrier(nc);
	for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("workload type: ");
      for(int j = 0; j < total_workloads; j++){
        printf(" %d,", total_queue_type[j]);
      }
      printf("\n");
    }
    barrier(nc);
  }

  for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("workload target: ");
      for(int j = 0; j < total_workloads; j++){
        printf(" %llu,", total_queue_target[j]);
      }
      printf("\n");
    }
    barrier(nc);
  }

  for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("workload priority: ");
      for(int j = 0; j < total_workloads; j++){
        printf(" %d,", total_queue_priority[j]);
      }
      printf("\n");
    }
    barrier(nc);
  }


  for (int i = 0; i < nc; i++) {
    if (i == cid && i == 1) {
      printf("workload dispatch time: ");
      for(int j = 0; j < total_workloads; j++){
        printf(" %llu,", total_queue_dispatch[j]);
      }
      printf("\n");
    }
    barrier(nc);
  }




  uint64_t cycle = 0;//{0, 84364462, 218804975, 279434942, 305350463, 354981134, 372031312};
  int c = 0;
  while(!done){
    cycle = 1000000*c;
      for(int j = 0; j < nc; j++){
        if(j == cid && j == 0){
          workload_groups = workload_priority_sp(total_workloads, cycle);
          //workload_groups = workload_priority_mp(total_workloads, NUM_ITER, cycle); 
          printf("number of groups: %d\n", workload_groups);
          //if (k == 1) c += 1;
          c++;
          done = (workload_groups == -1);
          //done = (workload_groups == 0 && total_queue_status[total_workloads - 1] != -1);
          workload_grouping(workload_groups);
          //printf("finished grouping\n");
        }
      }
      barrier(nc);
      if(done)
        break;
/*
      for(int k = 0;  k < NUM_GROUP; k++)
      for (int i = 0; i < nc; i++) {
        if (i == cid && i == 1) {
          for(int x = 0; x < workload_groups; x++){
            for(int y = 0; y < SUB_GROUP; y++){
              printf("queue %d, group %d: ", x, y);
              for(int j = 0; j < QUEUE_DEPTH; j++)
                printf("%d, ", gemmini_workload_assigned[k][y][x][j]);
              printf("\n");
            }
          }
          printf("\n");
        }
      }
*/
      for(int i = 0; i < nc; i++){
        if(i == cid && i == 1){
          for(int j = 0; j < workload_groups; j++)
            printf("%d, ", gemmini_workload_assigned[0][0][0][j]);
          printf("\n");
        }
      }
      barrier(nc);
/*

      for (int i = 0; i < nc; i++) {
        if (i == cid && i == 1) {
          for(int x = 0; x < workload_groups; x++){
            for(int y = 0; y < SUB_GROUP; y++){
              printf("queue %d, group %d: ", x, y);
              for(int j = 0; j < QUEUE_DEPTH; j++){
                int index = gemmini_workload_assigned[k][y][x][j]; 
                if (index != -1) printf("%d, ", total_queue_type[k][index]);
                else printf("%d, ", -1);
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
          for(int x = 0; x < workload_groups; x++){
            for(int y = 0; y < SUB_GROUP; y++){
              printf("queue %d, group %d: ", x, y);
              for(int j = 0; j < QUEUE_DEPTH; j++){
                int index = gemmini_workload_assigned[k][y][x][j]; 
                if (index != -1) printf("%d, ", total_queue_priority[k][index]);
                else printf("%d, ", -1);
              }
              printf("\n");
            }
          }
          printf("\n");
        }
      }
      barrier(nc);
    }
*/
      for(int k = 0; k < NUM_GROUP; k++)
      for (int i = 0; i < nc; i++) {
        if (i == cid && i == 1) {
          printf("grouped \n");
          for(int x = 0; x < workload_groups; x++){
            for(int y = 0; y < SUB_GROUP; y++){
              printf("group %d queue %d, sub group %d: ", k, x, y);
              for(int j = 0; j < QUEUE_DEPTH; j++)
                printf("%d, ", gemmini_workload_grouped[k][y][x][j]);
              printf("\n");
            }
          }
          printf("\n");
        }
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

