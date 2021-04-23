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
#include "include/threadpool.h"

int t1, t2, t3, t4;

void t1_task() {
  printf("t1: %d\n", t1);
}

void t2_task() {
  printf("t2: %d\n", t2);
}

void t3_task() {
  printf("t3: %d\n", t3);
}

void t4_task() {
  printf("t4: %d\n", t4);
}

void t1_task_2() {
  printf("t1: %d\n", -t1);
}

void t2_task_2() {
  printf("t2: %d\n", -t2);
}

void t3_task_2() {
  printf("t3: %d\n", -t3);
}

void t4_task_2() {
  printf("t4: %d\n", -t4);
}

void * thread_main() {
  t1 = 10;
  t2 = 20;
  t3 = 30;
  t4 = 40;

  thread_tasks[0] = t1_task;
  thread_tasks[1] = t2_task;
  thread_tasks[2] = t3_task;
  thread_tasks[3] = t4_task;

  RUN_TASKS();

  thread_tasks[0] = t1_task_2;
  thread_tasks[1] = t2_task_2;
  thread_tasks[2] = t3_task_2;
  thread_tasks[3] = t4_task_2;

  RUN_TASKS();

  end_threads = 1;
  RUN_TASKS();
}

MAIN()
