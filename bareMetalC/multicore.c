// See LICENSE for license details.

#include "include/threadpool.h"
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct task_args_t {
  int i;
  int cid;
} task_args_t;

void task1(void * args_ptr) {
  task_args_t * args = args_ptr;
  printf("Task 1, Thread %d output: %d\n", args->cid, args->i);
}

void task2(void * args_ptr) {
  task_args_t * args = args_ptr;
  printf("Task 2, Thread %d output: %d\n", args->cid, args->i * 10);
}

void * thread_main() {
  task_args_t task_args[THREADS];

  for (int t = 0; t < THREADS; t++) {
    task_args[t].i = t;
    task_args[t].cid = t;
    SET_TASK(t, task1, &task_args[t]);
  }
  RUN_TASKS();

  for (int t = 0; t < THREADS; t++) {
    task_args[t].i = t;
    task_args[t].cid = t;
    SET_TASK(t, task2, &task_args[t]);
  }
  RUN_TASKS();

  END_THREADS();
}

START_THREADPOOL()

