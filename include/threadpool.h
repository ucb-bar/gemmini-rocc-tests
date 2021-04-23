#ifndef THREADPOOL_HEADER
#define THREADPOOL_HEADER

#ifndef BAREMETAL
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <pthread.h>
#else
#include "util.h"
#endif

#define THREADS 4

void (*_thread_pool_impl_thread_tasks[THREADS])(void*);
void * _thread_pool_impl_thread_task_args[THREADS];
int _thread_pool_impl_end_threads = 0;

#ifndef BAREMETAL
pthread_t _thread_pool_impl_thread_id_main;
pthread_t _thread_pool_impl_thread_ids[THREADS];
pthread_barrier_t _thread_pool_impl_threadpool_barrier;
pthread_attr_t _thread_pool_impl_attr;
#endif

#ifndef BAREMETAL
#define BARRIER() pthread_barrier_wait(&_thread_pool_impl_threadpool_barrier);
#else
#define BARRIER() barrier(THREADS+1);
#endif

#define RUN_TASKS() { BARRIER(); BARRIER(); }

#define END_THREADS() { _thread_pool_impl_end_threads = 1; RUN_TASKS(); }

#define SET_TASK(cid,func,argptr) { \
  _thread_pool_impl_thread_tasks[cid] = func; \
  _thread_pool_impl_thread_task_args[cid] = argptr; \
}

void * _thread_pool_impl_thread_worker(int cid) {
  if (THREADS <= cid) {
    return 0;
  }

#ifndef BAREMETAL
  int s, j;
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  /* Set affinity mask to include CPUs 0 to THREADS */

  int speid = cid;

  CPU_ZERO(&cpuset);
  CPU_SET(speid, &cpuset);

  s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0) {
    printf("Failed to set pthread affinity\n");
    exit(1);
  }

  /* Check the actual affinity mask assigned to the thread */
  s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0) {
    printf("pthread_getaffinity_np failed\n");
    exit(1);
  }

  printf("Set returned by pthread_getaffinity_np() contained:\n");
  for (j = 0; j < CPU_SETSIZE; j++) {
    if (CPU_ISSET(j, &cpuset)) {
      printf("%d CPU %d\n", speid, j);
    }
  }
#endif

  while(1) {
    BARRIER();
    if (_thread_pool_impl_end_threads) {
      BARRIER();
      return 0;
    } else if (_thread_pool_impl_thread_tasks[cid] != 0) {
      (*_thread_pool_impl_thread_tasks[cid])(_thread_pool_impl_thread_task_args[cid]);
    }
    BARRIER();
  }
  return 0;
}

void * _thread_pool_impl_thread0() {
  _thread_pool_impl_thread_worker(0);
}

void * _thread_pool_impl_thread1() {
  _thread_pool_impl_thread_worker(1);
}

void * _thread_pool_impl_thread2() {
  _thread_pool_impl_thread_worker(2);
}

void * _thread_pool_impl_thread3() {
  _thread_pool_impl_thread_worker(3);
}

#ifndef BAREMETAL

#define THREADPOOL_CREATE() \
{ \
  int ret; \
  void *res; \
  pthread_barrier_init(&_thread_pool_impl_threadpool_barrier,0,THREADS+1); \
  ret=pthread_create(&_thread_pool_impl_thread_id_main,0,&thread_main,0); \
  if(ret!=0) { \
    printf("Unable to create thread_main"); \
  } \
  ret=pthread_create(&_thread_pool_impl_thread_ids[0],0,&_thread_pool_impl_thread0,0); \
  if(ret!=0) { \
    printf("Unable to create thread0"); \
  } \
  ret=pthread_create(&_thread_pool_impl_thread_ids[1],0,&_thread_pool_impl_thread1,0); \
\
  if(ret!=0) { \
    printf("Unable to create thread1"); \
  } \
  ret=pthread_create(&_thread_pool_impl_thread_ids[2],0,&_thread_pool_impl_thread2,0); \
\
  if(ret!=0) { \
    printf("Unable to create thread2"); \
  } \
  ret=pthread_create(&_thread_pool_impl_thread_ids[3],0,&_thread_pool_impl_thread3,0); \
\
  if(ret!=0) { \
    printf("Unable to create thread3"); \
  } \
  printf("\n Created threads \n"); \
}

#define THREADPOOL_DESTROY() { \
  pthread_join(_thread_pool_impl_thread_id_main,0);\
  pthread_join(_thread_pool_impl_thread_ids[0],0);\
  pthread_join(_thread_pool_impl_thread_ids[1],0);\
  pthread_join(_thread_pool_impl_thread_ids[2],0);\
  pthread_join(_thread_pool_impl_thread_ids[3],0);\
  pthread_barrier_destroy(&_thread_pool_impl_threadpool_barrier); \
}

#define START_THREADPOOL() int main() { THREADPOOL_CREATE(); THREADPOOL_DESTROY(); }

#else

#define START_THREADPOOL() void thread_entry(int cid, int nc) { \
  if (cid == 0) { \
    thread_main(); \
  } \
  else if (cid == 1){ \
    _thread_pool_impl_thread0(); \
  } \
  else if (cid == 2){ \
    _thread_pool_impl_thread1(); \
  } \
  else if (cid == 3){ \
    _thread_pool_impl_thread2(); \
  } \
  else if (cid == 4){ \
    _thread_pool_impl_thread3(); \
  } \
}

int main() {}
#endif

#endif

