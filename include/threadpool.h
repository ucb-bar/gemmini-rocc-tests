#ifndef THREADPOOL_HEADER
#define THREADPOOL_HEADER

#ifndef BAREMETAL
#include <pthread.h>
#else
#include "util.h"
#endif

#define THREADS 4

#if THREADS > 4
#error No more than four threads allowed
#endif

void (*thread_tasks[THREADS])();
int end_threads = 0;

#ifndef BAREMETAL
pthread_t thread_id_main, thread_id_1, thread_id_2, thread_id_3, thread_id_4;
pthread_barrier_t threadpool_barrier;
pthread_attr_t attr;
#endif

#ifndef BAREMETAL
#define BARRIER() pthread_barrier_wait(&threadpool_barrier);
#else
#define BARRIER() barrier(THREADS+1);
#endif

#define RUN_TASKS() { BARRIER(); BARRIER(); }

void * thread1() {
  if (THREADS < 1) {
    return NULL;
  }

  while(1) {
    BARRIER();
    if (end_threads) {
      BARRIER();
      return NULL;
    } else if (thread_tasks[0] != NULL) {
      (*thread_tasks[0])();
    }
    BARRIER();
  }
  return NULL;
}

void * thread2() {
  if (THREADS < 2) {
    return NULL;
  }

  while(1) {
    BARRIER();
    if (end_threads) {
      BARRIER();
      return NULL;
    } else if (thread_tasks[1] != NULL) {
      (*thread_tasks[1])();
    }
    BARRIER();
  }
  return NULL;
}

void * thread3() {
  if (THREADS < 3) {
    return NULL;
  }

  while(1) {
    BARRIER();
    if (end_threads) {
      BARRIER();
      return NULL;
    } else if (thread_tasks[2] != NULL) {
      (*thread_tasks[2])();
    }
    BARRIER();
  }
  return NULL;
}

void * thread4() {
  if (THREADS < 4) {
    return NULL;
  }

  while(1) {
    BARRIER();
    if (end_threads) {
      BARRIER();
      return NULL;
    } else if (thread_tasks[3] != NULL) {
      (*thread_tasks[3])();
    }
    BARRIER();
  }
  return NULL;
}

#ifndef BAREMETAL

#define THREADPOOL_CREATE() \
{ \
  int ret; \
  void *res; \
  pthread_barrier_init(&threadpool_barrier,NULL,THREADS+1); \
  ret=pthread_create(&thread_id_main,NULL,&thread_main,NULL); \
  if(ret!=0) { \
    printf("Unable to create thread_main"); \
  } \
  ret=pthread_create(&thread_id_1,NULL,&thread1,NULL); \
  if(ret!=0) { \
    printf("Unable to create thread1"); \
  } \
  ret=pthread_create(&thread_id_2,NULL,&thread2,NULL); \
\
  if(ret!=0) { \
    printf("Unable to create thread2"); \
  } \
  ret=pthread_create(&thread_id_3,NULL,&thread3,NULL); \
\
  if(ret!=0) { \
    printf("Unable to create thread3"); \
  } \
  ret=pthread_create(&thread_id_4,NULL,&thread4,NULL); \
\
  if(ret!=0) { \
    printf("Unable to create thread4"); \
  } \
  printf("\n Created threads \n"); \
}

#define THREADPOOL_DESTROY() { \
  pthread_join(thread_id_main,NULL);\
  pthread_join(thread_id_1,NULL);\
  pthread_join(thread_id_2,NULL);\
  pthread_join(thread_id_3,NULL);\
  pthread_join(thread_id_4,NULL);\
  pthread_barrier_destroy(&threadpool_barrier); \
}

#define MAIN() int main() { THREADPOOL_CREATE(); THREADPOOL_DESTROY(); }

#else

#define MAIN() void thread_entry(int cid, int nc) { \
  if (cid == 0) { \
    thread_main(); \
  } \
  else if (cid == 1){ \
    thread1(); \
  } \
  else if (cid == 2){ \
    thread2(); \
  } \
  else if (cid == 3){ \
    thread3(); \
  } \
  else if (cid == 4){ \
    thread4(); \
  } \
}

int main() {}
#endif

#endif
