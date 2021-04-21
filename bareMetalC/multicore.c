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
#include "util.h"

int thread_entry(int cid, int nc) {
  printf("Hello world from %d.\n", cid);
  barrier(nc);
}

int main() {
}
