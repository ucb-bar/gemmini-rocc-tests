// build test -- helloworld
//

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"


int main() {
  printf("Hello World!\n");
  exit(0);
}
