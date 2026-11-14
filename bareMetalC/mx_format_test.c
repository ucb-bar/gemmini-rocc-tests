// Test to verify mx_format fields in gemmini_extended3_config_ex
// Check simulation waveform for activation_mx_format, weight_mx_format,
// output_mx_format registers in ExecuteController

#include <stdint.h>
#include <stdio.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"

// MX format encoding: FP8=0, FP6=1, FP4=2
#define GEMMINI_CTRL 0x40084000
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)

#define GEMMINI_LUT0_ADDR (GEMMINI_CTRL + 0x80)
#define GEMMINI_LUT1_ADDR (GEMMINI_CTRL + 0x200)
#define GEMMINI_LUT2_ADDR (GEMMINI_CTRL + 0x380)

#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}

int main(void) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  gemmini_flush(0);

  // Test 1: Set all formats to FP8 (0)
  printf("Test 1: act=FP8(0), wt=FP8(0), out=FP8(0)\n");
  gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 0);
  gemmini_fence();

  // Test 2: Set all formats to FP6 (1)
  printf("Test 2: act=FP6(1), wt=FP6(1), out=FP6(1)\n");
  gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 1, 1, 1, 1);
  gemmini_fence();

  // Test 3: Set all formats to FP4 (2)
  printf("Test 3: act=FP4(2), wt=FP4(2), out=FP4(2)\n");
  gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 2, 2, 2, 0);
  gemmini_fence();

  // Test 4: Set mixed formats: act=FP8(0), wt=FP6(1), out=FP4(2)
  printf("Test 4: act=FP8(0), wt=FP6(1), out=FP4(2)\n");
  gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 1, 3, 1);
  gemmini_fence();

  // Test 5: Set mixed formats: act=FP4(2), wt=FP8(0), out=FP6(1)
  printf("Test 5: act=FP4(2), wt=FP8(0), out=FP6(1)\n");
  gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 2, 0, 1, 1);
  gemmini_fence();

  // Test 6: Verify uselut bit works too
  printf("Test 6: act=FP8(0), wt=FP8(0), out=FP8(0), uselut=1\n");
  gemmini_extended3_config_ex(0, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 1);
  gemmini_fence();


  exit(0);
}
