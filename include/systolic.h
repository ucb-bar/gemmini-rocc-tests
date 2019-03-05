// See LICENSE for license details.

#ifndef SRC_MAIN_C_SYSTOLIC_H
#define SRC_MAIN_C_SYSTOLIC_H

#include "rocc-software/src/xcustom.h"

#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 8
#define k_SETMODE 9

#define XCUSTOM_ACC 3

#define matmul_mvin(y, dram_addr, spad_addr)                        \
  ROCC_INSTRUCTION(XCUSTOM_ACC, y, dram_addr, spad_addr, k_MVIN);
#define matmul_mvout(dram_addr, spad_addr)                              \
  ROCC_INSTRUCTION(XCUSTOM_ACC, 0, spad_addr, dram_addr, k_MVOUT);

#endif  // SRC_MAIN_C_SYSTOLIC_H
