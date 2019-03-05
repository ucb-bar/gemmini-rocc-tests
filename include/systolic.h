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

#define matmul_mvin(dummy, dram_addr, spad_addr)                        \
  ROCC_INSTRUCTION(XCUSTOM_ACC, dummy, dram_addr, spad_addr, k_MVIN);
#define matmul_mvout(dummy, dram_addr, spad_addr)                              \
  ROCC_INSTRUCTION(XCUSTOM_ACC, dummy, dram_addr, spad_addr, k_MVOUT);

#define matmul_compute_preloaded(dummy, A, B)                              \
  ROCC_INSTRUCTION(XCUSTOM_ACC, dummy, A, B, k_COMPUTE_PRELOADED);

#define matmul_preload(rd, C, D)                              \
  ROCC_INSTRUCTION(XCUSTOM_ACC, rd, C, D, k_PRELOAD);
#define matmul_setmode(dummy, mode)                              \
  ROCC_INSTRUCTION(XCUSTOM_ACC, dummy, mode, 0, k_SETMODE);

#endif  // SRC_MAIN_C_SYSTOLIC_H
