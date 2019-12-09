#include "riscv_test.h"
#include "riscv-tests/isa/macros/scalar/test_macros.h"
#include "include/accumulator.h"
#include "rocc-software/src/riscv_test_rocc.h"

#define CUSTOM_X 3

  RVTEST_WITH_ROCC

start:

  RVTEST_CODE_BEGIN

  li a0, 1
  lw a1, data
  ROCC_INSTRUCTION_RAW_R_R_R(CUSTOM_X,  0, 10, 10, 9)
  li a0, 0
  ROCC_INSTRUCTION_RAW_R_R_R(CUSTOM_X,  0, 10, 10, 9)
  TEST_CASE(1, a1, 0xdead, )

  TEST_PASSFAIL

  RVTEST_CODE_END

  .data
  RVTEST_DATA_BEGIN

  TEST_DATA
data:
  .dword 0xdead
  .dword 0xbeef
  .dword 0x0bad
  .dword 0xf00d

  RVTEST_DATA_END
