// See LICENSE for license details.

#ifndef SRC_MAIN_C_DMA_H
#define SRC_MAIN_C_DMA_H

#undef abs

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#include "include/dma_params.h"

#define DMA_ASSERTIONS

// Accelerator interface
#include "rocc-software/src/xcustom.h"

#define k_CONFIG 0
#define k_MEMCPY 1

//#define CONFIG_EX 0
//#define CONFIG_LD 1
//#define CONFIG_ST 2
//#define CONFIG_BERT 3

#define LOAD 0
#define STORE 1


#define ROCC_INSTRUCTION_RD_RS1_RS2(X, rd, rs1, rs2, funct) \
	ROCC_INSTRUCTION_R_R_R(X, rd, rs1, rs2, funct)

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)

#define dma_config(idle, mode, dram_addr, dram_stride, spad_stride) \
  ROCC_INSTRUCTION_RD_RS1_RS2(XCUSTOM_DMA, idle, dram_addr, ((uint64_t) mode << 63) | ((uint64_t) spad_stride << 32) | dram_stride, k_CONFIG)

#define dma_memcpy(spad_addr, row_dram_offset, col_dram_offset, rows, cols) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_DMA, ((uint64_t) spad_addr << 32) | ((uint64_t) rows << 16) | row_dram_offset, ((uint64_t) cols << 32) | col_dram_offset, k_MEMCPY)

// fence
#define dma_fence() asm volatile("fence")


#endif // SRC_MAIN_C_DMA_H

