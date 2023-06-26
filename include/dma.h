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
#define k_PROBE 2

#define k_FLUSH 7

#define LOAD 0
#define STORE 1


#define ROCC_INSTRUCTION_RD_RS1_RS2(X, rd, rs1, rs2, funct) \
  ROCC_INSTRUCTION_R_R_R(X, rd, rs1, rs2, funct)

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)


#define dma_config(channel, mode, dram_addr, spad_addr, dram_stride, spad_stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_DMA, ((uint64_t) dram_addr << 5) | ((uint64_t) mode << 4) | ((uint64_t) channel), ((uint64_t) spad_addr << 40) | ((uint64_t) spad_stride << 20) | dram_stride, k_CONFIG)

#define dma_memcpy(channel, dram_offset, num_row_tile, num_col_tile, total_rows, total_bytes_per_row, tile_rows, tile_bytes_per_row) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_DMA, ((uint64_t) channel << 60) | ((uint64_t) total_bytes_per_row << 32) | ((uint64_t) total_rows << 16) | ((uint64_t) num_row_tile << 8) | ((uint64_t) num_col_tile), ((uint64_t) dram_offset << 32) | ((uint64_t) tile_rows << 20) | ((uint64_t) tile_bytes_per_row), k_MEMCPY)

// probe which tile is done
#define dma_probe(done_tile, channel) \
  ROCC_INSTRUCTION_RD_RS1_RS2(XCUSTOM_DMA, done_tile, channel, 0, k_PROBE)


// maybe not needed once we have rerocc
#define dma_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_DMA, skip, 0, k_FLUSH)


#endif // SRC_MAIN_C_DMA_H

