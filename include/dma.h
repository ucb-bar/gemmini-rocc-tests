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
#define k_MEMCPY_SUB 2
#define k_PROBE 3
#define k_SET 4

#define k_FLUSH 7

#define LOAD 0
#define STORE 1


#define ROCC_INSTRUCTION_RD_RS1_RS2(X, rd, rs1, rs2, funct) \
  ROCC_INSTRUCTION_R_R_R(X, rd, rs1, rs2, funct)

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)

// base dram & spad address, stride
// mode: memcpy in / out
#define dma_config(channel, mode, dram_addr, spad_addr, dram_stride, spad_stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_DMA, ((uint64_t) dram_addr << 5) | ((uint64_t) mode << 4) | ((uint64_t) channel), ((uint64_t) spad_addr << 40) | ((uint64_t) spad_stride << 20) | dram_stride, k_CONFIG)

// copy 1 tile
// granted: return value whether DMA granted the requested ISA
// offset: offset address from the config base address
// tile_rows: number of rows of this tile to memcpy in/out ; tile_bytes_per_row: number of bytes per row
// index: index tag of this tile to track 
#define dma_memcpy_tile(channel, granted, dram_offset, spad_offset, index, tile_rows, tile_bytes_per_row) \
  ROCC_INSTRUCTION_RD_RS1_RS2(XCUSTOM_DMA, granted, ((uint64_t) channel << 60) | ((uint64_t) index << 52) | ((uint64_t) tile_rows << 32) | ((uint64_t) spad_offset), ((uint64_t) dram_offset << 32) | ((uint64_t) tile_bytes_per_row), k_MEMCPY)

// copy multiple tile
// inter_tile_offset: starting address offset between the tiles (bytes)
// num_tile: number of multi-tile to memcpy
#define dma_memcpy_multitile(channel, granted, dram_offset, spad_offset, index, inter_tile_dram_offset, inter_tile_spad_offset, num_tile, tile_rows, tile_bytes_per_row) \
  ROCC_INSTRUCTION_RD_RS1_RS2(XCUSTOM_DMA, granted, ((uint64_t) channel << 60) | ((uint64_t) index << 54) | ((uint64_t) tile_rows << 46) | ((uint64_t) tile_bytes_per_row << 34) | ((uint64_t) inter_tile_spad_offset << 17) | ((uint64_t) inter_tile_dram_offset), ((uint64_t) num_tile << 56) | ((uint64_t) spad_offset << 32) | ((uint64_t) dram_offset), k_MEMCPY_SUB)

// probe state register to know which tile is done
// memcpy accelerator with set DMA state register to indicate up until which tile index has finished memcpy
#define dma_probe_state(done_tile, channel) \
  ROCC_INSTRUCTION_RD_RS1_RS2(XCUSTOM_DMA, done_tile, channel, 0, k_PROBE)

// CPU writes finished state index
// e.g. when accelerator output needs to be used by others, CPU sets the index of the tile that accel has finihsed mvout, so that the consumer can probe to track dependency without synchronization
// since accelerator cannot set DMA channel's state register, unlike memcpy accelerator
#define dma_set_state(done_tile, channel) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_DMA, channel, done_tile, k_SET)

// maybe not needed once we have rerocc
#define dma_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_DMA, skip, 0, k_FLUSH)


#endif // SRC_MAIN_C_DMA_H

