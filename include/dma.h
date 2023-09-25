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

#define k_CONFIG_S 0 // source (source) configuration
#define k_CONFIG_D 1 // destination (scratchpad) configuration
#define k_MEMCPY 2
#define k_MEMCPY_MULTI 3
#define k_PROBE 4
#define k_SET 5

#define k_FLUSH 7


#define ROCC_INSTRUCTION_RD_RS1_RS2(X, rd, rs1, rs2, funct) \
  ROCC_INSTRUCTION_R_R_R(X, rd, rs1, rs2, funct)

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)

// base source & dest address, stride
// mode: memcpy in / out
#define dma_source_config(channel, source_addr, source_stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_DMA, ((uint64_t) source_stride << 5) | ((uint64_t) channel), source_addr, k_CONFIG_S)

#define dma_dest_config(channel, dest_addr, dest_stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_DMA, ((uint64_t) dest_stride << 5) | ((uint64_t) channel), dest_addr, k_CONFIG_D)

// copy 1 tile
// granted: return value whether DMA granted the requested ISA
// offset: offset address from the config base address
// tile_rows: number of rows of this tile to memcpy in/out ; tile_bytes_per_row: number of bytes per row
// index: index tag of this tile to track 
#define dma_memcpy_tile(channel, granted, source_offset, dest_offset, index, tile_rows, tile_bytes_per_row) \
  ROCC_INSTRUCTION_RD_RS1_RS2(XCUSTOM_DMA, granted, ((uint64_t) channel << 60) | ((uint64_t) index << 52) | ((uint64_t) tile_rows << 32) | ((uint64_t) dest_offset), ((uint64_t) source_offset << 32) | ((uint64_t) tile_bytes_per_row), k_MEMCPY)


// copy multiple tile
// inter_tile_offset: starting address offset between the tiles (bytes)
// num_tile: number of multi-tile to memcpy
#define dma_memcpy_multitile(channel, granted, source_offset, dest_offset, index, inter_tile_source_offset, inter_tile_dest_offset, num_tile, tile_rows, tile_bytes_per_row) \
  ROCC_INSTRUCTION_RD_RS1_RS2(XCUSTOM_DMA, granted, ((uint64_t) channel << 60) | ((uint64_t) index << 54) | ((uint64_t) tile_rows << 46) | ((uint64_t) tile_bytes_per_row << 34) | ((uint64_t) inter_tile_dest_offset << 17) | ((uint64_t) inter_tile_source_offset), ((uint64_t) num_tile << 56) | ((uint64_t) dest_offset << 32) | ((uint64_t) source_offset), k_MEMCPY_MULTI)

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

