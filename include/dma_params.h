#ifndef DMA_PARAMS_H
#define DMA_PARAMS_H

#include <stdint.h>
#include <limits.h>

#define XCUSTOM_DMA 1
//#define ADDR_LEN 64

// DMA channel configuration
#define NUM_DMA 1
#define NUM_CHANNEL 4
#define TOTAL_CHANNEL (NUM_DMA * NUM_CHANNEL) // assume 1 for now
#define DMA_MAX_BYTES 16 // 128 bits

// Scratchpad configuration
#define SPAD_BANK_NUM 2
#define SPAD_BANK_ROWS 4096
#define BASE_ADDR 0x4000000 

#endif // DMA_PARAMS_H
