#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_128x128x256.h"

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define SMEM 0x40000000
#undef GEMMINI_BUSY_ADDR
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)

#define DIM 16

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)


#define ADDR_LEN 32

// BF16 values packed 4 per uint64_t output word
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_M / BF16_PER_WORD)

typedef uint8_t  elem_t;   // fp8:e4m3 input elements
typedef uint8_t  welem_t;  // fp8:e4m3 weight elements
typedef uint64_t out_t;    // 4x bf16 packed per word

// ---- Scale factor loader ----
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int n) {
  for (size_t i = 0; i < n / 8; i++) {
    sf_mem[i] = 0x7f7f7f7f7f7f7f7f;
  }
}



int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  // ---- Output buffer ----
  static out_t C_hw[MATMUL_M][OUT_COLS];
  uint32_t scale_factors[512] = {0};
  memset(C_hw, 0, sizeof(C_hw));

  // ---- Tile dimensions ----
  int tiles_I = MATMUL_M / DIM;
  int tiles_J = MATMUL_N / DIM;
  int tiles_K = MATMUL_K / DIM;

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  uint32_t acc_addr = (1u << (ADDR_LEN - 1));

  // ---- Gemmini setup ----
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();

  // ---- MVIN A: tile (i,k) -> a_base + (i*tiles_K + k)*DIM ----
  gemmini_config_ld(MATMUL_K * sizeof(elem_t));

  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_K + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  gemmini_config_ld(MATMUL_N * sizeof(elem_t));

  // ---- MVIN B: tile (k,j) -> b_base + (j*tiles_K + k)*DIM ----
  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j++) {
      elem_t *dram_ptr = ((elem_t*)B_in) + k * DIM * MATMUL_N + j * DIM;
      uint32_t sp_addr = b_base + (k * tiles_J + j) * DIM;
//      printf("j: %d, k: %d, sp_addr: %d \n", j, k, sp_addr);
//      printf("first elem: %x \n", dram_ptr[0]);
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  int SPAD_DEST = 256;

  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

  // Spike's MX datapath has NO accumulator destination. mx_loop_ws_spad derives its output index
  // as smem_base = C_spad * DIM (gemmini.cc:1154) and indexes mx_smem with it unchecked
  // (gemmini.cc:1216), so handing it acc_addr = 1<<31 gives smem_base = 0x800000000 and segfaults
  // the simulator. The one MX output route Spike models is the internal scratchpad -- which is
  // also where gemmini_mx_read_smem reads from below, at offset 0.
#ifdef SPIKE_SIM
  uint32_t c_dest = 0;
  uint32_t c_flag = 0x38;      // keep the spad store; there is no accumulator to skip to
#else
  uint32_t c_dest = acc_addr;
  uint32_t c_flag = 0xb8;      // skips the spad store; the output lands in the accumulator
#endif

  // ---- Compute ----
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base,
      BANK_NUM * BANK_ROWS,
      0,
      c_dest,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      c_flag);

//  for (int i = 0; i < tiles_I; i++) {
//    for (int j = 0; j < tiles_J; j++) {
//      uint32_t acc_tile_addr = acc_addr + (i * tiles_J + j) * DIM;
//      out_t *dram_ptr = &C_hw[i * DIM][j * DIM];
//      gemmini_mvout((void *) dram_ptr, acc_tile_addr);
//    }
//  }


//  gemmini_mvout((void*)&C_hw[0][0], 128 )
  gemmini_fence();

#ifdef SPIKE_SIM
  gemmini_mx_read_smem(&C_hw[0][0], 0, MATMUL_M * MATMUL_N);
#else
  // FP8xFP8 2x32 acc layout: acc tile = 2*DIM x 2*DIM C block; mvout rows count C rows (2 per acc row)
  gemmini_config_st(OUT_COLS * sizeof(out_t));
  for (int I = 0; I < tiles_I / 2; I++)
    for (int J = 0; J < (tiles_J + 1) / 2; J++)
      gemmini_extended_mvout((void *) &C_hw[I * 2 * DIM][J * 2 * DIM / BF16_PER_WORD],
                             acc_addr + (I * ((tiles_J + 1) / 2) + J) * DIM, DIM, 2 * DIM);
  gemmini_fence();
#endif

//  uint64_t* smem_start_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2;
//  printf("Address: %p \n", smem_start_addr);
//  for (int i = 0; i < MATMUL_M; i ++) {
//    for (int j = 0; j < OUT_COLS; j++) {
////       printf("addr: %p \n", smem_start_addr + (i*8 + j) );
////       printf("Elem: %d = %lx \n", i * MATMUL_M + j, *(smem_start_addr + (i*OUT_COLS + j)));
//        C_hw[i][j] = *(smem_start_addr + (i*OUT_COLS + j));
//    }
//  }

  // ---- Debug print tile (0,0) ----
//  printf("=== Tile (0,0) - acc_addr=0x%08x ===\n", acc_addr);
//  for (int i = 0; i < DIM; i++) {
//    for (int j = 0; j < DIM; j++) {
//      printf("C_hw[%d][%d] = 0x%016lx\n", i, j, (unsigned long)C_hw[i][j]);
//    }
//  }

  // ---- Elementwise check against C_out_bf16 ----
  int errors = 0;

  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t got = C_hw[i][j];

      // Pack 4 consecutive bf16 golden values into expected uint64_t
      uint64_t exp = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                     ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                     ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                     ((uint64_t)C_out_bf16[i][j*4 + 0]);

      if (got != exp) {
        for (int lane = 0; lane < BF16_PER_WORD; lane++) {
          uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
          uint16_t exp_bf16 = C_out_bf16[i][j * BF16_PER_WORD + lane];
          if (got_bf16 != exp_bf16) {
            printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n",
                   i, j * BF16_PER_WORD + lane, got_bf16, exp_bf16);
            errors++;
          }
        }
      }
    }
  }

  if (errors == 0) {
    printf("fp8 WS matmul test PASSED (no mismatches).\n");
  } else {
    printf("fp8 WS matmul test FAILED with %d mismatches.\n", errors);
  }

#ifndef BAREMETAL
  exit(errors != 0);
#else
  return errors != 0;
#endif
}