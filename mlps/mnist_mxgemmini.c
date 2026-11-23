#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "mnist_mxgemmini_params.h"

#define DIM 16
#define ADDR_LEN 32
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_N / BF16_PER_WORD)

#ifdef BAREMETAL
// Override the weak handle_trap (which silently exits 1337) so an unexpected
// trap reports its cause/epc instead. mcause: 1=instr access, 2=illegal instr,
// 5=load access fault, 7=store access fault, etc.
extern void exit(int);
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  printf("\n*** TRAP mcause=0x%lx mepc=0x%lx ***\n",
         (unsigned long)cause, (unsigned long)epc);
  exit(2);
  return 0;
}
#endif

// The RTL core has no FP unit, so compare bf16 logits with integer ops only.
// Map each bf16 (IEEE sign/exp/mantissa) bit pattern to a monotonically
// ordered unsigned key: positives get the sign bit set (so they sort above
// negatives), negatives are bit-inverted (so larger magnitude sorts lower).
// argmax over the keys is then argmax over the real values.
static uint16_t bf16_order_key(uint16_t b) {
  return b ^ ((b & 0x8000) ? 0xFFFF : 0x8000);
}

static int argmax_first_n(const uint16_t row[MATMUL_N], int n) {
  int best = 0;
  uint16_t best_k = bf16_order_key(row[0]);
  for (int j = 1; j < n; j++) {
    uint16_t k = bf16_order_key(row[j]);
    if (k > best_k) { best_k = k; best = j; }
  }
  return best;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  static uint64_t C_hw[MATMUL_M][OUT_COLS];
  static uint32_t out_scales[512] = {0};
  memset(C_hw, 0, sizeof(C_hw));

  int tiles_I = MATMUL_M / DIM;
  int tiles_J = MATMUL_N / DIM;
  int tiles_K = MATMUL_K / DIM;

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;

  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
                              1, 1, 0, 0, false, 0, 0, 3, 0);

  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

  gemmini_config_ld(MATMUL_K * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      const uint8_t *dram_ptr = ((const uint8_t*)A_in) + i * DIM * MATMUL_K + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  // B is [K][N] row-major in DRAM, so the scratchpad tiles must be laid out
  // k-major (k*tiles_J + j) to match what gemmini_loop_ws_spad expects -- same
  // as bareMetalC/matmul_tiled_fp8_128x128x256.c. (A j-major layout happens to
  // coincide only when tiles_J == 1, which is why the single-tile case worked.)
  gemmini_config_ld(MATMUL_N * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j++) {
      const uint8_t *dram_ptr = ((const uint8_t*)B_in) + k * DIM * MATMUL_N + j * DIM;
      uint32_t sp_addr = b_base + (k * tiles_J + j) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  int SPAD_DEST = 128;

  gemmini_config_st(OUT_COLS * sizeof(uint64_t));
  gemmini_mxquant_config_mvout((uint64_t)out_scales, tiles_I, tiles_J, tiles_K, 0, 0, 1);

  uint64_t start = read_cycles();
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base,
      BANK_NUM * BANK_ROWS,
      0,
      SPAD_DEST,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      0x38);
  uint64_t end = read_cycles();

  // Drain the mvout store DMA before reading results back.
  gemmini_fence();

  // BF16 C in the internal spad (row-major) -> flat spad->DRAM mvout
  gemmini_config_st(DIM * sizeof(uint8_t));
  for (int r = 0; r < MATMUL_M * MATMUL_N * 2 / DIM; r += DIM)
    gemmini_extended_mvout((uint8_t *) C_hw + r * DIM, SPAD_DEST + r, DIM, DIM);
  gemmini_fence();

  int bf16_errors = 0;
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t got = C_hw[i][j];
      uint64_t exp = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                     ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                     ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                     ((uint64_t)C_out_bf16[i][j*4 + 0]);
      if (got != exp) {
        for (int lane = 0; lane < BF16_PER_WORD; lane++) {
          uint16_t g = (got >> (lane * 16)) & 0xFFFF;
          uint16_t e = C_out_bf16[i][j * BF16_PER_WORD + lane];
          if (g != e) bf16_errors++;
        }
      }
    }
  }

  // static: at 64x64 this is 8 KB and would otherwise blow the baremetal stack.
  static uint16_t logits[MATMUL_M][MATMUL_N];
  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t w = C_hw[i][j];
      for (int lane = 0; lane < BF16_PER_WORD; lane++) {
        logits[i][j * BF16_PER_WORD + lane] = (w >> (lane * 16)) & 0xFFFF;
      }
    }
  }

  int correct = 0;
  for (int i = 0; i < MNIST_NUM_CLASSES; i++) {
    int pred = argmax_first_n(logits[i], MNIST_NUM_CLASSES);
    int label = mnist_labels[i];
    printf("img %d: predicted=%d label=%d %s\n",
           i, pred, label, pred == label ? "OK" : "WRONG");
    if (pred == label) correct++;
  }

  printf("MX-FP8 MNIST: %d/%d correct, bf16 mismatches=%d, cycles=%llu\n",
         correct, MNIST_NUM_CLASSES, bf16_errors, (unsigned long long)(end - start));

  int pass = (bf16_errors == 0) && (correct == MNIST_NUM_CLASSES);
  if (pass) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }

#ifndef BAREMETAL
  exit(pass ? 0 : 1);
#else
  return pass ? 0 : 1;
#endif
}
