// Chained back-to-back fp8 matmul on MxGemminiRocketConfig: C2 = (A1 @ B1) @ B2.
//
// MM1 (A1 @ B1) requantizes to fp8 and leaves C1 RESIDENT in the scratchpad (V1 path,
// ex_write_to_spad=true). MM2 reuses C1 as operand A and reuses MM1's output block-scales as MM2's
// input A-scales. This file is built in stages:
//   Step C2 (this commit): run MM1, mvout C1 from the scratchpad, check it == golden C1_out and the
//     requantizer's block scales == C1_scales_out. Proves C1 is correctly resident/readable.
//   Step C3 (next): feed C1 as MM2's operand A + reuse scales -> C2, check == C2_out/C2_scales_out.
//
// Unified real-RoCC instruction stream for Spike and RTL, mirroring
// matmul_tiled_fp8_64x64_requant.c (the passing single-matmul V1 test).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64_chain.h"

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define SMEM 0x40000000

#define DIM 16

// LOOP_WS rs2 bit10: deposit the requant->spad output in the BLOCK-TILED operand-A layout (instead
// of flat row-major) so C1 can be re-read in place as MM2's operand A with zero DRAM traffic.
// OR'd into the gemmini_loop_ws_spad `skips` field. Handled by Spike (mx_loop_ws_spad) and RTL
// (LoopMatmul/Scratchpad). Default 0 = flat (all existing requant tests unchanged).
#define LOOP_WS_REQUANT_TILED (1u << 10)

// C8: automatic scale residency (always on for the chain). MM1's requant writes C1's output
// act-scales directly into the on-chip act-scale window (transposed [GN][M]) via
// gemmini_mxquant_config_mvout_resident (MX_SCALE_RESIDENT = mxquant config rs1 bit 63), so MM2
// reads its A-scales in place -- no DRAM buffer, no SW transpose/reload. Data + scales resident.
#define CHAIN_FLAGS (0x38 | LOOP_WS_REQUANT_TILED)
#define MXQUANT_CFG(...) gemmini_mxquant_config_mvout_resident(__VA_ARGS__)  // automatic resident scale reuse

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)

#define ADDR_LEN 32

typedef uint8_t  elem_t;   // fp8:e4m3 input elements
typedef uint8_t  welem_t;  // fp8:e4m3 weight elements

static inline int popcount8(uint8_t x) {
  int count = 0;
  while (x) { count += x & 1; x >>= 1; }
  return count;
}

// Load a flat scale-factor window (weight or activation), packed 8 bytes per uint64_t word.
// Only used on the non-Spike/non-MX legacy path; MX/Spike use gemmini_mx_load_scales.
static void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int INDIM, int K) {
  for (size_t k = 0; k < K/32; k++)
    for (size_t i = 0; i < INDIM / 8; i++)
      sf_mem[k*INDIM/8 + i] = ((uint64_t*) scale_factors)[k * INDIM/8 + i];
}

// Read a BLOCK-TILED fp8 output resident in the scratchpad back to a flat [M][N] DRAM buffer.
// Reads CONTIGUOUSLY (config_st=DIM) into a temp, then de-tiles in SW. A strided de-tile mvout
// (config_st=N, 16-byte rows N bytes apart) makes the writer DMA emit whole 64-byte cache lines and
// ZERO-fill the gap lines on RTL -> corrupts the output at N>=128 (rows read back as 0). Contiguous
// read + SW de-tile avoids that. tile (i,nt) is at spad (i*tiles_N+nt)*DIM, 16 contiguous rows.
static void mvout_detile(uint8_t *dst, uint32_t spad_base, int M, int N) {
  int tiles_I = M / DIM, tiles_N = N / DIM;
  int total_rows = M * N / DIM;
  static uint8_t tiled[128 * 128];
  gemmini_config_st(DIM * sizeof(uint8_t));
  for (int r = 0; r < total_rows; r += DIM)
    gemmini_extended_mvout(tiled + r * DIM, spad_base + r, DIM, DIM);
  gemmini_fence();
  for (int i = 0; i < tiles_I; i++)
    for (int nt = 0; nt < tiles_N; nt++)
      for (int r = 0; r < DIM; r++)
        for (int c = 0; c < DIM; c++)
          dst[(i * DIM + r) * N + nt * DIM + c] = tiled[((i * tiles_N + nt) * DIM + r) * DIM + c];
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif

  // ---- Tile dimensions (square 64x64 fp8) ----
  int tiles_I = MATMUL_M / DIM;
  int tiles_J = MATMUL_N / DIM;
  int tiles_K = MATMUL_K / DIM;

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  int SPAD_DEST1 = 128;   // MM1 requant output lands here (V1 internal-spad)

  // C1 output buffer (fp8 bytes, 64x64) for the residency readback + check.
  static uint8_t C1_hw[MATMUL_M][MATMUL_N];
  uint32_t c1_scales[512] __attribute__((aligned(32))) = {0};
  memset(C1_hw, 0, sizeof(C1_hw));

  // ---- Gemmini setup ----
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 0);

  // ================= MM1: C1 = requant(A1 @ B1) =================
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);   // A/activation
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);   // B/weight
  gemmini_fence();

  // MVIN A1: tile (i,k) -> a_base + (i*tiles_K + k)*DIM
  gemmini_config_ld(MATMUL_M * sizeof(elem_t));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
      gemmini_extended_mvin((void *) dram_ptr, a_base + (i * tiles_K + k) * DIM, DIM, DIM);
    }

  // MVIN B1: tile (k,j) -> b_base + (j*tiles_K + k)*DIM
  for (int j = 0; j < tiles_J; j++)
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
      gemmini_extended_mvin((void *) dram_ptr, b_base + (j * tiles_K + k) * DIM, DIM, DIM);
    }

  gemmini_config_st(1 * sizeof(uint16_t));   // matches the passing requant test's store config
  MXQUANT_CFG((uint64_t)c1_scales, tiles_I, tiles_J, tiles_K, 0, 0, 1);

  // V1 + reuse: requant FP8 output -> internal scratchpad at SPAD_DEST1, TILED (0x38 = requant-to-spad,
  // LOOP_WS_REQUANT_TILED = block-tiled operand layout so MM2 can read C1 in place).
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base,
      BANK_NUM * BANK_ROWS,
      0,
      SPAD_DEST1,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      CHAIN_FLAGS);
  gemmini_fence();

  // ---- Residency readback: C1 is stored BLOCK-TILED (tile (i,nt) at SPAD_DEST1+(i*tiles_N+nt)*DIM,
  //      16 contiguous rows). De-tile on mvout: each 16x16 tile -> DRAM C1_hw[i*DIM][nt*DIM], with
  //      config_st = N (full-row stride), reconstructing flat C1_hw[M][N]. ----
  mvout_detile((uint8_t *) C1_hw, SPAD_DEST1, MATMUL_M, MATMUL_N);

  // ---- Check C1 codes vs golden C1_out, byte by byte ----
  int errors = 0, diff1 = 0, diff2 = 0, diff3plus = 0;
  uint8_t *hw = (uint8_t *) C1_hw;
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < MATMUL_N; j++) {
      uint8_t got = hw[i * MATMUL_N + j], exp = C1_out[i][j];
      if (got != exp) {
        errors++;
        if (errors <= 64) printf("C1[%d][%d], Got: %x, Exp: %x\n", i, j, got, exp);
        int bits = popcount8(got ^ exp);
        if (bits == 1) diff1++; else if (bits == 2) diff2++; else diff3plus++;
      }
    }

  // ---- Check the E8M0 block scales MM1's requantizer wrote vs golden C1_scales_out ----
  int scale_errors = 0;
  uint8_t *sf = (uint8_t *) c1_scales;
  for (int i = 0; i < MATMUL_M; i++)
    for (int b = 0; b < MATMUL_GN; b++) {
      uint8_t got = sf[b * MATMUL_M + i], exp = C1_scales_out[i][b];
      if (got != exp) {
        scale_errors++;
        if (scale_errors <= 64) printf("C1_scale[%d][%d], Got: %x, Exp: %x\n", i, b, got, exp);
      }
    }

  if (errors == 0 && scale_errors == 0)
    printf("MM1 residency OK (C1 codes+scales exact).\n");
  else
    printf("MM1 residency FAILED: %d code, %d scale mismatches (1b=%d 2b=%d 3+b=%d).\n",
           errors, scale_errors, diff1, diff2, diff3plus);

  // ================= MM2: C2 = requant(C1 @ B2), C1 read IN PLACE from the scratchpad =================
  // C1 is resident at SPAD_DEST1 in the block-tiled operand layout, so MM2's operand A points straight
  // at it (no A mvin, no DRAM). MM1's output block-scales are REUSED as MM2's input A-scales.
  int SPAD_DEST2 = 512;                        // C2 lands here (clear of C1 @128..383 and B2 @ b_base)
  static uint8_t C2_hw[MATMUL_M][MATMUL_N];
  uint32_t c2_scales[512] __attribute__((aligned(32))) = {0};
  memset(C2_hw, 0, sizeof(C2_hw));

  // Reuse MM1's OUTPUT scales as MM2's A-scales.

  // SCALE_RESIDENT: MM1's requant already wrote C1's act-scales into the on-chip window -> no A load.
  gemmini_mx_load_scales((uint64_t)&B2_scales_col,  sizeof(B2_scales_col),  1);  // B = fresh B2 scales
  gemmini_fence();

  // MVIN only B2 (C1 is already resident as operand A). tile (k,j) -> b_base + (j*tiles_K + k)*DIM.
  gemmini_config_ld(MATMUL_M * sizeof(elem_t));
  for (int j = 0; j < tiles_J; j++)
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)B2_in) + j * DIM * MATMUL_M + k * DIM;
      gemmini_extended_mvin((void *) dram_ptr, b_base + (j * tiles_K + k) * DIM, DIM, DIM);
    }

  gemmini_config_st(1 * sizeof(uint16_t));
  // C1's resident scales landed in the act-window half MM1 did not read (half 1): MM2 reads that half
  MXQUANT_CFG((uint64_t)c2_scales, tiles_I, tiles_J, tiles_K, 1, 0, 1);

  // Operand A = C1 resident at SPAD_DEST1 (skip A mvin via 0x38); output C2 tiled at SPAD_DEST2.
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      SPAD_DEST1,                 // A operand = C1, read in place from the scratchpad
      BANK_NUM * BANK_ROWS,
      0,
      SPAD_DEST2,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      CHAIN_FLAGS);
  gemmini_fence();

  // ---- C2 readback (de-tile, same as C1) ----
  mvout_detile((uint8_t *) C2_hw, SPAD_DEST2, MATMUL_M, MATMUL_N);

  // ---- Check C2 codes vs golden C2_out and C2 scales vs C2_scales_out ----
  int c2_errors = 0, c2_scale_errors = 0;
  uint8_t *hw2 = (uint8_t *) C2_hw;
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < MATMUL_N; j++) {
      uint8_t got = hw2[i * MATMUL_N + j], exp = C2_out[i][j];
      if (got != exp) {
        c2_errors++;
        if (c2_errors <= 64) printf("C2[%d][%d], Got: %x, Exp: %x\n", i, j, got, exp);
      }
    }
  uint8_t *sf2 = (uint8_t *) c2_scales;
  for (int i = 0; i < MATMUL_M; i++)
    for (int b = 0; b < MATMUL_GN; b++) {
      uint8_t got = sf2[b * MATMUL_M + i], exp = C2_scales_out[i][b];
      if (got != exp) {
        c2_scale_errors++;
        if (c2_scale_errors <= 64) printf("C2_scale[%d][%d], Got: %x, Exp: %x\n", i, b, got, exp);
      }
    }
  if (c2_errors == 0 && c2_scale_errors == 0)
    printf("MM2 chain OK (C2 codes+scales exact; C1 reused in place, scales reused).\n");
  else
    printf("MM2 chain FAILED: %d code, %d scale mismatches.\n", c2_errors, c2_scale_errors);

  int total = errors + scale_errors + c2_errors + c2_scale_errors;
  if (total == 0)
    printf("fp8 chain test PASSED (MM1 resident + reused as MM2 operand, scales reused).\n");
  else
    printf("fp8 chain test FAILED: %d total mismatches.\n", total);

#ifndef BAREMETAL
  exit(total != 0);
#else
  return total != 0;
#endif
}
