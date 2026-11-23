// Two-tile on-chip chain, three issue orders, to measure what the 4th RS queue overlaps.
// Per tile t: mvin BF16 (C1_out_bf16) -> VPU *= f_t (2.0 / 4.0, exact) -> SPAD_REQUANT (tiled + resident scales)
// -> MM2 = requant(C1_t @ B2) (A read in place). B2 is loaded once. A power-of-two scale leaves every E4M3 code unchanged
// and moves every unclamped E8M0 by log2(f_t), so the header goldens stay exact: C1 codes == C1_out, C1 scales ==
// C1_scales_out + log2(f_t), C2 codes == C2_out, C2 scales == C2_scales_out + log2(f_t). (No host-side requant golden:
// it costs ~2M cycles on the RTL Rocket.)
//   fenced   : a fence after every stage (no overlap; baseline)
//   program  : tile 0's chain then tile 1's, unfenced (overlap found by the RS alone)
//   pipelined: tile 1's mvin and VPU op issued ahead of tile 0's matmul
// A warm-up pass runs first so every measured pass sees the same cache state.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64_chain.h"
#include "include/vpu_ref.h"   // VPU op codes


#define LOOP_WS_REQUANT_TILED (1u << 10)
#define CHAIN_FLAGS (0x38 | LOOP_WS_REQUANT_TILED)

#define M MATMUL_M
#define N MATMUL_N
#define K MATMUL_K
#define T 2
static const uint32_t SP_BF16[T] = {0x1000, 0x1200};   // bank 1, 512 rows each
static const uint32_t SP_C1[T]   = {128, 1024};        // bank 0, 256 rows each (tiled)
static const uint32_t SP_C2[T]   = {512, 1536};        // bank 0, 256 rows each (tiled)
static const uint16_t FACT[T]    = {0x4000, 0x4080};   // 2.0, 4.0
static const int      LOG2F[T]   = {1, 2};

static uint16_t c1_bf16[M][N] __attribute__((aligned(64)));
static uint8_t  c1_scales[T][M * MATMUL_GN] __attribute__((aligned(64)));
static uint8_t  c2_scales[T][M * MATMUL_GN] __attribute__((aligned(64)));
static uint8_t  tmp[M * N] __attribute__((aligned(64)));

static const int tiles_I = M / DIM, tiles_J = N / DIM, tiles_K = K / DIM;
#define B_BASE (BANK_NUM * BANK_ROWS - (K / DIM) * (N / DIM) * DIM)

// mvout a tiled M x N E4M3 image and compare it, one 16-byte spad row at a time, with golden[M][N]
static int mvout_compare_tiled(uint32_t sp, const uint8_t golden[M][N]) {
  gemmini_config_st(DIM);
  for (int r = 0; r < M * N / DIM; r += DIM) gemmini_extended_mvout(tmp + r * DIM, sp + r, DIM, DIM);
  gemmini_fence();
  int bad = 0;
  for (int row = 0; row < M * N / DIM; row++) {   // row = ((i * N/16) + nt) * 16 + r
    const int r = row % DIM, nt = (row / DIM) % (N / DIM), i = row / DIM / (N / DIM);
    bad += memcmp(tmp + row * DIM, &golden[i * DIM + r][nt * DIM], DIM) != 0;
  }
  return bad;   // mismatching rows
}

static int compare_scales(const uint8_t *hw, const uint8_t golden[M][MATMUL_GN], int shift, int *skipped) {
  int bad = 0;
  for (int m = 0; m < M; m++)
    for (int b = 0; b < MATMUL_GN; b++) {
      int ref = golden[m][b];
      if (ref <= 104 || ref + shift > 254) { (*skipped)++; continue; }   // clamped scales do not move
      bad += hw[b * MATMUL_M + m] != ref + shift;
    }
  return bad;
}

static int fenced;
static void stage_end(void) { if (fenced) gemmini_fence(); }

static void mvin_tile(int t) {
  gemmini_config_ld(DIM);
  for (int r = 0; r < M * N / 8; r += DIM) gemmini_extended_mvin((uint8_t *)c1_bf16 + r * DIM, SP_BF16[t] + r, DIM, DIM);
  stage_end();
}
static void vpu_tile(int t) {
  gemmini_vpu_scalar(VPU_MULS, SP_BF16[t], SP_BF16[t], FACT[t], M * N / 8);
  stage_end();
}
static void requant_tile(int t) {
  gemmini_spad_requant(SP_C1[t], SP_BF16[t], M, N, 1, (uint64_t)c1_scales[t], 1);
  stage_end();
}
static void mm_tile(int t) {
  gemmini_config_st(1 * sizeof(uint16_t));
  gemmini_mxquant_config_mvout_resident((uint64_t)c2_scales[t], tiles_I, tiles_J, tiles_K, 0, 0, 1);
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, SP_C1[t], BANK_NUM * BANK_ROWS, 0, SP_C2[t],
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false, CHAIN_FLAGS);
  stage_end();
}

static uint64_t run(int mode) {   // 0 fenced, 1 program order, 2 pipelined
  memset(c1_scales, 0xa5, sizeof(c1_scales)); memset(c2_scales, 0xa5, sizeof(c2_scales));
  gemmini_fence();
  fenced = mode == 0;
  uint64_t t0 = read_cycles();
  if (mode != 2) {
    for (int t = 0; t < T; t++) { mvin_tile(t); vpu_tile(t); requant_tile(t); mm_tile(t); }
  } else {
    mvin_tile(0); mvin_tile(1);
    vpu_tile(0); requant_tile(0);
    vpu_tile(1);          // runs while tile 0's matmul computes
    mm_tile(0);
    requant_tile(1);      // waits for tile 0's stores (shared requantizer / scale state)
    mm_tile(1);
  }
  gemmini_fence();
  return read_cycles() - t0;
}

static int check(const char *name) {
  int bad = 0, skipped = 0;
  for (int t = 0; t < T; t++) {
    int b1 = mvout_compare_tiled(SP_C1[t], C1_out), b1s = compare_scales(c1_scales[t], C1_scales_out, LOG2F[t], &skipped);
    int b2 = mvout_compare_tiled(SP_C2[t], C2_out), b2s = compare_scales(c2_scales[t], C2_scales_out, LOG2F[t], &skipped);
    if (b1 || b1s || b2 || b2s)
      printf("  %s tile %d: C1 rows %d, C1 scales %d, C2 rows %d, C2 scales %d mismatches\n", name, t, b1, b1s, b2, b2s);
    bad += b1 + b1s + b2 + b2s;
  }
  printf("%-9s check: %s (%d clamped scales skipped)\n", name, bad ? "FAIL" : "ok", skipped);
  return bad != 0;
}

int main() {
  memcpy(c1_bf16, C1_out_bf16, sizeof(c1_bf16));

  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 0);
  gemmini_mx_load_scales((uint64_t)&B2_scales_col, sizeof(B2_scales_col), 1);
  gemmini_config_ld(N * sizeof(uint8_t));
  for (int j = 0; j < tiles_J; j++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((uint8_t *)B2_in + j * DIM * N + k * DIM, B_BASE + (j * tiles_K + k) * DIM, DIM, DIM);
  gemmini_fence();

  int fail = 0;
  run(2);                                    // warm-up
  fail |= check("warmup");
  const char *names[3] = {"fenced", "program", "pipelined"};
  uint64_t cyc[3];
  for (int mode = 0; mode < 3; mode++) {
    cyc[mode] = run(mode);
    fail |= check(names[mode]);
  }
  printf("PERF chain_pipelined (2 tiles: mvin 8 KB, vpu 512 rows, spad_requant 128 blocks, mm2 64x64x64 each):\n");
  for (int mode = 0; mode < 3; mode++)
    printf("PERF   %-9s %6llu cycles (%llu%% of fenced)\n", names[mode], (unsigned long long)cyc[mode],
           (unsigned long long)(cyc[mode] * 100 / cyc[0]));
  printf("chain_pipelined %s\n", fail ? "FAILED" : "PASSED");
  return fail;
}
