// VPU softmax over a 16 x 32 BF16 score tile (one logical row = 32 BF16 = 4 scratchpad rows), entirely on the
// VPU with no fences between steps: rmax -> sub(bcast) -> exp -> rsum -> rcp -> mul(bcast). Checked bit-exact
// against include/vpu_ref.h, and against an FP32 softmax for accuracy.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"


#define M 16
#define L 32
#define RL (L / VPU_LANES)    // 4 scratchpad rows per logical row
#define ROWS (M * RL)
#define SP_S   0x0000              // bank 0 (BANK_ROWS units: DIM=16 is 0x0, 0x400, 0x1000, 0x2000, 0x3000)
#define SP_P   (BANK_ROWS / 4)     // bank 0
#define SP_MX  (1 * BANK_ROWS)     // bank 1
#define SP_X   (2 * BANK_ROWS)     // bank 2
#define SP_SUM (3 * BANK_ROWS)     // bank 3

static uint16_t S[ROWS][VPU_LANES] __attribute__((aligned(64)));
static uint16_t P_hw[ROWS][VPU_LANES] __attribute__((aligned(64)));
static uint16_t mx[M][VPU_LANES], X[ROWS][VPU_LANES], sum[M][VPU_LANES], P_ref[ROWS][VPU_LANES];

static uint32_t lcg = 777;
static uint32_t rnd(void) { lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }

static void mvin_rows(const void *src, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM) gemmini_extended_mvin((const uint8_t *)src + r * DIM * sizeof(elem_t), sp + r, DIM, DIM);
}
static void mvout_rows(void *dst, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM) gemmini_extended_mvout((uint8_t *)dst + r * DIM * sizeof(elem_t), sp + r, DIM, DIM);
}

int main() {
  for (int r = 0; r < ROWS; r++)
    for (int l = 0; l < VPU_LANES; l++)
      S[r][l] = vpu_f_to_bf16(((float)(int)(rnd() % 2001) - 1000.0f) / 125.0f);   // [-8, 8]

  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));
  uint64_t t0 = read_cycles();
  mvin_rows(S, SP_S, ROWS);
  uint64_t t1 = read_cycles();
  gemmini_vpu_reduce(VPU_RMAX, SP_MX, SP_S, ROWS, RL);
  gemmini_vpu_bcast(VPU_SUB, SP_X, SP_S, SP_MX, ROWS, RL);
  gemmini_vpu_unary(VPU_EXP, SP_X, SP_X, ROWS);
  gemmini_vpu_reduce(VPU_RSUM, SP_SUM, SP_X, ROWS, RL);
  gemmini_vpu_unary(VPU_RCP, SP_SUM, SP_SUM, M);
  gemmini_vpu_bcast(VPU_MUL, SP_P, SP_X, SP_SUM, ROWS, RL);
  gemmini_fence();
  uint64_t t2 = read_cycles();
  mvout_rows(P_hw, SP_P, ROWS);
  gemmini_fence();

  vpu_ref_exec(VPU_RMAX, mx, S, 0, ROWS, RL, 0, 0);
  vpu_ref_exec(VPU_SUB, X, S, mx, ROWS, RL, 1, 0);
  vpu_ref_exec(VPU_EXP, X, X, 0, ROWS, 1, 0, 0);
  vpu_ref_exec(VPU_RSUM, sum, X, 0, ROWS, RL, 0, 0);
  vpu_ref_exec(VPU_RCP, sum, sum, 0, M, 1, 0, 0);
  vpu_ref_exec(VPU_MUL, P_ref, X, sum, ROWS, RL, 1, 0);

  int bad = 0;
  for (int r = 0; r < ROWS; r++)
    for (int l = 0; l < VPU_LANES; l++)
      if (P_hw[r][l] != P_ref[r][l] && bad++ < 8)
        printf("  P row %d lane %d: hw %04x ref %04x\n", r, l, P_hw[r][l], P_ref[r][l]);

  // accuracy vs FP32 softmax (integer-scaled print: no printf %f on bare metal)
  int worst_ppm = 0, worst_sum_ppm = 0;
  for (int m = 0; m < M; m++) {
    float v[L], mxf = -1e30f, s = 0, hs = 0;
    for (int c = 0; c < L; c++) { v[c] = vpu_bf16_to_f(S[m * RL + c / VPU_LANES][c % VPU_LANES]); if (v[c] > mxf) mxf = v[c]; }
    for (int c = 0; c < L; c++) {
      float d = v[c] - mxf, e = 1.0f, t = 1.0f;   // exp by series on [-16, 0]: e^d = (e^(d/16))^16
      float y = d / 16.0f;
      for (int k = 1; k < 12; k++) { t *= y / k; e += t; }
      for (int k = 0; k < 4; k++) e *= e;
      v[c] = e; s += e;
    }
    for (int c = 0; c < L; c++) {
      float p = v[c] / s, h = vpu_bf16_to_f(P_hw[m * RL + c / VPU_LANES][c % VPU_LANES]);
      int ppm = (int)((h > p ? h - p : p - h) * 1e6f);
      if (ppm > worst_ppm) worst_ppm = ppm;
      hs += h;
    }
    int sppm = (int)((hs > 1.0f ? hs - 1.0f : 1.0f - hs) * 1e6f);
    if (sppm > worst_sum_ppm) worst_sum_ppm = sppm;
  }
  printf("softmax %dx%d: %d mismatches vs ref; max |p - p_fp32| = %d ppm, max |sum - 1| = %d ppm\n",
         M, L, bad, worst_ppm, worst_sum_ppm);
  printf("cycles: mvin %llu, 6 VPU ops %llu\n", (unsigned long long)(t1 - t0), (unsigned long long)(t2 - t1));
  printf("vpu_softmax %s\n", bad ? "FAILED" : "PASSED");
  return bad != 0;
}
