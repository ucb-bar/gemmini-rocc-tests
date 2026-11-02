// VPU op test (MxFp4E3M2E4M3VpuGemminiRocketConfig / Spike): random BF16 tiles -> mvin -> VPU_EXEC -> mvout, compared
// bit-exact against include/vpu_ref.h. No fences between mvin, VPU and mvout (ordering is the hardware's job);
// VPU_FENCE=1 adds them, to separate an ordering failure from a datapath one.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: VPU config / Spike-only test\n"); return 0; }
#else

#ifndef VPU_FENCE
#define VPU_FENCE 0
#endif
#define VFENCE() do { if (VPU_FENCE) gemmini_fence(); } while (0)
// 1: also test the optional fused ops (EXPSUB, EXPSUM); needs a VPU built with VpuParams(expSub, expSum)
#ifndef VPU_FUSED
#define VPU_FUSED 0
#endif

#define N 64          // scratchpad rows per operand (512 BF16)
#define RL 4          // rows per logical row for reductions / broadcast
// operands spread over banks: A bank 0, B bank 1, A2 bank 0 (same-bank pair), outputs bank 2/3
// in BANK_ROWS units so every DIM stays in its own bank (DIM=16: 0x0, 0x800, 0x1000, 0x2000, 0x3000, off 0x400)
#define SP_A   0x0000
#define SP_A2  (BANK_ROWS / 2)
#define SP_B   (1 * BANK_ROWS)
#define SP_D   (2 * BANK_ROWS)
#define SP_R   (3 * BANK_ROWS)
#define SP_OFF (BANK_ROWS / 4)

static uint16_t A[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t A2[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t B[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t P[N][VPU_LANES] __attribute__((aligned(64)));   // positive (rsqrt)
static uint16_t X[N][VPU_LANES] __attribute__((aligned(64)));   // |x| <= ~80 (exp)
static uint16_t out_hw[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t out_ref[N][VPU_LANES];

static uint32_t lcg = 12345;
static uint32_t rnd(void) { lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }
static uint16_t rand_bf16(int elo, int ehi, int sign) {
  uint16_t s = (sign && (rnd() & 1)) ? 0x8000 : 0;
  return (uint16_t)(s | ((elo + rnd() % (ehi - elo + 1)) << 7) | (rnd() & 0x7f));
}

static void mvin_rows(const void *src, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM)
    gemmini_extended_mvin((const uint8_t *)src + r * DIM * sizeof(elem_t), sp + r, DIM, rows - r < DIM ? rows - r : DIM);
}
static void mvout_rows(void *dst, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM)
    gemmini_extended_mvout((uint8_t *)dst + r * DIM * sizeof(elem_t), sp + r, DIM, rows - r < DIM ? rows - r : DIM);
}

static int check(const char *name, int rows) {
  int bad = 0;
  for (int r = 0; r < rows; r++)
    for (int l = 0; l < VPU_LANES; l++)
      if (out_hw[r][l] != out_ref[r][l] && bad++ < 4)
        printf("  %s row %d lane %d: hw %04x ref %04x\n", name, r, l, out_hw[r][l], out_ref[r][l]);
  printf("%-14s %s (%d mismatches)\n", name, bad ? "FAIL" : "ok", bad);
  return bad != 0;
}

// run one op: VPU -> mvout -> fence -> compare with the reference on the host copies
static int run(const char *name, int op, uint32_t dst, uint32_t s1, uint32_t s2, const uint16_t (*h1)[VPU_LANES],
               const uint16_t (*h2)[VPU_LANES], int rows, int rlen, int bcast, uint16_t imm) {
  gemmini_vpu(op, dst, s1, s2, rows, rlen, bcast, imm);
  VFENCE();
  int nout = vpu_is_reduction(op) ? rows / rlen : rows;
  memset(out_hw, 0xa5, sizeof(out_hw));
  mvout_rows(out_hw, dst, nout);
  gemmini_fence();
  vpu_ref_exec(op, out_ref, h1, h2, rows, rlen, bcast, imm);
  return check(name, nout);
}

int main() {
  for (int r = 0; r < N; r++)
    for (int l = 0; l < VPU_LANES; l++) {
      A[r][l] = rand_bf16(110, 140, 1);
      A2[r][l] = rand_bf16(110, 140, 1);
      B[r][l] = rand_bf16(110, 140, 1);
      P[r][l] = rand_bf16(100, 154, 0);
      uint16_t x;
      do x = rand_bf16(100, 133, 1); while (vpu_bf16_to_f(x) > 80.0f || vpu_bf16_to_f(x) < -80.0f);
      X[r][l] = x;
    }
  // a few specials in the unary inputs
  X[0][0] = 0x0000; X[0][1] = 0x8000; X[0][2] = 0x7f80; X[0][3] = 0xff80; X[0][4] = 0x7fc1; X[0][5] = 0x0001;
  P[0][0] = 0x0000; P[0][1] = 0x7f80; P[0][2] = 0x0001;

  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));

  mvin_rows(A, SP_A, N);
  mvin_rows(A2, SP_A2, N);
  mvin_rows(B, SP_B, N);
  mvin_rows(P, SP_D + SP_OFF, N);
  mvin_rows(X, SP_R + SP_OFF, N);
  VFENCE();

  int fail = 0;
  const uint16_t bf_half = 0x3f00, bf_3 = 0x4040;
  fail |= run("add",          VPU_ADD,  SP_D, SP_A, SP_B,  A, B,  N, 1, 0, 0);
  fail |= run("sub",          VPU_SUB,  SP_D, SP_A, SP_B,  A, B,  N, 1, 0, 0);
  fail |= run("mul",          VPU_MUL,  SP_D, SP_A, SP_B,  A, B,  N, 1, 0, 0);
  fail |= run("mul same-bank",VPU_MUL,  SP_D, SP_A, SP_A2, A, A2, N, 1, 0, 0);
  fail |= run("mul bcast",    VPU_MUL,  SP_D, SP_A, SP_B,  A, B,  N, RL, 1, 0);
  fail |= run("max",          VPU_MAX,  SP_D, SP_A, SP_B,  A, B,  N, 1, 0, 0);
  fail |= run("max same-bank",VPU_MAX,  SP_D, SP_A, SP_A2, A, A2, N, 1, 0, 0);
  fail |= run("max bcast",    VPU_MAX,  SP_D, SP_A, SP_B,  A, B,  N, RL, 1, 0);
  fail |= run("sub bcast sb", VPU_SUB,  SP_D, SP_A, SP_A2, A, A2, N, RL, 1, 0);
#if VPU_FUSED
  fail |= run("expsub",       VPU_EXPSUB, SP_D, SP_A, SP_B,  A, B,  N, 1, 0, 0);
  fail |= run("expsub bcast", VPU_EXPSUB, SP_D, SP_A, SP_B,  A, B,  N, RL, 1, 0);
  fail |= run("expsub sb",    VPU_EXPSUB, SP_D, SP_A, SP_A2, A, A2, N, RL, 1, 0);
  {   // EXPSUM: the exp rows (as EXPSUB) and, at SP_R, one row sum per logical row (as RSUM over them)
    static uint16_t sums_ref[N][VPU_LANES];
    const uint32_t pairs[2] = {SP_B, SP_A2};
    const uint16_t (*h2s[2])[VPU_LANES] = {B, A2};
    for (int t = 0; t < 2; t++) {
      fail |= run(t ? "expsum sb" : "expsum bcast", VPU_EXPSUM, SP_D, SP_A, pairs[t], A, h2s[t], N, RL, 1, SP_R);
      vpu_ref_exec(VPU_RSUM, sums_ref, (const uint16_t (*)[VPU_LANES])out_ref, 0, N, RL, 0, 0);
      memset(out_hw, 0xa5, sizeof(out_hw));
      mvout_rows(out_hw, SP_R, N / RL);
      gemmini_fence();
      memcpy(out_ref, sums_ref, sizeof(sums_ref));
      fail |= check(t ? "expsum sb sums" : "expsum sums", N / RL);
    }
  }
#endif
  fail |= run("adds",         VPU_ADDS, SP_D, SP_A, 0,     A, 0,  N, 1, 0, bf_3);
  fail |= run("muls",         VPU_MULS, SP_D, SP_A, 0,     A, 0,  N, 1, 0, bf_half);
  fail |= run("exp",          VPU_EXP,  SP_D, SP_R + SP_OFF, 0, X, 0, N, 1, 0, 0);
  fail |= run("rcp",          VPU_RCP,  SP_D, SP_A, 0,     A, 0,  N, 1, 0, 0);
  fail |= run("rsqrt",        VPU_RSQRT,SP_R, SP_D + SP_OFF, 0, P, 0, N, 1, 0, 0);
  fail |= run("rmax",         VPU_RMAX, SP_R, SP_A, 0,     A, 0,  N, RL, 0, 0);
  fail |= run("ramax",        VPU_RAMAX,SP_R, SP_A, 0,     A, 0,  N, RL, 0, 0);
  fail |= run("rsum",         VPU_RSUM, SP_R, SP_A, 0,     A, 0,  N, RL, 0, 0);
  fail |= run("rsum rlen1",   VPU_RSUM, SP_R, SP_A, 0,     A, 0,  N, 1, 0, 0);

  // chained VPU -> VPU (RAW on the same bank, in place) without fences: ((A + B) * 0.5) -> exp-safe range
  gemmini_vpu(VPU_ADD,  SP_D, SP_A, SP_B, N, 1, 0, 0);
  gemmini_vpu(VPU_MULS, SP_D, SP_D, 0,    N, 1, 0, bf_half);
  gemmini_vpu(VPU_RMAX, SP_R, SP_D, 0,    N, RL, 0, 0);
  mvout_rows(out_hw, SP_R, N / RL);
  gemmini_fence();
  vpu_ref_exec(VPU_ADD,  out_ref, A, B, N, 1, 0, 0);
  vpu_ref_exec(VPU_MULS, out_ref, out_ref, 0, N, 1, 0, bf_half);
  vpu_ref_exec(VPU_RMAX, out_ref, out_ref, 0, N, RL, 0, 0);
  fail |= check("chain", N / RL);

  // WAR: mvin overwrites a VPU source right after the VPU op is issued
  gemmini_vpu(VPU_ADD, SP_D, SP_A, SP_B, N, 1, 0, 0);
  mvin_rows(A2, SP_A, N);
  mvout_rows(out_hw, SP_D, N);
  gemmini_fence();
  vpu_ref_exec(VPU_ADD, out_ref, A, B, N, 1, 0, 0);
  fail |= check("war mvin", N);

  // two VPUs: independent ops in disjoint banks run side by side (X: bank 0, Y: bank 3); Z reads X's result on the
  // other VPU (RAW across VPUs). No fences: ordering is the RS's.
  static uint16_t ref_x[N][VPU_LANES], ref_y[N][VPU_LANES], out_x[N][VPU_LANES] __attribute__((aligned(64)));
  static uint16_t out_y[N][VPU_LANES] __attribute__((aligned(64)));
  gemmini_vpu(VPU_ADDS, SP_A + SP_OFF, SP_A, 0, N, 1, 0, bf_3);        // X  (bank 0)
  gemmini_vpu(VPU_EXP,  SP_R, SP_R + SP_OFF, 0, N, 1, 0, 0);           // Y  (bank 3)
  gemmini_vpu(VPU_MUL,  SP_D, SP_A + SP_OFF, SP_B, N, 1, 0, 0);        // Z = X * B (banks 0, 1 -> 2)
  mvout_rows(out_x, SP_A + SP_OFF, N);
  mvout_rows(out_y, SP_R, N);
  mvout_rows(out_hw, SP_D, N);
  gemmini_fence();
  vpu_ref_exec(VPU_ADDS, ref_x, A2, 0, N, 1, 0, bf_3);   // SP_A holds A2 since the WAR case above
  vpu_ref_exec(VPU_EXP,  ref_y, X, 0, N, 1, 0, 0);
  vpu_ref_exec(VPU_MUL,  out_ref, ref_x, B, N, 1, 0, 0);
  int dual = 0;
  for (int r = 0; r < N; r++)
    for (int l = 0; l < VPU_LANES; l++)
      dual += (out_x[r][l] != ref_x[r][l]) + (out_y[r][l] != ref_y[r][l]);
  printf("%-14s %s (%d mismatches)\n", "dual X,Y", dual ? "FAIL" : "ok", dual);
  fail |= dual != 0;
  fail |= check("dual Z=X*B", N);

  printf("vpu_ops %s\n", fail ? "FAILED" : "PASSED");
  return fail;
}
#endif
