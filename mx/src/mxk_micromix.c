// MicroMix linear layer (Liu et al., ICLR 2026) on the DIM>=16 BF16 BOOM configs: the input channels are permuted into
// contiguous groups, each one MX format on both operands, all accumulated into ONE accumulator (shared act-quad layout):
//   FP4 group: MXFP4 x MXFP4, mode0 (A row pairs / W column pairs in nibbles), issued first
//   FP6 group: MXFP6 E3M2 x E3M2 direct (config_ex code 1, uselut 0), mode8 on the shared layout
//   FP8 group: MXFP8 E4M3 x E4M3, mode8 on the shared layout; the last group's last loop stores C (BF16)
// Fenced between groups (each group sets its own config_ex); -DMM_NOFENCE: one fence at the end only. Data: gen/gen_mxk_micromix.py (golden = one BF16
// accumulation over the K-blocks in this issue order).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_params.h"
#include "include/gemmini_testutils.h"
#include MM_DATA_H
#include "include/mx_native.h"
#include "include/mxk_counters.h"

#ifndef MM_RUN
#define MM_RUN 16   // K-blocks per loop at most
#endif
#ifdef MK_CTR
#define MM_ITERS MXK_CTR_ITERS
#endif
#ifndef MM_ITERS
#define MM_ITERS 2  // timed repeats; the last is reported
#endif
#ifdef MK_POWER   // compute only (power runs): groups back to back, one final fence; no checks, prints, counters, memset
#ifdef MK_CTR
#error "MK_POWER: build without MK_CTR"
#endif
#define printf(...) 0   // error paths stay silent
#ifndef MK_PWR_ITERS
#define MK_PWR_ITERS 8
#endif
#endif
#define GRP 32
#define M MM_M
#define K MM_K
#define N MM_N
#define OFF(p) (MM_BLOB + MM_OFF_##p)
#define GOLD_(d) OFF(C_D##d)
#define GOLD(d) GOLD_(d)

#if DIM < 16
int main() { printf("mxk_micromix: the shared accumulator layout needs DIM >= 16\n"); return 1; }
#else

static uint16_t C[M][N] __attribute__((aligned(64)));

typedef struct { int fmt, kb, nb; const uint8_t *A, *B; } mm_group_t;

static int pick_nc(void) {   // widest chunk whose quad-layout C fits an acc half and whose runs fit a spad half
  for (int nc = 512; nc >= 2 * DIM; nc >>= 1)
    if (N % nc == 0 && M * nc / (2 * DIM) <= ACC_ROWS / 2 &&
        M * GRP * MM_RUN / DIM + GRP * MM_RUN * nc / DIM <= MXN_HALF_ROWS) return nc;
  return 0;
}

// one group's loops for column chunk c; first: its first loop overwrites the acc; last: its last loop stores C
static void issue_group(const mm_group_t *g, int NC, int c, int first, int last) {
  const uint8_t *AS = OFF(AS), *BS = OFF(BS);
  const int kg = g->nb * GRP;
  if (g->fmt == 4) {
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, /*FP4*/ 2, 2, 3, 0);
    gemmini_extended3_config_ld(kg, MVIN_SCALE_IDENTITY, false, 0);       // A [M/2][kg]
    gemmini_extended3_config_ld(N / 2, MVIN_SCALE_IDENTITY, false, 1);    // B [kg][N/2]
    gemmini_config_st(2 * N * sizeof(uint16_t));
  } else {
    const int code = g->fmt == 6 ? 1 : 0;   // E3M2 direct (uselut 0) or E4M3
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, code, code, 3, 0);
    gemmini_extended3_config_ld(kg, MVIN_SCALE_IDENTITY, false, 0);       // A [M][kg]
    gemmini_extended3_config_ld(N, MVIN_SCALE_IDENTITY, false, 1);        // B [kg][N]
    gemmini_config_st(N * sizeof(uint16_t));
  }
  for (int k = 0; k < g->nb; k += MM_RUN) {
    const int r = g->nb - k < MM_RUN ? g->nb - k : MM_RUN;
    const int acc = !(first && k == 0), st = last && k + r == g->nb;
    const int h = 1 + (int) (mxn_loops++ & 1);
    const uint8_t *asc = AS + (size_t) (g->kb + k) * M, *bsc = BS + (size_t) (g->kb + k) * N + c * NC;
    uint16_t *dst = st ? &C[0][0] + c * NC : NULL;
    if (g->fmt == 4)
      gemmini_loop_ws_mx(M / (2 * DIM), NC / (2 * DIM), r * GRP / DIM, g->A + k * GRP, g->B + (size_t) k * GRP * (N / 2) + c * NC / 2,
                         dst, kg, N / 2, N, asc, bsc, M, N, acc, h, h)
    else
      gemmini_loop_ws_mx_shared(M / DIM, NC / DIM, r * GRP / DIM, g->A + k * GRP, g->B + (size_t) k * GRP * N + c * NC,
                                dst, kg, N, N, asc, bsc, M, N, acc, h, h)
  }
}

static int check(const uint16_t *got, const uint8_t *exp_bytes) {
  const uint16_t *exp = (const uint16_t *) exp_bytes;
  int bad = 0;
  for (int i = 0; i < M * N; i++)
    if (got[i] != exp[i] && bad++ < 8) printf("C[%d][%d] got %04x exp %04x\n", i / N, i % N, got[i], exp[i]);
  printf("C: %d mismatches\n", bad);
  for (int r = 0; bad && r < M; r++) {   // mismatch map: one char per 8 columns, failing rows only
    char line[N / 8 + 1]; int any = 0;
    for (int g = 0; g < N / 8; g++) {
      int b = 0;
      for (int c = g * 8; c < g * 8 + 8; c++) b |= got[r * N + c] != exp[r * N + c];
      line[g] = b ? 'x' : '.'; any |= b;
    }
    line[N / 8] = 0;
    if (any) printf("  r%02d %s\n", r, line);
  }
  return bad;
}

int main() {
  gemmini_flush(0);
#ifdef MM_SFMT_A   // eXmY block scales (MX_FLEX_SCALES build)
  gemmini_mx_scale_fmt(MM_SFMT_A, MM_SBASE_A, MM_SFMT_W, MM_SBASE_W, MX_SFMT_E8M0, 0);
#endif
  mm_group_t gs[3];
  int ng = 0;
#if MM_NB4
  gs[ng++] = (mm_group_t) {4, MM_KB4, MM_NB4, OFF(A4), OFF(B4)};
#endif
#if MM_NB6
  gs[ng++] = (mm_group_t) {6, MM_KB6, MM_NB6, OFF(A6), OFF(B6)};
#endif
#if MM_NB8
  gs[ng++] = (mm_group_t) {8, MM_KB8, MM_NB8, OFF(A8), OFF(B8)};
#endif
  const int NC = pick_nc();
#ifdef MK_POWER
  if (!NC || M % (2 * DIM)) return 1;
  for (int it = 0; it < MK_PWR_ITERS; it++)
    for (int c = 0; c < N / NC; c++)
      for (int i = 0; i < ng; i++) issue_group(&gs[i], NC, c, i == 0, i == ng - 1);
  gemmini_fence();
  return 0;
#else
  if (!NC || M % (2 * DIM)) { printf("mxk_micromix: no tiling for M=%d N=%d\n", M, N); return 1; }
  uint64_t cyc = 0;
  memset(C, 0, sizeof(C));   // once: a CPU write inside the timed loop leaves C dirty in the cache (mvout waits)
  for (int it = 0; it < MM_ITERS; it++) {
    mxk_ctr_begin(it);
    const uint64_t t0 = read_cycles();
    for (int c = 0; c < N / NC; c++)
      for (int i = 0; i < ng; i++) {
        issue_group(&gs[i], NC, c, i == 0, i == ng - 1);
#ifndef MM_NOFENCE   // -DMM_NOFENCE: no CPU fence between groups (each group's config_ex waits for running loops)
        gemmini_fence();
#endif
      }
    gemmini_fence();
    cyc = read_cycles() - t0;
    mxk_ctr_end(it, cyc);
  }
  const int bad = check(&C[0][0], GOLD(DIM));
  // ideal: MXFP4 x MXFP4 at 4 MACs per PE per cycle, FP6 / FP8 at 1
  const uint64_t ideal = (uint64_t) M * N * GRP * (MM_NB6 + MM_NB8 + MM_NB4 / 4.0) / (DIM * DIM);
  const uint64_t ideal_fp8 = (uint64_t) M * K * N / (DIM * DIM);
  printf("mxk_micromix %dx%dx%d dim%d groups FP4 %d / FP6 %d / FP8 %d K-blocks, NC %d\n", M, K, N, DIM,
         MM_NB4, MM_NB6, MM_NB8, NC);
  printf("PERF cycles=%lu ideal=%lu ideal_fp8=%lu util=%lu.%lu%%\n", (unsigned long) cyc, (unsigned long) ideal,
         (unsigned long) ideal_fp8, (unsigned long) (ideal * 100 / cyc), (unsigned long) (ideal * 1000 / cyc % 10));
  printf(bad ? "mxk_micromix FAILED\n" : "mxk_micromix PASSED\n");
  return bad != 0;
#endif
}
#endif
