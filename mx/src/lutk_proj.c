// Weights-only per-K-block LUT linear layer (MXQuant "32xN usage, kblock tables") on the BF16 BOOM configs
// (MxE4M3LutBf16Dim{16,32}GemminiBoomConfig): C[M][N] = A[M][K] (E4M3 single) @ W[K][N], each 32-channel K-block of W
// either MXFP8 E4M3 (mode8) or low precision: LUT (4-bit index, E4M3 codebook per 2 columns per K-block, mode13) or
// plain MXFP4 (mode6, -DLK_LO_FP4). Data: gen/gen_lutk_proj.py.
//
// Phase lo: low-precision K-blocks; LUT: one loop per K-block per column chunk (its own codebooks); FP4: contiguous
// K-blocks merged into one loop. Phase hi: MXFP8 K-blocks, contiguous runs merged (<= LK_HI_RUN blocks per loop).
// DIM>=16 (shared accumulator layout): per column chunk, phase lo then phase hi accumulate into the same acc rows and
// phase hi's last loop stores C. -DLK_SPLIT (and DIM 8): phase lo -> P_lo, phase hi -> P_hi, combine C = P_lo + P_hi
// on the VPU (exact sum, one RNE to BF16; -DLK_CPU_ADD: on the CPU). A fence + LUT disable separates the phases (the
// weight-LUT enable is one Controller flag, not ordered with compute).
// -DLK_VAR=<i>: mask i of the data header; -DLK_VAR=-1: all MXFP8 (the baseline, straight to C).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_params.h"   // the config's params header: -iquote bf16/dim<16|32>
#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"
#include LK_DATA_H
#include "include/mx_native.h"

#ifndef LK_VAR
#define LK_VAR 0
#endif
#ifndef LK_HI_RUN
#define LK_HI_RUN 16   // MXFP8 K-blocks per loop at most (512 channels, as mxn_matmul)
#endif
#define LK_G 1                       // one codebook per 2 output columns
#ifndef LK_PADI
#define LK_PADI 0   // test: phase lo pads the last LK_PADI C rows away (all-low variants; those rows must stay untouched)
#endif
#define LK_SENT 0xDEAD
#if DIM >= 16
#define LK_LOOP_HI gemmini_loop_ws_mx_shared   // MXFP8 on the shared acc layout (same acc rows as phase lo)
#else
#define LK_LOOP_HI gemmini_loop_ws_mx
#endif
#ifndef LK_ITERS
#define LK_ITERS 2   // timed section repeats; the last is reported (the first warms the CPU caches)
#endif
#ifndef LK_LUT_KB1
#define LK_LUT_KB1 0   // 1: one K-block per LUT loop (pre-K2 behaviour)
#endif
#define LK_I ((LK_M - LK_PADI + DIM - 1) / DIM)        // phase lo row-tiles over the M - LK_PADI valid rows
#define LK_PI (LK_I * DIM - (LK_M - LK_PADI))           // and the last tile's pad
#define LK_LOOP_MX(I, J, Kt, A, B, C, As, Bs, Cs, Asc, Bsc, Ap, Bp, acc, ha, hb) { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(Asc), (uint64_t)(Bsc), k_LOOP_WS_CONFIG_SCALES) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(Ap), (uint64_t)(Bp), k_LOOP_WS_CONFIG_SCALE_STRIDES) \
    gemmini_loop_ws(I, J, Kt, LK_PI, 0, 0, A, B, NULL, C, As, Bs, 0, Cs, false, false, false, false, acc, \
                    NO_ACTIVATION, ha, hb, false) }
// LK_LOOP_MX with extra LOOP_WS rs2 flags (LK_LUT_KB_FLAG: one weight-codebook set per K-block of the loop)
#define LK_LUT_KB_FLAG (1ULL << 12)
#define LK_LOOP_MXF(I, J, Kt, A, B, C, As, Bs, Cs, Asc, Bsc, Ap, Bp, acc, ha, hb, flags) { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(Asc), (uint64_t)(Bsc), k_LOOP_WS_CONFIG_SCALES) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(Ap), (uint64_t)(Bp), k_LOOP_WS_CONFIG_SCALE_STRIDES) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(LK_PI), ((uint64_t)(Kt) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), \
                             k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, NULL, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, As, Bs, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, Cs, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(ha) << 18) | ((uint64_t)(hb) << 16) | \
                             ((uint64_t)(NO_ACTIVATION) << 8) | (acc), (flags), k_LOOP_WS) }
#ifdef LUT_W_CODEBOOKS               // weight codebooks per loop half (params header), 32 on older builds
#define LK_LUT_HALF (LUT_W_CODEBOOKS / 2)
#else
#define LK_LUT_HALF 32
#endif
#define LK_LUT_BYTES 16
#define GRP 32
#if DIM >= 16 && !defined(LK_SPLIT)
#define LK_SHARED 1
#else
#define LK_SHARED 0
#endif


#define CAT4_(a, b, c, d) a##b##c##d
#define CAT4(a, b, c, d) CAT4_(a, b, c, d)
#define OFF(p) (LK_BLOB + CAT4(LK_OFF_, p, , ))
#define GOFF(p, v) (LK_BLOB + CAT4(LK_OFF_, p, v, CAT4(_D, DIM, , )))
#define M LK_M
#define K LK_K
#define N LK_N

static uint16_t Plo[M][N] __attribute__((aligned(64)));
static uint16_t Phi[M][N] __attribute__((aligned(64)));
static uint16_t C[M][N] __attribute__((aligned(64)));

static int run_len(const uint8_t *m, int kb, int want, int cap) {
  int r = 0;
  while (kb + r < LK_NKB && m[kb + r] == want && r < cap) r++;
  return r;
}

// Phase hi (or the baseline with every block): MXFP8 runs of K-blocks where m[kb] == want, mode8.
static int nc_hi(void) {   // column chunk, 0 = no tiling
  const int NC = mxn_pick_nc(M, N);
  return (!NC || M * GRP * LK_HI_RUN / DIM + GRP * LK_HI_RUN * NC / DIM > MXN_HALF_ROWS) ? 0 : NC;
}
static void hi_config(void) {
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, /*BF16*/ 3, 0);
  gemmini_extended3_config_ld(K, MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(N, MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(N * sizeof(uint16_t));
}
// one column chunk c of NC columns; acc0: the first loop accumulates (onto phase lo); its last loop stores to dst
static void phase_hi_chunk(const uint8_t *m, int want, uint16_t *dst, int NC, int c, int acc0) {
  const uint8_t *A = OFF(A), *AS = OFF(AS), *B = OFF(B8), *BS = OFF(BS);
  int nruns = 0;
  for (int kb = 0; kb < LK_NKB;) { const int r = run_len(m, kb, want, LK_HI_RUN); if (r) { nruns++; kb += r; } else kb++; }
  int p = 0;
  for (int kb = 0; kb < LK_NKB;) {
    const int r = run_len(m, kb, want, LK_HI_RUN);
    if (!r) { kb++; continue; }
    const int h = 1 + (int) (mxn_loops++ & 1);
    LK_LOOP_HI(M / DIM, NC / DIM, r * GRP / DIM,
                       A + kb * GRP, B + (size_t) kb * GRP * N + c * NC,
                       p == nruns - 1 ? dst + c * NC : NULL, K, N, N,
                       AS + kb * M, BS + kb * N + c * NC, M, N, /*accumulate*/ p > 0 || acc0, h, h);
    p++; kb += r;
  }
}
static int phase_hi(const uint8_t *m, int want, uint16_t *dst) {
  const int NC = nc_hi();
  if (!NC) { printf("phase hi: no tiling for M=%d N=%d\n", M, N); return 1; }
  hi_config();
  for (int c = 0; c < N / NC; c++) phase_hi_chunk(m, want, dst, NC, c, 0);
  return 0;
}

// Phase lo: the low-precision K-blocks (m[kb] == 1).
static int nc_lo(void) {   // column chunk, 0 = no tiling
  int NC = 0;
#ifdef LK_LO_FP4
  for (int nc = 512; nc >= 2 * DIM && !NC; nc >>= 1)
    if (N % nc == 0 && M * nc / (2 * DIM) <= ACC_ROWS / 2) NC = nc;
  if (NC && M * GRP * LK_HI_RUN / DIM + GRP * LK_HI_RUN * NC / (2 * DIM) > MXN_HALF_ROWS) NC = 0;
#else   // the widest chunk whose codebooks fit a loop half and whose C fits an acc half
  for (int nc = LK_LUT_HALF << LK_G; nc >= 2 * DIM && !NC; nc >>= 1)
    if (N % nc == 0 && M * nc / (2 * DIM) <= ACC_ROWS / 2) NC = nc;
#endif
  return NC;
}
static void lo_config_g(int g) {   // g: also (re)set the codebook granularity (a non-managed CONFIG_SCALE_MEM: drains)
#ifdef LK_LO_FP4
  (void) g;
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, /*FP4*/ 2, 3, 0);
#else
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, /*uselut*/ 1);
  if (g) gemmini_mxquant_config_mvout(0, 1, 1, 1, 0, 0, LK_G);
#endif
  gemmini_extended3_config_ld(K, MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(N / 2, MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(N * sizeof(uint16_t));
}
static void lo_config(void) { lo_config_g(1); }
// LUT and FP8 phases differ only in their B load stride (config_ex differs only in the unused uselut, config_st not
// at all), so a fence-free LUT kernel re-issues just the config_ld (which passes running loops)
static void lo_ld(void) { gemmini_extended3_config_ld(K, MVIN_SCALE_IDENTITY, false, 0); gemmini_extended3_config_ld(N / 2, MVIN_SCALE_IDENTITY, false, 1); }
static void hi_ld(void) { gemmini_extended3_config_ld(K, MVIN_SCALE_IDENTITY, false, 0); gemmini_extended3_config_ld(N, MVIN_SCALE_IDENTITY, false, 1); }
// one column chunk c of NC columns; its last loop stores to dst (NULL: leave the partial in the accumulator)
static void phase_lo_chunk(const uint8_t *m, uint16_t *dst, int NC, int c) {
  const uint8_t *A = OFF(A), *AS = OFF(AS), *BS = OFF(BS);
#ifdef LK_LO_FP4
  const uint8_t *B = OFF(B4);
  int nruns = 0;
  for (int kb = 0; kb < LK_NKB;) { const int r = run_len(m, kb, 1, LK_HI_RUN); if (r) { nruns++; kb += r; } else kb++; }
  int p = 0;
  for (int kb = 0; kb < LK_NKB;) {
    const int r = run_len(m, kb, 1, LK_HI_RUN);
    if (!r) { kb++; continue; }
    const int h = 1 + (int) (mxn_loops++ & 1);
    LK_LOOP_MX(LK_I, NC / (2 * DIM), r * GRP / DIM,
               A + kb * GRP, B + (size_t) kb * GRP * (N / 2) + c * NC / 2,
               dst && p == nruns - 1 ? dst + c * NC : NULL, K, N / 2, N,
               AS + kb * M, BS + kb * N + c * NC, M, N, p > 0, h, h);
    p++; kb += r;
  }
#else
  const uint8_t *B = OFF(BL), *LUT = OFF(BLUT);
  // K-blocks per loop: as many codebook sets as fit a loop half; > 1 needs them contiguous (chunk = full N)
  int kbs = (NC == N && !LK_LUT_KB1) ? LK_LUT_HALF / (NC >> LK_G) : 1;
#ifdef LK_KB_UNIFORM   // only with one loop length (a bounds change between loops forces an execute drain)
  for (int kb = 0; kbs > 1 && kb < LK_NKB;) {
    const int r = run_len(m, kb, 1, LK_NKB);
    if (r % kbs) kbs = 1;
    kb += r ? r : 1;
  }
#endif
  int nruns = 0;
  for (int kb = 0; kb < LK_NKB;) { const int r = run_len(m, kb, 1, kbs); if (r) { nruns++; kb += r; } else kb++; }
  int p = 0;
  for (int kb = 0; kb < LK_NKB;) {
    const int r = run_len(m, kb, 1, kbs);
    if (!r) { kb++; continue; }
    const int h = 1 + (int) (mxn_loops++ & 1);
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, (uint64_t) (LUT + ((size_t) kb * (N / 2) + ((c * NC) >> LK_G)) * LK_LUT_BYTES),
                             k_LOOP_WS_CONFIG_LUTS)
    LK_LOOP_MXF(LK_I, NC / (2 * DIM), r * GRP / DIM,
                A + kb * GRP, B + (size_t) kb * GRP * (N / 2) + c * NC / 2,
                dst && p == nruns - 1 ? dst + c * NC : NULL, K, N / 2, N,
                AS + kb * M, BS + kb * N + c * NC, M, N, p > 0, h, h, kbs > 1 ? LK_LUT_KB_FLAG : 0);
    p++; kb += r;
  }
#endif
}
#ifdef LK_NOFENCE
// Shared fence-free path: each phase's runs planned once before the timed loop (run scan + addresses), so the timed
// loop only issues commands (the CPU, not Gemmini, was the limit: ~1k cycles of planning per phase, ~75/loop)
typedef struct { uint64_t lut, a, b, c, as, bs; uint32_t kt, flags; uint8_t acc; } lk_run_t;
static lk_run_t lk_plan_lo[LK_NKB], lk_plan_hi[LK_NKB];
static int lk_n_lo, lk_n_hi;
static void plan_lo(const uint8_t *m, int NC, int c) {
  const uint8_t *A = OFF(A), *AS = OFF(AS), *BS = OFF(BS);
#ifdef LK_LO_FP4
  const uint8_t *B = OFF(B4);
  const int cap = LK_HI_RUN;
#else
  const uint8_t *B = OFF(BL), *LUT = OFF(BLUT);
  const int cap = (NC == N && !LK_LUT_KB1) ? LK_LUT_HALF / (NC >> LK_G) : 1;
#endif
  int n = 0;
  for (int kb = 0; kb < LK_NKB;) {
    const int r = run_len(m, kb, 1, cap);
    if (!r) { kb++; continue; }
    lk_run_t *e = &lk_plan_lo[n];
    e->kt = r * GRP / DIM; e->a = (uint64_t) (A + kb * GRP); e->c = 0; e->acc = n > 0;
    e->b = (uint64_t) (B + (size_t) kb * GRP * (N / 2) + c * NC / 2);
    e->as = (uint64_t) (AS + kb * M); e->bs = (uint64_t) (BS + kb * N + c * NC);
#ifdef LK_LO_FP4
    e->lut = 0; e->flags = 0;
#else
    e->lut = (uint64_t) (LUT + ((size_t) kb * (N / 2) + ((c * NC) >> LK_G)) * LK_LUT_BYTES);
    e->flags = cap > 1 ? LK_LUT_KB_FLAG : 0;
#endif
    n++; kb += r;
  }
  lk_n_lo = n;
}
static void plan_hi(const uint8_t *m, uint16_t *dst, int NC, int c) {   // accumulates onto phase lo; last loop stores
  const uint8_t *A = OFF(A), *AS = OFF(AS), *B = OFF(B8), *BS = OFF(BS);
  int n = 0;
  for (int kb = 0; kb < LK_NKB;) {
    const int r = run_len(m, kb, 0, LK_HI_RUN);
    if (!r) { kb++; continue; }
    lk_run_t *e = &lk_plan_hi[n];
    e->kt = r * GRP / DIM; e->a = (uint64_t) (A + kb * GRP); e->c = 0; e->acc = 1;
    e->b = (uint64_t) (B + (size_t) kb * GRP * N + c * NC);
    e->as = (uint64_t) (AS + kb * M); e->bs = (uint64_t) (BS + kb * N + c * NC);
    n++; kb += r;
  }
  if (n) lk_plan_hi[n - 1].c = (uint64_t) (dst + c * NC);
  lk_n_hi = n;
}
static void issue_lo(int NC) {
  for (int i = 0; i < lk_n_lo; i++) {
    const lk_run_t *e = &lk_plan_lo[i];
    const int h = 1 + (int) (mxn_loops++ & 1);
#ifdef LK_LO_FP4
    LK_LOOP_MX(LK_I, NC / (2 * DIM), e->kt, e->a, e->b, e->c, K, N / 2, N, e->as, e->bs, M, N, e->acc, h, h);
#else
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, e->lut, k_LOOP_WS_CONFIG_LUTS)
    LK_LOOP_MXF(LK_I, NC / (2 * DIM), e->kt, e->a, e->b, e->c, K, N / 2, N, e->as, e->bs, M, N, e->acc, h, h, e->flags);
#endif
  }
}
static void issue_hi(int NC) {
  for (int i = 0; i < lk_n_hi; i++) {
    const lk_run_t *e = &lk_plan_hi[i];
    const int h = 1 + (int) (mxn_loops++ & 1);
    LK_LOOP_HI(M / DIM, NC / DIM, e->kt, e->a, e->b, e->c, K, N, N, e->as, e->bs, M, N, e->acc, h, h);
  }
}
#endif
#ifdef LK_MIX
// Mixed-K loops (M2a): one LOOP_WS spans consecutive K-blocks of either kind; its kmask picks low precision (LUT/FP4)
// or MXFP8 per K-block. At most LK_HI_RUN K-blocks (the MXFP8 B footprint per K-block fits a spad half) and, for LUT,
// at most one loop half of codebook sets (no streaming yet). LUT codebooks are kept in LUT-ordinal order (static).
#ifndef LK_LO_FP4
static uint8_t lk_cbc[LK_NKB * (LK_N / 2) * LK_LUT_BYTES] __attribute__((aligned(64)));
#endif
typedef struct { uint64_t a, b, b8, as, bs, lut, mlo, mhi; uint32_t kt; uint8_t acc, last; } lk_mix_t;
static lk_mix_t lk_mix[LK_NKB];
static int lk_n_mix;
static int plan_mix(const uint8_t *m, uint16_t *dst, int NC) {
  if (NC != N) { printf("mix: needs one column chunk\n"); return 1; }
  const uint8_t *A = OFF(A), *AS = OFF(AS), *BS = OFF(BS), *B8 = OFF(B8);
#ifdef LK_LO_FP4
  const uint8_t *B = OFF(B4);
  const int lcap = LK_NKB;
#else
  const uint8_t *B = OFF(BL), *LUT = OFF(BLUT);
  const int lcap = LK_LUT_HALF / (NC >> LK_G);
  int ord = 0;
  for (int kb = 0; kb < LK_NKB; kb++)
    if (m[kb]) memcpy(lk_cbc + (size_t) ord++ * (N / 2) * LK_LUT_BYTES, LUT + (size_t) kb * (N / 2) * LK_LUT_BYTES,
                      (N / 2) * LK_LUT_BYTES);
  ord = 0;
#endif
  int n = 0;
  for (int kb = 0; kb < LK_NKB;) {
    int r = 0, nl = 0;
    while (kb + r < LK_NKB && r < LK_HI_RUN && r < 128 && (!m[kb + r] || nl < lcap)) nl += m[kb + r++];
    lk_mix_t *e = &lk_mix[n];
    e->mlo = e->mhi = 0;
    for (int i = 0; i < r; i++) if (m[kb + i]) { if (i < 64) e->mlo |= 1ULL << i; else e->mhi |= 1ULL << (i - 64); }
    e->kt = r * GRP / DIM; e->acc = n > 0; e->last = 0;
    e->a = (uint64_t) (A + kb * GRP);
    e->b = (uint64_t) (B + (size_t) kb * GRP * (N / 2));
    e->b8 = (uint64_t) (B8 + (size_t) kb * GRP * N);
    e->as = (uint64_t) (AS + kb * M); e->bs = (uint64_t) (BS + kb * N);
#ifdef LK_LO_FP4
    e->lut = 0;
#else
    e->lut = (uint64_t) (lk_cbc + (size_t) ord * (N / 2) * LK_LUT_BYTES); ord += nl;
#endif
    n++; kb += r;
  }
  lk_mix[n - 1].last = 1;
  lk_n_mix = n;
  (void) dst;
  return 0;
}
static void mix_config(void) {
  lo_config_g(1);
  gemmini_extended3_config_ld(N, MVIN_SCALE_IDENTITY, false, 2);   // MXFP8 B K-blocks load with state 2 (row = N)
}
static void issue_mix(uint16_t *dst, int NC) {
  for (int i = 0; i < lk_n_mix; i++) {
    const lk_mix_t *e = &lk_mix[i];
    const int h = 1 + (int) (mxn_loops++ & 1);
    const uint64_t c = e->last ? (uint64_t) dst : 0;
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, e->b8, (uint64_t) N, k_LOOP_WS_CONFIG_KMIX)
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, e->mlo, e->mhi, k_LOOP_WS_CONFIG_KMASK)
#ifdef LK_LO_FP4
    LK_LOOP_MX(LK_I, NC / (2 * DIM), e->kt, e->a, e->b, c, K, N / 2, N, e->as, e->bs, M, N, e->acc, h, h);
#else
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, e->lut, k_LOOP_WS_CONFIG_LUTS)
    LK_LOOP_MXF(LK_I, NC / (2 * DIM), e->kt, e->a, e->b, c, K, N / 2, N, e->as, e->bs, M, N, e->acc, h, h, 0);
#endif
  }
}
#endif
static int phase_lo(const uint8_t *m, uint16_t *dst) {   // dst: P_lo, or C when every K-block is low precision
  const int NC = nc_lo();
  if (!NC) { printf("phase lo: no tiling\n"); return 1; }
  lo_config();
  for (int c = 0; c < N / NC; c++) phase_lo_chunk(m, dst, NC, c);
  return 0;
}

#ifdef LK_CPU_ADD
static void combine(void) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) C[i][j] = vpu_add(Plo[i][j], Phi[i][j]);
}
#else
// two streams (one per VPU unit): rows [s*half, (s+1)*half) of P_lo in bank 2s, of P_hi in bank 2s+1
#define SPR(b, r) ((uint32_t) ((b) * BANK_ROWS + (r)))
static void mv_rows(int out, void *dram, uint32_t sp, int rows) {
  for (int r = 0; r < rows; r += DIM) {
    if (out) { gemmini_extended_mvout((uint8_t *) dram + r * DIM, sp + r, DIM, DIM); }
    else { gemmini_extended_mvin((uint8_t *) dram + r * DIM, sp + r, DIM, DIM); }
  }
}
static void combine(void) {
  const int half = M * N * 2 / DIM / 2;
  gemmini_config_ld(DIM);
  gemmini_config_st(DIM);
  for (int s = 0; s < 2; s++) {
    mv_rows(0, (uint8_t *) Plo + s * half * DIM, SPR(2 * s, 0), half);
    mv_rows(0, (uint8_t *) Phi + s * half * DIM, SPR(2 * s + 1, 0), half);
  }
  for (int s = 0; s < 2; s++) gemmini_vpu_binary(VPU_ADD, SPR(2 * s, 0), SPR(2 * s, 0), SPR(2 * s + 1, 0), half);
  for (int s = 0; s < 2; s++) mv_rows(1, (uint8_t *) C + s * half * DIM, SPR(2 * s, 0), half);
  gemmini_fence();
}
#endif

static int check(const char *name, const uint16_t *got, const uint8_t *exp_bytes) {
  const uint16_t *exp = (const uint16_t *) exp_bytes;
  int bad = 0;
  for (int i = 0; i < M * N; i++) {
    const uint16_t e = (LK_PADI && got == &C[0][0] && i / N >= M - LK_PADI) ? LK_SENT : exp[i];
    if (got[i] != e && bad++ < 8) printf("%s[%d][%d] got %04x exp %04x\n", name, i / N, i % N, got[i], e);
  }
  printf("%s: %d mismatches\n", name, bad);
  for (int r = 0; bad && r < M; r++) {   // mismatch map: one char per 8 columns, failing rows only
    char line[N / 8 + 1]; int any = 0;
    for (int g = 0; g < N / 8; g++) {
      int b = 0;
      for (int c = g * 8; c < g * 8 + 8; c++) {
        const int i = r * N + c;
        const uint16_t e = (LK_PADI && got == &C[0][0] && r >= M - LK_PADI) ? LK_SENT : exp[i];
        b |= got[i] != e;
      }
      line[g] = b ? 'x' : '.'; any |= b;
    }
    line[N / 8] = 0;
    if (any) printf("  %s r%02d %s\n", name, r, line);
  }
  return bad;
}

int main() {
  gemmini_flush(0);
  const uint64_t ideal_fp8 = (uint64_t) M * K * N / (DIM * DIM);
#if LK_VAR < 0
  static const uint8_t none[LK_NKB];
  uint64_t cyc = 0;
  for (int it = 0; it < LK_ITERS; it++) {
    const uint64_t t0 = read_cycles();
    if (phase_hi(none, 0, &C[0][0])) return 1;
    gemmini_fence();
    cyc = read_cycles() - t0;
  }
  const int bad = check("C", &C[0][0], OFF(CAT4(FP8_D, DIM, , )));
  printf("lutk %dx%dx%d dim%d variant=fp8 (all MXFP8, %d K-blocks)\n", M, K, N, DIM, LK_NKB);
  printf("PERF cycles=%lu ideal=%lu util=%lu.%lu%%\n", (unsigned long) cyc, (unsigned long) ideal_fp8,
         (unsigned long) (ideal_fp8 * 100 / cyc), (unsigned long) (ideal_fp8 * 1000 / cyc % 10));
#else
  const uint8_t *m = lk_mask[LK_VAR];
  int nlo = 0;
  for (int kb = 0; kb < LK_NKB; kb++) nlo += m[kb];
  const int nhi = LK_NKB - nlo;
  memset(Plo, 0, sizeof(Plo));
  memset(Phi, 0, sizeof(Phi));
  if (LK_PADI && nhi) { printf("LK_PADI: all-low variants only\n"); return 1; }
  for (int i = 0; LK_PADI && i < M * N; i++) (&C[0][0])[i] = LK_SENT;
  const int shared = LK_SHARED && nlo && nhi;
  uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
#ifdef LK_MIX
  if (!shared) { printf("mix: needs a mixed mask (LK_SHARED)\n"); return 1; }
  int lk_mix_nc;
  { const int nl = nc_lo(), nh = nc_hi(); lk_mix_nc = nl < nh ? nl : nh; }
  if (!lk_mix_nc || plan_mix(m, &C[0][0], lk_mix_nc)) return 1;
  printf("mix: %d loops for %d K-blocks\n", lk_n_mix, LK_NKB);
#endif
#ifdef LK_NOFENCE
  int lk_nc = 0;
  if (shared) {
    const int nl = nc_lo(), nh = nc_hi();
    lk_nc = nl < nh ? nl : nh;
    if (lk_nc && N / lk_nc == 1) { plan_lo(m, lk_nc, 0); plan_hi(m, &C[0][0], lk_nc, 0); }
  }
#endif
  for (int it = 0; it < LK_ITERS; it++) {
  t0 = read_cycles();
#ifdef LK_MIX   // mixed-K loops in K order into one accumulator (planned before the timed loop); the last stores C
  mix_config();
  issue_mix(&C[0][0], lk_mix_nc);
  gemmini_fence();
  t1 = t2 = t3 = read_cycles();
  if (0)
#endif
  if (shared) {   // per chunk: phase lo then phase hi into one accumulator, phase hi stores C
#ifdef LK_NOFENCE
    const int NC = lk_nc;
    if (!NC) { printf("shared: no tiling\n"); return 1; }
#else
    const int nl = nc_lo(), nh = nc_hi(), NC = nl < nh ? nl : nh;
    if (!NC) { printf("shared: no tiling (lo %d hi %d)\n", nl, nh); return 1; }
#endif
    uint64_t tlo = 0, thi = 0;
#ifdef LK_NOFENCE   // no fence / LUT disable between phases or chunks: each command carries its own mode
#ifndef LK_LO_FP4
    gemmini_mxquant_config_mvout(0, 1, 1, 1, 0, 0, LK_G);
#endif
    for (int c = 0; c < N / NC; c++) {
      if (N / NC > 1) { plan_lo(m, NC, c); plan_hi(m, &C[0][0], NC, c); }
#ifdef LK_LO_FP4
      lo_config_g(0);
      issue_lo(NC);
      hi_config();
#else
      if (c == 0 && it == 0) lo_config_g(0); else lo_ld();
      issue_lo(NC);
      hi_ld();
#endif
      issue_hi(NC);
    }
    gemmini_fence();
    tlo = read_cycles() - t0;   // lo/hi split not measurable without fences: all in lo
#else
    for (int c = 0; c < N / NC; c++) {
      uint64_t ta = read_cycles();
      lo_config();
      phase_lo_chunk(m, NULL, NC, c);
      gemmini_fence();
      gemmini_mx_lut_disable();
      uint64_t tb = read_cycles();
      hi_config();
      phase_hi_chunk(m, 0, &C[0][0], NC, c, 1);
      gemmini_fence();
      uint64_t tc = read_cycles();
      tlo += tb - ta; thi += tc - tb;
    }
#endif
    t1 = t0 + tlo; t2 = t1 + thi; t3 = t2;
  } else {
#ifndef LK_SKIP_LO
    if (nlo && phase_lo(m, nhi ? &Plo[0][0] : &C[0][0])) return 1;
#endif   // -DLK_SKIP_LO: debug, phase hi alone (only P_hi is meaningful)
    gemmini_fence();
    t1 = read_cycles();
    gemmini_mx_lut_disable();
    if (nhi && phase_hi(m, 0, &Phi[0][0])) return 1;
    gemmini_fence();
    t2 = read_cycles();
    if (nhi) combine();
    t3 = read_cycles();
  }
  }
#ifdef LK_LO_FP4
  const char *lo_name = "fp4";
#ifdef LK_MIX
  int bad = check("C", &C[0][0], GOFF(CM4, LK_VAR));
#else
  int bad = shared ? check("C", &C[0][0], GOFF(CS4, LK_VAR)) :
    check("P_lo", nhi ? &Plo[0][0] : &C[0][0], GOFF(PFP4, LK_VAR)) + check("C", &C[0][0], GOFF(CFP4, LK_VAR));
#endif
#else
  const char *lo_name = "lut";
#ifdef LK_MIX
  int bad = check("C", &C[0][0], GOFF(CML, LK_VAR));
#else
  int bad = shared ? check("C", &C[0][0], GOFF(CSL, LK_VAR)) :
    check("P_lo", nhi ? &Plo[0][0] : &C[0][0], GOFF(PLUT, LK_VAR)) + check("C", &C[0][0], GOFF(CLUT, LK_VAR));
#endif
#endif
#ifndef LK_MIX
  if (!shared) bad += check("P_hi", &Phi[0][0], GOFF(PHI, LK_VAR));
#endif
  // ideal: low-precision K-blocks at 2 MACs per PE per cycle, MXFP8 at 1
  const uint64_t ideal = (uint64_t) M * N * (nhi * GRP + nlo * GRP / 2) / (DIM * DIM);
  const uint64_t mm = t2 - t0, tot = t3 - t0;
  printf("lutk %dx%dx%d dim%d variant=%s lo=%s: %d/%d K-blocks low precision\n", M, K, N, DIM, lk_var_name[LK_VAR],
         lo_name, nlo, LK_NKB);
  printf("PERF cycles=%lu lo=%lu hi=%lu add=%lu ideal=%lu ideal_fp8=%lu util_mm=%lu.%lu%% util=%lu.%lu%%\n",
         (unsigned long) tot, (unsigned long) (t1 - t0), (unsigned long) (t2 - t1), (unsigned long) (t3 - t2),
         (unsigned long) ideal, (unsigned long) ideal_fp8,
         (unsigned long) (ideal * 100 / mm), (unsigned long) (ideal * 1000 / mm % 10),
         (unsigned long) (ideal * 100 / tot), (unsigned long) (ideal * 1000 / tot % 10));
#endif
  printf(bad ? "lutk_proj FAILED\n" : "lutk_proj PASSED\n");
  return bad != 0;
}
