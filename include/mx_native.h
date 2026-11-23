// Native MX matmul on MxGemmini: one call = a sequence of native DRAM loops (gemmini_loop_ws_mx).
//
//   C[M][N] (BF16, row pitch ldc) = A[M][K] (fp8 E4M3, pitch lda) @ B[K][N] (fp8 E4M3, pitch ldb)
//   A_sc = [K/32][M] E8M0 rows (pitch sc_lda), B_sc = [K/32][N] E8M0 rows (pitch sc_ldb)
//
// Loop structure: N-chunks of NC columns (one accumulator half) x K-tiles of Kt (accumulate; C only on
// the last K-tile). Every loop keeps its A and B in ONE scratchpad half and consecutive loops alternate
// halves (mxn_loops is global, so this also holds across calls): the next loop's operands and its
// loop-managed scale slices stream in under the current loop's compute, the chunk's C store overlaps the
// next chunk (acc halves alternate), and the cross-loop WAR hold never triggers. Cost: A is re-read per
// loop (+M/NC of the weight traffic) and each loop pays its scale-config drain (~35 cycles).
//
// The caller defines DIM (16) and pins BANK_ROWS before including this.
#ifndef INCLUDE_MX_NATIVE_H
#define INCLUDE_MX_NATIVE_H

#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#ifndef DIM
#error "mx_native.h needs DIM defined by the caller"
#endif
#define MXN_HALF_ROWS ((BANK_NUM * BANK_ROWS) / 2)

// Tile choice: the largest N-chunk NC (<= 512) whose C fits one acc half (E4M3-single: M*NC/64 rows) --
// larger NC means less A re-read -- then the largest K-tile Kt (<= 512) whose A (M*Kt/16) + B (Kt*NC/16)
// fit one spad half and whose scale slices fit 4 KB. At M = 32 this is NC 512, Kt 128.
static int mxn_pick_nc(int M, int N) {
  for (int nc = 512; nc >= DIM; nc >>= 1)
    if (N % nc == 0 && M * nc / (DIM * 4) <= ACC_ROWS / 2) return nc;
  return 0;
}
static int mxn_pick_kt(int M, int K, int NC) {
  for (int kt = 512; kt >= 32; kt >>= 1)
    if (K % kt == 0 && M * kt / DIM + kt * NC / DIM <= MXN_HALF_ROWS && (kt / 32) * NC <= 4096 &&
        (kt / 32) * M <= 4096) return kt;
  return 0;
}

static unsigned mxn_loops;   // loops issued so far: the parity picks the scratchpad half

// blk_nc/blk_kt != 0: B is PRE-BLOCKED for that tiling -- block (c, t) = B[t*Kt:+Kt][c*NC:+NC] stored
// contiguously (row pitch NC), blocks c-major -- so each loop streams one contiguous run from DRAM instead
// of Kt rows ldb apart. B_sc stays [K/32][N] (pitch sc_ldb). The chosen tiling must equal (blk_nc, blk_kt).
// a_kt != 0: A is K-BLOCKED -- K-tile t = A + t*M*a_kt, a contiguous [M][a_kt] (lda = a_kt) -- and the K-tile is a_kt.
static int mxn_matmul_core(const uint8_t *A, int lda, const uint8_t *B, int ldb, uint16_t *C, int ldc,
                           const uint8_t *A_sc, int sc_lda, const uint8_t *B_sc, int sc_ldb,
                           int M, int K, int N, int blk_nc, int blk_kt, int a_kt) {
  const int NC = mxn_pick_nc(M, N);
  int Kt = NC ? mxn_pick_kt(M, K, NC) : 0;
  if (a_kt) {
    if (K % a_kt || M * a_kt / DIM + a_kt * NC / DIM > MXN_HALF_ROWS) Kt = 0;
    else Kt = a_kt;
    lda = a_kt;
  }
  const int blocked = blk_nc != 0;
  if (blocked && (NC != blk_nc || Kt != blk_kt)) {
    printf("mxn_matmul: B blocked for NC=%d Kt=%d but tiling is NC=%d Kt=%d\n", blk_nc, blk_kt, NC, Kt);
    return 1;
  }
  if (!NC || !Kt || M % DIM ||
      (((uintptr_t) A_sc | (uintptr_t) B_sc | (uintptr_t) sc_lda | (uintptr_t) sc_ldb) & 7)) {
    printf("mxn_matmul: unsupported shape/alignment M=%d K=%d N=%d (Kt=%d NC=%d)\n", M, K, N, Kt, NC);
    return 1;
  }
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0,
                              /*BF16*/ 3, 0);
  gemmini_extended3_config_ld(lda * sizeof(uint8_t), MVIN_SCALE_IDENTITY, false, 0);
  const int ldb_loop = blocked ? NC : ldb;
  gemmini_extended3_config_ld(ldb_loop * sizeof(uint8_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(ldc * sizeof(uint16_t));
  const int kts = K / Kt;
  for (int c = 0; c < N / NC; c++)
    for (int t = 0; t < kts; t++) {
      const int h = 1 + (int) (mxn_loops++ & 1);
      gemmini_loop_ws_mx(M / DIM, NC / DIM, Kt / DIM,
                         A + (size_t) t * Kt * (a_kt ? M : 1),
                         blocked ? B + ((size_t) c * kts + t) * Kt * NC : B + (size_t) t * Kt * ldb + (size_t) c * NC,
                         t == kts - 1 ? C + (size_t) c * NC : NULL,
                         lda, ldb_loop, ldc,
                         A_sc + (size_t) (t * Kt / 32) * sc_lda,
                         B_sc + (size_t) (t * Kt / 32) * sc_ldb + (size_t) c * NC, sc_lda, sc_ldb,
                         /*accumulate*/ t > 0, h, h);
    }
  return 0;
}

static int mxn_matmul_ex(const uint8_t *A, int lda, const uint8_t *B, int ldb, uint16_t *C, int ldc,
                         const uint8_t *A_sc, int sc_lda, const uint8_t *B_sc, int sc_ldb,
                         int M, int K, int N, int blk_nc, int blk_kt) {
  return mxn_matmul_core(A, lda, B, ldb, C, ldc, A_sc, sc_lda, B_sc, sc_ldb, M, K, N, blk_nc, blk_kt, 0);
}

static int mxn_matmul(const uint8_t *A, int lda, const uint8_t *B, int ldb, uint16_t *C, int ldc,
                      const uint8_t *A_sc, int sc_lda, const uint8_t *B_sc, int sc_ldb, int M, int K, int N) {
  return mxn_matmul_ex(A, lda, B, ldb, C, ldc, A_sc, sc_lda, B_sc, sc_ldb, M, K, N, 0, 0);
}

// FP4 (E2M1) x FP4 on the quad mesh (32 x 32 output tile per pass): A [M/2][K] bytes, row pair (2r, 2r+1) of
// column k in byte (r, k) (low nibble = row 2r), pitch lda = K bytes; B [K][N/2] packed nibbles (low = even
// column), pitch ldb bytes; scales and C as mxn_matmul. Same N-chunk / K-tile / half alternation; operands take
// half the spad rows of E4M3, the accumulator the same. config_st is two C rows (an acc row holds two).
static int mxn_pick_kt_fp4(int M, int K, int NC) {
  for (int kt = 512; kt >= 32; kt >>= 1)
    if (K % kt == 0 && M * kt / (2 * DIM) + kt * NC / (2 * DIM) <= MXN_HALF_ROWS && (kt / 32) * NC <= 4096 &&
        (kt / 32) * M <= 4096) return kt;
  return 0;
}

static int mxn_matmul_fp4(const uint8_t *A, int lda, const uint8_t *B, int ldb, uint16_t *C, int ldc,
                          const uint8_t *A_sc, int sc_lda, const uint8_t *B_sc, int sc_ldb, int M, int K, int N) {
  int NC = 0;
  for (int nc = 512; nc >= 2 * DIM; nc >>= 1)
    if (N % nc == 0 && M * nc / (DIM * 4) <= ACC_ROWS / 2) { NC = nc; break; }
  const int Kt = NC ? mxn_pick_kt_fp4(M, K, NC) : 0;
  if (!NC || !Kt || M % (2 * DIM) ||
      (((uintptr_t) A_sc | (uintptr_t) B_sc | (uintptr_t) sc_lda | (uintptr_t) sc_ldb) & 7)) {
    printf("mxn_matmul_fp4: unsupported shape/alignment M=%d K=%d N=%d (Kt=%d NC=%d)\n", M, K, N, Kt, NC);
    return 1;
  }
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, /*FP4*/ 2, 2,
                              /*BF16*/ 3, 0);
  gemmini_extended3_config_ld(lda, MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(ldb, MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(2 * ldc * sizeof(uint16_t));
  const int kts = K / Kt;
  for (int c = 0; c < N / NC; c++)
    for (int t = 0; t < kts; t++) {
      const int h = 1 + (int) (mxn_loops++ & 1);
      gemmini_loop_ws_mx(M / (2 * DIM), NC / (2 * DIM), Kt / DIM,
                         A + (size_t) t * Kt, B + (size_t) t * Kt * ldb + (size_t) c * NC / 2,
                         t == kts - 1 ? C + (size_t) c * NC : NULL,
                         lda, ldb, ldc,
                         A_sc + (size_t) (t * Kt / 32) * sc_lda,
                         B_sc + (size_t) (t * Kt / 32) * sc_ldb + (size_t) c * NC, sc_lda, sc_ldb,
                         /*accumulate*/ t > 0, h, h);
    }
  return 0;
}

// LUT x LUT (FP6 E3M2/E2M3, FP8 E4M3-quad/E5M2) on the quad mesh: operands are 4-bit codebook indices packed like FP4
// (A [M/2][K] row pairs, B [K][N/2]); scales and C as mxn_matmul_fp4. The caller has set config_ex (format + LUT), the
// LUT granularity G (CONFIG_SCALE_MEM) and loaded the act codebooks flat (MX_LOAD_LUT sel 1). Weight codebooks are
// loop-managed: N-chunk c's loops load B_lut + ((c*NC) >> G) codebooks of lut_bytes (12 FP6, 16 FP8) into their
// slot's weight-LUT half (32 codebooks), so NC <= 32 << G.
// _ex: A_lut != NULL makes the act-in codebooks loop-managed too (each loop loads its M >> G <= 32 into its slot's act
// half), so the caller need not load them flat and back-to-back matmuls can use different codebooks with no fence;
// set_st = 0 skips config_st (it waits for every loop to finish), the caller having set the store stride (2 * ldc * 2).
static int mxn_matmul_lut_ex(const uint8_t *A, int lda, const uint8_t *B, int ldb, uint16_t *C, int ldc,
                             const uint8_t *A_sc, int sc_lda, const uint8_t *B_sc, int sc_ldb,
                             const uint8_t *B_lut, int lut_bytes, int G, int M, int K, int N,
                             const uint8_t *A_lut, int set_st) {
  int NC = 0;
  for (int nc = 512; nc >= 2 * DIM; nc >>= 1)
    if (N % nc == 0 && M * nc / (DIM * 4) <= ACC_ROWS / 2 && (nc >> G) <= 32) { NC = nc; break; }
  const int Kt = NC ? mxn_pick_kt_fp4(M, K, NC) : 0;
  if (!NC || !Kt || M % (2 * DIM) || ((uintptr_t) B_lut & 7) || (((NC >> G) * lut_bytes) & 7) ||
      (((uintptr_t) A_sc | (uintptr_t) B_sc | (uintptr_t) sc_lda | (uintptr_t) sc_ldb) & 7)) {
    printf("mxn_matmul_lut: unsupported shape/alignment M=%d K=%d N=%d G=%d (Kt=%d NC=%d)\n", M, K, N, G, Kt, NC);
    return 1;
  }
  gemmini_extended3_config_ld(lda, MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(ldb, MVIN_SCALE_IDENTITY, false, 1);
  if (A_lut && (((M >> G) > 32) || ((uintptr_t) A_lut & 7) || (((M >> G) * lut_bytes) & 7))) {
    printf("mxn_matmul_lut: loop-managed act codebooks need M >> G <= 32 (M=%d G=%d)\n", M, G);
    return 1;
  }
  if (set_st) gemmini_config_st(2 * ldc * sizeof(uint16_t));
  const int kts = K / Kt;
  for (int c = 0; c < N / NC; c++)
    for (int t = 0; t < kts; t++) {
      const int h = 1 + (int) (mxn_loops++ & 1);
      gemmini_loop_ws_mx_lut(M / (2 * DIM), NC / (2 * DIM), Kt / DIM,
                             A + (size_t) t * Kt, B + (size_t) t * Kt * ldb + (size_t) c * NC / 2,
                             t == kts - 1 ? C + (size_t) c * NC : NULL,
                             lda, ldb, ldc,
                             A_sc + (size_t) (t * Kt / 32) * sc_lda,
                             B_sc + (size_t) (t * Kt / 32) * sc_ldb + (size_t) c * NC, sc_lda, sc_ldb,
                             A_lut, B_lut + (size_t) ((c * NC) >> G) * lut_bytes,
                             /*accumulate*/ t > 0, h, h);
    }
  return 0;
}
static int mxn_matmul_lut(const uint8_t *A, int lda, const uint8_t *B, int ldb, uint16_t *C, int ldc,
                          const uint8_t *A_sc, int sc_lda, const uint8_t *B_sc, int sc_ldb,
                          const uint8_t *B_lut, int lut_bytes, int G, int M, int K, int N) {
  return mxn_matmul_lut_ex(A, lda, B, ldb, C, ldc, A_sc, sc_lda, B_sc, sc_ldb, B_lut, lut_bytes, G, M, K, N, NULL, 1);
}

// mxn_matmul_fp4 with A RESIDENT ("A anywhere"): A is loaded into the scratchpad ONCE, K-tile t as one block of
// (M/32) x (Kt/16) tiles at row (a_base + t * blk) mod the scratchpad, and every loop reads its K-tile there (A = NULL:
// no A load, a_spad_id 0 -> the loop's a_addr_start from LOOP_WS_CONFIG_SPAD_AB). B double-buffers in two explicit
// regions from row b_base (b_spad_id 0 -> b_addr_end). A's DRAM traffic drops from N / NC passes to one.
// The pieces (plan, configs, A block load, one loop) let a caller schedule A blocks under another matmul's loops: a raw
// mvin passes the running loops once they have issued their loads and its rows miss theirs (LoopMatmul ld_pass), and
// so does a config_ld; a config_ex / config_st waits for every loop to finish.
typedef struct {
  const uint8_t *A, *B, *A_sc, *B_sc; uint16_t *C;
  int lda, ldb, ldc, sc_lda, sc_ldb, M, K, N, a_base, b_base, NC, Kt, TI, TK, kts, blk, b_rows;
} mxn_fp4_mm;

// Tile choice; returns 1 when A's blocks and two B buffers do not fit.
static int mxn_fp4_plan(mxn_fp4_mm *p, const uint8_t *A, int lda, const uint8_t *B, int ldb, uint16_t *C, int ldc,
                        const uint8_t *A_sc, int sc_lda, const uint8_t *B_sc, int sc_ldb, int M, int K, int N,
                        int a_base, int b_base) {
  const int spad_rows = BANK_NUM * BANK_ROWS;
  int NC = 0, Kt = 0;
  for (int nc = 512; nc >= 2 * DIM && !NC; nc >>= 1)
    if (N % nc == 0 && M * nc / (DIM * 4) <= ACC_ROWS / 2) NC = nc;
  for (int kt = 512; kt >= 32 && NC && !Kt; kt >>= 1)
    if (K % kt == 0 && b_base + 2 * (kt * NC / (2 * DIM)) <= spad_rows && (kt / 32) * NC <= 4096 && (kt / 32) * M <= 4096)
      Kt = kt;
  if (!NC || !Kt || M % (2 * DIM) || (((uintptr_t) A_sc | (uintptr_t) B_sc | (uintptr_t) sc_lda | (uintptr_t) sc_ldb) & 7))
    return 1;
  *p = (mxn_fp4_mm) {A, B, A_sc, B_sc, C, lda, ldb, ldc, sc_lda, sc_ldb, M, K, N, a_base, b_base, NC, Kt,
                     M / (2 * DIM), Kt / DIM, K / Kt, M / (2 * DIM) * (Kt / DIM) * DIM, Kt * NC / (2 * DIM)};
  for (int t = 0; t < p->kts; t++) {   // every A block inside the scratchpad, clear of the B buffers
    const int r = (a_base + t * p->blk) % spad_rows;
    if (r + p->blk > spad_rows || (r < b_base + 2 * p->b_rows && b_base < r + p->blk)) return 1;
  }
  return 0;
}

static int mxn_fp4_a_row(const mxn_fp4_mm *p, int t) { return (p->a_base + t * p->blk) % (BANK_NUM * BANK_ROWS); }

static void mxn_fp4_cfg_ex_st(const mxn_fp4_mm *p) {
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, /*FP4*/ 2, 2,
                              /*BF16*/ 3, 0);
  gemmini_config_st(2 * p->ldc * sizeof(uint16_t));
}
static void mxn_fp4_cfg_a(const mxn_fp4_mm *p) { gemmini_extended3_config_ld(p->lda, MVIN_SCALE_IDENTITY, false, 0); }
static void mxn_fp4_cfg_b(const mxn_fp4_mm *p) { gemmini_extended3_config_ld(p->ldb, MVIN_SCALE_IDENTITY, false, 1); }

// A's K-tile t, k-major (the loop computes k-outer): row tile i, k-tiles 0..TK-1 contiguous, 4 per mvin (load state 0)
static void mxn_fp4_ld_a(const mxn_fp4_mm *p, int t) {
  for (int k = 0; k < p->TK; k += 4)
    for (int i = 0; i < p->TI; i++) {
      const int w = p->TK - k < 4 ? p->TK - k : 4;
      gemmini_extended_mvin(p->A + (size_t) i * DIM * p->lda + (size_t) t * p->Kt + k * DIM,
                            mxn_fp4_a_row(p, t) + (i * p->TK + k) * DIM, DIM * w, DIM);
    }
}

// Loop (N-chunk c, K-tile t); C stored on the last K-tile
static void mxn_fp4_loop(const mxn_fp4_mm *p, int c, int t) {
  const int s = (int) (mxn_loops++ & 1);   // this loop's B buffer: [b_base + s * b_rows, b_base + (s + 1) * b_rows)
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t) mxn_fp4_a_row(p, t), (uint64_t) (p->b_base + (s + 1) * p->b_rows),
                           k_LOOP_WS_CONFIG_SPAD_AB)
  gemmini_loop_ws_mx(p->M / (2 * DIM), p->NC / (2 * DIM), p->Kt / DIM,
                     NULL, p->B + (size_t) t * p->Kt * p->ldb + (size_t) c * p->NC / 2,
                     t == p->kts - 1 ? p->C + (size_t) c * p->NC : NULL,
                     p->lda, p->ldb, p->ldc,
                     p->A_sc + (size_t) (t * p->Kt / 32) * p->sc_lda,
                     p->B_sc + (size_t) (t * p->Kt / 32) * p->sc_ldb + (size_t) c * p->NC, p->sc_lda, p->sc_ldb,
                     /*accumulate*/ t > 0, 0, 0);
}

// The whole matmul, A block t loaded right before the first loop that reads it. Returns 1 (and does nothing) when it
// does not fit; the caller then falls back to mxn_matmul_fp4.
static int mxn_matmul_fp4_ares_at(const uint8_t *A, int lda, const uint8_t *B, int ldb, uint16_t *C, int ldc,
                                  const uint8_t *A_sc, int sc_lda, const uint8_t *B_sc, int sc_ldb, int M, int K, int N,
                                  int a_base, int b_base) {
  mxn_fp4_mm p;
  if (mxn_fp4_plan(&p, A, lda, B, ldb, C, ldc, A_sc, sc_lda, B_sc, sc_ldb, M, K, N, a_base, b_base)) return 1;
  mxn_fp4_cfg_ex_st(&p);
  mxn_fp4_cfg_a(&p);
  mxn_fp4_cfg_b(&p);
  for (int c = 0; c < N / p.NC; c++)
    for (int t = 0; t < p.kts; t++) {
      if (c == 0) mxn_fp4_ld_a(&p, t);
      mxn_fp4_loop(&p, c, t);
    }
  return 0;
}

static int mxn_matmul_fp4_ares(const uint8_t *A, int lda, const uint8_t *B, int ldb, uint16_t *C, int ldc,
                               const uint8_t *A_sc, int sc_lda, const uint8_t *B_sc, int sc_ldb, int M, int K, int N) {
  const int a_rows = M * K / (2 * DIM * DIM) * DIM;
  return mxn_matmul_fp4_ares_at(A, lda, B, ldb, C, ldc, A_sc, sc_lda, B_sc, sc_ldb, M, K, N, 0, a_rows);
}

// Ideal mesh cycles of one M x K x N matmul on the 16x16 mesh.
#define MXN_IDEAL(m, k, n) ((uint64_t) (m) * (k) * (n) / (DIM * DIM))

#endif  // INCLUDE_MX_NATIVE_H
