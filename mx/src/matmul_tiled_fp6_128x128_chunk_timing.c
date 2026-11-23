// B0 timing experiment: does an FP6 E3M2 LUT requant matmul's store tail hide under the next row chunk's compute?
// MM1 of matmul_tiled_fp6_128x128_chain (requant -> tiled spad, resident scales), run twice:
//   1) one loop over all rows (timed, C1 checked against the golden),
//   2) NCHUNKS row-chunk loops on alternating acc halves (inc_acc_addr), scale config sized to one chunk.
// Run 2 is TIMING ONLY: the LUT row tables and the output-scale coalescer are not chunk-aware in RTL, so only chunk 0's
// codes are expected to match on RTL (printed as a sanity check; Spike reads scales per loop, so it differs). See planning (B0).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/matmul_fp6_128x128_chain.h"
#include "include/mx_perf.h"

#ifndef NCHUNKS
#define NCHUNKS 2
#endif
#define CHUNK_STR_(x) #x
#define CHUNK_STR(x) CHUNK_STR_(x)

#define TILE 16
#define VALUES_PER_BYTE 2
#define USE_LUT 1
#define QUANT_LUT_UPDATE_GRANULARITY 1
#define DIM 16
#define GEMMINI_FORMAT 1   // fp6
#define LOOP_WS_REQUANT_TILED (1u << 10)
#define LOOP_WS_INC_ACC_ADDR (1u << 8)
#define CHAIN_FLAGS (0x38 | LOOP_WS_REQUANT_TILED)


#define CHUNK_M (MATMUL_M / NCHUNKS)
// Run 2 scales: the scale read row is loop_bound_i * (k >> s) + i with free-running counters, so with the config sized to
// a chunk and loop_bound_k = NCHUNKS * tiles_K (k runs on across chunks), chunk c reads A groups [c*GK, (c+1)*GK) of a
// [NCHUNKS][GK][CHUNK_M] layout and B groups of B repeated NCHUNKS times (as matmul_tiled_fp8_128x128_chunked).
static uint8_t A_scales_chunked[NCHUNKS][MATMUL_GK][CHUNK_M] __attribute__((aligned(64)));
static uint8_t B_scales_rep[NCHUNKS][MATMUL_GK][MATMUL_N] __attribute__((aligned(64)));

static void mvout_detile(uint8_t *dst, uint32_t spad_base, int M, int N) {
  int tiles_I = M / DIM, tiles_N = N / DIM;
  static uint8_t tiled[128 * 128];
  gemmini_config_st(DIM * sizeof(uint8_t));
  for (int r = 0; r < M * N / DIM; r += DIM)
    gemmini_extended_mvout(tiled + r * DIM, spad_base + r, DIM, DIM);
  gemmini_fence();
  for (int i = 0; i < tiles_I; i++)
    for (int nt = 0; nt < tiles_N; nt++)
      for (int r = 0; r < DIM; r++)
        for (int c = 0; c < DIM; c++)
          dst[(i * DIM + r) * N + nt * DIM + c] = tiled[((i * tiles_N + nt) * DIM + r) * DIM + c];
}

static int count_bad(const uint8_t *got, int Mp, int N) {
  int bad = 0;
  for (int i = 0; i < Mp; i++)
    for (int j = 0; j < N; j++) bad += got[i * N + j] != C1_out[i][j];
  return bad;
}

static inline void fp6_config_ex(void) {
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16) | (GEMMINI_FORMAT << 14) | (GEMMINI_FORMAT << 12) | (GEMMINI_FORMAT << 10)
    | ((USE_LUT) << 4) | ((WEIGHT_STATIONARY) << 2) | CONFIG_EX,
    ((uint64_t)(1) << 48), k_CONFIG);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 1, 1, 1, 1);
}

int main() {
  const int tiles_I = MATMUL_M / 32, tiles_J = MATMUL_N / 32, tiles_K = MATMUL_K / 16;
  const int chunk_I = tiles_I / NCHUNKS;
  const uint32_t a_base = 0, b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * TILE;
  const int DEST_ONE = 2048, DEST_CHUNK = 4096;
  const int c_rows_per_i = (MATMUL_N / DIM) * DIM;   // tiled output: one 32-row block = N/16 tiles x 16 rows
  static uint8_t C_hw[MATMUL_M / 2][MATMUL_N];
  static uint32_t c1_scales[512] __attribute__((aligned(32)));

  printf("fp6 e3m2 LUT chunk timing: M=N=K=%d, NCHUNKS=%d (chunk_I=%d)\n", MATMUL_M, NCHUNKS, chunk_I);
  if (tiles_I % NCHUNKS) { printf("NCHUNKS must divide tiles_I=%d\n", tiles_I); return 1; }

  gemmini_flush(0);
  fp6_config_ex();
  gemmini_config_st(1 * sizeof(uint64_t));
  gemmini_mx_load_lut_dt((uint64_t)&B_lut[0][0], LUT_GROUPS_B, 0, 6);
  gemmini_mx_load_lut_dt((uint64_t)&A_lut[0][0], LUT_GROUPS_A, 1, 6);
  gemmini_mx_load_lut_dt((uint64_t)&C1_lut[0][0], LUT_GROUPS_A, 2, 6);
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();
  gemmini_mxquant_config_mvout_resident((uint64_t)c1_scales, tiles_I, tiles_J, tiles_K, 0, 0, QUANT_LUT_UPDATE_GRANULARITY);

  gemmini_config_ld(MATMUL_K * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void *) ((uint8_t *) A_in_hw + i * DIM * MATMUL_K + k * DIM), a_base + (i * tiles_K + k) * DIM, DIM, DIM);
  gemmini_config_ld((MATMUL_N / VALUES_PER_BYTE) * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++)
      gemmini_extended_mvin((void *) ((uint8_t *) B_in + k * TILE * (MATMUL_N / VALUES_PER_BYTE) + j * DIM),
                            b_base + (k * tiles_J + j) * TILE, DIM, DIM);
  gemmini_fence();

  // ---- 1) one loop ----
  mx_perf_t p1 = {0};
  mx_perf_ctr_start(&p1);
  MX_PERF_MARK(p1.t0);
  p1.t_ld = p1.t0;
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0, DEST_ONE,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false, CHAIN_FLAGS);
  MX_PERF_MARK(p1.t_ex);
  mx_perf_ctr_stop(&p1);
  p1.t_st = p1.t_ex;
  mvout_detile((uint8_t *) C_hw, DEST_ONE, MATMUL_M / 2, MATMUL_N);
  const int bad_one = count_bad((uint8_t *) C_hw, MATMUL_M / 2, MATMUL_N);

  // ---- 2) NCHUNKS row-chunk loops, alternating acc halves (timing only) ----
  for (int c = 0; c < NCHUNKS; c++)
    for (int g = 0; g < MATMUL_GK; g++) {
      memcpy(A_scales_chunked[c][g], &A_scales_row[g][c * CHUNK_M], CHUNK_M);
      memcpy(B_scales_rep[c][g], B_scales_col[g], MATMUL_N);
    }
  gemmini_mx_load_scales((uint64_t)A_scales_chunked, sizeof(A_scales_chunked), 0);   // run 1's resident flush overwrote them
  gemmini_mx_load_scales((uint64_t)B_scales_rep, sizeof(B_scales_rep), 1);
  gemmini_fence();
  gemmini_config_st(1 * sizeof(uint64_t));
  gemmini_mxquant_config_mvout_resident((uint64_t)c1_scales, chunk_I, tiles_J, NCHUNKS * tiles_K, 0, 0, QUANT_LUT_UPDATE_GRANULARITY);
  gemmini_fence();
  mx_perf_t p2 = {0};
  mx_perf_ctr_start(&p2);
  MX_PERF_MARK(p2.t0);
  p2.t_ld = p2.t0;
  for (int c = 0; c < NCHUNKS; c++)
    gemmini_loop_ws_spad(chunk_I, tiles_J, tiles_K, 0, 0, 0,
                         a_base + c * chunk_I * tiles_K * DIM, BANK_NUM * BANK_ROWS, 0, DEST_CHUNK + c * chunk_I * c_rows_per_i,
                         false, false, false, false, false, NO_ACTIVATION, 0, 0, false, CHAIN_FLAGS | LOOP_WS_INC_ACC_ADDR);
  MX_PERF_MARK(p2.t_ex);
  mx_perf_ctr_stop(&p2);
  p2.t_st = p2.t_ex;
  mvout_detile((uint8_t *) C_hw, DEST_CHUNK, MATMUL_M / 2, MATMUL_N);
  const int bad_c0 = count_bad((uint8_t *) C_hw, MATMUL_M / 2 / NCHUNKS, MATMUL_N);

  const uint64_t ideal = MX_PERF_IDEAL_QUAD(MATMUL_M, MATMUL_N, MATMUL_K, DIM);
  mx_perf_report_ideal("fp6_e3m2_lut_one_loop", MATMUL_M, MATMUL_N, MATMUL_K, DIM, ideal, &p1);
  mx_perf_report_ideal("fp6_e3m2_lut_chunk" CHUNK_STR(NCHUNKS), MATMUL_M, MATMUL_N, MATMUL_K, DIM, ideal, &p2);
  printf("one loop: C1 %d bad bytes (expect 0); chunked: chunk 0 %d bad bytes (expect 0), later chunks not checked\n",
         bad_one, bad_c0);
  const int fail = bad_one != 0;
  printf("fp6 chunk timing %s\n", fail ? "FAILED (one-loop C1 mismatch)" : "DONE");
  return fail;
}
