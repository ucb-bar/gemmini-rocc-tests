// Native multi-loop MX: matmul_tiled_fp8_128x128 (e4m3 single, BF16 out) as NCHUNKS back-to-back gemmini_loop_ws
// over column chunks of C (the N-chunking tiled_matmul / llama do). A loads once (first loop) and stays resident
// in spad half 1; B alternates halves 1/2 (double buffer); C goes acc -> DRAM with row pitch N (stride fix);
// DRAM-C loops alternate the two acc banks, so loop c's store overlaps loop c+1's compute.
// Scales: legacy (software) flow = per chunk one scale load + CONFIG_SCALE_MEM (bounds = that loop) + fence, since
// scale positions are loop-local (MxScalePos); LOOP_SCALES = loop-managed scales, no command between the loops.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_128x128.h"
#include "include/mx_perf.h"


#define DIM 16
#ifndef NCHUNKS
#define NCHUNKS 2
#endif
#define NC (MATMUL_N / NCHUNKS)
#ifndef K_TILES
#define K_TILES 1   // >1: each chunk is K_TILES accumulating loops (C only on the last), loop-managed scales
#endif
#define KT (MATMUL_K / K_TILES)
#define STR_NC_(x) #x
#define STR_NC(x) STR_NC_(x)

typedef uint8_t elem_t;

static uint16_t C_hw[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));
static uint8_t  a_sc[NCHUNKS][MATMUL_GK][MATMUL_M] __attribute__((aligned(64)));
static uint8_t  b_sc[NCHUNKS][MATMUL_GK][NC] __attribute__((aligned(64)));
static uint32_t scale_sink[512] __attribute__((aligned(32)));

// Complete MAIN_* busy-state split of the loop window (all internal counters: snapshot-safe).
static const int ev[8] = { EXE_ACTIVE_CYCLE, MAIN_EX_CYCLES, MAIN_LD_CYCLES, MAIN_ST_CYCLES,
                           MAIN_LD_EX_CYCLES, MAIN_ST_EX_CYCLES, MAIN_LD_ST_CYCLES, MAIN_LD_ST_EX_CYCLES };
static const char *evn[8] = { "exe_active", "ex", "ld", "st", "ld_ex", "st_ex", "ld_st", "ld_st_ex" };

int main() {
  memset(C_hw, 0, sizeof(C_hw));
  const int I = MATMUL_M / DIM, J = NC / DIM, K = MATMUL_K / DIM;
  for (int c = 0; c < NCHUNKS; c++)
    for (int kb = 0; kb < MATMUL_GK; kb++) {
      memcpy(a_sc[c][kb], A_scales_row[kb], MATMUL_M);
      memcpy(b_sc[c][kb], &B_scales_col[kb][c * NC], NC);
    }

  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

  uint64_t t0 = read_cycles();
#if defined(LOOP_SCALES) || K_TILES == 1
  // nothing here: loop-managed scales, or legacy per-chunk loads in the chunk loop
#elif defined(SCALES_2D)
  // 2-D loads: A repeats per chunk; chunk c's B scales are a pitch-N slice of [GK][N].
  for (int c = 0; c < NCHUNKS; c++) {
    gemmini_mx_load_scales_2d(A_scales_row, sizeof(A_scales_row), 1, 0, c * sizeof(A_scales_row), 0);
    gemmini_mx_load_scales_2d(&B_scales_col[0][c * NC], NC, MATMUL_GK, MATMUL_N, c * MATMUL_GK * NC, 1);
  }
#else
  gemmini_mx_load_scales((uint64_t) a_sc, sizeof(a_sc), 0);
  gemmini_mx_load_scales((uint64_t) b_sc, sizeof(b_sc), 1);
#endif
#if !defined(SCALE_WAIT) && !defined(LOOP_SCALES) && K_TILES > 1
  gemmini_fence();
#endif
  uint64_t t1 = read_cycles();

  gemmini_extended3_config_ld(MATMUL_K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(MATMUL_N * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
#ifdef RELU_OUT
  // ReLU on the accumulated output (store-path activation, applied before the BF16 store).
  gemmini_extended_config_st(MATMUL_N * sizeof(uint16_t), RELU, ACC_SCALE_IDENTITY);
#else
  gemmini_config_st(MATMUL_N * sizeof(uint16_t));
#endif
#if defined(LOOP_SCALES) || K_TILES == 1
  // no CONFIG_SCALE_MEM here: loop-managed, or per chunk below
#elif defined(SCALE_WAIT)
  // No fence: the execute unit holds this config until the scale loads above have landed (rs2[16]).
  gemmini_mxquant_config_mvout_wait((uint64_t) scale_sink, I, J, NCHUNKS * K, 0, 0, 1);
#else
  gemmini_mxquant_config_mvout((uint64_t) scale_sink, I, J, NCHUNKS * K, 0, 0, 1);
#endif

  counter_snapshot_reset();
  for (int i = 0; i < 8; i++) counter_configure(i, ev[i]);
  uint64_t t2 = read_cycles();
  for (int c = 0; c < NCHUNKS; c++) {
#if K_TILES > 1
    // K-tile t: A columns [t*KT, +KT) (pitch K), B rows [t*KT, +KT) of chunk c, scale rows t*KT/32 onward.
    for (int t = 0; t < K_TILES; t++)
      gemmini_loop_ws_mx(I, J, KT / DIM,
                         (const uint8_t *) A_in + t * KT, (const uint8_t *) B_in + (size_t) t * KT * MATMUL_N + c * NC,
                         t == K_TILES - 1 ? &C_hw[0][c * NC] : NULL,
                         MATMUL_K, MATMUL_N, MATMUL_N,
                         &A_scales_row[t * KT / 32][0], &B_scales_col[t * KT / 32][c * NC], MATMUL_M, MATMUL_N,
                         t > 0, 1, 1 + ((c * K_TILES + t) & 1));
#elif defined(LOOP_SCALES)
    // chunk c: A scales = all of [GK][M] (pitch M), B scales = [GK][NC] slice of [GK][N] (pitch N)
    gemmini_loop_ws_mx(I, J, K,
                       c == 0 ? (const void *) A_in : NULL, (const uint8_t *) B_in + c * NC, &C_hw[0][c * NC],
                       MATMUL_K, MATMUL_N, MATMUL_N,
                       A_scales_row, &B_scales_col[0][c * NC], MATMUL_M, MATMUL_N,
                       false, 1, 1 + (c & 1));
#else
    // chunk c's scales (A: [GK][M], B: [GK][NC] slice), its own config, then the loop; fence before the next load
#if defined(SCALES_2D)
    gemmini_mx_load_scales_2d(A_scales_row, sizeof(A_scales_row), 1, 0, 0, 0);
    gemmini_mx_load_scales_2d(&B_scales_col[0][c * NC], NC, MATMUL_GK, MATMUL_N, 0, 1);
#else
    gemmini_mx_load_scales((uint64_t) a_sc[c], sizeof(a_sc[c]), 0);
    gemmini_mx_load_scales((uint64_t) b_sc[c], sizeof(b_sc[c]), 1);
#endif
#if defined(SCALE_WAIT)
    gemmini_mxquant_config_mvout_wait((uint64_t) scale_sink, I, J, K, 0, 0, 1);
#else
    gemmini_fence();
    gemmini_mxquant_config_mvout((uint64_t) scale_sink, I, J, K, 0, 0, 1);
#endif
    gemmini_loop_ws(I, J, K, 0, 0, 0,
                    c == 0 ? (const void *) A_in : NULL, (const uint8_t *) B_in + c * NC, NULL, &C_hw[0][c * NC],
                    MATMUL_K, MATMUL_N, 0, MATMUL_N,
                    false, false, false, false, false, NO_ACTIVATION,
                    1, 1 + (c & 1), false);
    gemmini_fence();
#endif
  }
  gemmini_fence();
  uint64_t t3 = read_cycles();
  uint32_t cnt[8];
  counter_snapshot_take();
  for (int i = 0; i < 8; i++) cnt[i] = counter_read(i);
  counter_snapshot_reset();

  int errors = 0;
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < MATMUL_N; j++)
    {
#ifdef RELU_OUT
      const uint16_t exp = (C_out_bf16[i][j] & 0x8000) ? 0 : C_out_bf16[i][j];   // relu(golden)
#else
      const uint16_t exp = C_out_bf16[i][j];
#endif
      if (C_hw[i][j] != exp) {
        if (errors < 16)
          printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n", i, j, C_hw[i][j], exp);
        errors++;
      }
    }
  if (errors == 0)
    printf("fp8 WS native multi-loop (%d chunks) test PASSED (no mismatches).\n", NCHUNKS);
  else
    printf("fp8 WS native multi-loop (%d chunks) test FAILED with %d mismatches.\n", NCHUNKS, errors);

  uint64_t loop = t3 - t2, ideal = (uint64_t) MATMUL_M * MATMUL_N * MATMUL_K / (DIM * DIM);
  printf("PERF fp8_128x128_dramloop_nc" STR_NC(NCHUNKS) " scales=%lu loops=%lu total=%lu ideal=%lu util(loops)=%lu%%\n",
         (unsigned long) (t1 - t0), (unsigned long) loop, (unsigned long) (t3 - t0), (unsigned long) ideal,
         (unsigned long) (ideal * 100 / loop));
  uint64_t sum = 0;
  printf("PERF_HW fp8_128x128_dramloop_nc" STR_NC(NCHUNKS));
  for (int i = 0; i < 8; i++) { printf(" %s=%u", evn[i], cnt[i]); if (i) sum += cnt[i]; }
  printf(" idle=%ld\n", (long) loop - (long) sum);
  return errors != 0;
}
