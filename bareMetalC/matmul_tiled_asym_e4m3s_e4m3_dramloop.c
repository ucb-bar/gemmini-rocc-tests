// E4M3-single act (no LUT) x E4M3-quad wei (LUT, mode13) through native DRAM loops: loop-managed scales and weight
// codebooks (each loop loads its N-chunk's NC >> G codebooks), split-K accumulate, BF16 C to DRAM. Prints utilization.
//   -DE4Q_HDR='"include/matmul_data_asym_e4m3s_e4m3_128x512x1024_dim32_bf16.h"' -DE4Q_NC=64 -DE4Q_KT=512
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#ifndef E4Q_HDR
#if defined(E4Q_SMALL) && DIM == 32   // the 64x64 data the spad-path _bf16 tests pass on
#define E4Q_HDR "include/matmul_data_asym_e4m3s_e4m3_dim32_bf16.h"
#elif defined(E4Q_SMALL)
#define E4Q_HDR "include/matmul_data_asym_e4m3s_e4m3_bf16.h"
#elif DIM == 32
#define E4Q_HDR "include/matmul_data_asym_e4m3s_e4m3_128x512x1024_dim32_bf16.h"
#else
#define E4Q_HDR "include/matmul_data_asym_e4m3s_e4m3_128x512x1024_bf16.h"
#endif
#endif
#include E4Q_HDR
#ifndef MATMUL_LUT_G
#define MATMUL_LUT_G 1
#endif
#ifndef E4Q_NC
#define E4Q_NC 64
#endif
#ifndef E4Q_KT
#define E4Q_KT 512
#endif
#define LUT_BYTES 16   // one E4M3 codebook: 16 x 8-bit
#ifndef E4Q_ARES
#define E4Q_ARES 1     // A resident: each K-tile of A loaded once (first N-chunk), loops stream only B; 0 = A per loop
#endif

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: MX config / Spike-only test\n"); return 0; }
#else

static uint16_t C[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));

int main() {
  const int NC = E4Q_NC, KT = E4Q_KT, nk = MATMUL_K / KT;
  const int TI = MATMUL_M / DIM, TK = KT / DIM, blk = TI * TK * DIM, b_rows = KT * NC / (2 * DIM);
  const int a_total = MATMUL_M * MATMUL_K / DIM;   // A-resident: all K-tiles at rows [0, a_total), then 2 B buffers
  if (MATMUL_N % NC || MATMUL_K % KT || (NC >> MATMUL_LUT_G) > 32 || MATMUL_M * NC / (2 * DIM) > ACC_ROWS / 2 ||
      (E4Q_ARES ? a_total + 2 * b_rows > BANK_NUM * BANK_ROWS
                : MATMUL_M * KT / DIM + KT * NC / (2 * DIM) > BANK_NUM * BANK_ROWS / 2)) {
    printf("e4m3s x e4m3 dramloop: unsupported NC=%d KT=%d for %dx%dx%d dim%d\n", NC, KT, MATMUL_M, MATMUL_K, MATMUL_N, DIM);
    return 1;
  }
  gemmini_flush(0);
  // A fp8 code0 single (no act LUT); B fp8 code0 quad via the weight LUT; out BF16; uselut
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 1);
  gemmini_mxquant_config_mvout(0, 1, 1, 1, 0, 0, MATMUL_LUT_G);
  gemmini_fence();
  gemmini_extended3_config_ld(MATMUL_K, MVIN_SCALE_IDENTITY, false, 0);       // A [M][K]
  gemmini_extended3_config_ld(MATMUL_N / 2, MVIN_SCALE_IDENTITY, false, 1);   // B [K][N/2] LUT-index nibbles
  gemmini_config_st(MATMUL_N * sizeof(uint16_t));                             // C [M][N] BF16

  int loops = 0;
  uint64_t t0 = read_cycles();
  for (int c = 0; c < MATMUL_N / NC; c++)
    for (int t = 0; t < nk; t++, loops++) {
#if E4Q_ARES
      if (c == 0)   // A K-tile t: row tile i, k-tiles k..k+w-1 per mvin, at (t * blk) + (i * TK + k) * DIM
        for (int k = 0; k < TK; k += MAX_BLOCK_LEN)
          for (int i = 0; i < TI; i++) {
            const int w = TK - k < MAX_BLOCK_LEN ? TK - k : MAX_BLOCK_LEN;
            gemmini_extended_mvin((uint8_t *) A_in + i * DIM * MATMUL_K + t * KT + k * DIM,
                                  t * blk + (i * TK + k) * DIM, DIM * w, DIM);
          }
      const int s = loops & 1;   // this loop's B buffer, ending at a_total + (s + 1) * b_rows
      ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t) (t * blk), (uint64_t) (a_total + (s + 1) * b_rows),
                               k_LOOP_WS_CONFIG_SPAD_AB)
      const int h = 0;
      const uint8_t *a_dram = NULL;
#else
      const int h = 1 + (loops & 1);
      const uint8_t *a_dram = (uint8_t *) A_in + t * KT;
#endif
      gemmini_loop_ws_mx_lut(MATMUL_M / DIM, NC / (2 * DIM), KT / DIM,
                             a_dram, (uint8_t *) B_in + t * KT * (MATMUL_N / 2) + c * NC / 2,
                             t == nk - 1 ? (void *) &C[0][c * NC] : NULL,
                             MATMUL_K, MATMUL_N / 2, MATMUL_N,
                             &A_scales_row[t * KT / 32][0], &B_scales_col[t * KT / 32][c * NC], MATMUL_M, MATMUL_N,
                             NULL, (uint8_t *) &B_lut[0][0] + ((c * NC) >> MATMUL_LUT_G) * LUT_BYTES,
                             t > 0, h, h);
    }
  gemmini_fence();
  uint64_t cyc = read_cycles() - t0;

  int bad = 0;
  for (int i = 0; i < MATMUL_M; i++)
    for (int j = 0; j < MATMUL_N; j++)
      if (C[i][j] != C_out_bf16[i][j] && bad++ < 20)
        printf("C[%d][%d] got %04x exp %04x\n", i, j, C[i][j], C_out_bf16[i][j]);
  // ideal: M*N*K MACs at 2 per PE per cycle (mode13)
  const uint64_t ideal = (uint64_t) MATMUL_M * MATMUL_N * MATMUL_K / (2 * DIM * DIM);
  printf("e4m3s x e4m3 dramloop %dx%dx%d dim%d NC=%d KT=%d G=%d A-%s (%d loops): %d mismatches\n",
         MATMUL_M, MATMUL_K, MATMUL_N, DIM, NC, KT, MATMUL_LUT_G, E4Q_ARES ? "resident" : "streamed", loops, bad);
  printf("PERF cycles=%lu ideal=%lu util=%lu.%lu%%\n", (unsigned long) cyc, (unsigned long) ideal,
         (unsigned long) (ideal * 100 / cyc), (unsigned long) (ideal * 1000 / cyc % 10));
  printf(bad ? "e4m3s x e4m3 dramloop test FAILED\n" : "e4m3s x e4m3 dramloop test PASSED\n");
  return bad != 0;
}
#endif
