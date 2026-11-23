// One mixed-K LOOP_WS whose ACTIVATION format changes per K-block (LOOP_WS_CONFIG_KMIX_A, DIM >= 16).
//   KA_BASE 44: loop = MXFP4 x MXFP4 (mode0); plain K-blocks = E4M3 x E4M3 (or E3M2 direct, KA_PLAIN 6): KMIX + KMIX_A
//   KA_BASE 48: loop = MXFP4 A x E4M3-single W (mode2); plain = E4M3 x E4M3, same W: KMIX_A only
// Data: gen/gen_mxk_kmixa.py (golden = one BF16 accumulation over the K-blocks in K order).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_params.h"
#include "include/gemmini_testutils.h"
#include KA_DATA_H

#define M KA_M
#define K KA_K
#define N KA_N
#define OFF(p) (KA_BLOB + KA_OFF_##p)
#define GOLD_(d) OFF(C_D##d)
#define GOLD(d) GOLD_(d)
#if KA_BASE == 44
#define KA_BLOOP OFF(B4)
#else
#define KA_BLOOP OFF(BP)
#endif
#define KA_FP6 (KA_PLAIN == 6 ? (1ULL << 63) : 0ULL)   // KMIX / KMIX_A rs2[63:62]: plain operand is direct E3M2

#if DIM < 16
int main() { printf("mxk_kmixa: mixed-K loops need DIM >= 16\n"); return 1; }
#else

static uint16_t C[M][N] __attribute__((aligned(64)));

int main() {
  gemmini_flush(0);
#ifdef KA_SFMT_A
  gemmini_mx_scale_fmt(KA_SFMT_A, KA_SBASE_A, KA_SFMT_W, KA_SBASE_W, MX_SFMT_E8M0, 0);
#endif
  const int wq = KA_BASE == 44;   // the loop's W is FP4 (quad)
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, /*A FP4*/ 2,
                              wq ? 2 : 0, /*BF16*/ 3, 0);
  gemmini_extended3_config_ld(K, MVIN_SCALE_IDENTITY, false, 0);                // A4 [M/2][K], AP [M][K]: same pitch
  gemmini_extended3_config_ld(wq ? N / 2 : N, MVIN_SCALE_IDENTITY, false, 1);   // loop B
  gemmini_extended3_config_ld(N, MVIN_SCALE_IDENTITY, false, 2);                // plain B (KMIX, state 2)
  gemmini_config_st(2 * N * sizeof(uint16_t));                                  // quad-act C rows
  memset(C, 0, sizeof(C));
  gemmini_fence();

  const uint64_t t0 = read_cycles();
  if (wq) ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t) OFF(BP), (uint64_t) N | KA_FP6, k_LOOP_WS_CONFIG_KMIX)
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t) OFF(AP), KA_FP6, k_LOOP_WS_CONFIG_KMIX_A)
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, KA_MLO, KA_MHI, k_LOOP_WS_CONFIG_KMASK)
  gemmini_loop_ws_mx(M / (2 * DIM), wq ? N / (2 * DIM) : N / DIM, K / DIM, OFF(A4), KA_BLOOP, &C[0][0],
                     K, wq ? N / 2 : N, N, OFF(AS), OFF(BS), M, N, 0, 1, 1);
  gemmini_fence();
  const uint64_t cyc = read_cycles() - t0;

  const uint16_t *exp = (const uint16_t *) GOLD(DIM);
  int bad = 0;
  for (int i = 0; i < M * N; i++)
    if (C[i / N][i % N] != exp[i] && bad++ < 8) printf("C[%d][%d] got %04x exp %04x\n", i / N, i % N, C[i / N][i % N], exp[i]);
  for (int r = 0; bad && r < M; r++) {   // one char per 8 columns, failing rows only
    char line[N / 8 + 1]; int any = 0;
    for (int g = 0; g < N / 8; g++) {
      int b = 0;
      for (int c = g * 8; c < g * 8 + 8; c++) b |= C[r][c] != exp[r * N + c];
      line[g] = b ? 'x' : '.'; any |= b;
    }
    line[N / 8] = 0;
    if (any) printf("  r%02d %s\n", r, line);
  }
  printf("mxk_kmixa %dx%dx%d dim%d base %d plain FP%d, %d of %d K-blocks in the loop mode: %d mismatches, %lu cycles\n",
         M, K, N, DIM, KA_BASE, KA_PLAIN, KA_NL, KA_NKB, bad, (unsigned long) cyc);
  printf(bad ? "mxk_kmixa FAILED\n" : "mxk_kmixa PASSED\n");
  return bad != 0;
}
#endif
