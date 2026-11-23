// On-chip chain: BF16 tile -> VPU -> SPAD_REQUANT (tiled + resident scales) -> LOOP_WS matmul, no DRAM in between.
// Data = matmul_fp8_64x64_chain.h: C1_out_bf16 (MM1's BF16 output) is mvin'd, scaled by 2.0 on the VPU (exact),
// requantized in the scratchpad into the operand-A tile layout with its E8M0 scales written resident into the act-scale
// memory, and consumed in place by MM2 = requant(C1 @ B2). The x2 is a power of two, so the goldens stay exact:
// C1 codes == C1_out (scales +1, recomputed by mx_e4m3_ref.h) and C2 codes == C2_out (scales C2_scales_out + 1).
// The whole chain is issued without fences (ordering is the hardware's); one fence at the end.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64_chain.h"
#include "include/vpu_ref.h"
#include "include/mx_e4m3_ref.h"


#define LOOP_WS_REQUANT_TILED (1u << 10)
#define CHAIN_FLAGS (0x38 | LOOP_WS_REQUANT_TILED)   // A read in place from the spad, requant output tiled

#define M MATMUL_M
#define N MATMUL_N
#define K MATMUL_K
#define SP_BF16  0x1000   // bank 1: C1 BF16, row-major, M*N/8 = 512 rows (VPU works in place)
#define SP_C1    128      // bank 0: C1 E4M3, operand-A tiles (256 rows)
#define SP_C2    512      // bank 0: C2 E4M3, tiled (256 rows)
#define BF16_TWO 0x4000

static uint16_t c1_bf16[M][N] __attribute__((aligned(64)));
static uint8_t  c1_scales[M * MATMUL_GN] __attribute__((aligned(64)));
static uint8_t  c2_scales[M * MATMUL_GN] __attribute__((aligned(64)));
static uint8_t  tmp[M * N] __attribute__((aligned(64)));
static uint8_t  c1_codes_ref[M][N], c1_scales_ref[M][MATMUL_GN];

static void mvout_detile(uint8_t *dst, uint32_t sp, int rows, int cols) {   // contiguous mvout, de-tile in SW
  gemmini_config_st(DIM);
  for (int r = 0; r < rows * cols / DIM; r += DIM) gemmini_extended_mvout(tmp + r * DIM, sp + r, DIM, DIM);
  gemmini_fence();
  for (int m = 0; m < rows; m++)
    for (int n = 0; n < cols; n++)
      dst[m * cols + n] = tmp[(((m / DIM) * (cols / DIM) + n / DIM) * DIM + m % DIM) * DIM + n % DIM];
}

static uint64_t t0, t1;

static void chain(void) {
  const int tiles_I = M / DIM, tiles_J = N / DIM, tiles_K = K / DIM;
  const uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 0, 0);
  gemmini_mx_load_scales((uint64_t)&B2_scales_col, sizeof(B2_scales_col), 1);   // MM2 weight scales
  gemmini_fence();
  t0 = read_cycles();

  // MM2's B2 first: independent of the VPU/requant work, so it loads while that runs (bank 3)
  gemmini_config_ld(N * sizeof(uint8_t));
  for (int j = 0; j < tiles_J; j++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((uint8_t *)B2_in + j * DIM * N + k * DIM, b_base + (j * tiles_K + k) * DIM, DIM, DIM);

  // C1 BF16 -> spad (row-major)
  gemmini_config_ld(DIM);
  for (int r = 0; r < M * N / 8; r += DIM) gemmini_extended_mvin((uint8_t *)c1_bf16 + r * DIM, SP_BF16 + r, DIM, DIM);

  // VPU: C1 *= 2.0 in place (512 spad rows)
  gemmini_vpu_scalar(VPU_MULS, SP_BF16, SP_BF16, BF16_TWO, M * N / 8);

  // SPAD_REQUANT: tiled E4M3 at SP_C1, scales -> DRAM c1_scales and resident act-scale memory
  gemmini_spad_requant(SP_C1, SP_BF16, M, N, 1, (uint64_t)c1_scales, 1);

  // MM2 = requant(C1 @ B2): A = C1 in place, A-scales resident, B2 already in the spad
  gemmini_config_st(1 * sizeof(uint16_t));
  gemmini_mxquant_config_mvout_resident((uint64_t)c2_scales, tiles_I, tiles_J, tiles_K, 0, 0, 1);
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, SP_C1, BANK_NUM * BANK_ROWS, 0, SP_C2,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false, CHAIN_FLAGS);
  gemmini_fence();
  t1 = read_cycles();
}

static int check(const char *pass) {
  static uint8_t got[M * N];
  int bad1 = 0, bad1s = 0, bad2 = 0, bad2s = 0, skipped = 0, c1_vs_golden = 0;
  mvout_detile(got, SP_C1, M, N);
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      if (got[m * N + n] != c1_codes_ref[m][n] && bad1++ < 4)
        printf("  %s C1[%d][%d]: hw %02x ref %02x\n", pass, m, n, got[m * N + n], c1_codes_ref[m][n]);
      c1_vs_golden += c1_codes_ref[m][n] != C1_out[m][n];
    }
  for (int m = 0; m < M; m++)
    for (int b = 0; b < MATMUL_GN; b++)
      if (c1_scales[b * MATMUL_M + m] != c1_scales_ref[m][b] && bad1s++ < 4)
        printf("  %s C1 scale[%d][%d]: hw %02x ref %02x\n", pass, m, b, c1_scales[b * MATMUL_M + m], c1_scales_ref[m][b]);
  mvout_detile(got, SP_C2, M, N);
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++)
      if (got[m * N + n] != C2_out[m][n] && bad2++ < 4)
        printf("  %s C2[%d][%d]: hw %02x ref %02x\n", pass, m, n, got[m * N + n], C2_out[m][n]);
  for (int m = 0; m < M; m++)
    for (int b = 0; b < MATMUL_GN; b++) {
      uint8_t ref = C2_scales_out[m][b];
      if (ref <= 104 || ref >= 254) { skipped++; continue; }   // clamped scales do not move with the x2
      if (c2_scales[b * MATMUL_M + m] != ref + 1 && bad2s++ < 4)
        printf("  %s C2 scale[%d][%d]: hw %02x ref %02x\n", pass, m, b, c2_scales[b * MATMUL_M + m], ref + 1);
    }
  printf("%s: C1 codes %d, C1 scales %d, C2 codes %d, C2 scales %d mismatches (%d clamped C2 scales skipped; "
         "x2 C1 codes vs C1_out differ in %d)\n", pass, bad1, bad1s, bad2, bad2s, skipped, c1_vs_golden);
  return bad1 || bad1s || bad2 || bad2s || c1_vs_golden;
}

int main() {
  // host goldens for the VPU + requant stages
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) c1_bf16[m][n] = C1_out_bf16[m][n];
  for (int m = 0; m < M; m++)
    for (int b = 0; b < MATMUL_GN; b++) {
      uint16_t x[32];
      for (int k = 0; k < 32; k++) x[k] = vpu_mul(C1_out_bf16[m][32 * b + k], BF16_TWO);
      mxr_quant_block(x, &c1_codes_ref[m][32 * b], &c1_scales_ref[m][b]);
    }

  memset(c1_scales, 0xa5, sizeof(c1_scales)); memset(c2_scales, 0xa5, sizeof(c2_scales));
  chain();
  int fail = check("chain");
  printf("PERF chain: mvin 4 KB B2 + 8 KB BF16 -> vpu x2 (512 rows) -> spad_requant (128 blocks) -> mm2 64x64x64: %llu cycles\n",
         (unsigned long long)(t1 - t0));

  printf("chain_vpu_spad_requant %s\n", fail ? "FAILED" : "PASSED");
  return fail;
}
