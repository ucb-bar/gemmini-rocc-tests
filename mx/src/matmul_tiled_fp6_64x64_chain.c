// Chained back-to-back fp6 (e3m2, LUT-indexed) matmul on MxGemminiRocketConfig: C2 = (A1 @ B1) @ B2.
//
// fp6 operands are 4-bit LUT indices; a per-group 16-entry LUT maps an index to an fp6 value. MM1's
// requantized output C1 is 4-bit indices into an OUTPUT LUT (C1_lut), stored RESIDENT in the tiled
// operand layout (LOOP_WS_REQUANT_TILED). MM2 reuses C1 in place as operand A -- so MM2's
// activation-in LUT is loaded with MM1's OUTPUT LUT (C1_lut), and MM1's output block-scales are
// reused as MM2's input A-scales. Unified Spike / RTL stream, mirroring
// matmul_tiled_fp6_128x128x512_requant.c + the fp4/fp8 chain tests.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_fp6_64x64_chain.h"

#define TILE 16
#define VALUES_PER_BYTE 2
#define USE_LUT 1
#define QUANT_LUT_UPDATE_GRANULARITY 1
#define DIM 16
#define ADDR_LEN 32
#define GEMMINI_FORMAT 1   // fp6

#define LOOP_WS_REQUANT_TILED (1u << 10)

// C8: automatic scale residency (always on for the chain). MM1's requant writes C1's output
// act-scales directly into the on-chip act-scale window (transposed [GN][M]) via
// gemmini_mxquant_config_mvout_resident (MX_SCALE_RESIDENT = mxquant config rs1 bit 63), so MM2
// reads its A-scales in place -- no DRAM buffer, no SW transpose/reload. Data + scales resident.
#define CHAIN_FLAGS (0x38 | LOOP_WS_REQUANT_TILED)
#define MXQUANT_CFG(...) gemmini_mxquant_config_mvout_resident(__VA_ARGS__)  // automatic resident scale reuse

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#undef GEMMINI_BUSY_ADDR
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)


typedef uint8_t elem_t;

static inline int popcount8(uint8_t x) { int c = 0; while (x) { c += x & 1; x >>= 1; } return c; }

// Contiguous read + SW de-tile (see fp4/fp8 chain tests). For fp6 pass M=packed rows (M/2), N=cols.
static void mvout_detile(uint8_t *dst, uint32_t spad_base, int M, int N) {
  int tiles_I = M / DIM, tiles_N = N / DIM;
  int total_rows = M * N / DIM;
  static uint8_t tiled[128 * 128];
  gemmini_config_st(DIM * sizeof(uint8_t));
  for (int r = 0; r < total_rows; r += DIM)
    gemmini_extended_mvout(tiled + r * DIM, spad_base + r, DIM, DIM);
  gemmini_fence();
  for (int i = 0; i < tiles_I; i++)
    for (int nt = 0; nt < tiles_N; nt++)
      for (int r = 0; r < DIM; r++)
        for (int c = 0; c < DIM; c++)
          dst[(i * DIM + r) * N + nt * DIM + c] = tiled[((i * tiles_N + nt) * DIM + r) * DIM + c];
}

static int check_nibbles(const char *tag, uint8_t *got, const uint8_t *exp, int Mp, int N) {
  int errors = 0;
  for (int i = 0; i < Mp; i++)
    for (int j = 0; j < N; j++) {
      uint8_t g = got[i * N + j], e = exp[i * N + j];
      if ((g & 0xF) != (e & 0xF)) { errors++; if (errors <= 40) printf("%s[%d][%d].lo got %x exp %x\n", tag, i, j, g & 0xF, e & 0xF); }
      if ((g >> 4) != (e >> 4))   { errors++; if (errors <= 40) printf("%s[%d][%d].hi got %x exp %x\n", tag, i, j, g >> 4, e >> 4); }
    }
  return errors;
}

// fp6 format config: raw ROCC config (format + USE_LUT) then extended3 (a/b/out fmt + use_lut).
static inline void fp6_config_ex(void) {
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    | ((uint64_t)(1) << 16) | (GEMMINI_FORMAT << 14) | (GEMMINI_FORMAT << 12) | (GEMMINI_FORMAT << 10)
    | (0 << 9) | (0 << 8) | ((false) << 7) | ((USE_LUT) << 4) | ((0) << 3)
    | ((WEIGHT_STATIONARY) << 2) | CONFIG_EX,
    ((uint64_t)(1) << 48) | (0), k_CONFIG);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 1, 1, 1, 1);
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif

  int tiles_I = MATMUL_M / 32;
  int tiles_J = MATMUL_N / 32;
  int tiles_K = MATMUL_K / 16;

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * TILE;
  int SPAD_DEST1 = 2048;
  int SPAD_DEST2 = 4096;

  static uint8_t C1_hw[MATMUL_M / 2][MATMUL_N];
  static uint8_t C2_hw[MATMUL_M / 2][MATMUL_N];
  uint32_t c1_scales[512] __attribute__((aligned(32))) = {0};
  uint32_t c2_scales[512] __attribute__((aligned(32))) = {0};
  memset(C1_hw, 0, sizeof(C1_hw));
  memset(C2_hw, 0, sizeof(C2_hw));

  gemmini_flush(0);
  fp6_config_ex();

  // ================= MM1: C1 = requant(A1 @ B1), stored TILED =================
  gemmini_config_st(1 * sizeof(uint64_t));

  gemmini_mx_load_lut_dt((uint64_t)&B_lut[0][0], LUT_GROUPS_B, 0, 6);   // weight
  gemmini_mx_load_lut_dt((uint64_t)&A_lut[0][0], LUT_GROUPS_A, 1, 6);   // activation-in
  gemmini_mx_load_lut_dt((uint64_t)&C1_lut[0][0], LUT_GROUPS_A, 2, 6);  // activation-out
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();
  // MXQUANT_CFG sets scale_resident, which makes the Controller act-scale mux hold off
  // scale_loader_act -> it MUST come AFTER MM1's A-scale load or that load hangs (the fp6 RTL bug).
  MXQUANT_CFG((uint64_t)c1_scales, tiles_I, tiles_J, tiles_K, 0, 0, QUANT_LUT_UPDATE_GRANULARITY);

  // MVIN A1 (HW-tiled [M/2][K]): tile (i,k) -> a_base + (i*tiles_K + k)*DIM
  gemmini_config_ld(MATMUL_K * sizeof(uint8_t));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++) {
      gemmini_extended_mvin((void *) ((uint8_t*)A_in_hw + i * DIM * MATMUL_K + k * DIM),
                            a_base + (i * tiles_K + k) * DIM, DIM, DIM);
      gemmini_fence();
    }
  // MVIN B1 (packed [K][N/2]): tile (k,j) -> b_base + (k*tiles_J + j)*TILE
  gemmini_config_ld((MATMUL_N / VALUES_PER_BYTE) * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++) {
      gemmini_extended_mvin((void *) ((uint8_t*)B_in + k * TILE * (MATMUL_N / VALUES_PER_BYTE) + j * DIM),
                            b_base + (k * tiles_J + j) * TILE, DIM, DIM);
      gemmini_fence();
    }

  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0,
                       a_base, BANK_NUM * BANK_ROWS, 0, SPAD_DEST1,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false,
                       CHAIN_FLAGS);
  gemmini_fence();

  mvout_detile((uint8_t *) C1_hw, SPAD_DEST1, MATMUL_M / 2, MATMUL_N);
  int c1_err = check_nibbles("C1", (uint8_t *) C1_hw, (const uint8_t *) C1_out, MATMUL_M / 2, MATMUL_N);
  int c1_scale_err = 0;
  { uint8_t *sf = (uint8_t *) c1_scales;
    for (int i = 0; i < MATMUL_M; i++)
      for (int b = 0; b < MATMUL_GN; b++)
        if (sf[b * MATMUL_M + i] != C1_scales_out[i][b]) {
          c1_scale_err++;
          if (c1_scale_err <= 40) printf("C1_scale[%d][%d] got %x exp %x\n", i, b, sf[b * MATMUL_M + i], C1_scales_out[i][b]);
        } }
  if (c1_err == 0 && c1_scale_err == 0) printf("MM1 residency OK (C1 codes+scales exact).\n");
  else printf("MM1 residency FAILED: %d code, %d scale mismatches.\n", c1_err, c1_scale_err);

  // ================= MM2: C2 = requant(C1 @ B2), C1 read IN PLACE, LUT reused =================
  // MM2's activation-in LUT = MM1's OUTPUT LUT (C1_lut). Reuse c1_scales as A-scales ([M][GN]->[GK][M]).

  gemmini_config_st(1 * sizeof(uint64_t));
  // C1's resident scales landed in the act-window half MM1 did not read (half 1): MM2 reads that half
  MXQUANT_CFG((uint64_t)c2_scales, tiles_I, tiles_J, tiles_K, 1, 0, QUANT_LUT_UPDATE_GRANULARITY);

  gemmini_mx_load_lut_dt((uint64_t)&B2_lut[0][0], LUT_GROUPS_B, 0, 6);  // weight = B2
  gemmini_mx_load_lut_dt((uint64_t)&C1_lut[0][0], LUT_GROUPS_A, 1, 6);  // activation-in = MM1 output LUT (REUSED)
  gemmini_mx_load_lut_dt((uint64_t)&C2_lut[0][0], LUT_GROUPS_A, 2, 6);  // activation-out = C2 LUT
  gemmini_mx_load_scales((uint64_t)&B2_scales_col, sizeof(B2_scales_col), 1);
  gemmini_fence();

  // MVIN only B2 (C1 already resident as operand A).
  gemmini_config_ld((MATMUL_N / VALUES_PER_BYTE) * sizeof(uint8_t));
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++) {
      gemmini_extended_mvin((void *) ((uint8_t*)B2_in + k * TILE * (MATMUL_N / VALUES_PER_BYTE) + j * DIM),
                            b_base + (k * tiles_J + j) * TILE, DIM, DIM);
      gemmini_fence();
    }

  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0,
                       SPAD_DEST1, BANK_NUM * BANK_ROWS, 0, SPAD_DEST2,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false,
                       CHAIN_FLAGS);
  gemmini_fence();

  mvout_detile((uint8_t *) C2_hw, SPAD_DEST2, MATMUL_M / 2, MATMUL_N);
  int c2_err = check_nibbles("C2", (uint8_t *) C2_hw, (const uint8_t *) C2_out, MATMUL_M / 2, MATMUL_N);
  int c2_scale_err = 0;
  { uint8_t *sf = (uint8_t *) c2_scales;
    for (int i = 0; i < MATMUL_M; i++)
      for (int b = 0; b < MATMUL_GN; b++)
        if (sf[b * MATMUL_M + i] != C2_scales_out[i][b]) {
          c2_scale_err++;
          if (c2_scale_err <= 40) printf("C2_scale[%d][%d] got %x exp %x\n", i, b, sf[b * MATMUL_M + i], C2_scales_out[i][b]);
        } }
  if (c2_err == 0 && c2_scale_err == 0) printf("MM2 chain OK (C2 codes+scales exact; C1 reused in place, LUT+scales reused).\n");
  else printf("MM2 chain FAILED: %d code, %d scale mismatches.\n", c2_err, c2_scale_err);

  int total = c1_err + c1_scale_err + c2_err + c2_scale_err;
  if (total == 0) printf("fp6 chain test PASSED (MM1 resident + reused as MM2 operand, LUT+scales reused).\n");
  else printf("fp6 chain test FAILED: %d total mismatches.\n", total);

#ifndef BAREMETAL
  exit(total != 0);
#else
  return total != 0;
#endif
}
