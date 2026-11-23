// Template: chained fp4 matmuls, C1 stays in the spad. Built per line of mx/tests.txt.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include MX_DATA_H

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define SMEM 0x40000000

#undef DIM
#define DIM MESH_DIM
#define ADDR_LEN 32

// LOOP_WS rs2 bit10: deposit requant->spad output BLOCK-TILED (operand layout) for in-place reuse.
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


typedef uint8_t elem_t;   // 2 fp4 codes per byte

static inline int popcount8(uint8_t x) {
  int count = 0;
  while (x) { count += x & 1; x >>= 1; }
  return count;
}

// Read a BLOCK-TILED output resident in the scratchpad back to a flat [M][N]-byte DRAM buffer.
// Reads CONTIGUOUSLY (config_st=DIM) into a temp then de-tiles in SW -- a strided de-tile mvout
// (config_st=N) makes the writer DMA zero-fill gap cache lines on RTL at N>=128 and corrupts the
// output. For fp4 pass M=packed-rows (M/2), N=cols: tile (i,nt) at spad (i*tiles_N+nt)*DIM.
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

// Compare a resident nibble output (de-tiled to [M/2][N]) vs golden, nibble by nibble.
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

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif

  // fp4 tiling: 32x32 mesh tiles -> tiles_I=M/32, tiles_J=N/32; K in 16-wide tiles.
  int tiles_I = MATMUL_M / DIM / 2;
  int tiles_J = MATMUL_N / DIM / 2;
  int tiles_K = MATMUL_K / DIM;

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  int SPAD_DEST1 = 2048;   // C1 resident here; fp4 64x64 C1 footprint = (M/2)*N/16 = 128 rows
  int SPAD_DEST2 = 4096;   // C2 here (clear of C1 and B2)

  static uint8_t C1_hw[MATMUL_M / 2][MATMUL_N];
  static uint8_t C2_hw[MATMUL_M / 2][MATMUL_N];
  uint32_t c1_scales[512] __attribute__((aligned(32))) = {0};
  uint32_t c2_scales[512] __attribute__((aligned(32))) = {0};
  memset(C1_hw, 0, sizeof(C1_hw));
  memset(C2_hw, 0, sizeof(C2_hw));

  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 2, 2, 2, 0);

  // ================= MM1: C1 = requant(A1 @ B1), stored TILED =================
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_fence();

  // MVIN A1 (HW-tiled [M/2][K]): tile (i,k) -> a_base + (i*tiles_K + k)*DIM
  gemmini_config_ld(MATMUL_M * sizeof(elem_t));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++)
      gemmini_extended_mvin((void *) (((elem_t*)A_in_hw) + i * DIM * MATMUL_M + k * DIM),
                            a_base + (i * tiles_K + k) * DIM, DIM, DIM);

  // MVIN B1 (packed [K][N/2]): tile (k,j) -> b_base + (k*tiles_J + j)*DIM
  gemmini_config_ld(MATMUL_N * sizeof(elem_t) / 2);
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++)
      gemmini_extended_mvin((void *) (((elem_t*)B_in) + k * DIM * MATMUL_N / 2 + j * DIM),
                            b_base + (k * tiles_J + j) * DIM, DIM, DIM);

  gemmini_config_st(1 * sizeof(uint16_t));
  MXQUANT_CFG((uint64_t)c1_scales, tiles_I, tiles_J, tiles_K, 0, 0, 1);

  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0,
                       a_base, BANK_NUM * BANK_ROWS, 0, SPAD_DEST1,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false,
                       CHAIN_FLAGS);
  gemmini_fence();

  // Residency readback (de-tile) + checks. C1 packed layout is [M/2][N].
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

  // ================= MM2: C2 = requant(C1 @ B2), C1 read IN PLACE =================
  // Reuse MM1's output block-scales as MM2's A-scales: c1_scales is [M][GN]; the A-scale window
  // wants [GK][M] (a_off = group*M + row), and N1/32 == K2/32 so GN == GK -> transpose (tiny).

  gemmini_mx_load_scales((uint64_t)&B2_scales_col, sizeof(B2_scales_col), 1);  // B = fresh B2 scales
  gemmini_fence();

  // MVIN only B2 (C1 already resident as operand A). packed [K2][N2/2], K2=N1 (square).
  gemmini_config_ld(MATMUL_N * sizeof(elem_t) / 2);
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++)
      gemmini_extended_mvin((void *) (((elem_t*)B2_in) + k * DIM * MATMUL_N / 2 + j * DIM),
                            b_base + (k * tiles_J + j) * DIM, DIM, DIM);

  gemmini_config_st(1 * sizeof(uint16_t));
  // C1's resident scales landed in the act-window half MM1 did not read (half 1): MM2 reads that half
  MXQUANT_CFG((uint64_t)c2_scales, tiles_I, tiles_J, tiles_K, 1, 0, 1);

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
  if (c2_err == 0 && c2_scale_err == 0) printf("MM2 chain OK (C2 codes+scales exact; C1 reused in place, scales reused).\n");
  else printf("MM2 chain FAILED: %d code, %d scale mismatches.\n", c2_err, c2_scale_err);

  int total = c1_err + c1_scale_err + c2_err + c2_scale_err;
  if (total == 0) printf("fp4 chain test PASSED (MM1 resident + reused as MM2 operand, scales reused).\n");
  else printf("fp4 chain test FAILED: %d total mismatches.\n", total);

#ifndef BAREMETAL
  exit(total != 0);
#else
  return total != 0;
#endif
}
