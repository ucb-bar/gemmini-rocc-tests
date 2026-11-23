// See LICENSE for license details.

// Plain BF16 Gemmini (gemmini_params_bf16_dim*_vpu.h): matmul output moved from the accumulator into the scratchpad
// (MVOUT_SPAD), then read there by the VPU.
//   acc = D + A0*B0 + A1*B1        (two K tiles accumulated on an FP32 bias)
//   S   = mvout_spad(acc)          (BF16 rows in the scratchpad)
//   V   = VPU ADDS(S, +1.0)        (VPU reads S, writes V)
// S and V are moved to DRAM and compared with the expected values. Integer-valued data keeps every step exact.

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"

#if !defined(ELEM_T_IS_FLOAT) || ELEM_T_EXP_BITS != 8 || ELEM_T_SIG_BITS != 8
#error "mvout_spad_vpu_bf16 needs a BF16 Gemmini header (gemmini_params_bf16_dim*_vpu.h)"
#endif

#define k_VPU_EXEC 33
#define VPU_F 14          // VPU_EXEC row-address field width (RTL VpuCmd.fieldW = max(14, spad row bits))
#define VPU_ADDS 3        // dst = src1 + imm
#define BF16_ONE 0x3F80

#define ACC_ADDR (1u << (ADDR_LEN - 1))
#define ACC_ACCUMULATE (1u << (ADDR_LEN - 2))

// scratchpad rows
#define A_SP(t) ((t) * DIM)
#define B_SP(t) ((2 + (t)) * DIM)
#define S_SP (4 * DIM)
#define V_SP (5 * DIM)

static uint16_t f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}

static void vpu_exec(uint32_t src1, uint32_t src2, uint32_t dst, uint32_t rows, int op, uint16_t imm) {
  const uint64_t rs1 = (uint64_t)src1 | ((uint64_t)src2 << VPU_F) | ((uint64_t)dst << (2 * VPU_F)) |
                       ((uint64_t)rows << (3 * VPU_F));
  const uint64_t rs2 = (uint64_t)op | ((uint64_t)imm << 16);   // bcast 0, rlen 0
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, rs1, rs2, k_VPU_EXEC);
}

static uint32_t lcg_state = 4321;
static uint32_t lcg(void) { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state >> 8; }

int main() {
  printf("BF16 MVOUT_SPAD -> VPU, DIM %d\n", DIM);
  gemmini_flush(0);

  static int8_t Ai[DIM][2 * DIM], Bi[2 * DIM][DIM];
  static int Di[DIM][DIM];
  static elem_t A[DIM][2 * DIM] row_align(1);
  static elem_t B[2 * DIM][DIM] row_align(1);
  static acc_t D[DIM][DIM] row_align_acc(1);
  static elem_t S[DIM][DIM] row_align(1);
  static elem_t V[DIM][DIM] row_align(1);

  for (int i = 0; i < DIM; i++)
    for (int k = 0; k < 2 * DIM; k++) { Ai[i][k] = (int8_t)(lcg() % 3) - 1; A[i][k] = f32_to_bf16(Ai[i][k]); }
  for (int k = 0; k < 2 * DIM; k++)
    for (int j = 0; j < DIM; j++) { Bi[k][j] = (int8_t)(lcg() % 3) - 1; B[k][j] = f32_to_bf16(Bi[k][j]); }
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++) { Di[i][j] = (int)(lcg() % 9) - 4; D[i][j] = (float)Di[i][j]; }

  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_extended_config_st(DIM * sizeof(elem_t), NO_ACTIVATION, ACC_SCALE_IDENTITY);

  // A tiles (rows of 2*DIM elements in DRAM), B tiles, bias D (full FP32 into the accumulator)
  gemmini_config_ld(2 * DIM * sizeof(elem_t));
  for (int t = 0; t < 2; t++) gemmini_extended_mvin(&A[0][t * DIM], A_SP(t), DIM, DIM);
  gemmini_config_ld(DIM * sizeof(elem_t));
  for (int t = 0; t < 2; t++) gemmini_extended_mvin(&B[t * DIM][0], B_SP(t), DIM, DIM);
  gemmini_config_ld(DIM * sizeof(acc_t));
  gemmini_extended_mvin(&D[0][0], ACC_ADDR, DIM, DIM);

  for (int t = 0; t < 2; t++) {
    gemmini_preload(B_SP(t), ACC_ADDR | ACC_ACCUMULATE);
    gemmini_compute_preloaded(A_SP(t), GARBAGE_ADDR);
  }

  gemmini_mvout_spad(S_SP, ACC_ADDR);                       // accumulator -> BF16 scratchpad rows
  vpu_exec(S_SP, 0, V_SP, DIM, VPU_ADDS, BF16_ONE);        // VPU reads the stored rows

  gemmini_config_st(DIM * sizeof(elem_t));
  gemmini_extended_mvout(&S[0][0], S_SP, DIM, DIM);
  gemmini_extended_mvout(&V[0][0], V_SP, DIM, DIM);
  gemmini_fence();

  int errors = 0;
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++) {
      int acc = Di[i][j];
      for (int k = 0; k < 2 * DIM; k++) acc += Ai[i][k] * Bi[k][j];
      const uint16_t s_exp = f32_to_bf16((float)acc), v_exp = f32_to_bf16((float)(acc + 1));
      if (S[i][j] != s_exp || V[i][j] != v_exp) {
        if (errors < 10)
          printf("MISMATCH [%d][%d]: spad 0x%04x (expected 0x%04x), vpu 0x%04x (expected 0x%04x)\n",
                 i, j, S[i][j], s_exp, V[i][j], v_exp);
        errors++;
      }
    }
  if (errors) {
    printf("FAIL: %d of %d outputs differ\n", errors, DIM * DIM);
    exit(1);
  }
  printf("PASS\n");
  exit(0);
}
