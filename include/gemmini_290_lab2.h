// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_LAB2_H
#define SRC_MAIN_C_GEMMINI_LAB2_H

#include "gemmini.h"
static uint32_t static_alloc_end = 0x0;

// #define STATIC_ANNOTATE 1

static void sp_tiled_matmul_ws_simple(const elem_t * A, const elem_t * B,
        const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        int act,
        int a_spad_id, int b_spad_id) {


  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2) | (full_C << (ADDR_LEN-3));
  const int A_blocks = a_transpose ? (I <= MAX_BLOCK_LEN ? I : MAX_BLOCK_LEN) :
    (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN);
  const int B_blocks = b_transpose ? (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN) :
    (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  const int D_blocks = low_D ? (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN) :
    (J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC);
  const int C_blocks = full_C ? 1 : (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  // Move-in D
  if (D != NULL && !no_bias) {
    for (size_t i = 0; i < I; i++) {
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const void * const D_dram_addr = (int8_t *)D + (bias_row * D_row_stride + j)*DIM*sizeof_D;
        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;
        size_t blocks = j + D_blocks <= J ? D_blocks : J-j;
        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        gemmini_extended_mvin3(D_dram_addr, D_sp_addr_acc, cols, rows);
        #ifdef STATIC_ANNOTATE
        printf("gemmini_extended_mvin3(D + %p, %p, %u, %u);\n", (bias_row * D_row_stride + j)*DIM*sizeof_D, D_sp_addr_acc, cols, rows);
        #endif
      }
    }
  }
  for (size_t k = 0; k < K; k++) {
    for (size_t j = 0; j < J; j++) {
      for (size_t i = 0; i < I; i++) {
        const uint32_t A_sp_addr = a_transpose ? (A_sp_addr_start + (k*I + i)*DIM) :
          (A_sp_addr_start + (i*K + k)*DIM);
        const uint32_t B_sp_addr = b_transpose ? (B_sp_addr_start + (j*K + k)*DIM) :
          (B_sp_addr_start + (k*J + j)*DIM);
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;
        // Mvin A
        if (a_transpose) {
          if (j == 0 && i % A_blocks == 0) {
            const elem_t * const A_dram_addr = A + (k*A_row_stride + i)*DIM;
            const size_t blocks = i + A_blocks <= I ? A_blocks : I-i;
            const size_t cols = blocks * DIM - (i + blocks >= I ? pad_I : 0);
            const size_t rows = DIM - (k == K-1 ? pad_K : 0);
            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
            #ifdef STATIC_ANNOTATE
            printf("gemmini_extended_mvin(A + %p, %p, %u, %u);\n", (k*A_row_stride+i)*DIM, A_sp_addr, cols, rows);
            #endif
          }
        } else {
          if (j == 0 && k % A_blocks == 0) {
            const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
            const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
            const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
            const size_t rows = DIM - (i == I-1 ? pad_I : 0);
            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
            #ifdef STATIC_ANNOTATE
            printf("gemmini_extended_mvin(A + %p, %p, %u, %u);\n", (i*A_row_stride+k)*DIM, A_sp_addr, cols, rows);
            #endif
          }
        }
        // Mvin B
        if (b_transpose) {
          if (i == 0 && k % B_blocks == 0) {
            const elem_t * const B_dram_addr = B + (j*B_row_stride + k)*DIM;
            const size_t blocks = k + B_blocks <= K ? B_blocks : K-k;
            const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
            const size_t rows = DIM - (j == J-1 ? pad_J : 0);
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
            #ifdef STATIC_ANNOTATE
            printf("gemmini_extended_mvin2(B + %p, %p, %u, %u);\n", (j*B_row_stride+k)*DIM, B_sp_addr, cols, rows);
            #endif
          }
        } else {
          if (i == 0 && j % B_blocks == 0) {
            const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
            const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
            const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
            const size_t rows = DIM - (k == K-1 ? pad_K : 0);
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
            #ifdef STATIC_ANNOTATE
            printf("gemmini_extended_mvin2(B + %p, %p, %u, %u);\n", (k*B_row_stride+j)*DIM, B_sp_addr, cols, rows);
            #endif
          }
        }
        // Compute
        {
          uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          uint32_t out_sp_addr = C_sp_addr;
          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it
          int no_bias_new_matrix = no_bias && D != NULL && k == 0;
          if (no_bias_new_matrix) {
            out_sp_addr &= ~(1 << (ADDR_LEN-2));
          }
          const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
          const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
          const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
          const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
          const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
          const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);
          gemmini_extended_preload(pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);
          #ifdef STATIC_ANNOTATE
          printf("gemmini_extended_preload(%p, 0x%x, %u, %u, %u, %u);\n", pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);
          #endif
          if (i == 0) { // First iteration
            gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            #ifdef STATIC_ANNOTATE
            printf("gemmini_extended_compute_preloaded(%p, 0x%x, %u, %u, %u, %u);\n", A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            #endif
          } else { // All other iterations
            gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            #ifdef STATIC_ANNOTATE
            printf("gemmini_extended_compute_accumulated(%p, 0x%x, %u, %u, %u, %u);\n", A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            #endif
          }
        }
        if (C != NULL && k == K-1) {
          // Move-out C (if not normalizing)
          if (((act != LAYERNORM) && (act != SOFTMAX)) && (j == J-1 || j % C_blocks == C_blocks-1)) {
            const size_t rounded_j = (j / C_blocks) * C_blocks;
            const uint32_t rounded_C_sp_addr = C_sp_addr_start + (i*J + rounded_j)*DIM;
            void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + rounded_j)*DIM*sizeof_C;
            const size_t blocks = rounded_j + C_blocks <= J ? C_blocks : J-rounded_j;
            const size_t cols = blocks * DIM - (rounded_j + blocks >= J ? pad_J : 0);
            const size_t rows = DIM - (i == I - 1 ? pad_I : 0);
            gemmini_extended_mvout(C_dram_addr, rounded_C_sp_addr, cols, rows);
            #ifdef STATIC_ANNOTATE
            printf("gemmini_extended_mvout(C + %p, 0x%x, %u, %u);\n", (i*C_row_stride+rounded_j)*DIM*sizeof_C, rounded_C_sp_addr, cols, rows);
            #endif
          }
        }
      } 
    }
  }
  gemmini_fence();
}

static void sp_tiled_matmul_dram_spad_ws(const elem_t * A, const uint32_t B_sp_addr_start,
        const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        int act,
        int a_spad_id, int b_spad_id) {

  // TODO Lab2
}

static void sp_tiled_matmul_full_spad_ws(const uint32_t A_sp_addr_start, const uint32_t B_sp_addr_start,
        const void * D, const uint32_t C_dst_sp_addr_start,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        int act,
        int a_spad_id, int b_spad_id) {
  
  // TODO Lab2
}


static void sp_tiled_matmul_auto_ws(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        int act) {
        
        size_t stride_A = dim_K;
        size_t stride_B = dim_J;
        size_t stride_D = dim_J;
        size_t stride_C = dim_J;

        scale_t A_scale_factor = MVIN_SCALE_IDENTITY;
        scale_t B_scale_factor = MVIN_SCALE_IDENTITY;
        scale_acc_t D_scale_factor = MVIN_SCALE_IDENTITY;
        acc_scale_t scale = ACC_SCALE_IDENTITY;
        acc_scale_t bert_scale = ACC_SCALE_IDENTITY;
        bool repeating_bias = false;
        bool transpose_A = false;
        bool transpose_B = false;

        bool full_C = false;
        bool low_D = 0;
        uint8_t weightA = 0;
        enum tiled_matmul_type_t tiled_matmul_type = WS;
        const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
        const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

        gemmini_extended_config_ex(WS, act & 3, 0, 1, transpose_A, transpose_B);
        gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
        gemmini_extended3_config_ld(stride_A * sizeof(elem_t), A_scale_factor, false, 0);
        gemmini_extended3_config_ld(stride_B * sizeof(elem_t), B_scale_factor, false, 1)
        gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);

        #ifdef STATIC_ANNOTATE
        printf("====================================\n");
        printf("STARTING ANNOTATION\n");
        printf("====================================\n");
        printf("gemmini_extended_config_ex(WS, %d, 0, 1, %d, %d);\n", act & 3, transpose_A, transpose_B);
        printf("gemmini_extended_config_st(%d, %d, ACC_SCALE_IDENTITY);\n", stride_C * sizeof_C, act & 3);
        printf("gemmini_extended3_config_ld(%d, MVIN_SCALE_IDENTITY, %d, %d);\n", stride_A * sizeof(elem_t), false, 0);
        printf("gemmini_extended3_config_ld(%d, MVIN_SCALE_IDENTITY, %d, %d);\n", stride_B * sizeof(elem_t), false, 1);
        printf("gemmini_extended3_config_ld(%d, MVIN_SCALE_IDENTITY, %d, %d);\n", repeating_bias ? 0 : (stride_D * sizeof_D), low_D, 2);
        #endif
        
        sp_tiled_matmul_ws_simple(A, B, D == NULL ? 0x1 : D, C, 
          A_scale_factor, B_scale_factor, D_scale_factor,
          dim_I / DIM, dim_J / DIM, dim_K / DIM, 0, 0, 0,
          stride_A, stride_B, stride_D, stride_C,
          transpose_A, transpose_B,
          full_C, low_D,
          true, repeating_bias,
          act,
          1, 1);
        
        
        #ifdef STATIC_ANNOTATE
        printf("gemmini_fence();\n");
        printf("====================================\n");
        printf("ENDING ANNOTATION\n");
        printf("====================================\n");
        #endif
}

static void sp_tiled_matmul_auto_dram_spad_ws(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, uint32_t B_spad_addr_start,
        const void * D, void * C,
        int act) {
        
        size_t stride_A = dim_K;
        size_t stride_B = dim_J;
        size_t stride_D = dim_J;
        size_t stride_C = dim_J;

        scale_t A_scale_factor = MVIN_SCALE_IDENTITY;
        scale_t B_scale_factor = MVIN_SCALE_IDENTITY;
        scale_acc_t D_scale_factor = MVIN_SCALE_IDENTITY;
        acc_scale_t scale = ACC_SCALE_IDENTITY;
        acc_scale_t bert_scale = ACC_SCALE_IDENTITY;
        bool repeating_bias = false;
        bool transpose_A = false;
        bool transpose_B = false;

        bool full_C = false;
        bool low_D = 0;
        uint8_t weightA = 0;
        enum tiled_matmul_type_t tiled_matmul_type = WS;
        const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
        const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

        gemmini_extended_config_ex(WS, act & 3, 0, 1, transpose_A, transpose_B);
        gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
        gemmini_extended3_config_ld(stride_A * sizeof(elem_t), A_scale_factor, false, 0);
        gemmini_extended3_config_ld(stride_B * sizeof(elem_t), B_scale_factor, false, 1)
        gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
        
        sp_tiled_matmul_dram_spad_ws(A, B_spad_addr_start, D == NULL ? 0x1 : D, C, 
          A_scale_factor, B_scale_factor, D_scale_factor,
          dim_I / DIM, dim_J / DIM, dim_K / DIM, 0, 0, 0,
          stride_A, stride_B, stride_D, stride_C,
          transpose_A, transpose_B,
          full_C, low_D,
          true, repeating_bias,
          act,
          1, 1);
}

static void mvin_matrix(size_t I, size_t J, const elem_t * A, uint32_t spad_addr) {

  gemmini_extended3_config_ld(J * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
  for(int i = 0; i < I/DIM; i++) {
    for(int j = 0; j < J/DIM; j++) {
      gemmini_extended_mvin(&A[i*DIM*J + j*DIM], spad_addr + j*DIM + i*DIM*(J/DIM), DIM, DIM);
    }
  }
  gemmini_fence();
}

static void mvout_matrix(size_t I, size_t J, uint32_t spad_addr, elem_t * A) {

  gemmini_extended_config_st(J * sizeof(elem_t), NO_ACTIVATION, MVIN_SCALE_IDENTITY);
  for(int i = 0; i < I/DIM; i++) {
    for(int j = 0; j < J/DIM; j++) {
      gemmini_extended_mvout(&A[i*DIM*J + j*DIM], spad_addr + j*DIM + i*DIM*(J/DIM), DIM, DIM);
    }
  }
  gemmini_fence();
}

static void sp_tiled_matmul_auto_full_spad_ws(size_t dim_I, size_t dim_J, size_t dim_K,
        uint32_t A_spad_addr_start, uint32_t B_spad_addr_start,
        const void * D, uint32_t C_spad_addr_start,
        int act) {
        
        size_t stride_A = dim_K;
        size_t stride_B = dim_J;
        size_t stride_D = dim_J;
        size_t stride_C = dim_J;

        scale_t A_scale_factor = MVIN_SCALE_IDENTITY;
        scale_t B_scale_factor = MVIN_SCALE_IDENTITY;
        scale_acc_t D_scale_factor = MVIN_SCALE_IDENTITY;
        acc_scale_t scale = ACC_SCALE_IDENTITY;
        acc_scale_t bert_scale = ACC_SCALE_IDENTITY;
        bool repeating_bias = false;
        bool transpose_A = false;
        bool transpose_B = false;

        bool full_C = false;
        bool low_D = 0;
        uint8_t weightA = 0;
        enum tiled_matmul_type_t tiled_matmul_type = WS;
        const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
        const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

        gemmini_extended_config_ex(WS, act & 3, 0, 1, transpose_A, transpose_B);
        gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
        gemmini_extended3_config_ld(stride_A * sizeof(elem_t), A_scale_factor, false, 0);
        gemmini_extended3_config_ld(stride_B * sizeof(elem_t), B_scale_factor, false, 1)
        gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
        
        sp_tiled_matmul_full_spad_ws(A_spad_addr_start, B_spad_addr_start, D == NULL ? 0x1 : D, C_spad_addr_start, 
          A_scale_factor, B_scale_factor, D_scale_factor,
          dim_I / DIM, dim_J / DIM, dim_K / DIM, 0, 0, 0,
          stride_A, stride_B, stride_D, stride_C,
          transpose_A, transpose_B,
          full_C, low_D,
          true, repeating_bias,
          act,
          1, 1);
}

static void sp_tiled_matmul_layer0_ws(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        int act) {
        
gemmini_extended_config_ex(WS, 1, 0, 1, 0, 0);
gemmini_extended_config_st(256, 1, ACC_SCALE_IDENTITY);
gemmini_extended3_config_ld(64, MVIN_SCALE_IDENTITY, 0, 0);
gemmini_extended3_config_ld(256, MVIN_SCALE_IDENTITY, 0, 1);
gemmini_extended3_config_ld(1024, MVIN_SCALE_IDENTITY, 0, 2);
gemmini_extended_mvin(A + 0x0, 0x0, 64, 16);
gemmini_extended_mvin2(B + 0x0, 0x3c00, 64, 16);
gemmini_extended_preload(0x3c00, 0x80000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c10, 0x80000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c20, 0x80000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c30, 0x80000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x40, 0x3c40, 64, 16);
gemmini_extended_preload(0x3c40, 0x80000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c50, 0x80000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c60, 0x80000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c70, 0x80000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x80, 0x3c80, 64, 16);
gemmini_extended_preload(0x3c80, 0x80000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c90, 0x80000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ca0, 0x800000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3cb0, 0x800000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xc0, 0x3cc0, 64, 16);
gemmini_extended_preload(0x3cc0, 0x800000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3cd0, 0x800000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ce0, 0x800000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3cf0, 0x800000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x1000, 0x3d00, 64, 16);
gemmini_extended_preload(0x3d00, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d10, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d20, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d30, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x1040, 0x3d40, 64, 16);
gemmini_extended_preload(0x3d40, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d50, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d60, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d70, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x1080, 0x3d80, 64, 16);
gemmini_extended_preload(0x3d80, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d90, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3da0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3db0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x10c0, 0x3dc0, 64, 16);
gemmini_extended_preload(0x3dc0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3dd0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3de0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3df0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x2000, 0x3e00, 64, 16);
gemmini_extended_preload(0x3e00, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e10, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e20, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e30, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x2040, 0x3e40, 64, 16);
gemmini_extended_preload(0x3e40, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e50, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e60, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e70, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x2080, 0x3e80, 64, 16);
gemmini_extended_preload(0x3e80, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e90, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ea0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3eb0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x20c0, 0x3ec0, 64, 16);
gemmini_extended_preload(0x3ec0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ed0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ee0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ef0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x3000, 0x3f00, 64, 16);
gemmini_extended_preload(0x3f00, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f10, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f20, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f30, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvout(C + 0x0, 0xc0000000, 64, 16);
gemmini_extended_mvin2(B + 0x3040, 0x3f40, 64, 16);
gemmini_extended_preload(0x3f40, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f50, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f60, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f70, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvout(C + 0x40, 0xc0000040, 64, 16);
gemmini_extended_mvin2(B + 0x3080, 0x3f80, 64, 16);
gemmini_extended_preload(0x3f80, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f90, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3fa0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3fb0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvout(C + 0x80, 0xc0000080, 64, 16);
gemmini_extended_mvin2(B + 0x30c0, 0x3fc0, 64, 16);
gemmini_extended_preload(0x3fc0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3fd0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3fe0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ff0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvout(C + 0xc0, 0xc00000c0, 64, 16);
gemmini_fence();
}

static void sp_tiled_matmul_layer1_ws(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        int act) {

gemmini_extended_config_ex(WS, 1, 0, 1, 0, 0);
gemmini_extended_config_st(256, 1, ACC_SCALE_IDENTITY);
gemmini_extended3_config_ld(256, MVIN_SCALE_IDENTITY, 0, 0);
gemmini_extended3_config_ld(256, MVIN_SCALE_IDENTITY, 0, 1);
gemmini_extended3_config_ld(1024, MVIN_SCALE_IDENTITY, 0, 2);
gemmini_extended_mvin(A + 0x0, 0x0, 64, 16);
gemmini_extended_mvin2(B + 0x0, 0x3000, 64, 16);
gemmini_extended_preload(0x3000, 0x80000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3010, 0x80000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3020, 0x80000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3030, 0x80000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x40, 0x3040, 64, 16);
gemmini_extended_preload(0x3040, 0x80000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3050, 0x80000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3060, 0x80000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3070, 0x80000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x80, 0x3080, 64, 16);
gemmini_extended_preload(0x3080, 0x80000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3090, 0x80000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x30a0, 0x800000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x30b0, 0x800000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xc0, 0x30c0, 64, 16);
gemmini_extended_preload(0x30c0, 0x800000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x30d0, 0x800000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x30e0, 0x800000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x30f0, 0x800000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x1000, 0x3100, 64, 16);
gemmini_extended_preload(0x3100, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3110, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3120, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3130, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x1040, 0x3140, 64, 16);
gemmini_extended_preload(0x3140, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3150, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3160, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3170, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x1080, 0x3180, 64, 16);
gemmini_extended_preload(0x3180, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3190, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x31a0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x31b0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x10c0, 0x31c0, 64, 16);
gemmini_extended_preload(0x31c0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x31d0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x31e0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x31f0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x2000, 0x3200, 64, 16);
gemmini_extended_preload(0x3200, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3210, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3220, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3230, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x2040, 0x3240, 64, 16);
gemmini_extended_preload(0x3240, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3250, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3260, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3270, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x2080, 0x3280, 64, 16);
gemmini_extended_preload(0x3280, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3290, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x32a0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x32b0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x20c0, 0x32c0, 64, 16);
gemmini_extended_preload(0x32c0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x32d0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x32e0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x32f0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x3000, 0x3300, 64, 16);
gemmini_extended_preload(0x3300, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3310, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3320, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3330, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x3040, 0x3340, 64, 16);
gemmini_extended_preload(0x3340, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3350, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3360, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3370, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x3080, 0x3380, 64, 16);
gemmini_extended_preload(0x3380, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3390, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x33a0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x33b0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x30c0, 0x33c0, 64, 16);
gemmini_extended_preload(0x33c0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x33d0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x33e0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x33f0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin(A + 0x40, 0x40, 64, 16);
gemmini_extended_mvin2(B + 0x4000, 0x3400, 64, 16);
gemmini_extended_preload(0x3400, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3410, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3420, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3430, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x4040, 0x3440, 64, 16);
gemmini_extended_preload(0x3440, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3450, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3460, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3470, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x4080, 0x3480, 64, 16);
gemmini_extended_preload(0x3480, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3490, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x34a0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x34b0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x40c0, 0x34c0, 64, 16);
gemmini_extended_preload(0x34c0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x34d0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x34e0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x34f0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x5000, 0x3500, 64, 16);
gemmini_extended_preload(0x3500, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3510, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3520, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3530, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x5040, 0x3540, 64, 16);
gemmini_extended_preload(0x3540, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3550, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3560, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3570, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x5080, 0x3580, 64, 16);
gemmini_extended_preload(0x3580, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3590, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x35a0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x35b0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x50c0, 0x35c0, 64, 16);
gemmini_extended_preload(0x35c0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x35d0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x35e0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x35f0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x6000, 0x3600, 64, 16);
gemmini_extended_preload(0x3600, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3610, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3620, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3630, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x6040, 0x3640, 64, 16);
gemmini_extended_preload(0x3640, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3650, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3660, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3670, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x6080, 0x3680, 64, 16);
gemmini_extended_preload(0x3680, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3690, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x36a0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x36b0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x60c0, 0x36c0, 64, 16);
gemmini_extended_preload(0x36c0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x36d0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x36e0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x36f0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x7000, 0x3700, 64, 16);
gemmini_extended_preload(0x3700, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3710, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3720, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3730, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x7040, 0x3740, 64, 16);
gemmini_extended_preload(0x3740, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3750, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3760, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3770, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x7080, 0x3780, 64, 16);
gemmini_extended_preload(0x3780, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3790, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x37a0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x37b0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x70c0, 0x37c0, 64, 16);
gemmini_extended_preload(0x37c0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x37d0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x37e0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x37f0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin(A + 0x80, 0x80, 64, 16);
gemmini_extended_mvin2(B + 0x8000, 0x3800, 64, 16);
gemmini_extended_preload(0x3800, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3810, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3820, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3830, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x8040, 0x3840, 64, 16);
gemmini_extended_preload(0x3840, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3850, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3860, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3870, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x8080, 0x3880, 64, 16);
gemmini_extended_preload(0x3880, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3890, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x38a0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x38b0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x80c0, 0x38c0, 64, 16);
gemmini_extended_preload(0x38c0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x38d0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x38e0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x38f0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x9000, 0x3900, 64, 16);
gemmini_extended_preload(0x3900, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3910, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3920, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3930, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x9040, 0x3940, 64, 16);
gemmini_extended_preload(0x3940, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3950, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3960, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3970, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x9080, 0x3980, 64, 16);
gemmini_extended_preload(0x3980, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3990, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x39a0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x39b0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x90c0, 0x39c0, 64, 16);
gemmini_extended_preload(0x39c0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x39d0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x39e0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x39f0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xa000, 0x3a00, 64, 16);
gemmini_extended_preload(0x3a00, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3a10, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3a20, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3a30, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xa040, 0x3a40, 64, 16);
gemmini_extended_preload(0x3a40, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3a50, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3a60, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3a70, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xa080, 0x3a80, 64, 16);
gemmini_extended_preload(0x3a80, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3a90, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3aa0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ab0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xa0c0, 0x3ac0, 64, 16);
gemmini_extended_preload(0x3ac0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ad0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ae0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3af0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xb000, 0x3b00, 64, 16);
gemmini_extended_preload(0x3b00, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3b10, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3b20, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3b30, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xb040, 0x3b40, 64, 16);
gemmini_extended_preload(0x3b40, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3b50, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3b60, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3b70, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xb080, 0x3b80, 64, 16);
gemmini_extended_preload(0x3b80, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3b90, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ba0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3bb0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xb0c0, 0x3bc0, 64, 16);
gemmini_extended_preload(0x3bc0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3bd0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3be0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3bf0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin(A + 0xc0, 0xc0, 64, 16);
gemmini_extended_mvin2(B + 0xc000, 0x3c00, 64, 16);
gemmini_extended_preload(0x3c00, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c10, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c20, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c30, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xc040, 0x3c40, 64, 16);
gemmini_extended_preload(0x3c40, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c50, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c60, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c70, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xc080, 0x3c80, 64, 16);
gemmini_extended_preload(0x3c80, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3c90, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ca0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3cb0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xc0c0, 0x3cc0, 64, 16);
gemmini_extended_preload(0x3cc0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3cd0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ce0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3cf0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xd000, 0x3d00, 64, 16);
gemmini_extended_preload(0x3d00, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d10, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d20, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d30, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xd040, 0x3d40, 64, 16);
gemmini_extended_preload(0x3d40, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d50, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d60, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d70, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xd080, 0x3d80, 64, 16);
gemmini_extended_preload(0x3d80, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3d90, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3da0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3db0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xd0c0, 0x3dc0, 64, 16);
gemmini_extended_preload(0x3dc0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3dd0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3de0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3df0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xe000, 0x3e00, 64, 16);
gemmini_extended_preload(0x3e00, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e10, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e20, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e30, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xe040, 0x3e40, 64, 16);
gemmini_extended_preload(0x3e40, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e50, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e60, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e70, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xe080, 0x3e80, 64, 16);
gemmini_extended_preload(0x3e80, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3e90, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ea0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3eb0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xe0c0, 0x3ec0, 64, 16);
gemmini_extended_preload(0x3ec0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ed0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ee0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ef0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xf000, 0x3f00, 64, 16);
gemmini_extended_preload(0x3f00, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f10, 0xc0000010, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f20, 0xc0000020, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f30, 0xc0000030, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvout(C + 0x0, 0xc0000000, 64, 16);
gemmini_extended_mvin2(B + 0xf040, 0x3f40, 64, 16);
gemmini_extended_preload(0x3f40, 0xc0000040, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f50, 0xc0000050, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f60, 0xc0000060, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f70, 0xc0000070, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvout(C + 0x40, 0xc0000040, 64, 16);
gemmini_extended_mvin2(B + 0xf080, 0x3f80, 64, 16);
gemmini_extended_preload(0x3f80, 0xc0000080, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3f90, 0xc0000090, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3fa0, 0xc00000a0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3fb0, 0xc00000b0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvout(C + 0x80, 0xc0000080, 64, 16);
gemmini_extended_mvin2(B + 0xf0c0, 0x3fc0, 64, 16);
gemmini_extended_preload(0x3fc0, 0xc00000c0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3fd0, 0xc00000d0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3fe0, 0xc00000e0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_preload(0x3ff0, 0xc00000f0, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvout(C + 0xc0, 0xc00000c0, 64, 16);
gemmini_fence();

}

static void sp_tiled_matmul_layer2_ws(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        int act) {

gemmini_extended_config_ex(WS, 1, 0, 1, 0, 0);
gemmini_extended_config_st(16, 1, ACC_SCALE_IDENTITY);
gemmini_extended3_config_ld(256, MVIN_SCALE_IDENTITY, 0, 0);
gemmini_extended3_config_ld(16, MVIN_SCALE_IDENTITY, 0, 1);
gemmini_extended3_config_ld(64, MVIN_SCALE_IDENTITY, 0, 2);
gemmini_extended_mvin(A + 0x0, 0x0, 64, 16);
gemmini_extended_mvin2(B + 0x0, 0x3f00, 16, 16);
gemmini_extended_preload(0x3f00, 0x80000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x100, 0x3f10, 16, 16);
gemmini_extended_preload(0x3f10, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x10, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x200, 0x3f20, 16, 16);
gemmini_extended_preload(0x3f20, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x20, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x300, 0x3f30, 16, 16);
gemmini_extended_preload(0x3f30, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x30, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin(A + 0x40, 0x40, 64, 16);
gemmini_extended_mvin2(B + 0x400, 0x3f40, 16, 16);
gemmini_extended_preload(0x3f40, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x40, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x500, 0x3f50, 16, 16);
gemmini_extended_preload(0x3f50, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x50, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x600, 0x3f60, 16, 16);
gemmini_extended_preload(0x3f60, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x60, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x700, 0x3f70, 16, 16);
gemmini_extended_preload(0x3f70, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x70, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin(A + 0x80, 0x80, 64, 16);
gemmini_extended_mvin2(B + 0x800, 0x3f80, 16, 16);
gemmini_extended_preload(0x3f80, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x80, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0x900, 0x3f90, 16, 16);
gemmini_extended_preload(0x3f90, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0x90, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xa00, 0x3fa0, 16, 16);
gemmini_extended_preload(0x3fa0, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xa0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xb00, 0x3fb0, 16, 16);
gemmini_extended_preload(0x3fb0, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xb0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin(A + 0xc0, 0xc0, 64, 16);
gemmini_extended_mvin2(B + 0xc00, 0x3fc0, 16, 16);
gemmini_extended_preload(0x3fc0, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xc0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xd00, 0x3fd0, 16, 16);
gemmini_extended_preload(0x3fd0, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xd0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xe00, 0x3fe0, 16, 16);
gemmini_extended_preload(0x3fe0, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xe0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvin2(B + 0xf00, 0x3ff0, 16, 16);
gemmini_extended_preload(0x3ff0, 0xc0000000, 16, 16, 16, 16);
gemmini_extended_compute_preloaded(0xf0, 0xffffffff, 16, 16, 16, 16);
gemmini_extended_mvout(C + 0x0, 0xc0000000, 16, 16);
gemmini_fence();

}


#endif // SRC_MAIN_C_GEMMINI_LAB2_H

