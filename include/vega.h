// See LICENSE for license details.

#ifndef SRC_MAIN_C_VEGA_H
#define SRC_MAIN_C_VEGA_H

#undef abs

#define FINE_ISA 0

#define k_CONFIG_VEGA 24
#define k_LOOP_VEGA 25
#define k_LOOP_VEGA_CONFIG_BOUNDS 26
#define k_LOOP_VEGA_CONFIG_ADDRS_AB 27
#define k_LOOP_VEGA_CONFIG_ADDRS_DC 28


#define VEGA_BANK_NUM 2
// assume gemmini has 2 acc banks (todo: parameterize)
#define VEGA_ACC_ROWS (ACC_ROWS / 2)

// mvin and mvout
#define vega_extended_mvin(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN)

#define vega_extended_mvin2(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN2)

#define vega_extended_mvin3(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN3)

#define vega_block_mvin(dram_addr, spad_addr, len) \
  vega_extended_mvin(dram_addr, spad_addr, (len) * DIM, DIM)

#define vega_mvin(dram_addr, spad_addr) \
  vega_extended_mvin(dram_addr, spad_addr, DIM, DIM)

#define vega_extended_mvout(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(spad_addr), k_MVOUT)

#define vega_mvout(dram_addr, spad_addr) \
  vega_extended_mvout(dram_addr, spad_addr, DIM, DIM)

// compute
#define vega_extended_compute_preloaded(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_PRELOADED)

#define vega_extended_compute_accumulated(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_ACCUMULATE)

#define vega_compute_preloaded(A, BD) \
  vega_extended_compute_preloaded(A, BD, DIM, DIM, DIM, DIM)

#define vega_compute_accumulated(A, BD) \
  vega_extended_compute_accumulated(A, BD, DIM, DIM, DIM, DIM)

// preload
#define vega_extended_preload(BD, C, BD_cols, BD_rows, C_cols, C_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), ((uint64_t)(C_rows) << (ADDR_LEN + 16)) | ((uint64_t)(C_cols) << ADDR_LEN) | (uint64_t)(C), k_PRELOAD)

#define vega_preload(BD, C) \
  vega_extended_preload(BD, C, DIM, DIM, DIM, DIM)

#define vega_preload_zeros(C) \
  vega_preload(GARBAGE_ADDR, C)

// config
#define vega_extended3_config_ex(dataflow, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride, A_transpose, B_transpose, set_only_strides) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)sys_acc_scale) << 32) | ((uint64_t)(A_stride) << 16) | (B_transpose << 9) | (A_transpose << 8) | ((set_only_strides) << 7) | ((sys_act) << 3) | ((dataflow) << 2) | CONFIG_EX, ((uint64_t)(C_stride) << 48) | (sys_shift), k_CONFIG_VEGA); \

#define vega_extended2_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  vega_extended3_config_ex(dataflow, sys_act, sys_shift, ACC_SCALE_IDENTITY, 1, A_stride, A_transpose, B_transpose, false)

#define vega_extended_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  vega_extended2_config_ex(dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose)

#define vega_config_ex(dataflow, sys_act, sys_shift) \
    vega_extended_config_ex(dataflow, sys_act, sys_shift, 1, 0, 0)

// Note: The "pixel_repeats" parameter below is still experimental, andthere is
// a high chance that it will be removed in future releases.
#define vega_extended5_config_ld(stride, scale, shrunk, block_mvin_stride, pixel_repeats, id) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(block_mvin_stride) << 16) | ((uint64_t)(pixel_repeats) << 8) | ((id) << 3) | ((shrunk) << 2) | CONFIG_LD, stride, k_CONFIG)

#define vega_extended4_config_ld(stride, scale, shrunk, block_mvin_stride, id) \
  vega_extended5_config_ld(stride, scale, shrunk, block_mvin_stride, 1, id) \

#define vega_extended3_config_ld(stride, scale, shrunk, id) \
  vega_extended4_config_ld(stride, scale, shrunk, DIM, id)

#define vega_extended2_config_ld(stride, scale, shrunk) \
  vega_extended3_config_ld(stride, scale, shrunk, 0)

#define vega_extended_config_ld(stride, scale) \
  vega_extended2_config_ld(stride, scale, false)

#define vega_config_ld(stride) \
  vega_extended_config_ld(stride, MVIN_SCALE_IDENTITY)

#define vega_extended2_config_st(stride, acc_act, acc_scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, upad, lpad) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(ocols) << 56) | ((uint64_t)(orows) << 48) | ((uint64_t)(pocols) << 40) | ((uint64_t)(porows) << 32) | ((uint64_t)(pool_out_dim) << 24) | ((uint64_t)(lpad) << 10) | ((uint64_t)(upad) << 8) | ((uint64_t)(pool_size) << 6) | ((uint64_t)(pool_stride) << 4) | ((uint64_t)(acc_act) << 2) | CONFIG_ST, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)acc_scale) << 32) | ((uint32_t)stride), k_CONFIG)

#define vega_extended_config_st(stride, acc_act, acc_scale) \
    vega_extended2_config_st(stride, acc_act, acc_scale, 0, 0, 0, 0, 0, 0, 0, 0, 0)

#define vega_config_st(stride) \
    vega_extended_config_st(stride, NO_ACTIVATION, ACC_SCALE_IDENTITY)

// flush
#define vega_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

// fence
#define vega_fence() asm volatile("fence")

// weight-stationary gemv loop
#define vega_loop_ws(I, K, pad_I, pad_K, A, B, D, C, A_stride, full_C, low_D, ex_accumulate, act, a_spad_id, b_spad_id, is_add) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | (uint64_t)(I), k_LOOP_VEGA_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_VEGA_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_VEGA_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(is_add) << 32) | ((uint64_t)(a_spad_id) << 18) | ((uint64_t)(b_spad_id) << 16) | ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), A_stride, k_LOOP_VEGA) \
  }

#define vega_clock_gate(entire_en, gemmini_en, vega_en) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(vega_en) << 2) | ((uint64_t)(gemmini_en) << 1) | ((uint64_t) entire_en), 0, k_CLK_GATE)


static void sp_tiled_vector_scale(const size_t I, 
        const scale_t A_scale,
        const elem_t * A, elem_t * C,
        //size_t A_row_stride,
        bool relu) {

    int pad_I = ((I%DIM) == 0) ? 0 : DIM - (I % DIM);
    int tile_I = (I%DIM == 0) ? (int)(I/DIM) : (int)(I/DIM) + 1;
//    printf("I: %d, pad I: %d, tile_I: %d\n", I, pad_I, tile_I);
    vega_loop_ws(1, tile_I, pad_I, pad_I, NULL, A, NULL, C, 0, false, false, false, relu, 0, 0, true);
/*
    vega_fence();
    const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);// | (full_C << (ADDR_LEN-3));
    for(int i = 0; i < I; i += DIM){
        size_t rows = 1;
        size_t cols = i + DIM > I ? DIM : DIM - pad_I;
        void * const C_dram_addr = (int8_t*)C + i*sizeof(elem_t);
        uint32_t C_sp_addr = C_sp_addr_start + (i / DIM);
        printf("i: %d, cols: %d, c_dram_addr: 0x%08lx, C_sp_addr: %d\n", i, cols, C_dram_addr, C_sp_addr);
        vega_extended_mvout(C_dram_addr, C_sp_addr, cols, rows);
        vega_fence();
    }
*/    
}

static void tiled_vector_scale(const size_t dim_I, const scale_t A_scale, 
    elem_t * A, elem_t* C, bool relu){

  vega_extended_config_ex(WEIGHT_STATIONARY, 0 , 0, 1, false, false);// a_transpose, b_transpose);

  vega_extended_config_st(DIM * sizeof(elem_t), 0, 1);
  vega_extended4_config_ld(0 * sizeof(elem_t), A_scale, true, 1, 1);
  //vega_extended3_config_ld(0 * sizeof(elem_t), A_scale, false, 1);
  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  size_t max_acc_rows = VEGA_ACC_ROWS / 2;
  size_t tile_I = dim_I;
  if (max_acc_rows > dim_I_padded / DIM){
      tile_I = max_acc_rows * DIM;
  }
  for(int i = 0; i < dim_I; i += tile_I){
      elem_t * a = A + i;
      elem_t * c = C + i;
      size_t I_tile = i + tile_I <= dim_I ? tile_I : dim_I - i;
      sp_tiled_vector_scale(I_tile, A_scale, a, c, relu);
  }
  vega_fence();
}


static void sp_tiled_gemv(const elem_t * A, const elem_t * B,
        const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t K, size_t pad_I, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        int act,
        int a_spad_id, int b_spad_id) {

#if FINE_ISA == 1
  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = VEGA_BANK_NUM * BANK_ROWS - K;// - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2) | (full_C << (ADDR_LEN-3));
  const int A_blocks = a_transpose ? (I <= MAX_BLOCK_LEN ? I : MAX_BLOCK_LEN) :
    (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN);
  const int B_blocks = (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN);
    //(J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  const int D_blocks = 1;//low_D ? (I <= MAX_BLOCK_LEN ? I : MAX_BLOCK_LEN) :
    //(I <= MAX_BLOCK_LEN_ACC ? I : MAX_BLOCK_LEN_ACC);
  const int C_blocks = 1;//full_C ? 1 : (I <= MAX_BLOCK_LEN ? I : MAX_BLOCK_LEN);
  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);
  // Move-in D
  if (D != NULL && !no_bias) {
    //for (size_t j = 0; j < J; j++) {
      const size_t rows = 1;//DIM - (j == J-1 ? pad_J : 0);
      for (size_t i = 0; i < I; i += D_blocks) {
        //const size_t bias_row = repeating_bias ? 0 : j;
        const void * const D_dram_addr = (int8_t *)D + i*DIM*sizeof_D;
        const uint32_t D_sp_addr_acc = D_sp_addr_start + i;//i*DIM;
        size_t blocks = i + D_blocks <= I ? D_blocks : I-i;
        const size_t cols = blocks * DIM - (i + blocks >= I ? pad_I : 0);
        vega_extended_mvin3(D_dram_addr, D_sp_addr_acc, cols, rows);
      }
    //}
  }
  for (size_t k = 0; k < K; k++) {
    for (size_t i = 0; i < I; i++) {
      const uint32_t A_sp_addr = a_transpose ? (A_sp_addr_start + (k*I + i)*DIM) :
        (A_sp_addr_start + (i*K + k)*DIM);
      //const uint32_t B_sp_addr = b_transpose ? (B_sp_addr_start + k*DIM) :
      //  B_sp_addr_start + k;//(k*J + j)*DIM);
      const uint32_t B_sp_addr = B_sp_addr_start + k;
      const uint32_t C_sp_addr = C_sp_addr_start + i;
       
      //vega_fence();

      //printf("mvin A\n");
      // Mvin A
      
      if (a_transpose) {
        if (i % A_blocks == 0) {
          const elem_t * const A_dram_addr = A + (k*A_row_stride + i)*DIM;
          const size_t blocks = i + A_blocks <= I ? A_blocks : I-i;
          const size_t cols = blocks * DIM - (i + blocks >= I ? pad_I : 0);
          const size_t rows = DIM - (k == K-1 ? pad_K : 0);
          vega_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
        }
      } else {
        if (k % A_blocks == 0) {
          const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
          const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
          const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
          const size_t rows = DIM - (i == I-1 ? pad_I : 0);
          vega_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
        }
      }
      
      // Mvin B
        //if (i == 0 && k % DIM == 0){
          //printf("mvin B\n");
        if (i == 0 && k % B_blocks == 0) {
          const elem_t * const B_dram_addr = B + k*DIM;
          const size_t blocks = k + B_blocks <= K ? B_blocks : K-k;
          const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
          const size_t rows = 1;//(k + DIM > K) ? K%DIM : DIM;
          //printf("cols: %d, rows: %d, blocks: %d, B_sp_addr: %d\n", cols, rows, blocks, B_sp_addr);
          vega_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
        }
    
      //vega_fence();
      //vega_fence();
      // Compute
      {
        //printf("compute\n");
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
        const size_t B_cols = 1;//DIM - (j == J - 1 ? pad_J : 0);
        const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
        const size_t C_cols = 1;//DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);
        //printf("out_sp_addr: 0x%08lx\n", out_sp_addr);
        //printf("A_sp_addr: 0x%08lx, pre_sp_addr: 0x%08lx, out_sp_addr: 0x%08lx, b rows: %d, b cols: %d\n", A_sp_addr, pre_sp_addr, out_sp_addr, B_rows, B_cols);
        vega_extended_preload(pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);
        if (i == 0) { // First iteration
          vega_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
        } else { // All other iterations
          vega_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
        }
      }
      //vega_fence();
      //printf("done compute\n");
    
      /*
       // other mvout version
      if (C != NULL && k == K-1 &&(i == I-1 || i % C_blocks == C_blocks-1)) {
        //printf("mvout \n");
        // Move-out C (if not normalizing)
        const size_t rounded_i = (i / C_blocks) * C_blocks;
        const uint32_t rounded_C_sp_addr = C_sp_addr_start + rounded_i;//(rounded_i)*DIM;
        //printf("spad addr: 0x%08lx\n", rounded_C_sp_addr);
        void * const C_dram_addr = (int8_t*)C + (rounded_i)*DIM*sizeof_C;
        const size_t blocks = rounded_i + C_blocks <= I ? C_blocks : I-rounded_i;
        const size_t cols = blocks * DIM - (rounded_i + blocks >= I ? pad_I : 0);
        const size_t rows = 1;//DIM - (j == J - 1 ? pad_J : 0);
        vega_extended_mvout(C_dram_addr, rounded_C_sp_addr, cols, rows);
      }
        */  
      // make C_stride DIM
      if (C != NULL && k == K-1 &&(i == I-1 || i % DIM == DIM-1)) {
       // vega_fence();
       // printf("mvout \n");
        // Move-out C (if not normalizing)
        const size_t rounded_i = (i / DIM) * DIM;
        const uint32_t rounded_C_sp_addr = C_sp_addr_start + rounded_i;//(rounded_i)*DIM;
        //printf("spad addr: 0x%08lx\n", rounded_C_sp_addr);
        void * const C_dram_addr = (int8_t*)C + (rounded_i)*DIM*sizeof_C;
        const size_t blocks = rounded_i + C_blocks <= I ? C_blocks : I-rounded_i;
        const size_t cols = DIM;//blocks * DIM - (rounded_i + blocks >= I ? pad_I : 0);
        size_t rows = i == I-1 ? I % DIM : DIM;
        
        //if(pad_I != 0 && i == I - 1){
        //  if(rows > 1) {
        //    rows -= 1;
        //    vega_extended_mvout((int8_t*)C + (i)*DIM*sizeof_C, C_sp_addr_start + i, DIM - pad_I, 1);
        //  }
        //  if(rows > 0) vega_extended_mvout(C_dram_addr, rounded_C_sp_addr, cols, rows);
        //}
        //else
        
          vega_extended_mvout(C_dram_addr, rounded_C_sp_addr, cols, rows);
      }    
    }
    
  }
    
#else
  // Combined loop
  vega_loop_ws(I, K, pad_I, pad_K, A, B, no_bias ? NULL : D, C,
    A_row_stride, full_C, low_D, !no_bias || D == NULL, act, a_spad_id, b_spad_id, false);
  //vega_fence();
  //printf("done\n");
#endif
}


static void tiled_gemv_outer(size_t dim_I, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t tile_I, size_t tile_K,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        uint8_t weightA,
        int dataflow) {

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  //const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const size_t I0 = dim_I_padded / (tile_I*DIM) + (dim_I_padded % (tile_I*DIM) != 0);
  //const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
  const size_t K0 = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (tile_I*DIM) == 0 ? tile_I : (dim_I_padded/DIM) % tile_I;
  //const size_t last_J = dim_J_padded % (tile_J*DIM) == 0 ? tile_J : (dim_J_padded/DIM) % tile_J;
  const size_t last_K = dim_K_padded % (tile_K*DIM) == 0 ? tile_K : (dim_K_padded/DIM) % tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - dim_I;
  const size_t padding_J = DIM - 1;//dim_J_padded - dim_J;
  const size_t padding_K = dim_K_padded - dim_K;

  const bool no_bias = D == NULL;

  if (no_bias) {
    D = (void*) 1; // Dummy address which isn't NULL
  }

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  vega_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose, b_transpose);
  vega_extended_config_st(DIM * sizeof_C, act & 3, scale);
  vega_extended3_config_ld(stride_A * sizeof(elem_t), A_scale_factor, false, 0);
  //vega_extended3_config_ld(0 * sizeof(elem_t), B_scale_factor, false, 1);
  vega_extended4_config_ld(0 * sizeof(elem_t), B_scale_factor, false, 1, 1);
  vega_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);

  // reuse operand if it fits scratchpad
  int a_spad_id = 0;
  int b_spad_id = 0;
  bool b_reuse = FINE_ISA == 1 ? false : K0 <= 2;
  //bool b_reuse = (J0 * K0 <= 2) && (dataflow == WEIGHT_STATIONARY);
  bool a_reuse = false;//(I0 * K0 <= 2) && (dataflow == WEIGHT_STATIONARY);

  for (size_t i0 = 0; i0 < I0; i0++)
      for (size_t k0 = 0; k0 < K0; k0++) {
        if(a_reuse)
          a_spad_id = ((i0+k0) == 0) ? 1 : 2;
        if(b_reuse)
          b_spad_id = ((k0) == 0) ? 1 : 2;

        const void * pre;
        if (k0 != 0) {
          pre = NULL;
        } else {
          size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
          // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
          pre = (int8_t*)D + (bias_row)*sizeof_D;
        }

        //void * out = k0 == K0-1 ? (int8_t*)C + (i0*tile_I*DIM*stride_C + j0*tile_J*DIM)*sizeof_C : NULL;
        void * out = k0 == K0-1 ? (int8_t*) C + (i0*tile_I*DIM) * sizeof_C : NULL;
        const size_t I = i0 < I0-1 ? tile_I : last_I;
        //const size_t J = j0 < J0-1 ? tile_J : last_J;
        const size_t K = k0 < K0-1 ? tile_K : last_K;

        const size_t pad_I = i0 == I0-1 ? padding_I : 0;
        const size_t pad_J = padding_J;//j0 == J0-1 ? padding_J : 0;
        const size_t pad_K = k0 == K0-1 ? padding_K : 0;

        const elem_t * a = a_transpose ? (A + k0*tile_K*DIM*stride_A + i0*tile_I*DIM)
          : (A + i0*tile_I*DIM*stride_A + k0*tile_K*DIM);

        const elem_t * b = (B + k0*tile_K*DIM);
          //: (B + k0*tile_K*DIM*stride_B + j0*tile_J*DIM);

        //if(a_reuse && j0 >= 1) a = NULL;
        if(b_reuse && i0 >= 1) b = NULL;
        //printf("i0: %d, k0: %d, I: %d, K: %d, out offset: %d\n", i0, k0, I, K, i0*tile_I*DIM);
        //printf("a_reuse: %d, b_reuse: %d, a_spad_id: %d, b_spad_id: %d, a: %llu, b: %llu \n", a_reuse, b_reuse, a_spad_id, b_spad_id, a, b);
        sp_tiled_gemv(a, b, pre, out,
            A_scale_factor, B_scale_factor, D_scale_factor,
            I, K,
            pad_I, pad_K,
            stride_A, stride_B, stride_D, stride_C,
            a_transpose, b_transpose,
            full_C, low_D,
            no_bias, repeating_bias,
            act, a_spad_id, b_spad_id);
      }

  vega_fence();
}

/*
#ifdef HAS_MVIN_SCALE
#define VEGA_SCALE(x, scale) MVIN_SCALE((x), (scale))
#else
#define VEGA_SCALE(x, scale) (x)
#endif

#ifdef HAS_MVIN_ACC_SCALE
#define VEGA_ACC_SCALE(x, scale) MVIN_SCALE_ACC((x), (scale))
#else
#define VEGA_ACC_SCALE(x, scale) (x)
#endif
*/

// This function runs a tiled matrix mulctiplication, with hardcoded tiling
// factors
static void tiled_gemv(size_t dim_I, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        size_t tile_I, size_t tile_K,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {

  size_t dim_J = 1;
#ifdef VEGA_ASSERTIONS
  // Make sure that the tiling factors make sense
  if (tile_I <= 0) {
    printf("tile_I is non-positive\n");
    exit(1);
  } else if (tile_K <= 0) {
    printf("tile_K is non-positive\n");
    exit(1);
  }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  //const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  if (tile_I * DIM > dim_I_padded) {
    printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
    exit(1);
  } else if (tile_K * DIM > dim_K_padded) {
    printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
    exit(1);
  }

  //const bool double_buffered = true;
  const bool double_buffered = FINE_ISA == 1 ? false : true;
  const size_t total_spad_size = double_buffered ? VEGA_BANK_NUM * BANK_ROWS / 2 :
      VEGA_BANK_NUM * BANK_ROWS;
  const size_t total_acc_size = double_buffered ? VEGA_ACC_ROWS / 2 : VEGA_ACC_ROWS;

  const size_t total_spad_rows =
      (tile_I * tile_K * DIM) +   // Rows to store A
      (tile_K);    // Rows to store B

  if (total_spad_rows > total_spad_size) {
    printf("Not enough space in scratchpad to store A and B matrices\n");
    exit(1);
  }

  const size_t total_acc_rows =
      tile_I;      // Rows to store C

  if (total_acc_rows > total_acc_size) {
    printf("Not enough space in accumulator to store C\n");
    exit(1);
  }

  if (tile_I > 65535 || tile_K > 65535) {
    printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
    exit(1);
  }

  char gemv_type_str[][4] = {"OS", "WS", "CPU"};

  // Check if transpose options are correct
  if (((tiled_matmul_type == OS) && (transpose_A || transpose_B)) ||
    (tiled_matmul_type == WS && transpose_A && transpose_B)) {
    printf("Not implemented: %s gemv, a_transpose=%d, b_transpose=%d\n", gemv_type_str[tiled_matmul_type], transpose_A, transpose_B);
    exit(1);
  }

  // Check if full_C options are correct
  if ((tiled_matmul_type == CPU && (full_C || low_D)) ||
      (tiled_matmul_type == OS && low_D)) {
    printf("Not implemented: %s gemv, full_C=%d, low_D=%d\n", gemv_type_str[tiled_matmul_type], full_C, low_D);
  }

#endif

  // Run a tiled matrix multiplication on either vega or the CPU
  //if (tiled_matmul_type == OS || tiled_matmul_type == WS) {
    tiled_gemv_outer(dim_I, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        tile_I, tile_K,
        act, scale, bert_scale, repeating_bias,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        (int)tiled_matmul_type);
  //} 
/*
  else  {
    gemv_cpu(transpose_A, transpose_B, dim_I, dim_K,
            A, B, (const acc_t*) D, (elem_t*)C,
            stride_A, stride_B, stride_D, stride_C,
            A_scale_factor, B_scale_factor, D_scale_factor,
            act, scale, bert_scale, repeating_bias);
  }
  */
}


static size_t tiled_gemv_total_spad_rows(size_t I, size_t K) {
  return (I * K) * DIM + K;
}


static size_t tiled_gemv_total_acc_rows(size_t I) {
  return I;//(I * J) * DIM;
}

// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
static void tiled_gemv_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {

#define partition_rows (VEGA_BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (VEGA_ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

    // "db_" means "double-buffered"
#define db_partition_rows ((VEGA_BANK_NUM * BANK_ROWS / 2) / 2)
#define db_mats_in_partition (db_partition_rows / DIM)
#define db_mats_in_acc ((VEGA_ACC_ROWS / 2) / DIM)
#define db_max_tile_i_j db_mats_in_acc //((size_t)sqrt(db_mats_in_acc))
#define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)

    //printf("db max tile_i_j: %d, db_max_tile_k: %d\n", db_max_tile_i_j, db_max_tile_k);
    const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
    //const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
    const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

    //const bool double_buffered = true;
    const bool double_buffered = FINE_ISA == 1 ? false : true;//tiled_matmul_type == WS;

    const size_t max_spad_rows = double_buffered ? VEGA_BANK_NUM * BANK_ROWS / 2 :
      VEGA_BANK_NUM * BANK_ROWS;
    const size_t max_acc_rows = double_buffered ? VEGA_ACC_ROWS / 2 : VEGA_ACC_ROWS;

    size_t tile_I, tile_K;

    if (double_buffered) {
       tile_I = dim_I_padded/DIM < db_max_tile_i_j ? dim_I_padded/DIM : db_max_tile_i_j;
       //tile_J = dim_J_padded/DIM < db_max_tile_i_j ? dim_J_padded/DIM : db_max_tile_i_j;
       tile_K = dim_K_padded/DIM < db_max_tile_k ? dim_K_padded/DIM : db_max_tile_k;
    } else {
       tile_I = dim_I_padded/DIM < max_tile_i_j ? dim_I_padded/DIM : max_tile_i_j;
       //tile_J = dim_J_padded/DIM < max_tile_i_j ? dim_J_padded/DIM : max_tile_i_j;
       tile_K = dim_K_padded/DIM < max_tile_k ? dim_K_padded/DIM : max_tile_k;
    }

    // Fill scratchpad as much as possible
    while (true) {
      bool increased = false;

      if (tiled_gemv_total_spad_rows(tile_I+1, tile_K) <= max_spad_rows &&
          tiled_gemv_total_acc_rows(tile_I+1) <= max_acc_rows &&
          (tile_I+1) * DIM <= dim_I_padded) {
        tile_I++;
        increased = true;
      }

      if (tiled_gemv_total_spad_rows(tile_I, tile_K+1) <= max_spad_rows &&
          (tile_K+1) * DIM <= dim_K_padded) {
        tile_K++;
        increased = true;
      }

      if (!increased)
        break;
    }

#ifdef PRINT_TILE
#if PRINT_TILE
    const int spad_rows = tiled_gemv_total_spad_rows(tile_I, tile_K);
    const int acc_rows = tiled_gemv_total_acc_rows(tile_I);

    printf("tile_I: %d\n", tile_I);
    //printf("tile_J: %d\n", tile_J);
    printf("tile_K: %d\n\n", tile_K);

    printf("spad_rows: %d\n", spad_rows);
    printf("acc_rows: %d\n\n", acc_rows);

    printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
    printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

    //exit(EXIT_SUCCESS);
#endif
#endif

    tiled_gemv(dim_I, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        act, scale, bert_scale, repeating_bias,
        tile_I, tile_K,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        tiled_matmul_type);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
#undef db_partition_rows
#undef db_mats_in_partition
#undef db_mats_in_acc
#undef db_max_tile_i_j
#undef db_max_tile_k
}

#undef abs

#endif // SRC_MAIN_C_VEGA_H

