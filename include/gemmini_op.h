// See LICENSE for license details.
#ifndef BAREMETAL
#define OP0 0
#define OP1 1
#define OP2 2
#define OP3 3
#else
#define OP0 3
#define OP1 3
#define OP2 3
#define OP3 3
#endif


// mvin and mvout
#define gemmini_opcode_extended_mvin(OPCODE, dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(OPCODE, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN)

#define gemmini_opcode_extended_mvin2(OPCODE, dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(OPCODE, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN2)

#define gemmini_opcode_extended_mvin3(OPCODE, dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(OPCODE, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN3)

#define gemmini_opcode_block_mvin(OPCODE, dram_addr, spad_addr, len) \
  gemmini_extended_mvin(OPCODE, dram_addr, spad_addr, (len) * DIM, DIM)

#define gemmini_opcode_mvin(OPCODE, dram_addr, spad_addr) \
  gemmini_extended_mvin(OPCODE, dram_addr, spad_addr, DIM, DIM)

#define gemmini_opcode_extended_mvout(OPCODE, dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(OPCODE, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(spad_addr), k_MVOUT)

#define gemmini_opcode_mvout(OPCODE, dram_addr, spad_addr) \
  gemmini_extended_mvout(OPCODE, dram_addr, spad_addr, DIM, DIM)

// compute
#define gemmini_opcode_extended_compute_preloaded(OPCODE, A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_PRELOADED)

#define gemmini_opcode_extended_compute_accumulated(OPCODE, A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_ACCUMULATE)

#define gemmini_opcode_compute_preloaded(OPCODE, A, BD) \
  gemmini_extended_compute_preloaded(OPCODE, A, BD, DIM, DIM, DIM, DIM)

#define gemmini_opcode_compute_accumulated(OPCODE, A, BD) \
  gemmini_extended_compute_accumulated(OPCODE, A, BD, DIM, DIM, DIM, DIM)

// preload
#define gemmini_opcode_extended_preload(OPCODE, BD, C, BD_cols, BD_rows, C_cols, C_rows) \
  ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), ((uint64_t)(C_rows) << (ADDR_LEN + 16)) | ((uint64_t)(C_cols) << ADDR_LEN) | (uint64_t)(C), k_PRELOAD)

#define gemmini_opcode_preload(OPCODE, BD, C) \
  gemmini_extended_preload(OPCODE, BD, C, DIM, DIM, DIM, DIM)

#define gemmini_opcode_preload_zeros(OPCODE, C) \
  gemmini_preload(OPCODE, GARBAGE_ADDR, C)

// flush
#define gemmini_opcode_flush(OPCODE, skip) \
  ROCC_INSTRUCTION_RS1_RS2(OPCODE, skip, 0, k_FLUSH)

// fence
#define gemmini_opcode_fence() asm volatile("fence")

// Counter access
#define gemmini_opcode_counter_access(OPCODE, rd, config_reg) \
  { \
    uint32_t _placeholder; \
    ROCC_INSTRUCTION(OPCODE, rd, config_reg, _placeholder, k_COUNTER) \
  }

// weight-stationary conv loop
#define gemmini_opcode_loop_conv_ws(OPCODE, num_array, batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, activation, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, in_stride, weight_stride, out_stride, input_direct_dram, weight_direct_dram, output_direct_dram, bias_direct_dram, a_ex_id, b_ex_id) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(out_channels) << 48) | ((uint64_t)(in_channels) << 32) | ((uint64_t)(in_dim) << 16) | (uint64_t)(batch_size), \
      ((uint64_t)(padding) << 48) | ((uint64_t)(stride) << 32) | ((uint64_t)(pool_out_dim) << 16) | (uint64_t)(out_dim), k_LOOP_CONV_WS_CONFIG_1) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(kernel_dim) << 48) | ((uint64_t)(pool_size) << 32) | ((uint64_t)(pool_stride) << 16) | (uint64_t)(pool_padding), \
      ((uint64_t)(batches) << 48) | ((uint64_t)(porows) << 32) | ((uint64_t)(pocols) << 16) | (uint64_t)(pochs), k_LOOP_CONV_WS_CONFIG_2) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(krows) << 48) | ((uint64_t)(kcols) << 32) | ((uint64_t)(kchs) << 16) | (uint64_t)(lpad), \
      ((uint64_t)(rpad) << 48) | ((uint64_t)(upad) << 32) | ((uint64_t)(dpad) << 16) | (uint64_t)(plpad), k_LOOP_CONV_WS_CONFIG_3) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(orows) << 48) | ((uint64_t)(prpad) << 32) | ((uint64_t)(pupad) << 21) | ((uint64_t)(pdpad) << 10) | (uint64_t)(kernel_dilation), \
      ((uint64_t)(in_stride) << 48) | ((uint64_t)(weight_stride) << 32) | ((uint64_t)(out_stride) << 16) | (uint64_t)(ocols), k_LOOP_CONV_WS_CONFIG_4) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(weight_direct_dram) << 63) | (uint64_t) weights, \
      ((uint64_t)(output_direct_dram) << 63) | (uint64_t) output, k_LOOP_CONV_WS_CONFIG_5) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(bias_direct_dram) << 63) | (uint64_t) bias, \
      ((uint64_t)(input_direct_dram) << 63) | (uint64_t) input, k_LOOP_CONV_WS_CONFIG_6) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(num_array) << 32) | ((uint64_t)(a_ex_id) << 18) | ((uint64_t)(b_ex_id) << 16) | ((uint64_t)(max_pixels_per_row) << 8) | ((trans_input_3120) << 5) | ((trans_weight_0132) << 4) | ((trans_weight_1203) << 3) | ((trans_output_1203) << 2) | ((wrot180) << 1) | (no_bias), \
      ((activation) << 3)| ((input_dilated) << 2) | ((downsample) << 1) | (no_pool), \
      k_LOOP_CONV_WS) \
  }


// for resadd
#define gemmini_opcode_loop_one(OPCODE, dram_addr, spad_choice, dram_stride, rows, cols, cols_rounded, operation) \
  ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(operation) << 60) | ((uint64_t)(cols_rounded) << 48) | ((uint64_t) dram_addr), ((uint64_t)(rows) << 49) | ((uint64_t)(cols) << 34) | ((uint64_t)(spad_choice) << 32) | dram_stride, k_LOOP_ONE)


// for different opcodes
// weight-stationary matmul loop
#define gemmini_opcode_loop_ws(OPCODE, num_array, I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, weightA, a_ex_id, b_ex_id) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(num_array) << 32) | ((uint64_t)(a_ex_id) << 18) | ((uint64_t)(b_ex_id) << 16) | ((uint64_t)(weightA) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

// config
#define gemmini_opcode_extended3_config_ex(OPCODE, dataflow, sys_act, sys_shift, sys_acc_scale, relu6_shift, C_stride, A_stride, A_transpose, B_transpose, ocol, row_turn, kdim, stride, channel, row_left, kdim2, weight_double_bank, weight_triple_bank, set_only_strides) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)sys_acc_scale) << 32) | ((uint64_t)(A_stride) << 16) | (B_transpose << 9) | (A_transpose << 8) | ((set_only_strides) << 7) | ((sys_act) << 3) | ((dataflow) << 2) | CONFIG_EX, ((uint64_t)(C_stride) << 48) | ((uint64_t)(relu6_shift) << 32) | (sys_shift), k_CONFIG); \
    \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(weight_triple_bank) << 59) | ((uint64_t)(weight_double_bank) << 58) | ((uint64_t)(row_left) << 54) | ((uint64_t)(row_turn) << 42) | CONFIG_IM2COL, ((uint64_t)ocol << 56) | ((uint64_t)kdim2 << 48) | ((uint64_t)kdim << 44) | ((uint64_t)channel << 23) | ((uint64_t)stride << 20), k_CONFIG) \
  }

#define gemmini_opcode_extended2_config_ex(OPCODE, dataflow, sys_act, sys_shift, relu6_shift, A_stride, A_transpose, B_transpose, ocol, row_turn, kdim, stride, channel, row_left, kdim2, weight_double_bank, weight_triple_bank) \
  gemmini_opcode_extended3_config_ex(OPCODE, dataflow, sys_act, sys_shift, ACC_SCALE_IDENTITY, relu6_shift, 1, A_stride, A_transpose, B_transpose, 0, 0, 0, 0, 0, 0, 0, 0, 0, false)

#define gemmini_opcode_extended_config_ex(OPCODE, dataflow, sys_act, sys_shift, relu6_shift, A_stride, A_transpose, B_transpose) \
  gemmini_opcode_extended2_config_ex(OPCODE, dataflow, sys_act, sys_shift, relu6_shift, A_stride, A_transpose, B_transpose, 0, 0, 0, 0, 0, 0, 0, 0, 0)

#define gemmini_opcode_config_ex(OPCODE, dataflow, sys_act, sys_shift, relu6_shift) \
    gemmini_opcode_extended_config_ex(OPCODE, dataflow, sys_act, sys_shift, relu6_shift, 1, 0, 0)

// Note: The "pixel_repeats" parameter below is still experimental, andthere is
// a high chance that it will be removed in future releases.
#define gemmini_opcode_extended5_config_ld(OPCODE, direct_dram, stride, scale, shrunk, block_mvin_stride, pixel_repeats, id) \
  ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(block_mvin_stride) << 16) | ((uint64_t)(pixel_repeats) << 8) | ((uint64_t)(direct_dram) << 5) | ((id) << 3) | ((shrunk) << 2) | CONFIG_LD, stride, k_CONFIG)

#define gemmini_opcode_extended4_config_ld(OPCODE, direct_dram, stride, scale, shrunk, block_mvin_stride, id) \
  gemmini_opcode_extended5_config_ld(OPCODE, direct_dram, stride, scale, shrunk, block_mvin_stride, 1, id) \

#define gemmini_opcode_extended3_config_ld(OPCODE, direct_dram, stride, scale, shrunk, id) \
  gemmini_opcode_extended4_config_ld(OPCODE, direct_dram, stride, scale, shrunk, DIM, id)

#define gemmini_opcode_extended2_config_ld(OPCODE, stride, scale, shrunk) \
  gemmini_opcode_extended3_config_ld(OPCODE, false, stride, scale, shrunk, 0)

#define gemmini_opcode_extended_config_ld(OPCODE, stride, scale) \
  gemmini_opcode_extended2_config_ld(OPCODE, stride, scale, false)

#define gemmini_opcode_config_ld(OPCODE, stride) \
  gemmini_opcode_extended_config_ld(OPCODE, stride, MVIN_SCALE_IDENTITY)

#define gemmini_opcode_extended2_config_st(OPCODE, direct_dram, stride, acc_act, acc_scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, upad, lpad) \
  ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(ocols) << 56) | ((uint64_t)(orows) << 48) | ((uint64_t)(pocols) << 40) | ((uint64_t)(porows) << 32) | ((uint64_t)(pool_out_dim) << 24) | ((uint64_t)(direct_dram) << 14) |  ((uint64_t)(lpad) << 12) | ((uint64_t)(upad) << 10) | ((uint64_t)(pool_size) << 8) | ((uint64_t)(pool_stride) << 4) | ((acc_act) << 2) | CONFIG_ST, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)acc_scale) << 32) | ((uint32_t)stride), k_CONFIG)

#define gemmini_opcode_extended_config_st(OPCODE, direct_dram, stride, acc_act, acc_scale) \
    gemmini_opcode_extended2_config_st(OPCODE, direct_dram, stride, acc_act, acc_scale, 0, 0, 0, 0, 0, 0, 0, 0, 0)

#define gemmini_opcode_config_st(OPCODE, stride) \
    gemmini_opcode_extended_config_st(OPCODE, false, stride, NO_ACTIVATION, ACC_SCALE_IDENTITY)

/*
static void sp_tiled_opcode_matmul_ws(elem_t * A, elem_t * B,
        const void * D, void * C,
        //const size_t sub_num_I, const size_t sub_num_J, const size_t sub_num_K,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        uint8_t weightA,
        size_t num_array) {

  // Combined loop
    bool ex_acc = !no_bias || D==NULL;
    int op = 0;
    gemmini_opcode_loop_ws(OP3, num_array, I, J, K, pad_I, pad_J, pad_K, A + op*K*DIM, B + B_row_stride*op*K*DIM, no_bias ? NULL : D, C,
      A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
      a_transpose, b_transpose,
      full_C, low_D, ex_acc,
      weightA);

    if(!no_bias) ex_acc = false;
    if(num_array >= 2){
      op = 1;
      gemmini_opcode_loop_ws(OP2, num_array, I, J, K, pad_I, pad_J, pad_K, A + op*K*DIM, B + B_row_stride*op*K*DIM, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, ex_acc,
        weightA);
    }
    if(num_array >= 3){
      op = 2;
      gemmini_opcode_loop_ws(OP1, num_array, I, J, K, pad_I, pad_J, pad_K, A + op*K*DIM, B + B_row_stride*op*K*DIM, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, ex_acc,
        weightA);
    }
    if(num_array == 4){
      op = 3;
      gemmini_opcode_loop_ws(OP0, num_array, I, J, K, pad_I, pad_J, pad_K, A + op*K*DIM, B + B_row_stride*op*K*DIM, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, ex_acc,
        weightA);
    }  
}
*/

// for special case (lask K iterating factor)
static void sp_tiled_opcode_matmul_ws(elem_t * A, elem_t * B,
        void * D, void * C,
        //const size_t sub_num_I, const size_t sub_num_J, const size_t sub_num_K,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        uint8_t weightA,
        size_t num_array,
        size_t a_spad_id, size_t b_spad_id) {

  // Combined loop
    bool ex_acc = !no_bias || D==NULL;
    int op = 0;
    int acc_K = 0;
    int K1 = 0; int K2 = 0;
    int K0 = ceil_divide_int(K, num_array);

    gemmini_opcode_loop_ws(OP3, num_array, I, J, K0, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
      A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
      a_transpose, b_transpose,
      full_C, low_D, ex_acc,
      weightA, a_spad_id, b_spad_id);

    K -= K0;
    if(!no_bias) ex_acc = false;
    if(num_array >= 2){
      op = 1;
      acc_K += K0;
      K1 = ceil_divide_int(K, num_array - 1);
#ifndef BAREMETAL
      gemmini_opcode_loop_ws(OP2, num_array, I, J, K1, pad_I, pad_J, pad_K, A == NULL ? NULL : A + acc_K*DIM, B == NULL ? NULL : B + B_row_stride*acc_K*DIM, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, ex_acc,
        weightA, a_spad_id, b_spad_id);
#endif
      K -= K1;
    }
    if(num_array >= 3){
      op = 2;
      acc_K += K1;
      K2 = ceil_divide_int(K, num_array - 2);
#ifndef BAREMETAL
      gemmini_opcode_loop_ws(OP1, num_array, I, J, K2, pad_I, pad_J, pad_K, A == NULL ? NULL : A + acc_K*DIM, B == NULL ? NULL : B + B_row_stride*acc_K*DIM, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, ex_acc,
        weightA, a_spad_id, b_spad_id);
#endif
      K -= K2;
    }
    if(num_array == 4){
      op = 3;
      acc_K += K2;
#ifndef BAREMETAL
      gemmini_opcode_loop_ws(OP0, num_array, I, J, K, pad_I, pad_J, pad_K, A == NULL ? NULL : A + acc_K*DIM, B == NULL ? NULL : B + B_row_stride*acc_K*DIM, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, ex_acc,
        weightA, a_spad_id, b_spad_id);
#endif
    }
}

// dim_I, dim_J are already fully dividied into subarray size
static void tiled_opcode_matmul_outer(size_t dim_I_original, size_t dim_J_original, size_t dim_K_original,
        size_t dim_I, size_t dim_J, size_t dim_K,
        //const size_t sub_num_I, const size_t sub_num_J, const size_t sub_num_K,
        elem_t* A, elem_t* B,
        void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram, 
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t tile_I, size_t tile_J, size_t tile_K,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        uint8_t weightA,
        int dataflow,
        bool div_k,
        int num_array) {
  
  // based on original dimension
  const size_t dim_I_padded = (dim_I_original / DIM + (dim_I_original % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J_original / DIM + (dim_J_original % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K_original / DIM + (dim_K_original % DIM != 0)) * DIM;

  // need to iterate outer tile granularity
  size_t outer_tile_I = div_k ? tile_I : tile_I * num_array;
  size_t outer_tile_K = div_k ? tile_K * num_array : tile_K; 

  // to increase loop iter factor by +1
  const size_t I0 = dim_I_padded / (outer_tile_I*DIM) + (dim_I_padded % (outer_tile_I*DIM) != 0);
  const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
  const size_t K0 = dim_K_padded / (outer_tile_K*DIM) + (dim_K_padded % (outer_tile_K*DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (outer_tile_I*DIM) == 0 ? outer_tile_I : (dim_I_padded/DIM) % outer_tile_I;
  const size_t last_J = dim_J_padded % (tile_J*DIM) == 0 ? tile_J : (dim_J_padded/DIM) % tile_J;
  const size_t last_K = dim_K_padded % (outer_tile_K*DIM) == 0 ? outer_tile_K : (dim_K_padded/DIM) % outer_tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - dim_I_original;
  const size_t padding_J = dim_J_padded - dim_J_original;
  const size_t padding_K = dim_K_padded - dim_K_original;

  const bool no_bias = D == NULL;

  if (no_bias) {
    D = (void*) 1; // Dummy address which isn't NULL
  }

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

#ifndef BAREMETAL
  if(num_array == 4){
    gemmini_opcode_extended_config_ex(OP0, dataflow, act, 0, relu6_shift, 1, a_transpose, b_transpose);
    gemmini_opcode_extended_config_st(OP0, C_direct_dram, stride_C * sizeof_C, act, scale);
    gemmini_opcode_extended3_config_ld(OP0, A_direct_dram, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
    gemmini_opcode_extended3_config_ld(OP0, B_direct_dram, stride_B * sizeof(elem_t), B_scale_factor, false, 1)
    gemmini_opcode_extended3_config_ld(OP0, D_direct_dram, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
  }
  if(num_array >= 3){
    gemmini_opcode_extended_config_ex(OP1, dataflow, act, 0, relu6_shift, 1, a_transpose, b_transpose);
    gemmini_opcode_extended_config_st(OP1, C_direct_dram, stride_C * sizeof_C, act, scale);
    gemmini_opcode_extended3_config_ld(OP1, A_direct_dram, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
    gemmini_opcode_extended3_config_ld(OP1, B_direct_dram, stride_B * sizeof(elem_t), B_scale_factor, false, 1)
    gemmini_opcode_extended3_config_ld(OP1, D_direct_dram, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
  }
  if(num_array >= 2){
    gemmini_opcode_extended_config_ex(OP2, dataflow, act, 0, relu6_shift, 1, a_transpose, b_transpose);
    gemmini_opcode_extended_config_st(OP2, C_direct_dram, stride_C * sizeof_C, act, scale);
    gemmini_opcode_extended3_config_ld(OP2, A_direct_dram, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
    gemmini_opcode_extended3_config_ld(OP2, B_direct_dram, stride_B * sizeof(elem_t), B_scale_factor, false, 1)
    gemmini_opcode_extended3_config_ld(OP2, D_direct_dram, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
  }
#endif
    gemmini_opcode_extended_config_ex(OP3, dataflow, act, 0, relu6_shift, 1, a_transpose, b_transpose);
    gemmini_opcode_extended_config_st(OP3, C_direct_dram, stride_C * sizeof_C, act, scale);
    gemmini_opcode_extended3_config_ld(OP3, A_direct_dram, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
    gemmini_opcode_extended3_config_ld(OP3, B_direct_dram, stride_B * sizeof(elem_t), B_scale_factor, false, 1)
    gemmini_opcode_extended3_config_ld(OP3, D_direct_dram, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
             
//  }
/*
  void (*inner)(elem_t *, elem_t *, const void *, void *,
        //size_t, size_t, size_t, 
        scale_t, scale_t, scale_acc_t,
        size_t, size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t,
        bool, bool,
        bool, bool,
        bool, bool,
        uint8_t,
        size_t);
*/
  size_t a_spad_id = 0;
  size_t b_spad_id = 0;

  bool a_reuse = false;
  bool b_reuse = false;

  // printf("I0: %d, J0: %d, K0: %d\n", I0, J0, K0);
   
  if(J0 * K0 <= 2) 
    b_reuse = true;
  if(I0 * K0 <= 2)
    a_reuse = true;

  int num_array_store = num_array;
 
  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++) {
        if(a_reuse)
          a_spad_id = ((i0+k0) == 0) ? 1 : 2;
        if(b_reuse)
          b_spad_id = ((j0+k0) == 0) ? 1 : 2;


        void * pre;
        if (k0 != 0) {
          pre = NULL;
        } else {
          size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
          // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
          pre = (int8_t*)D + (bias_row * stride_D + j0 * tile_J * DIM)*sizeof_D;
        }

        void * out = k0 == K0-1 ? (int8_t*)C + (i0*outer_tile_I*DIM*stride_C + j0*tile_J*DIM)*sizeof_C : NULL;

        size_t I = i0 < I0-1 ? tile_I : last_I;
        size_t J = j0 < J0-1 ? tile_J : last_J;
        size_t K = k0 < K0-1 ? outer_tile_K : last_K; // entire K

        // if last iteration, I/K is the sum of all subarrays
        const bool last = div_k ? k0 == K0-1 : i0 == I0-1;         
        num_array_store = num_array;
        if(last && div_k){
          if(last_K <= num_array){
            num_array_store = last_K;
          }
        }
        const size_t pad_I = i0 == I0-1 ? padding_I : 0;
        const size_t pad_J = j0 == J0-1 ? padding_J : 0;
        const size_t pad_K = k0 == K0-1 ? padding_K : 0;

        elem_t * a = a_transpose ? (A + k0*num_array*outer_tile_K*DIM*stride_A + i0*outer_tile_I*DIM)
          : (A + i0*outer_tile_I*DIM*stride_A + k0*num_array*outer_tile_K*DIM);

        elem_t * b = b_transpose ? (B + j0*tile_J*DIM*stride_B + k0*num_array*outer_tile_K*DIM)
          : (B + k0*num_array*outer_tile_K*DIM*stride_B + j0*tile_J*DIM);

        if(a_reuse && j0 >= 1) a = NULL;
        if(b_reuse && i0 >= 1) b = NULL;
        /*
        if(k0 == K0 - 1){ // last
          sp_tiled_opcode_matmul_ws_last(a, b, pre, out,
              A_scale_factor, B_scale_factor, D_scale_factor,
              //sub_num_I, sub_num_J, sub_num_K,
              I, J, K,
              pad_I, pad_J, pad_K,
              stride_A, stride_B, stride_D, stride_C,
              a_transpose, b_transpose,
              full_C, low_D,
              no_bias, repeating_bias,
              weightA,
              num_array);
        }
        else{*/
          sp_tiled_opcode_matmul_ws(a, b, pre, out,// NULL,
              A_scale_factor, B_scale_factor, D_scale_factor,
              //sub_num_I, sub_num_J, sub_num_K,
              I, J, K,
              pad_I, pad_J, 0, // pad_K,
              stride_A, stride_B, stride_D, stride_C,
              a_transpose, b_transpose,
              full_C, low_D,
              no_bias, repeating_bias,
              weightA,
              num_array_store,
              a_spad_id, b_spad_id);
       // }
      }

  gemmini_opcode_fence();
}



static void tiled_opcode_matmul_nn_auto_multi(size_t dim_I, size_t dim_J, size_t dim_K,
  //const size_t sub_num_I, const size_t sub_num_J, const size_t sub_num_K,
  size_t stride_A, size_t stride_B, size_t stride_C,
  bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram,  
  elem_t* A, elem_t* B,
  void * D, elem_t* C,
  int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
  enum tiled_matmul_type_t tiled_matmul_type,
  //size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id,
  const size_t num_array)
// orow_divide -> total, num_array -> inner
// for now assume one workload (no cid, orow_divide)
{

  bool div_k = true;//false;
  size_t* args_out;
  size_t args[10];
  size_t dim_J_original = dim_J;
  size_t dim_K_original = dim_K;
  size_t dim_I_original = dim_I;
//  if(dim_K >= num_array * DIM * 2){
    dim_K = ceil_divide_int(dim_K, num_array);
    if(dim_K % DIM != 0){
      dim_K = ceil_divide_int(dim_K, DIM) * DIM;
    }
    //div_k = true;
    args_out = tiling_factor_matmul_calculate_auto(dim_I, dim_J, dim_K, 1, 1, 0, 0, args, 0);
//  }
//  else{
//    args_out = tiling_factor_matmul_calculate_auto(dim_I, dim_J, dim_K, num_array, 1, 0, 0, args, 0);
//  }
  dim_I = args_out[3];
  dim_J = args_out[4];
  dim_K = args_out[5];
  size_t tile_I = args_out[8];
  size_t tile_J = args_out[9];
  size_t tile_K = args_out[10];

  size_t orow_offset_floor = args_out[6];
  bool row_divisible = (args_out[7] != 0);
//  int window = args_out[0];
//  int target_load = args_out[1];

//  orow_divide = batch_divide * orow_divide;
//  batch_divide = 1;
  //size_t total_divide = orow_divide * batch_divide; // number of cores for this workload

//  if(!row_divisible) orow_divide = 1;
//  int out_offset = (row_divisible) ? 0 : dim_J * cid; // no need to apply offset if we divided row
//  int A_orow_offset = (row_divisible && cid != 0) ? stride_A * cid * dim_I + stride_A * orow_offset_floor : 0; // if row is divided, need offset it I dimension
//  int C_orow_offset = (row_divisible && cid != 0) ? stride_C * cid * dim_I + stride_C * orow_offset_floor : 0; // if row is divided, need offset it I dimension
//  printf("dim_I: %d, orow_offset_floor: %d, A_row_offset: %d \n", dim_I, orow_offset_floor, A_orow_offset);
/*
  int A_batch_offset = 0;
  int C_batch_offset = 0;
  if (batch_divide > 1){
     A_batch_offset = stride_A * cid * dim_I;
     C_batch_offset = stride_C * cid * dim_I;
  }
*/
  bool no_bias = (D == NULL);
  
  tiled_opcode_matmul_outer(dim_I_original, dim_J_original, dim_K_original,
      dim_I, dim_J, dim_K,
      //sub_num_I, sub_num_J, sub_num_K,
      A, B, no_bias ? NULL : D, C, // for now, disable global workload division
      //A + A_orow_offset + A_batch_offset, B + out_offset, no_bias ? NULL : D + out_offset*sizeof(acc_t), C + C_orow_offset + out_offset + C_batch_offset,
      stride_A, stride_B, stride_B, stride_C,
      A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      tile_I, tile_J, tile_K,
      act, scale, relu6_shift, repeating_bias,
      false, false, false, false, 3,
      (int)tiled_matmul_type,
      div_k,
      num_array);
}

static void tiled_opcode_matmul_nn_auto_cid(size_t dim_I, size_t dim_J, size_t dim_K,
  size_t stride_C,
  elem_t* A, elem_t* B,
  const void * D, elem_t* C,
  int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
  enum tiled_matmul_type_t tiled_matmul_type,
  size_t num_array){
  //size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id){

  size_t stride_A = (dim_K % 128 == 0) ? dim_K + 64 : dim_K;
  size_t stride_B = (dim_J % 128 == 0) ? dim_J + 64 : dim_J;

  //printf("A dram addr: 0x%08lx\n", A);
  tiled_opcode_matmul_nn_auto_multi(
      dim_I, dim_J, dim_K,
      stride_A, stride_B, stride_C,
      false, false, false, false, // direct dram
      A, B, D, C,
      act, scale, relu6_shift, repeating_bias,
      WS, num_array);
      //orow_divide, batch_divide, cid, group_id);

}
static void sp_tiled_opcode_conv(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim, int pool_out_dim,

        int stride, int padding, int kernel_dim, int kernel_dilation,

        int in_stride, int weight_stride, int out_stride,
        bool in_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool out_direct_dram,

        int pool_size, int pool_stride, int pool_padding,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs, // kchs is aggregated across all subarrays

        int lpad, int rpad, int upad, int dpad,
        int plpad, int prpad, int pupad, int pdpad,

        elem_t * input,
        elem_t * weights,
        elem_t * output,
        acc_t * bias,

        int act, acc_scale_t scale,

        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        bool no_bias, bool no_pool, bool downsample, bool input_dilated,
        size_t num_array, size_t a_spad_id, size_t b_spad_id) {

  const int orows = porows * pool_stride + pool_size - 1 - pupad - pdpad;
  const int ocols = pocols * pool_stride + pool_size - 1 - plpad - prpad;
  const int ochs = pochs;
  const int ichs = kchs;

  // Calculate image dimensions
  // Note: "irows" and "icols" includes padding
  const int dilated_krows = krows + (kernel_dilation - 1)*(krows - 1);
  const int dilated_kcols = kcols + (kernel_dilation - 1)*(kcols - 1);
  int irows = orows * stride + dilated_krows - 1;
  int icols = ocols * stride + dilated_kcols - 1;
  int irows_unpadded = irows - upad - dpad;
  int icols_unpadded = icols - lpad - rpad;
#define UNDILATED(x) ((input_dilated) ? (((x)+1)/2) : (x))

  if (input_dilated) {
    irows_unpadded = (irows_unpadded+1)/2;
    icols_unpadded = (icols_unpadded+1)/2;

    irows = irows_unpadded + UNDILATED(upad) + UNDILATED(dpad);
    icols = icols_unpadded + UNDILATED(lpad) + UNDILATED(rpad);
  }

#ifdef HAS_FIRST_LAYER_OPTIMIZATIONS
  const bool transposed = trans_output_1203 || trans_input_3120 ||
      trans_weight_1203 || trans_weight_0132;
  int max_pixels_per_row = transposed || wrot180 || downsample ||
      input_dilated || kernel_dilation > 1 ||
      ichs > DIM ? 1 : DIM/ichs;
  if (max_pixels_per_row > kcols) max_pixels_per_row = kcols;
#else
  const int max_pixels_per_row = 1;
#endif

  int op = 0;
  int acc_kch = 0;
  int kchs0 = ceil_divide_int(kchs, num_array * DIM) * DIM;
  int kchs1 = 0; int kchs2 = 0;
  /*
  printf("in_channels: %d, kchs0: %d, max_pixels_per_row: %d \n", in_channels, kchs0, max_pixels_per_row);
  gemmini_opcode_loop_conv_ws(OP3, num_array, batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs0, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, in_stride, weight_stride, out_stride, in_direct_dram, weight_direct_dram, out_direct_dram, bias_direct_dram, a_spad_id, b_spad_id);
*/
  kchs -= kchs0;

  if(num_array >= 2){
    op = 1;
    acc_kch += kchs0;
    kchs1 = ceil_divide_int(kchs, (num_array - 1) * DIM) * DIM;
    printf("in_channels: %d, kchs0: %d, kchs1: %d, acc_kch: %d \n", in_channels, kchs0, kchs1, acc_kch);
//#ifndef BAREMETAL
    gemmini_opcode_loop_conv_ws(OP2, num_array, batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs1, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights == NULL ? NULL : weights+acc_kch*weight_stride, output, bias, input == NULL? NULL : input+acc_kch, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, in_stride, weight_stride, out_stride, in_direct_dram, weight_direct_dram, out_direct_dram, bias_direct_dram, a_spad_id, b_spad_id);
//#endif
    kchs -= kchs1;
  }

  if(num_array >= 3){
    op = 2;
    acc_kch += kchs1;
    kchs2 = ceil_divide_int(kchs, (num_array - 2) * DIM) * DIM;
#ifndef BAREMETAL
    gemmini_opcode_loop_conv_ws(OP1, num_array, batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs2, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights == NULL ? NULL : weights+acc_kch*weight_stride, output, bias, input == NULL? NULL : input+acc_kch, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, in_stride, weight_stride, out_stride, in_direct_dram, weight_direct_dram, out_direct_dram, bias_direct_dram, a_spad_id, b_spad_id);
#endif
    kchs -= kchs2;
  }

  if(num_array == 4){
    op = 3;
    acc_kch += kchs2;
#ifndef BAREMETAL
    gemmini_opcode_loop_conv_ws(OP0, num_array, batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights == NULL ? NULL : weights+acc_kch*weight_stride, output, bias, input == NULL? NULL : input+acc_kch, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, in_stride, weight_stride, out_stride, in_direct_dram, weight_direct_dram, out_direct_dram, bias_direct_dram, a_spad_id, b_spad_id);
#endif
  }
}


static void tiled_opcode_conv(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        int in_stride, int weight_stride, int out_stride,
        bool in_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool out_direct_dram,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs,

        elem_t * input,
        elem_t * weights,
        acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,

        enum tiled_matmul_type_t tiled_conv_type,
        size_t num_array){
        //size_t orow_divide, size_t cid, size_t group_id) {

  size_t orow_divide = 1;
  size_t cid = 0;
  size_t group_id =0;
#ifdef GEMMINI_ASSERTIONS
  if (trans_weight_1203 && trans_weight_0132) {
    printf("Only one weight transformation can be applied at a time\n");
    exit(1);
  }
#endif

    if (tiled_conv_type == CPU) {
      if (pool_size == 1 && pool_stride == 1 && pool_padding == 0) {
        pool_stride = 0;
      }

      conv_cpu(
        batch_size, in_dim, in_channels,
        out_channels, out_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale, relu6_shift,
        pool_size, pool_stride, pool_padding);
      return;
    } else if (tiled_conv_type == OS) {
      printf("Gemmini convs do not currently support OS\n");
      exit(1);
    }

    // TODO move everything below this into a tiled_conv_outer function to match the tiled_matmul function

    bool no_bias = false;
    if (bias == NULL) {
        bias = (acc_t*)1;
        no_bias = true;
    }

    bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const bool downsample = stride == 2 && kernel_dim == 1 && in_dim % 2 == 0
      && padding == 0 && no_pool && input_dilation == 1 && !trans_input_3120;

    const int input_dilated = input_dilation == 2;

#ifdef GEMMINI_ASSERTIONS
    {
        // const int orows = porows * pool_stride + pool_size - 1;
        // const int ocols = pocols * pool_stride + pool_size - 1;

        // Check that data will fit in scratchpad
        const int spad_rows = tiled_conv_total_spad_rows(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);
        const int acc_rows = tiled_conv_total_spad_rows(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);

        if (spad_rows > BANK_NUM * BANK_ROWS / 2) {
            printf("not enough scratchpad space to store inputs and weights, %d\n", spad_rows);
            exit(1);
        }
        if (acc_rows > ACC_ROWS / 2) {
            printf("not enough accumulator space to store outputs\n");
            exit(1);
        }
        if (kernel_dim <= padding) {
            printf("kernel_dim must be larger than padding\n");
            exit(1);
        }
        if (input_dilation > 2) {
            printf("input_dilation > 2 is only supported on CPU\n");
            exit(1);
        }
        if (input_dilation > 1 && stride > 1) {
            printf("input input_dilation is only supported when stride == 1\n");
            exit(1);
        }
        if (trans_output_1203 && !no_pool) {
            printf("Output can only be transposed when pooling is disabled\n");
            exit(1);
        }
        if (trans_input_3120 && trans_weight_0132) {
            printf("Cannot transpose innermost dimensions of both inputs and weights on WS.\n");
            exit(1);
        }
    }
#endif

    const size_t st_dram_stride = trans_output_1203 ?
        batch_size * out_channels * sizeof(elem_t) :
        out_stride * sizeof(elem_t);

#ifndef BAREMETAL
    if(num_array == 4){
      gemmini_opcode_extended_config_st(OP0, out_direct_dram, st_dram_stride, act, scale);
      gemmini_opcode_extended3_config_ex(OP0, WEIGHT_STATIONARY, 0, 0, 0, relu6_shift, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
    }
    if(num_array >= 3){
      gemmini_opcode_extended_config_st(OP1, out_direct_dram, st_dram_stride, act, scale);
      gemmini_opcode_extended3_config_ex(OP1, WEIGHT_STATIONARY, 0, 0, 0, relu6_shift, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
    }
    if(num_array >= 2){
      gemmini_opcode_extended_config_st(OP2, out_direct_dram, st_dram_stride, act, scale);
      gemmini_opcode_extended3_config_ex(OP2, WEIGHT_STATIONARY, 0, 0, 0, relu6_shift, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
    }
#endif
    gemmini_opcode_extended_config_st(OP3, out_direct_dram, st_dram_stride, act, scale);
    gemmini_opcode_extended3_config_ex(OP3, WEIGHT_STATIONARY, 0, 0, 0, relu6_shift, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
    
    const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;
    const int dilated_in_dim = in_dim + (input_dilation-1)*(in_dim-1);

    int pool_out_row = (pool_out_dim % orow_divide == 0) ? pool_out_dim / orow_divide : ((int)(pool_out_dim/orow_divide)) + 1;
    int porow_start = (orow_divide == 1) ? 0 : pool_out_row * cid;
    int porow_end = (orow_divide == 1) ? pool_out_dim : ((cid == orow_divide - 1) ? pool_out_dim : pool_out_row * (cid + 1));

    int iter_kchs = kchs * num_array < in_channels ? kchs * num_array : in_channels;

    size_t a_spad_id = 0;
    size_t b_spad_id = 0;

    bool a_reuse = false;
    bool b_reuse = false;
    size_t num_kch = ceil_divide_int(in_channels, iter_kchs);
    size_t num_poch = ceil_divide_int(out_channels, pochs);
    size_t num_b = ceil_divide_int(batch_size, batches);
    size_t num_porow = ceil_divide_int((porow_end - porow_start), porows);
    size_t num_pocol = ceil_divide_int(pool_out_dim, pocols);
    size_t num_krow = ceil_divide_int(kernel_dim, krows);
    size_t num_kcol = ceil_divide_int(kernel_dim, kcols);

    printf("num_kch: %d, num_poch: %d, num_b: %d, num_porow: %d, num_pocol: %d, num_krow: %d, num_kcol: %d\n", num_kch, num_poch, num_b, num_porow, num_pocol, num_krow, num_kcol);

    if(num_kch * num_poch * num_krow * num_kcol <= 2) 
      b_reuse = true;
    if(num_kch * num_krow * num_kcol * num_b * num_porow * num_pocol <= 2)
      a_reuse = true;

    printf("iter_kchs: %d, kchs: %d, in_channels: %d\n", iter_kchs, kchs, in_channels);

    int num_array_store = num_array;

    for (int poch = 0; poch < out_channels; poch += pochs) {
      for (int b = 0; b < batch_size; b += batches) {
        for (int porow = porow_start; porow < porow_end; porow += porows) {
 //         printf("porow_start: %d, porow_end: %d, porow: %d \n", porow_start, porow_end, porow);
          const int orow = porow * pool_stride - pool_padding;
          for (int pocol = 0; pocol < pool_out_dim; pocol += pocols) {
            const int ocol = pocol * pool_stride - pool_padding;
            for (int krow = 0; krow < kernel_dim; krow += krows) {
              const int orow_floored = orow < 0 ? 0 : orow;
              const int irow = orow_floored * stride + krow*kernel_dilation - padding;
              for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                const int ocol_floored = ocol < 0 ? 0 : ocol;
                const int icol = ocol_floored * stride + kcol*kernel_dilation - padding;
                for (int kch = 0; kch < in_channels; kch += iter_kchs) { //aggregated
                  if(a_reuse)
                    a_spad_id = (kch + krow + kcol + b + (porow - porow_start) + pocol) == 0 ? 1 : 2;
                  if(b_reuse)
                    b_spad_id = (kch + poch + krow + kcol) == 0 ? 1 : 2;

                  elem_t * out = output + (b*pool_out_dim*pool_out_dim + porow*pool_out_dim + pocol) * out_stride + poch;
                  if (trans_output_1203) {
                    out = output + (porow*pool_out_dim*batch_size + pocol*batch_size + b) * out_channels + poch;
                  }

                  if (krow + krows < kernel_dim ||
                      kcol + kcols < kernel_dim ||
                      kch + iter_kchs < in_channels) {
                    out = NULL;
                  }

                  acc_t * bias_ = bias + poch;
                  if (krow > 0 ||
                          kcol > 0 ||
                          kch > 0) {
                      bias_ = NULL;
                  }
                  
                  bool last_kch = in_channels - kch <= iter_kchs;

                  const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                  const int porows_ = pool_out_dim - porow > porows ? porows : pool_out_dim - porow;
                  const int pocols_ = pool_out_dim - pocol > pocols ? pocols : pool_out_dim - pocol;
                  const int pochs_ = out_channels - poch > pochs ? pochs : out_channels - poch;
                  const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                  const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;
                  const int kchs_ = last_kch ? in_channels- kch : iter_kchs;// in_channels - kch > iter_kchs ? iter_kchs : in_channels - kch;
                  if(last_kch && kchs_ <= num_array * DIM){
                    num_array_store = ceil_divide_int(kchs_, DIM);
                  }
                  else num_array_store = num_array;

                  const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                  const int orows_ = porows_ * pool_stride + pool_size - 1;

                  const int plpad = ocol < 0 ? -ocol : 0;
                  const int prpad = ocol + ocols_ > out_dim ? ocol + ocols_ - out_dim : 0;
                  const int pupad = orow < 0 ? -orow : 0;
                  const int pdpad = orow + orows_ > out_dim ? orow + orows_ - out_dim : 0;

                  const int dilated_krows_ = krows_ + (kernel_dilation - 1)*(krows_ - 1);
                  const int dilated_kcols_ = kcols_ + (kernel_dilation - 1)*(kcols_ - 1);

                  const int icols_ = (ocols_ - plpad - prpad) * stride + dilated_kcols_ - 1;
                  const int irows_ = (orows_ - pupad - pdpad) * stride + dilated_krows_ - 1;

                  int lpad = icol < 0 ? -icol : 0;
                  int rpad = icol + icols_ > dilated_in_dim ? icol + icols_ - dilated_in_dim : 0;
                  int upad = irow < 0 ? -irow : 0;
                  int dpad = irow + irows_ > dilated_in_dim ? irow + irows_ - dilated_in_dim : 0;

                  if (input_dilated) {
                    lpad += lpad == 0 && icol % 2 != 0;
                    rpad += rpad == 0 && (icol + icols_) % 2 != 1;
                    upad += upad == 0 && irow % 2 != 0;
                    dpad += dpad == 0 && (irow + irows_) % 2 != 1;
                  }

                  int krow_ = krow;
                  int kcol_ = kcol;
                  if (wrot180) {
                    krow_ = kernel_dim - krow - krows_;
                    kcol_ = kernel_dim - kcol - kcols_;
                  }

                  elem_t * weights_slice = weights + (krow_*kernel_dim*in_channels + kcol_*in_channels + kch) * weight_stride + poch;
                  if (trans_weight_1203) {
                    weights_slice = weights + (kch*kernel_dim*kernel_dim + krow_*kernel_dim+kcol_) * out_channels + poch;
                  } else if (trans_weight_0132) {
                    weights_slice = weights + (krow_*kernel_dim*out_channels + kcol_*out_channels + poch) * in_channels + kch;
                  }

                  elem_t * in = input + (b*in_dim*in_dim + ((irow+upad)>>input_dilated)*in_dim + ((icol+lpad)>>input_dilated)) * in_stride + kch;
                  if (trans_input_3120) {
                    in = input + (kch*in_dim*in_dim + ((irow+upad)>>input_dilated)*in_dim + ((icol+lpad)>>input_dilated)) * batch_size + b;
                  }

                  if(b_reuse && (pocol + (porow - porow_start) + b > 0)) weights_slice = NULL;
                  if(a_reuse && (poch > 0)) in = NULL;

                  //printf("in: %llu, weights: %llu\n", in, weights_slice);

                  sp_tiled_opcode_conv(
                      batch_size, in_dim, in_channels,
                      out_channels, out_dim, pool_out_dim,

                      stride, padding, kernel_dim, kernel_dilation,

                      in_stride, weight_stride, out_stride,
                      in_direct_dram, weight_direct_dram, bias_direct_dram, out_direct_dram,

                      pool_size, pool_stride, pool_padding,

                      batches_,
                      porows_, pocols_, pochs_,
                      krows_, kcols_, kchs_,

                      lpad, rpad, upad, dpad,
                      plpad, prpad, pupad, pdpad,

                      in,
                      weights_slice,
                      out,
                      bias_,

                      act, scale,

                      wrot180, trans_output_1203, trans_input_3120,
                      trans_weight_1203, trans_weight_0132,

                      no_bias, no_pool, downsample, input_dilated,
                      num_array_store, a_spad_id, b_spad_id);
                    
                }
              }
            }
          }
        }
      }
    }
}

// division by row dimension
static void tiled_opcode_conv_auto_multi( // for sw padding
    int batch_size, int in_dim, int in_channels,
    int out_channels, int out_dim,
    int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,

    int in_stride, int weight_stride, int out_stride,
    bool in_direct_dram, bool weight_direct_dram, bool bias_direct_dram, bool out_direct_dram,

    bool wrot180, bool trans_output_1203, bool trans_input_3120,
    bool trans_weight_1203, bool trans_weight_0132,

    elem_t * input,
    elem_t * weights,
    acc_t * bias,
    elem_t * output,

    int act, acc_scale_t scale, size_t relu6_shift,
    int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,

    enum tiled_matmul_type_t tiled_conv_type,
    size_t num_array){
    //size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id){

  size_t orow_divide = 1;
  size_t batch_divide = 1;
  size_t cid = 0;
  size_t group_id = 0;

  int in_channels_orig = in_channels;
  in_channels = ceil_divide_int(in_channels, DIM * num_array) * DIM;

  int target_util = 0;
  const bool no_pool = pool_stride == 0;
   if (no_pool) {
      pool_size = 1;
      pool_stride = 1;
      pool_padding = 0;
   }
#ifdef GEMMINI_ASSERTIONS
   if(batch_size == 1 && batch_divide > 1){
     printf("batch_divide set wrong \n");
     exit(1);
   }
#endif
//   printf("%d, ", batch_size*out_dim*out_dim*in_channels*out_channels*kernel_dim*kernel_dim);
   
   // tiling, calm configure
   int args_in[10] = {target_util, 0, 0, 0, 0, 0, 0, 0, 0, 0};
   int * args = tiled_conv_bubble_calculate(args_in, batch_size, in_dim, in_channels, out_channels, out_dim, stride, kernel_dilation, padding, kernel_dim, pool_size, pool_stride, pool_padding, pool_ceil_dim, orow_divide, batch_divide, cid, group_id);

  int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;
  if (pool_ceil_dim)
    pool_out_dim += (out_dim + 2*pool_padding - pool_size) % pool_stride != 0;

  //size_t total_divide = orow_divide * batch_divide;
  size_t batch_cid = (size_t)(cid / orow_divide);
  size_t orow_cid = (size_t)(cid % orow_divide);

  // divide in batch dimension
  batch_size = batch_size / batch_divide;
  int batch_in_offset = (batch_divide > 1) ? batch_size*in_dim*in_dim*in_stride*batch_cid : 0;
  int batch_out_offset = (batch_divide > 1) ? batch_size*pool_out_dim*pool_out_dim*out_stride*batch_cid : 0; // not dividing in out_channel dimension
//printf("batch in offset: %d, batch out offset: %d\n", batch_in_offset, batch_out_offset);
// divide in row dimension (single batch)
	bool row_divisible = (orow_divide > 1) && ((pool_out_dim % orow_divide == 0) || (in_channels == 3 && padding == 0)) && (kernel_dilation <= 2);
  //bool row_divisible = (orow_divide > 1) && (pool_out_dim % orow_divide == 0) && (kernel_dilation <= 2);
  if (orow_divide > 1 && padding == 0) row_divisible = true;
  int pool_out_row = (row_divisible) ? (pool_out_dim / orow_divide) : pool_out_dim;
  if(pool_out_dim % orow_divide != 0) {
     if(orow_cid != orow_divide - 1) pool_out_row += 1;
     //else pool_out_row -= 1;
  }
  const size_t och_divide = (row_divisible) ? 1 : orow_divide; //if row isn't divisible, divide channel instead
  out_channels = out_channels / och_divide;
  
  const int out_offset = (och_divide > 1) ? out_channels * orow_cid : 0;

  /*
  int args_in[] = {batch_size, pool_out_row, pool_out_dim, out_channels, kernel_dim, kernel_dim, in_channels};
  int* args;
  args = tiling_factor_calculate(args_in, stride, pool_size, pool_stride, kernel_dilation, padding);
*/
  const int batches = args[0];
  const int orows = args[1];
  const int ocols = args[2];
  const int ochs = args[3];
  const int krows = args[4];
  const int kcols = args[5];
  const int kchs = args[6];

  const int window = args[7];
  const int target_load = args[8];
  const int ideal_cycle = args[9];
 // if(row_divisible){
      tiled_opcode_conv(
          batch_size, in_dim, in_channels_orig,
          out_channels, out_dim,
          stride, input_dilation, kernel_dilation, padding, kernel_dim,
          in_stride, weight_stride, out_stride,
          in_direct_dram, weight_direct_dram, bias_direct_dram, out_direct_dram,

          wrot180, trans_output_1203, trans_input_3120,
          trans_weight_1203, trans_weight_0132,

          batches,
          orows, ocols, ochs,
          krows, kcols, kchs,

          (elem_t*) input + batch_in_offset,
          (elem_t*) weights,
          (acc_t*) bias,
          output + batch_out_offset,

          act, scale, relu6_shift,
          pool_size, no_pool ? 0 : pool_stride, pool_padding, pool_ceil_dim,

          tiled_conv_type, num_array);
          //orow_divide, orow_cid, group_id);
         // window, target_load);
/*
  }else{
    bool no_bias = (bias == NULL);
    tiled_conv(
        batch_size, in_dim, in_channels,
        out_channels, out_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        in_stride, weight_stride, out_stride,
        in_direct_dram, weight_direct_dram, bias_direct_dram, out_direct_dram,

        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,

        batches,
        orows, ocols, ochs,
        krows, kcols, kchs,

        (elem_t*) input + batch_in_offset,
        (elem_t*) weights + out_offset,
        no_bias ? NULL : (acc_t*) bias + out_offset,
        output + out_offset + batch_out_offset,

        act, scale, relu6_shift,
        pool_size, no_pool ? 0 : pool_stride, pool_padding, pool_ceil_dim,

        tiled_conv_type, 1, orow_cid, group_id);
        //window, target_load);
  }
  */
}

// for convert
static void tiled_opcode_conv_auto_cid(
    int batch_size, int in_dim, int in_channels,
    int out_channels, int out_dim,
    int stride, int dilation, int padding, int kernel_dim,
    int out_stride, //for concatenation

    const elem_t * input,
    const elem_t * weights,
    const acc_t * bias,
    elem_t * output,

    int act, acc_scale_t scale, size_t relu6_shift,
    int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,

    enum tiled_matmul_type_t tiled_conv_type,
    size_t num_array){
    //size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id){

  int in_stride = (in_channels % 128 == 0) ? in_channels + 64 : in_channels;
  int weight_stride = (out_channels % 128 == 0) ? out_channels + 64 : out_channels;
#ifdef GEMMINI_ASSERTIONS
  if(out_stride % 128 == 0){
    printf("need padding\n");
    exit(1);
  }
#endif
  //printf("conv\n");

  tiled_opcode_conv_auto_multi(
     batch_size, in_dim, in_channels,
     out_channels, out_dim,
     stride, 1, dilation, padding, kernel_dim,
     in_stride, weight_stride, out_stride,
     false, false, false, false, // dfrom dram

     false, false, false, false, false,

     input, weights, bias, output,

     act, scale, relu6_shift,
     pool_size, pool_stride, pool_padding, pool_ceil_dim,
     tiled_conv_type,
     num_array);
     //orow_divide, batch_divide, cid, group_id);
}
static void sp_tiled_opcode_resadd(const size_t I, size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const elem_t * A, const elem_t * B, elem_t * C,
        size_t A_row_stride, size_t B_row_stride, size_t C_row_stride,
        bool relu, size_t num_array) {

    // Use the new mvin2 command to overlap mvin A, mvin B, and mvout C

    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
    const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

    int acc_J[4] = {0};
    int J_arr[4] = {0};
    J_arr[0] = ceil_divide_int(J, num_array * DIM) * DIM;
    acc_J[0] = 0;
    //printf("acc_J[0]: %d, J_arr[0]: %d\n", acc_J[0], J_arr[0]);
    size_t blocks[4] = {0};
    size_t rounded_up_J[4] = {0};
    blocks[0] = (J_arr[0]/DIM + (J_arr[0] % DIM != 0));
    if (blocks[0] > MAX_BLOCK_LEN) blocks[0] = MAX_BLOCK_LEN;
    rounded_up_J[0] = (J_arr[0] / DIM + (J_arr[0] % DIM != 0)) * DIM;

    for(int i = 1; i < num_array; i++){
      J -= J_arr[i-1];
      acc_J[i] += acc_J[i-1] + J_arr[i-1];
      J_arr[i] = ceil_divide_int(J, (num_array - i)*DIM) * DIM;
      //printf("acc_J[%d]: %d, J_arr[%d]: %d\n", i, acc_J[i], i, J_arr[i]);
      blocks[i] = (J_arr[i]/DIM + (J_arr[i] % DIM != 0));
      if (blocks[i] > MAX_BLOCK_LEN) blocks[i] = MAX_BLOCK_LEN;
      rounded_up_J[i] = (J_arr[i] / DIM + (J_arr[i] % DIM != 0)) * DIM;
    }

    // LD/ST
    // dram_addr, sp_addr, loop bounds, stride

    // Mvin A
    // printf("Mving A\n");
    gemmini_opcode_loop_one(OP3, A, 3, A_row_stride, I, J_arr[0], rounded_up_J[0]/DIM, 1);
    if(num_array >= 2)
      gemmini_opcode_loop_one(OP2, A+acc_J[1], 3, A_row_stride, I, J_arr[1], rounded_up_J[1]/DIM, 1);
    if(num_array >= 3)
      gemmini_opcode_loop_one(OP1, A+acc_J[2], 3, A_row_stride, I, J_arr[2], rounded_up_J[2]/DIM, 1);
       
    if(num_array == 4)
      gemmini_opcode_loop_one(OP0, A+acc_J[3], 3, A_row_stride, I, J_arr[3], rounded_up_J[3]/DIM, 1);
       
    /*
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J_arr[0]; j += blocks[0] * DIM) {
            const size_t cols = j + blocks[0]*DIM <= J_arr[0] ? blocks[0]*DIM : J_arr[0]-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            const elem_t * const A_dram_addr = A + i * A_row_stride + j;
            const uint32_t A_sp_addr = D_sp_addr_start + i * (rounded_up_J[0]/DIM) + j;

            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
        }
    }
    */


    // Mvin B
    // printf("Mving B\n");
    //gemmini_loop_one(B, 2, B_row_stride, I, J, rounded_up_J/DIM, 2);
    gemmini_opcode_loop_one(OP3, B, 2, B_row_stride, I, J_arr[0], rounded_up_J[0]/DIM, 2);
    if(num_array >= 2)
      gemmini_opcode_loop_one(OP2, B+acc_J[1], 2, B_row_stride, I, J_arr[1], rounded_up_J[1]/DIM, 2);
    if(num_array >= 3)
      gemmini_opcode_loop_one(OP1, B+acc_J[2], 2, B_row_stride, I, J_arr[2], rounded_up_J[2]/DIM, 2);
       
    if(num_array == 4)
      gemmini_opcode_loop_one(OP0, B+acc_J[3], 2, B_row_stride, I, J_arr[3], rounded_up_J[3]/DIM, 2);
       
/*
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            const elem_t * const B_dram_addr = B + i * B_row_stride + j;
            const uint32_t B_sp_addr = C_sp_addr_start + i * (rounded_up_J/DIM) + j;
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
        }
    }
*/
    // Mvout C from accumulator
    // printf("Mvout C from accumulator\n");
//    gemmini_loop_one(C, 2, C_row_stride, I, J, rounded_up_J/DIM, 0);
    gemmini_opcode_loop_one(OP3, C, 2, C_row_stride, I, J_arr[0], rounded_up_J[0]/DIM, 2);
    if(num_array >= 2)
      gemmini_opcode_loop_one(OP2, C, 2, C_row_stride, I, J_arr[1], rounded_up_J[1]/DIM, 2);
    if(num_array >= 3)
      gemmini_opcode_loop_one(OP1, C, 2, C_row_stride, I, J_arr[2], rounded_up_J[2]/DIM, 2);
       
    if(num_array == 4)
      gemmini_opcode_loop_one(OP0, C, 2, C_row_stride, I, J_arr[3], rounded_up_J[3]/DIM, 2);
    
  /*  
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            elem_t * const C_dram_addr = C + i * C_row_stride + j;
            const uint32_t C_sp_addr = D_sp_addr_start + i * (rounded_up_J/DIM) + j;
            gemmini_extended_mvout(C_dram_addr, C_sp_addr, cols, rows);
        }
    }
*/
}

// Compute MVIN_SCALE(A, A_scale) + MVIN_SCALE(B, B_scale) = C
static void tiled_opcode_resadd(const size_t I, const size_t J, const size_t stride,
        bool A_direct_dram, bool B_direct_dram, bool C_direct_dram,
        const size_t tile_I, const size_t tile_J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type,
        int num_array) {

  size_t outer_tile_J = num_array * tile_J;

    gemmini_opcode_extended_config_st(OP3, C_direct_dram, stride * sizeof(elem_t), relu ? RELU : NO_ACTIVATION, C_scale);
    gemmini_opcode_config_ex(OP3, WS, 0, 0, 0);

    gemmini_opcode_extended4_config_ld(OP3, A_direct_dram, stride * sizeof(elem_t), A_scale, true, DIM, 0);
    gemmini_opcode_extended4_config_ld(OP3, B_direct_dram, stride * sizeof(elem_t), B_scale, true, DIM, 1);

#ifndef BAREMETAL
    if(num_array >= 2){
      gemmini_opcode_extended_config_st(OP2, C_direct_dram, stride * sizeof(elem_t), relu ? RELU : NO_ACTIVATION, C_scale);
      gemmini_opcode_config_ex(OP2, WS, 0, 0, 0);

      gemmini_opcode_extended4_config_ld(OP2, A_direct_dram, stride * sizeof(elem_t), A_scale, true, DIM, 0);
      gemmini_opcode_extended4_config_ld(OP2, B_direct_dram, stride * sizeof(elem_t), B_scale, true, DIM, 1);
    }
    if(num_array >= 3){
      gemmini_opcode_extended_config_st(OP1, C_direct_dram, stride * sizeof(elem_t), relu ? RELU : NO_ACTIVATION, C_scale);
      gemmini_opcode_config_ex(OP1, WS, 0, 0, 0);

      gemmini_opcode_extended4_config_ld(OP1, A_direct_dram, stride * sizeof(elem_t), A_scale, true, DIM, 0);
      gemmini_opcode_extended4_config_ld(OP1, B_direct_dram, stride * sizeof(elem_t), B_scale, true, DIM, 1);
    }
    if(num_array == 4){
      gemmini_opcode_extended_config_st(OP0, C_direct_dram, stride * sizeof(elem_t), relu ? RELU : NO_ACTIVATION, C_scale);
      gemmini_opcode_config_ex(OP0, WS, 0, 0, 0);

      gemmini_opcode_extended4_config_ld(OP0, A_direct_dram, stride * sizeof(elem_t), A_scale, true, DIM, 0);
      gemmini_opcode_extended4_config_ld(OP0, B_direct_dram, stride * sizeof(elem_t), B_scale, true, DIM, 1);
    }
#endif

    for (size_t j = 0; j < J; j += outer_tile_J) {
      for (size_t i = 0; i < I; i += tile_I) {
            const size_t I_tile = i + tile_I <= I ? tile_I : I - i;
            const size_t J_tile = j + outer_tile_J <= J ? outer_tile_J : J - j; // aggregated
            const bool last = j + outer_tile_J > J;         
            if(last){
              if(J_tile <= num_array*DIM){
                num_array = ceil_divide_int(J_tile, DIM);
              }
            }
            const elem_t * a = A + i * stride + j;
            const elem_t * b = B + i * stride + j;
            elem_t * c = C + i * stride + j;

            sp_tiled_opcode_resadd(I_tile, J_tile,
                    A_scale, B_scale, a, b, c,
                    stride, stride, stride,
                    relu, num_array);
        }
    }

    gemmini_opcode_fence();
}



static void tiled_opcode_resadd_auto_multi(size_t I, size_t J,
    const scale_t A_scale,
    const scale_t B_scale,
    const acc_scale_t C_scale,
    const size_t J_stride,
    bool A_direct_dram, bool B_direct_dram, bool C_direct_dram,
    const elem_t * A,
    const elem_t * B,
    elem_t * C,
    bool relu,
    enum tiled_matmul_type_t matadd_type,
    size_t num_array){
//    size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id) {
  if (matadd_type == CPU) {
    resadd_cpu(I, J,
    A_scale, B_scale, C_scale, A, B, C,
    relu);
    return;
  }
  size_t orow_divide = 1;
  size_t batch_divide = 1;
  size_t cid = 0; size_t group_id = 0;

  size_t batch_cid = (size_t)(cid / orow_divide);
  size_t orow_cid = (size_t)(cid % orow_divide);

  size_t array_J = ceil_divide_int(J, num_array * DIM) * DIM;

  int args_in[] = {0, 0, 0, 0, 0};
  int target_util = 0;
  int * args = tiled_resadd_bubble_calculate(args_in, I, array_J, orow_divide, batch_divide, group_id, 0);

  size_t batch_size = I / batch_divide;
  I = batch_size;

  bool row_divisible = orow_divide > 1 && (I % orow_divide == 0);
  I = (row_divisible) ? I / orow_divide : I;
  size_t och_divide = (row_divisible) ? 1 : orow_divide; // if row is divisible, no need to divide channel

  size_t tile_I = I; // divide when it is divisible
  J = J / och_divide;
  size_t tile_J = J;

  int out_offset = (och_divide > 1) ? tile_J * orow_cid : 0; // no offset if divided in row dimension
  int orow_offset = (row_divisible) ? J_stride * orow_cid * I : 0;
  int batch_offset = (batch_divide > 1) ? batch_cid * batch_size * J_stride : 0;

  int window = args[0];
  int target_load = args[1];
  int ideal_cycles = args[2];
  tile_I = args[3];
  tile_J = args[4];

	//printf("ideal cycle: %d, target_cycle: %d, \n", ideal_cycles, target_cycles);
  //printf("priority: %d, window: %d, target_load: %d \n", priority, window, target_load);
 
    // printf("tile_I: %llu\n", tile_I);
    // printf("tile_J: %llu\n", tile_J);

  if (matadd_type == WS) {
    tiled_opcode_resadd(I, J, J_stride, A_direct_dram, B_direct_dram, C_direct_dram,
        tile_I, tile_J, 
        A_scale, B_scale, C_scale, A + batch_offset + out_offset + orow_offset, B + batch_offset + orow_offset + out_offset, C + batch_offset + orow_offset + out_offset,
        relu, matadd_type, num_array);
  } else if(matadd_type == CPU){
    resadd_cpu(I, J, A_scale, B_scale, C_scale,
        A, B, C, relu);
  }
  else {
    printf("Unsupported type\n");
    exit(1);
  }
}

static void tiled_opcode_resadd_auto_cid(size_t I, size_t J,
    const scale_t A_scale,
    const scale_t B_scale,
    const acc_scale_t C_scale,
    const elem_t * A,
    const elem_t * B,
    elem_t * C,
    bool relu,
    enum tiled_matmul_type_t matadd_type,
    size_t num_array){
    //size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id) {
 // printf("resadd\n");
  size_t J_stride = (J % 128 == 0) ? J + 64 : J;
  
  tiled_opcode_resadd_auto_multi(I, J, A_scale, B_scale, C_scale,
      J_stride,
      false, false, false,
      A, B, C,
      relu, matadd_type,
      num_array);
      //orow_divide, batch_divide, cid, group_id);

}
// assume DIM = 16, tile_poch = 16
static void sp_tiled_opcode_pool(
    int batch_size, int in_dim, int channels,
		int pool_out_dim, 
    int pool_size, int pool_stride, int pool_padding,
		int stride,
    bool input_direct_dram, bool output_direct_dram,

    int batches,
    int porows, int pocols, int pochs,
    int plpad, int prpad, int pupad, int pdpad,

    const elem_t * input,
    elem_t * output,
    
    int num_array)
{
    const int orows = porows * pool_stride + pool_size - 1 - pupad - pdpad;
    const int ocols = pocols * pool_stride + pool_size - 1 - plpad - prpad;
    const int ochs = pochs / num_array;

    int D_sp_addr_row = (D_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
    int C_sp_addr_row = (C_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;

    const uint32_t D_sp_addr_start = (1 << (ADDR_LEN - 1)) + D_sp_addr_row;
    const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) + C_sp_addr_row;
    int boroc = batches*orows*ocols;
    gemmini_opcode_extended2_config_st(OP3, input_direct_dram, stride * sizeof(elem_t), 0, 1, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, pupad, plpad);
    gemmini_opcode_extended4_config_ld(OP3, output_direct_dram, stride * sizeof(elem_t), MVIN_SCALE_IDENTITY, true, boroc, 2);
    if(num_array >= 2){
      gemmini_opcode_extended2_config_st(OP2, input_direct_dram, stride * sizeof(elem_t), 0, 1, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, pupad, plpad);
      gemmini_opcode_extended4_config_ld(OP2, output_direct_dram, stride * sizeof(elem_t), MVIN_SCALE_IDENTITY, true, boroc, 2);
    }
    if(num_array >= 3){
      gemmini_opcode_extended2_config_st(OP1, input_direct_dram, stride * sizeof(elem_t), 0, 1, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, pupad, plpad);
      gemmini_opcode_extended4_config_ld(OP1, output_direct_dram, stride * sizeof(elem_t), MVIN_SCALE_IDENTITY, true, boroc, 2);
    }
    if(num_array == 4){
      gemmini_opcode_extended2_config_st(OP0, input_direct_dram, stride * sizeof(elem_t), 0, 1, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, pupad, plpad);
      gemmini_opcode_extended4_config_ld(OP0, output_direct_dram, stride * sizeof(elem_t), MVIN_SCALE_IDENTITY, true, boroc, 2);
    }



  //  gemmini_opcode_extended4_config_ld(J_stride * sizeof(elem_t), B_scale, true, DIM, 1);


    const int max_ochs_per_mvin = ochs < MAX_BLOCK_LEN_ACC * DIM ? ochs : MAX_BLOCK_LEN_ACC * DIM;

	  for (int b = 0; b < batches; b++)
			for (int orow = 0; orow < orows; orow++)
				 for (int ocol = 0; ocol < ocols; ocol += DIM) {
					  const int I = ocols - ocol > DIM ? DIM : ocols - ocol;
            const int input_offset = (b*in_dim*in_dim + orow*in_dim + ocol) * stride;
            const int spad_offset = b * orows * ocols + orow * ocols + ocol;
            const int J = DIM;

					  //for (int och = 0; och < ochs; och += max_ochs_per_mvin) {
						//	const int J = ochs - och > max_ochs_per_mvin ? max_ochs_per_mvin : ochs - och;

							const uint32_t D_sp_addr = D_sp_addr_start + spad_offset;// (och / DIM) * batches * orows * ocols + ;

							gemmini_opcode_extended_mvin3(OP3, input + input_offset,
									  D_sp_addr,
									  J, I);
              if(num_array >= 2)
                gemmini_opcode_extended_mvin3(OP2, input + input_offset + DIM,
                      D_sp_addr,
                      J, I);
              if(num_array >= 3)
                gemmini_opcode_extended_mvin3(OP1, input + input_offset + 2*DIM,
                      D_sp_addr,
                      J, I);
              if(num_array == 4)
                gemmini_opcode_extended_mvin3(OP0, input + input_offset + 3*DIM,
                      D_sp_addr,
                      J, I);
			//		  }
				 }

		for (int b = 0; b < batches; b++) {
			 //for (int poch = 0; poch < pochs; poch += DIM) {
				  const int out_channels = DIM;//poch + DIM >= pochs ? pochs - poch : DIM;

				  elem_t * const pout = output + (b * pool_out_dim * pool_out_dim)*stride;// + poch;

				  const uint32_t C_sp_addr = C_sp_addr_start + b*orows*ocols;//(poch / DIM) * batches * orows * ocols + b * orows * ocols;

				  gemmini_opcode_extended_mvout(OP3, pout,
							 C_sp_addr,
							 out_channels, 0);
          if(num_array >= 2)
            gemmini_opcode_extended_mvout(OP2, pout + DIM,
                 C_sp_addr,
                 out_channels, 0);
          if(num_array >= 3)
            gemmini_opcode_extended_mvout(OP1, pout + 2 * DIM,
                 C_sp_addr,
                 out_channels, 0);
          if(num_array == 4)
            gemmini_opcode_extended_mvout(OP0, pout + 3 * DIM,
                 C_sp_addr,
                 out_channels, 0);
		//	 }
		}

}

static void tiled_opcode_pool(
    int batch_size, int in_dim, int channels,
		int pool_out_dim,
		int batches,
    int porows, int pocols, int pochs, // fix it to DIM (16)
    int out_stride,

    bool input_direct_dram, bool output_direct_dram,

		const elem_t * input,
    elem_t * pool_output,
		  
    int act, acc_scale_t scale, size_t relu6_shift,
    int pool_size, int pool_stride, int pool_padding,

    size_t num_array){
		//size_t orow_divide, size_t cid, size_t group_id, int window, int target_load) {

  size_t orow_divide = 1; size_t cid = 0; size_t group_id = 0; size_t window = 0; size_t target_load = 0;
	 //int out_stride = channels * och_divide;

    //gemmini_opcode_extended_config_st(out_stride * sizeof(elem_t), RELU, MVIN_SCALE_IDENTITY);
    gemmini_opcode_extended_config_ex(OP3, WEIGHT_STATIONARY, 0, 0, 0, 1, false, false);
    if(num_array >= 2)
      gemmini_opcode_extended_config_ex(OP2, WEIGHT_STATIONARY, 0, 0, 0, 1, false, false);
    if(num_array >= 3)
      gemmini_opcode_extended_config_ex(OP1, WEIGHT_STATIONARY, 0, 0, 0, 1, false, false);
    if(num_array >= 4)
      gemmini_opcode_extended_config_ex(OP0, WEIGHT_STATIONARY, 0, 0, 0, 1, false, false);
          
   
    int iter_pochs = pochs * num_array;

    bool row_divide = (orow_divide > 1);
    int out_row = (row_divide) ? pool_out_dim / orow_divide : pool_out_dim;
    size_t och_cid = (size_t)(cid % orow_divide);
    int porow_start = row_divide ? out_row * och_cid : 0;
    int porow_end = row_divide ? out_row * (och_cid + 1) : pool_out_dim;
 
    int num_array_store = num_array;
    for (int poch = 0; poch < channels; poch += iter_pochs) {
        const int pochs_ = channels - poch > iter_pochs ? iter_pochs : channels - poch;
        const bool last = channels - poch <= iter_pochs;         
        num_array_store = num_array;
        if(last){
          if(pochs_ <= num_array*DIM){
            num_array_store = ceil_divide_int(pochs_, DIM);
          }
        }
        for (int b = 0; b < batch_size; b += batches) {
           for (int porow = porow_start; porow < porow_end; porow += porows) {
               const int orow = porow * pool_stride - pool_padding;
               const int orow_floored = orow < 0 ? 0 : orow;        
               for (int pocol = 0; pocol < pool_out_dim; pocol += pocols) {
                  const int ocol = pocol * pool_stride - pool_padding;
                  const int ocol_floored = ocol < 0 ? 0 : ocol;
             
                  elem_t * out = pool_output + (b*pool_out_dim*pool_out_dim + porow*pool_out_dim + pocol) * out_stride + poch;
                  const elem_t * in = input + (b*in_dim*in_dim + orow_floored*in_dim + ocol_floored) * out_stride + poch;

                  // printf("batch: %d, poch: %d, porow: %d, pocol: %d\n", b, poch, porow, pocol);
                  const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                  const int porows_ = porow_end - porow > porows ? porows : porow_end - porow;
                  const int pocols_ = pool_out_dim - pocol > pocols ? pocols : pool_out_dim - pocol;
                  const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                  const int orows_ = porows_ * pool_stride + pool_size - 1;

                  const int plpad = ocol < 0 ? -ocol : 0;
                  const int prpad = ocol + ocols_ > in_dim ? ocol + ocols_ - in_dim : 0;
                  const int pupad = orow < 0 ? -orow : 0;
                  const int pdpad = orow + orows_ > in_dim ? orow + orows_ - in_dim : 0;

                 sp_tiled_opcode_pool(
                  batch_size, in_dim, channels,
                  pool_out_dim,
                  pool_size, pool_stride, pool_padding,
                  out_stride,
                  input_direct_dram, output_direct_dram,

                  batches_,
                  porows_, pocols_, pochs_,
                  plpad, prpad, pupad, pdpad,

                  in,
                  out, num_array_store);
               }
            }
        }
    }
    gemmini_opcode_fence();
}

// pooling using Gemmini DMA
static void tiled_opcode_pool_auto_multi(int batch_size, int channels, int in_dim,
    int pool_out_dim, int stride,
    int pool_size, int pool_stride, int pool_padding,
    bool input_direct_dram, bool output_direct_dram,
    const elem_t * A,
    elem_t * C,
    size_t num_array){
    //size_t och_divide, size_t batch_divide, size_t cid, size_t group_id) {
  
  bool relu = true;
	//int stride = channels;

  size_t och_divide = 1; size_t batch_divide = 1; size_t cid = 0; size_t group_id = 0;

  bool row_divide = (och_divide > 1 && channels < 64);
  int * args;
  int args_in[] = {0, 0, 0, 0};
  int target_util = 0;
  size_t och = DIM; // fix this to DIM

  args = tiled_pool_bubble_calculate(args_in, batch_size, in_dim, och, pool_out_dim, pool_size, pool_stride, pool_padding, row_divide, och_divide, batch_divide, cid, group_id);
  
  size_t batch_cid = (size_t)(cid / och_divide);
  size_t och_cid = (size_t)(cid % och_divide);


  batch_size = batch_size/batch_divide;
  channels = (row_divide) ? channels : channels / och_divide;
  //int pool_out_dim = (in_dim + 2*pool_padding - pool_size) / pool_stride + 1;
	int batch_in_offset = (batch_divide > 1) ? batch_size*in_dim*in_dim*stride*batch_cid : 0;
	int batch_out_offset = (batch_divide > 1) ? batch_size*pool_out_dim*pool_out_dim*stride*batch_cid : 0; // not dividing in out_channel dimension
 	const int out_offset = (och_divide > 1 && !row_divide) ? channels * och_cid : 0;
  if(!row_divide) och_divide = 1;

  int window = args[0];
  int target_load = args[1]; 
  const int batches = args[3];
  const int porows = args[4];
  const int pocols = args[5];
  const int pochs = args[6];
  //printf("window: %d, target_load: %d \n", window, target_load);

  window = 0;
  target_load = 0; // for now, disable CALM on pooling
  //printf("C dram addr before pool: 0x%08lx\n", C);
  tiled_opcode_pool(batch_size, in_dim, channels, pool_out_dim,
				batches, porows, pocols, pochs,
        stride,
        input_direct_dram, output_direct_dram, 
        A + batch_in_offset + out_offset, C + batch_out_offset + out_offset,	
				RELU, MVIN_SCALE_IDENTITY, 0,
				pool_size, pool_stride, pool_padding,
			  num_array);
        //och_divide, cid, group_id, window, target_load);
  
  //printf("C dram addr after pool: 0x%08lx\n", C);
}

static void tiled_opcode_pool_auto_cid(int batch_size, int channels, int in_dim,
    int pool_out_dim, int stride,
    int pool_size, int pool_stride, int pool_padding,
    const elem_t * A,
    elem_t * C,
    size_t num_array){
    //size_t och_divide, size_t batch_divide, size_t cid, size_t group_id){
  
  tiled_opcode_pool_auto_multi(batch_size, channels, in_dim,
      pool_out_dim, stride,
      pool_size, pool_stride, pool_padding,
      false, false,
      A, C,
      num_array);
      //och_divide, batch_divide, cid, group_id);
}


