// See LICENSE for license details.
#define OP0 3
#define OP1 3
#define OP2 3
#define OP3 3

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
#define gemmini_opcode_loop_conv_ws(OPCODE, num_array, batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, activation, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, in_stride, weight_stride, out_stride, input_direct_dram, weight_direct_dram, output_direct_dram, bias_direct_dram) \
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
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(num_array) << 32) | ((uint64_t)(max_pixels_per_row) << 8) | ((trans_input_3120) << 5) | ((trans_weight_0132) << 4) | ((trans_weight_1203) << 3) | ((trans_output_1203) << 2) | ((wrot180) << 1) | (no_bias), \
      ((activation) << 3)| ((input_dilated) << 2) | ((downsample) << 1) | (no_pool), \
      k_LOOP_CONV_WS) \
  }


// for different opcodes
// weight-stationary matmul loop
#define gemmini_opcode_loop_ws(OPCODE, num_array, I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, weightA) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(OPCODE, ((uint64_t)(num_array) << 32) | ((uint64_t)(weightA) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
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


static void sp_tiled_opcode_matmul_ws(const elem_t * A, const elem_t * B,
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
//  for (int op = 0; op < num_array; op++){
    bool ex_acc = !no_bias || D==NULL;
    int op = 0;
    gemmini_opcode_loop_ws(OP3, I, J, K, pad_I, pad_J, pad_K, A + op*K*DIM, B + B_row_stride*op*K*DIM, no_bias ? NULL : D, C,
      A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
      a_transpose, b_transpose,
      full_C, low_D, ex_acc,
      weightA);

    if(!no_bias) ex_acc = false;

    if(num_array == 4){
      op = 3;
      gemmini_opcode_loop_ws(OP0, I, J, K, pad_I, pad_J, pad_K, A + op*K*DIM, B + B_row_stride*op*K*DIM, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, ex_acc,
        weightA);
    }
    if(num_array >= 3){
      op = 2;
      gemmini_opcode_loop_ws(OP1, I, J, K, pad_I, pad_J, pad_K, A + op*K*DIM, B + B_row_stride*op*K*DIM, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, ex_acc,
        weightA);
    }
    if(num_array >= 2){
      op = 1;
      gemmini_opcode_loop_ws(OP2, I, J, K, pad_I, pad_J, pad_K, A + op*K*DIM, B + B_row_stride*op*K*DIM, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, ex_acc,
        weightA);
    }
//  }
  
}


// dim_I, dim_J are already fully dividied into subarray size
static void tiled_opcode_matmul_outer(size_t dim_I, size_t dim_J, size_t dim_K,
        //const size_t sub_num_I, const size_t sub_num_J, const size_t sub_num_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram, 
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t tile_I, size_t tile_J, size_t tile_K,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        uint8_t weightA,
        int dataflow,
        int num_array) {
  
  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const size_t I0 = dim_I_padded / (tile_I*DIM) + (dim_I_padded % (tile_I*DIM) != 0);
  const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
  const size_t K0 = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (tile_I*DIM) == 0 ? tile_I : (dim_I_padded/DIM) % tile_I;
  const size_t last_J = dim_J_padded % (tile_J*DIM) == 0 ? tile_J : (dim_J_padded/DIM) % tile_J;
  const size_t last_K = dim_K_padded % (tile_K*DIM) == 0 ? tile_K : (dim_K_padded/DIM) % tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - dim_I;
  const size_t padding_J = dim_J_padded - dim_J;
  const size_t padding_K = dim_K_padded - dim_K;

  const bool no_bias = D == NULL;

  if (no_bias) {
    D = (void*) 1; // Dummy address which isn't NULL
  }

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

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
    gemmini_opcode_extended_config_ex(OP3, dataflow, act, 0, relu6_shift, 1, a_transpose, b_transpose);
    gemmini_opcode_extended_config_st(OP3, C_direct_dram, stride_C * sizeof_C, act, scale);
    gemmini_opcode_extended3_config_ld(OP3, A_direct_dram, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
    gemmini_opcode_extended3_config_ld(OP3, B_direct_dram, stride_B * sizeof(elem_t), B_scale_factor, false, 1)
    gemmini_opcode_extended3_config_ld(OP3, D_direct_dram, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
             
//  }

  void (*inner)(const elem_t *, const elem_t *, const void *, void *,
        //size_t, size_t, size_t, 
        scale_t, scale_t, scale_acc_t,
        size_t, size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t,
        bool, bool,
        bool, bool,
        bool, bool,
        uint8_t,
        size_t);

  if (dataflow == OUTPUT_STATIONARY) {
    //inner = &sp_tiled_opcode_matmul_os;
  } else /* if (dataflow == WEIGHT_STATIONARY) */ {
    inner = &sp_tiled_opcode_matmul_ws;
  }
  // printf("I0: %d, J0: %d, K0: %d\n", I0, J0, K0);
  
  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++) {

        const void * pre;
        if (k0 != 0) {
          pre = NULL;
        } else {
          size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
          // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
          pre = (int8_t*)D + (bias_row * stride_D + j0 * tile_J * DIM)*sizeof_D;
        }

        void * out = k0 == K0-1 ? (int8_t*)C + (i0*tile_I*DIM*stride_C + j0*tile_J*DIM)*sizeof_C : NULL;

        const size_t I = i0 < I0-1 ? tile_I : last_I;
        const size_t J = j0 < J0-1 ? tile_J : last_J;
        const size_t K = k0 < K0-1 ? tile_K : last_K;

        const size_t pad_I = i0 == I0-1 ? padding_I : 0;
        const size_t pad_J = j0 == J0-1 ? padding_J : 0;
        const size_t pad_K = k0 == K0-1 ? padding_K : 0;

        const elem_t * a = a_transpose ? (A + k0*num_array*tile_K*DIM*stride_A + i0*tile_I*DIM)
          : (A + i0*tile_I*DIM*stride_A + k0*num_array*tile_K*DIM);

        const elem_t * b = b_transpose ? (B + j0*tile_J*DIM*stride_B + k0*num_array*tile_K*DIM)
          : (B + k0*num_array*tile_K*DIM*stride_B + j0*tile_J*DIM);

        (*inner)(a, b, pre, out,
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

  gemmini_opcode_fence();
}

// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
static void tiled_opcode_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        //const size_t sub_num_I, const size_t sub_num_J, const size_t sub_num_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram, 
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type,
        const size_t num_array) {

  dim_K = dim_K / num_array;

  size_t* args_out;
  size_t args[10];
  args_out = tiling_factor_matmul_calculate_auto(dim_I, dim_J, dim_K, 1, 1, 0, 0, args, 0);
  dim_I = args_out[3];
  dim_J = args_out[4];
  dim_K = args_out[5];
  size_t tile_I = args_out[8];
  size_t tile_J = args_out[9];
  size_t tile_K = args_out[10];

  /*
#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

    // "db_" means "double-buffered"
#define db_partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define db_mats_in_partition (db_partition_rows / DIM)
#define db_mats_in_acc ((ACC_ROWS / 2) / DIM)
#define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))
#define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)
    const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
    const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
    const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

    const bool double_buffered = tiled_matmul_type == WS;

    const size_t max_spad_rows = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
      BANK_NUM * BANK_ROWS;
    const size_t max_acc_rows = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

    size_t tile_I, tile_J, tile_K;

    if (double_buffered) {
       tile_I = dim_I_padded/DIM < db_max_tile_i_j ? dim_I_padded/DIM : db_max_tile_i_j;
       tile_J = dim_J_padded/DIM < db_max_tile_i_j ? dim_J_padded/DIM : db_max_tile_i_j;
       tile_K = dim_K_padded/DIM < db_max_tile_k ? dim_K_padded/DIM : db_max_tile_k;
    } else {
       tile_I = dim_I_padded/DIM < max_tile_i_j ? dim_I_padded/DIM : max_tile_i_j;
       tile_J = dim_J_padded/DIM < max_tile_i_j ? dim_J_padded/DIM : max_tile_i_j;
       tile_K = dim_K_padded/DIM < max_tile_k ? dim_K_padded/DIM : max_tile_k;
    }

    // Fill scratchpad as much as possible
    while (true) {
      bool increased = false;

      if (tiled_matmul_total_spad_rows(tile_I, tile_J+1, tile_K) <= max_spad_rows &&
          tiled_matmul_total_acc_rows(tile_I, tile_J+1) <= max_acc_rows &&
          (tile_J+1) * DIM <= dim_J_padded) {
        tile_J++;
        increased = true;
      }

      if (tiled_matmul_total_spad_rows(tile_I+1, tile_J, tile_K) <= max_spad_rows &&
          tiled_matmul_total_acc_rows(tile_I+1, tile_J) <= max_acc_rows &&
          (tile_I+1) * DIM <= dim_I_padded) {
        tile_I++;
        increased = true;
      }

      if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K+1) <= max_spad_rows &&
          (tile_K+1) * DIM <= dim_K_padded) {
        tile_K++;
        increased = true;
      }

      if (!increased)
        break;
    }
*/
    /*
    const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
    const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);

    printf("tile_I: %d\n", tile_I);
    printf("tile_J: %d\n", tile_J);
    printf("tile_K: %d\n\n", tile_J);

    printf("spad_rows: %d\n", spad_rows);
    printf("acc_rows: %d\n\n", acc_rows);

    printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
    printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);
#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
    */

    tiled_opcode_matmul_outer(dim_I, dim_J, dim_K,
        //sub_num_I, sub_num_J, sub_num_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        tile_I, tile_J, tile_K,
        act, scale, relu6_shift, repeating_bias,
        transpose_A, transpose_B,
        full_C, low_D, weightA,
        (int)tiled_matmul_type,
        num_array);
}

static void tiled_opcode_matmul_nn_auto_multi(size_t dim_I, size_t dim_J, size_t dim_K,
  //const size_t sub_num_I, const size_t sub_num_J, const size_t sub_num_K,
  size_t stride_A, size_t stride_B, size_t stride_C,
  bool A_direct_dram, bool B_direct_dram, bool D_direct_dram, bool C_direct_dram,  
  elem_t* A, elem_t* B,
  const void * D, elem_t* C,
  int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
  enum tiled_matmul_type_t tiled_matmul_type,
  size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id,
  const size_t num_array)
{

  dim_K = dim_K / num_array;


  size_t* args_out;
  size_t args[10];
  args_out = tiling_factor_matmul_calculate_auto(dim_I, dim_J, dim_K, 1, 1, 0, 0, args, 0);
  dim_I = args_out[3];
  dim_J = args_out[4];
  dim_K = args_out[5];
  size_t tile_I = args_out[8];
  size_t tile_J = args_out[9];
  size_t tile_K = args_out[10];

  size_t orow_offset_floor = args_out[6];
  bool row_divisible = (args_out[7] != 0);
  int window = args_out[0];
  int target_load = args_out[1];

  orow_divide = batch_divide * orow_divide;
  batch_divide = 1;
  //size_t total_divide = orow_divide * batch_divide; // number of cores for this workload

  if(!row_divisible) orow_divide = 1;
  int out_offset = (row_divisible) ? 0 : dim_J * cid; // no need to apply offset if we divided row
  int A_orow_offset = (row_divisible && cid != 0) ? stride_A * cid * dim_I + stride_A * orow_offset_floor : 0; // if row is divided, need offset it I dimension
  int C_orow_offset = (row_divisible && cid != 0) ? stride_C * cid * dim_I + stride_C * orow_offset_floor : 0; // if row is divided, need offset it I dimension
//  printf("dim_I: %d, orow_offset_floor: %d, A_row_offset: %d \n", dim_I, orow_offset_floor, A_orow_offset);
  int A_batch_offset = 0;
  int C_batch_offset = 0;
  if (batch_divide > 1){
     A_batch_offset = stride_A * cid * dim_I;
     C_batch_offset = stride_C * cid * dim_I;
  }

  bool no_bias = (D==NULL);
  
  tiled_opcode_matmul_outer(dim_I, dim_J, dim_K,
      //sub_num_I, sub_num_J, sub_num_K,
      A + A_orow_offset + A_batch_offset, B + out_offset, no_bias ? NULL : D + out_offset*sizeof(acc_t), C + C_orow_offset + out_offset + C_batch_offset,
      stride_A, stride_B, stride_B, stride_C,
      A_direct_dram, B_direct_dram, D_direct_dram, C_direct_dram,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      tile_I, tile_J, tile_K,
      act, scale, relu6_shift, repeating_bias,
      false, false, false, false, 3,
      (int)tiled_matmul_type,
      num_array);
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
        int krows, int kcols, int kchs,

        int lpad, int rpad, int upad, int dpad,
        int plpad, int prpad, int pupad, int pdpad,

        const elem_t * input,
        const elem_t * weights,
        elem_t * output,
        const acc_t * bias,

        int act, acc_scale_t scale,

        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        bool no_bias, bool no_pool, bool downsample, bool input_dilated,
        size_t num_array) {

  const int orows = porows * pool_stride + pool_size - 1 - pupad - pdpad;
  const int ocols = pocols * pool_stride + pool_size - 1 - plpad - prpad;
  const int ochs = pochs;

  // Calculate image dimensions
  // Note: "irows" and "icols" includes padding
  const int dilated_krows = krows + (kernel_dilation - 1)*(krows - 1);
  const int dilated_kcols = kcols + (kernel_dilation - 1)*(kcols - 1);
  int irows = orows * stride + dilated_krows - 1;
  int icols = ocols * stride + dilated_kcols - 1;
  int irows_unpadded = irows - upad - dpad;
  int icols_unpadded = icols - lpad - rpad;
  const int ichs = kchs;

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

  // Calculate spad address offsets
  const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
  const int in_channels_per_bank = kchs / DIM + (kchs % DIM != 0);
  const int B_rows = trans_weight_0132 ?
    in_channels_per_bank * kcols * krows * ochs :
    out_channels_per_bank * kcols * krows * kchs;

  static uint32_t D_sp_addr_row = 0;
  static uint32_t C_sp_addr_row = 0;

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - B_rows;
  const uint32_t D_sp_addr_start = (1 << (ADDR_LEN - 1)) + D_sp_addr_row;
  const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) + C_sp_addr_row;

  if (bias != 0) {
    D_sp_addr_row = (D_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  } 

  if (output != 0) {
    C_sp_addr_row = (C_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  }

  if(num_array == 4){
    gemmini_opcode_loop_conv_ws(OP0, batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, in_stride, weight_stride, out_stride, in_direct_dram, weight_direct_dram, out_direct_dram, bias_direct_dram);
  }
}
