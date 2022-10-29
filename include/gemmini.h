// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_H
#define SRC_MAIN_C_GEMMINI_H

#undef abs

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#include "include/gemmini_params.h"
#include "include/define.h"
#include "include/precompile.h"
//#define GEMMINI_ASSERTIONS
#define PRINT_MOCA 0

// Accelerator interface
#include "rocc-software/src/xcustom.h"

#define k_CONFIG 0
#define k_MVIN2 1
#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 6
#define k_FLUSH 7

#define k_LOOP_WS 8
#define k_LOOP_WS_CONFIG_BOUNDS 9
#define k_LOOP_WS_CONFIG_ADDRS_AB 10
#define k_LOOP_WS_CONFIG_ADDRS_DC 11
#define k_LOOP_WS_CONFIG_STRIDES_AB 12
#define k_LOOP_WS_CONFIG_STRIDES_DC 13

#define k_MVIN3 14

#define k_LOOP_CONV_WS 15
#define k_LOOP_CONV_WS_CONFIG_1 16
#define k_LOOP_CONV_WS_CONFIG_2 17
#define k_LOOP_CONV_WS_CONFIG_3 18
#define k_LOOP_CONV_WS_CONFIG_4 19
#define k_LOOP_CONV_WS_CONFIG_5 20
#define k_LOOP_CONV_WS_CONFIG_6 21

#define CONFIG_EX 0
#define CONFIG_LD 1
#define CONFIG_ST 2
#define CONFIG_MOCA 3

#define GARBAGE_ADDR ((uint32_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define NO_ACTIVATION 0
#define RELU 1
#define RELU6 2

#ifdef ELEM_T_IS_FLOAT
elem_t elem_t_bits_to_elem_t(elem_t_bits x) {
    union {
        elem_t_bits b;
        elem_t f;
    } un;

    un.b = x;
    return un.f;
}

elem_t_bits elem_t_to_elem_t_bits(elem_t x) {
    union {
        elem_t_bits b;
        elem_t f;
    } un;

    un.f = x;
    return un.b;
}

acc_t acc_t_bits_to_acc_t(acc_t_bits x) {
    union {
        acc_t_bits b;
        acc_t f;
    } un;

    un.b = x;
    return un.f;
}

acc_t_bits acc_t_to_acc_t_bits(acc_t x) {
    union {
        acc_t_bits b;
        acc_t f;
    } un;

    un.f = x;
    return un.b;
}

bool elem_t_isnan(elem_t x) {
    elem_t_bits bits = elem_t_to_elem_t_bits(x);
    uint64_t exp = (bits >> (ELEM_T_SIG_BITS-1)) & (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
    uint64_t sig = bits & (((uint64_t)1 << ELEM_T_SIG_BITS) - 1);
    bool is_nan_or_inf = exp == (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
    bool is_not_inf = sig != 0;
    return is_nan_or_inf && is_not_inf;
}

bool acc_t_isnan(acc_t x) {
    acc_t_bits bits = acc_t_to_acc_t_bits(x);
    uint64_t exp = (bits >> (ACC_T_SIG_BITS-1)) & (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
    uint64_t sig = bits & (((uint64_t)1 << ACC_T_SIG_BITS) - 1);
    bool is_nan_or_inf = exp == (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
    bool is_not_inf = sig != 0;
    return is_nan_or_inf && is_not_inf;
}
#endif

int ceil_divide_int(int a, int b){
  int c = (a % b == 0) ? ((int)(a/b)) :(((int)(a/b)) + 1); 
  if(a < b) c = 1;
  return c;
}

int round_divide_int(int a, int b){
  int c = (a % b == 0) ? ((int)(a/b)) : ((a % b) >= 0.5*b ? (((int)(a/b)) + 1) : (int)(a/b));
  if(a < b) c = 1;
  return c;
}

int round_int(float a){
  int int_a = (int)(a);
  if(int_a - a == 0){
    return int_a;
  }
  else
    return (int)(a + 0.5);
}

#ifdef HAS_MVIN_SCALE
static scale_t scale_t_bits_to_scale_t(scale_t_bits x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.b = x;
    return un.f;
}

static scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.f = x;
    return un.b;
}
#endif

#ifdef HAS_MVIN_ACC_SCALE
static scale_acc_t scale_acc_t_bits_to_scale_acc_t(scale_acc_t_bits x) {
    union {
        scale_acc_t_bits b;
        scale_acc_t f;
    } un;

    un.b = x;
    return un.f;
}

static scale_acc_t_bits scale_acc_t_to_scale_acc_t_bits(scale_acc_t x) {
    union {
        scale_acc_t_bits b;
        scale_acc_t f;
    } un;

    un.f = x;
    return un.b;
}
#endif

static acc_scale_t acc_scale_t_bits_to_acc_scale_t(acc_scale_t_bits x) {
    union {
        acc_scale_t_bits b;
        acc_scale_t f;
    } un;

    un.b = x;
    return un.f;
}

static acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
    union {
        acc_scale_t_bits b;
        acc_scale_t f;
    } un;

    un.f = x;
    return un.b;
}

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)

// mvin and mvout
#define gemmini_extended_mvin(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN)

#define gemmini_extended_mvin2(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN2)

#define gemmini_extended_mvin3(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN3)

#define gemmini_block_mvin(dram_addr, spad_addr, len) \
  gemmini_extended_mvin(dram_addr, spad_addr, (len) * DIM, DIM)

#define gemmini_mvin(dram_addr, spad_addr) \
  gemmini_extended_mvin(dram_addr, spad_addr, DIM, DIM)

#define gemmini_extended_mvout(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(spad_addr), k_MVOUT)

#define gemmini_mvout(dram_addr, spad_addr) \
  gemmini_extended_mvout(dram_addr, spad_addr, DIM, DIM)

// compute
#define gemmini_extended_compute_preloaded(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_PRELOADED)

#define gemmini_extended_compute_accumulated(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_ACCUMULATE)

#define gemmini_compute_preloaded(A, BD) \
  gemmini_extended_compute_preloaded(A, BD, DIM, DIM, DIM, DIM)

#define gemmini_compute_accumulated(A, BD) \
  gemmini_extended_compute_accumulated(A, BD, DIM, DIM, DIM, DIM)

// preload
#define gemmini_extended_preload(BD, C, BD_cols, BD_rows, C_cols, C_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), ((uint64_t)(C_rows) << (ADDR_LEN + 16)) | ((uint64_t)(C_cols) << ADDR_LEN) | (uint64_t)(C), k_PRELOAD)

#define gemmini_preload(BD, C) \
  gemmini_extended_preload(BD, C, DIM, DIM, DIM, DIM)

#define gemmini_preload_zeros(C) \
  gemmini_preload(GARBAGE_ADDR, C)

// config
#define gemmini_extended3_config_ex(dataflow, act, sys_shift, acc_scale, relu6_shift, C_stride, A_stride, A_transpose, B_transpose, ocol, row_turn, kdim, stride, channel, row_left, kdim2, weight_double_bank, weight_triple_bank, set_only_strides) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)acc_scale) << 32) | ((uint64_t)(A_stride) << 16) | (B_transpose << 9) | (A_transpose << 8) | ((set_only_strides) << 7) | ((act) << 3) | ((dataflow) << 2) | CONFIG_EX, ((uint64_t)(C_stride) << 48) | ((uint64_t)(relu6_shift) << 32) | (sys_shift), k_CONFIG)

#define gemmini_extended2_config_ex(dataflow, act, sys_shift, relu6_shift, A_stride, A_transpose, B_transpose, ocol, row_turn, kdim, stride, channel, row_left, kdim2, weight_double_bank, weight_triple_bank) \
  gemmini_extended3_config_ex(dataflow, act, sys_shift, ACC_SCALE_IDENTITY, relu6_shift, 1, A_stride, A_transpose, B_transpose, 0, 0, 0, 0, 0, 0, 0, 0, 0, false)
   
#define gemmini_extended_config_ex(dataflow, act, sys_shift, relu6_shift, A_stride, A_transpose, B_transpose) \
  gemmini_extended2_config_ex(dataflow, act, sys_shift, relu6_shift, A_stride, A_transpose, B_transpose, 0, 0, 0, 0, 0, 0, 0, 0, 0)

#define gemmini_config_ex(dataflow, act, sys_shift, relu6_shift) \
    gemmini_extended_config_ex(dataflow, act, sys_shift, relu6_shift, 1, 0, 0)

#define gemmini_extended4_config_ld(stride, scale, shrunk, block_mvin_stride, id) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(block_mvin_stride) << 16) | ((id) << 3) | ((shrunk) << 2) | CONFIG_LD, stride, k_CONFIG)

#define gemmini_extended3_config_ld(stride, scale, shrunk, id) \
  gemmini_extended4_config_ld(stride, scale, shrunk, DIM, id)

#define gemmini_extended2_config_ld(stride, scale, shrunk) \
  gemmini_extended3_config_ld(stride, scale, shrunk, 0)

#define gemmini_extended_config_ld(stride, scale) \
  gemmini_extended2_config_ld(stride, scale, false)

#define gemmini_config_ld(stride) \
  gemmini_extended_config_ld(stride, MVIN_SCALE_IDENTITY)

#define gemmini_extended2_config_st(stride, acc_act, acc_scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, upad, lpad) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(ocols) << 56) | ((uint64_t)(orows) << 48) | ((uint64_t)(pocols) << 40) | ((uint64_t)(porows) << 32) | ((uint64_t)(pool_out_dim) << 24) | ((uint64_t)(lpad) << 10) | ((uint64_t)(upad) << 8) | ((uint64_t)(pool_size) << 6) | ((uint64_t)(pool_stride) << 4) | ((acc_act) << 2) | CONFIG_ST, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)acc_scale) << 32) | ((uint32_t)stride), k_CONFIG)

#define gemmini_extended_config_st(stride, acc_act, acc_scale) \
    gemmini_extended2_config_st(stride, acc_act, acc_scale, 0, 0, 0, 0, 0, 0, 0, 0, 0)

#define gemmini_config_st(stride) \
    gemmini_extended_config_st(stride, NO_ACTIVATION, ACC_SCALE_IDENTITY)


// flush
#define gemmini_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

// fence
#define gemmini_fence() asm volatile("fence")

// for MOCA configuration
#define gemmini_config_calm(window, target_load) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_MOCA, ((uint64_t)(window) << 16) | ((uint64_t)(target_load)), k_CONFIG)

// weight-stationary matmul loop
#define gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, weightA) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(weightA) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

// weight-stationary matmul loop
#define gemmini_loop_conv_ws(batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, in_stride, weight_stride, out_stride) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(out_channels) << 48) | ((uint64_t)(in_channels) << 32) | ((uint64_t)(in_dim) << 16) | (uint64_t)(batch_size), \
      ((uint64_t)(padding) << 48) | ((uint64_t)(stride) << 32) | ((uint64_t)(pool_out_dim) << 16) | (uint64_t)(out_dim), k_LOOP_CONV_WS_CONFIG_1) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(kernel_dim) << 48) | ((uint64_t)(pool_size) << 32) | ((uint64_t)(pool_stride) << 16) | (uint64_t)(pool_padding), \
      ((uint64_t)(batches) << 48) | ((uint64_t)(porows) << 32) | ((uint64_t)(pocols) << 16) | (uint64_t)(pochs), k_LOOP_CONV_WS_CONFIG_2) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(krows) << 48) | ((uint64_t)(kcols) << 32) | ((uint64_t)(kchs) << 16) | (uint64_t)(lpad), \
      ((uint64_t)(rpad) << 48) | ((uint64_t)(upad) << 32) | ((uint64_t)(dpad) << 16) | (uint64_t)(plpad), k_LOOP_CONV_WS_CONFIG_3) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(orows) << 48) | ((uint64_t)(prpad) << 32) | ((uint64_t)(pupad) << 21) | ((uint64_t)(pdpad) << 10) | (uint64_t)(kernel_dilation), \
      ((uint64_t)(in_stride) << 48) | ((uint64_t)(weight_stride) << 32) | ((uint64_t)(out_stride) << 16) | (uint64_t)(ocols), k_LOOP_CONV_WS_CONFIG_4) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, weights, \
      output, k_LOOP_CONV_WS_CONFIG_5) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, bias, \
      input, k_LOOP_CONV_WS_CONFIG_6) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((trans_input_3120) << 5) | ((trans_weight_0132) << 4) | ((trans_weight_1203) << 3) | ((trans_output_1203) << 2) | ((wrot180) << 1) | (no_bias), \
      ((input_dilated) << 2) | ((downsample) << 1) | (no_pool), k_LOOP_CONV_WS) \
  }

static size_t tiled_matmul_total_spad_rows(size_t I, size_t J, size_t K) {
  return (I * K + K * J) * DIM;
}

static size_t tiled_matmul_total_acc_rows(size_t I, size_t J) {
  return (I * J) * DIM;
}

// MOCA: ran when executing each layer
// calculate tiling factor for matmul function
// detect contention using estimated BW needed
// calculate bubble, window cycles to configure MOCA hardware
size_t* tiling_factor_matmul_calculate_auto(size_t dim_I_in, size_t dim_J_in, size_t dim_K_in,
  size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id, size_t args[], int dram_util){

  int num_core = orow_divide > batch_divide ? orow_divide : batch_divide;
  if (orow_divide > 1 && batch_divide > 1)
    num_core = orow_divide + batch_divide;
  bool no_row_divide = false;
   if(dim_I_in <= DIM && dim_I_in % orow_divide == 0 && batch_divide == 1 && dim_J_in >= 1024)
     no_row_divide = true;

  bool fc_layer = (dim_I_in <= DIM) || (dim_J_in <= DIM);
  orow_divide = batch_divide * orow_divide;
  batch_divide = 1;

  bool row_divisible = (orow_divide > 1) && (dim_I_in % orow_divide == 0) && !no_row_divide;
  size_t orow_offset_floor = 0;
  size_t dim_I = dim_I_in;
  size_t dim_J = dim_J_in;
  size_t dim_K = dim_K_in;
  if(!row_divisible && orow_divide > 1 && (dim_J_in < DIM * orow_divide * 2 || dim_I_in > (DIM/2) * orow_divide)) {
  //if(!row_divisible && orow_divide > 1 && dim_I > DIM) { // for FC layers
    row_divisible = true;
    size_t dim_I_floor = dim_I_in / orow_divide;
    orow_offset_floor = dim_I_in - dim_I_floor * orow_divide;
    if(cid != 0) dim_I = dim_I_floor;
    else dim_I = dim_I_in - dim_I_floor * (orow_divide - 1);
  }
  else if(row_divisible){
    dim_I = dim_I_in / orow_divide;
  }
  size_t och_divide = (row_divisible) ? 1 : orow_divide;
  dim_I = dim_I / batch_divide; //batch dimension: I
  dim_J = dim_J / och_divide;


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

  const bool double_buffered = true;//tiled_matmul_type == WS;
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

  const size_t dim_I_in_padded = (dim_I_in / DIM + (dim_I_in % DIM != 0)) * DIM;
  const size_t dim_J_in_padded = (dim_J_in / DIM + (dim_J_in % DIM != 0)) * DIM;
  const size_t dim_K_in_padded = (dim_K_in / DIM + (dim_K_in % DIM != 0)) * DIM;

  size_t tile_I_in = dim_I_in_padded/DIM < db_max_tile_i_j ? dim_I_in_padded/DIM : db_max_tile_i_j;
  size_t tile_J_in = dim_J_in_padded/DIM < db_max_tile_i_j ? dim_J_in_padded/DIM : db_max_tile_i_j;
  size_t tile_K_in = dim_K_in_padded/DIM < db_max_tile_k ? dim_K_in_padded/DIM : db_max_tile_k;
  
  
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
  size_t row_divisible_size_t = (row_divisible) ? 1 : 0;
  args[8] = tile_I; args[9] = tile_J; args[10] = tile_K;
  args[3] = dim_I; args[4] = dim_J; args[5] = dim_K;
  args[6] = orow_offset_floor; args[7] = row_divisible_size_t;

  const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
  const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);
  const int spad_util = (spad_rows * 100) / max_spad_rows;
  const int acc_util = (acc_rows * 100) / max_acc_rows;

  const size_t I0 = dim_I_padded / (tile_I*DIM) + (dim_I_padded % (tile_I*DIM) != 0);
  const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
  const size_t K0 = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);


  // for pre-compilation of MOCA (number of load, runtime estimation)
  int A_load = 0;
  int B_load = 0;
  int D_load = 0;
  int A_size = dim_I_in * ceil_divide_int(dim_K_in, DIM);
  int B_size = dim_K_in * ceil_divide_int(dim_J_in, DIM);
  int C_size = dim_I_in * ceil_divide_int(dim_J_in, DIM);
  size_t num_tiles = I0 * J0 * K0;
  uint64_t total_from_dram = B_size + C_size;
  if(A_size > CACHE_SIZE || total_from_dram + A_size > CACHE_SIZE) {
//    added_image = 1;
    total_from_dram += A_size; // add for first layer
  }

   //MOCA config
  const uint64_t total_macs = dim_I * dim_J * dim_K;
  uint64_t ideal_runtime = (uint64_t)(total_macs / (DIM*DIM));
  if(fc_layer){
    A_load = dim_I * ceil_divide_int(dim_K, DIM);
    B_load = dim_K * ceil_divide_int(dim_J, DIM);
    D_load = dim_I * ceil_divide_int(dim_J, DIM) * 4; // bias has 4x bitwidth
    const int total_loads = A_load + B_load + D_load; 
    ideal_runtime = total_loads;
  }
  else{ 
    uint64_t num_K_tile = ceil_divide_int(dim_K, tile_K*DIM);
    uint64_t bias_time = (4 * tile_I * tile_J * DIM) / num_K_tile;
    //window = target tile runtime;
    for(size_t i0 = 0; i0 < dim_I; i0+=tile_I*DIM){
      int I = i0 + tile_I*DIM > dim_I ? dim_I - i0 : tile_I*DIM;
      for(size_t j0 = 0; j0 < dim_J; j0+=tile_J*DIM){
        int J = j0 + tile_J*DIM > dim_J ? dim_J - j0 : tile_J*DIM;
        int A_load_unit = I > DIM ? DIM : I;
        int B_load_unit = J > DIM ? DIM : J;
        D_load += ceil_divide_int(J * 4, B_load_unit) * ceil_divide_int(I, A_load_unit);//ceil_divide_int(I, A_load_unit) * 4; //ceil_divide_int(I*J, DIM);
        for (size_t k0 = 0; k0 < dim_K; k0+=tile_K*DIM) {
          int K = k0 + tile_K*DIM > dim_K ? dim_K - k0 : tile_K*DIM;
          int K_load_unit = K > DIM ? DIM : K;
          A_load += (ceil_divide_int(K, K_load_unit) * I);
          B_load += (ceil_divide_int(J, B_load_unit) * K);
        }
      }
    }
    bool full_power = false;
    uint64_t A_time = tile_I * DIM * tile_K;
    uint64_t B_time = tile_J * tile_K * DIM;
    int tile_ideal = tile_I * tile_J * tile_K * DIM;
    //if(bias_time + A_time + B_time > target_tile_runtime || (tile_ideal > target_tile_runtime)) full_power = true;
    
    int fresh_B_load = dim_J * dim_K / DIM;
    //printf("number of fresh B load: %d\n", fresh_B_load);
    //printf("number of tiles: %d, target runtime: %d, bias_time: %d,  target load: %d, window: %d, A load: %d, B load: %d, D load: %d\n", num_tiles, target_runtime, bias_time, target_load, window, A_load, B_load, D_load);
    //target_load = (int)((A_load + B_load + D_load)/num_tiles);
    if(spad_util < acc_util){
    //  target_load = (int)((target_load * acc_util) / spad_util);
    //  window = (int)((window * acc_util) / spad_util);
      num_tiles = (size_t)((num_tiles * spad_util) / acc_util);
    }
  }

  // figures out whether it is from cache or DRAM
  int inner_tile_A = tile_I_in * DIM * (ceil_divide_int)(dim_K_in, DIM);
  int inner_tile_B = ceil_divide_int(dim_J_in, DIM) * dim_K_in;
  int outer_loop_iter_A = 1;
  int outer_loop_iter_B = (ceil_divide_int)(dim_I_in, tile_I_in*DIM);
  num_tiles = I0 * J0 * K0;
  int D_from_dram = ceil_divide_int(dim_J_in, DIM) * 4 * ceil_divide_int(dim_I_in, tile_I_in * DIM);
  D_from_dram = D_from_dram / num_core;

  if(inner_tile_A + inner_tile_B + (C_size / outer_loop_iter_B) > CACHE_SIZE){
    total_from_dram += ((outer_loop_iter_B - 1) * inner_tile_B)/2;
  }
  total_from_dram += D_from_dram; //D_load * och_divide;
  uint64_t total_mem = A_load + B_load + D_load +dim_I * ceil_divide_int(dim_J, DIM); 
  uint64_t mem_ideal = total_from_dram / DRAM_BW + (total_mem - total_from_dram/num_core);
  uint64_t ideal_prediction = MAX(mem_ideal, ideal_runtime) + MIN(mem_ideal, ideal_runtime) * 0.5;

  int workload_type = total_queue_type[gemmini_queue_id[group_id]];
  int queue_id = gemmini_queue_id[group_id];
#if PRINT_MOCA != 1
  // replace with pre-compiled data
  total_from_dram = from_dram[workload_type-1][total_queue_conv[queue_id]];
  ideal_prediction = conv_prediction_cycles[workload_type-1][total_queue_conv[queue_id]]; 
#endif
  if(cid == 0 && dram_util == -1) gemmini_dram_util[group_id] = 0;

  // ideal exepected dram bandwidth
  int ideal_dram_bw_exp = (100 * total_from_dram) / ideal_prediction;
  int ideal_dram_util = (ideal_dram_bw_exp / DRAM_BW);

  uint64_t dispatch_cycle = total_queue_dispatch[queue_id];
  uint64_t end = read_cycles();
  uint64_t this_cycles = end - gemmini_start_time[group_id];
  uint64_t slack = (this_cycles > dispatch_cycle) ? this_cycles - dispatch_cycle : total_queue_target[queue_id];
  int priority = total_queue_priority[queue_id];
  // MOCA runtime dynamic priority score
  int this_score = (1+priority)/4 + round_divide_int(10*total_queue_togo[queue_id], slack);//max(1, (int)((10*total_queue_togo[queue_id])/slack));
  // update for the next conv layer
  if(cid == 0){ 
    //gemmini_estimate_togo[group_id] -= conv_prediction_cycles[workload_type][gemmini_num_conv[group_id]];
    //gemmini_num_conv[group_id] ++;
    gemmini_score[group_id] = this_score;
  }
  // detect contention and partition
  int other_dram_util = 0;
  int other_score = 0;
  int other_weight_sum = 0;
//  int this_score = gemmini_score[group_id];
  for(int i = 0; i < NUM_SUB_GROUP; i++)
    if(i != group_id) {
      other_score += this_score;//gemmini_score[i];
      other_dram_util += gemmini_dram_util[i];
      other_weight_sum += this_score * gemmini_dram_util[i];
    }

  if(dram_util == 0){
    // contention detected
    if(ideal_dram_util + other_dram_util > DRAM_MAX_UTIL){
      int excess = ideal_dram_util + other_dram_util - DRAM_MAX_UTIL;
      // partition overflowing amount
      dram_util = ideal_dram_util - (int)((excess * other_weight_sum) / (this_score * ideal_dram_util + other_weight_sum));
      //dram_util = ideal_dram_util - (int)((excess * ideal_dram_util * other_score) / (this_score * ideal_dram_util + other_score * other_dram_util));
      dram_util = MAX(25, dram_util); 
    }
    else{ // if no contention detected
      dram_util = -1; // don't really have to use memory modulation
    }
    if(cid == 0) gemmini_dram_util[group_id] = ideal_dram_util;//(dram_util != -1) ? dram_util : ideal_dram_util;//ideal_dram_util;
  }


  uint64_t prediction = (100 * total_from_dram) / (DRAM_BW * dram_util);

  int window = prediction / num_tiles;
  int target_load = (int)((A_load + B_load + D_load)/num_tiles);
  // if too short running layer, then skip
  if(dram_util >= ideal_dram_util || num_tiles < 4){
    window = 0;
    target_load = 0;
  } 
  window = (dram_util == -1) ? 0 : window;
  target_load = (dram_util == -1) ? 0 : target_load;

// prints for runtime estimation (offline pre-compiler)
#if PRINT_MOCA == 1
  printf("window: %d, target load: %d, prediction cycles: %llu \n", window, target_load, prediction);
  // for pre-compilation
  printf("compute_ideal: %llu, mem_ideal: %llu, ideal prediction cycles: %llu, ideal dram bw usage: %d, ideal dram bw util: %d, result dram bw util: %d\n", ideal_runtime, mem_ideal, ideal_prediction, ideal_dram_bw_exp, ideal_dram_util, dram_util);
  printf("total A load: %d, total B load: %d, total D load: %d, raw D: %d \n", A_load, B_load, D_load, D_from_dram);
  printf("A size: %d, B size: %d, C size: %d \n", A_size, B_size, C_size);
  printf("inner tile A: %d, inner tile B: %d, outer loop iteration A: %d, outer loop iteration B: %d \n", inner_tile_A, inner_tile_B, outer_loop_iter_A, outer_loop_iter_B);
  printf("number of tile: %d, target load per tile: %d, ideal runtime: %llu\n\n", num_tiles, (A_load + B_load + D_load) / num_tiles, ideal_runtime);
#endif
  args[0] = window;
  args[1] = target_load;
  args[2] = ideal_prediction;
  return args;
}

static void sp_tiled_matmul_os(const elem_t * A, const elem_t * B, const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        uint8_t weightA) {

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = (3 << (ADDR_LEN-2)) | (full_C << (ADDR_LEN-3));

  const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
  const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
  const int D_blocks = J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC;

  // Move-in D
  if (D != NULL && !no_bias) {
    const size_t D_stride = repeating_bias ? 0 : D_row_stride * sizeof(acc_t);
    gemmini_extended_config_ld(D_stride, D_scale_factor);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const acc_t * const D_dram_addr = (acc_t *)D + (bias_row * D_row_stride + j)*DIM;

        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;

        const size_t blocks = j + D_blocks <= J ? D_blocks : J-j;

        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        const size_t rows = DIM - (i == I-1 ? pad_I : 0);

        gemmini_extended_mvin(D_dram_addr, D_sp_addr_acc, cols, rows);
      }
    }
  }

  // Move-in B
  gemmini_extended_config_ld(B_row_stride * sizeof(elem_t), B_scale_factor);
  for (size_t j = 0; j < J; j += B_blocks) {
    for (size_t k = 0; k < K; k++) {
      const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
      const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
      const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
      const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
      const size_t rows = DIM - (k == K-1 ? pad_K : 0);
      gemmini_extended_mvin(B_dram_addr, B_sp_addr, cols, rows);
    }
  }

  // Move-in A
  gemmini_extended_config_ld(A_row_stride * sizeof(elem_t), A_scale_factor);
  for (size_t i = 0; i < I; i++) {
    for (size_t k = 0; k < K; k += A_blocks) {
      const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
      const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
      const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
      const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
    }
  }

  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j++) {
      const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

      for (size_t k = 0; k < K; k++) {

        const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

        uint32_t out_sp_addr = k == K-1 ? C_sp_addr : GARBAGE_ADDR;

        // If we're not using a bias, then we want to overwrite what's in the
        // accumulator, rather than writing over it
        int no_bias_new_matrix = no_bias && D != NULL && k == K-1;
        if (no_bias_new_matrix) {
          out_sp_addr &= ~(1 << (ADDR_LEN-2));
        }

        const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
        const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
        const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);

        if (k == 0) { // First iteration
          gemmini_extended_compute_preloaded(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
        } else { // All other iterations
          gemmini_extended_compute_accumulated(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
        }
      }
    }
  }

  // Move-out C
  if (C != NULL) {
    const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j++) {
        void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + j)*DIM*sizeof_C;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        gemmini_extended_mvout(C_dram_addr, C_sp_addr, C_cols, C_rows);
      }
    }
  }
}

static void sp_tiled_matmul_ws(const elem_t * A, const elem_t * B,
        const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        uint8_t weightA) {

  /*
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
      }
    }
  }

  for (size_t j = 0; j < J; j++) {
    for (size_t k = 0; k < K; k++) {
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
          }
        } else {
          if (j == 0 && k % A_blocks == 0) {
            const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
            const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
            const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
            const size_t rows = DIM - (i == I-1 ? pad_I : 0);
            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
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
          }
        } else {
          if (i == 0 && j % B_blocks == 0) {
            const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
            const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
            const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
            const size_t rows = DIM - (k == K-1 ? pad_K : 0);
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
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

          if (i == 0) { // First iteration
            gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
          } else { // All other iterations
            gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
          }
        }

        // Move-out C
        if (C != NULL && k == K-1 && (j == J-1 || j % C_blocks == C_blocks-1)) {
          const size_t rounded_j = (j / C_blocks) * C_blocks;

          const uint32_t rounded_C_sp_addr = C_sp_addr_start + (i*J + rounded_j)*DIM;
          void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + rounded_j)*DIM*sizeof_C;

          const size_t blocks = rounded_j + C_blocks <= J ? C_blocks : J-rounded_j;
          const size_t cols = blocks * DIM - (rounded_j + blocks >= J ? pad_J : 0);
          const size_t rows = DIM - (i == I - 1 ? pad_I : 0);

          gemmini_extended_mvout(C_dram_addr, rounded_C_sp_addr, cols, rows);
        }
      }
    }
  }
  */

   // Combined loop
  gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
    A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
    a_transpose, b_transpose,
    full_C, low_D, !no_bias || D == NULL,
    weightA);
}

static void tiled_matmul_outer(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t tile_I, size_t tile_J, size_t tile_K,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        uint8_t weightA,
        int dataflow,
        int window, int target_load) {
        // window: time window to monitor


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

  gemmini_extended_config_ex(dataflow, act, 0, relu6_shift, 1, a_transpose, b_transpose);
  gemmini_extended_config_st(stride_C * sizeof_C, act, scale);
  gemmini_config_calm(window, target_load);
  gemmini_extended3_config_ld(stride_A * sizeof(elem_t), A_scale_factor, false, 0);
  gemmini_extended3_config_ld(stride_B * sizeof(elem_t), B_scale_factor, false, 1)
  gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);

  void (*inner)(const elem_t *, const elem_t *, const void *, void *,
        scale_t, scale_t, scale_acc_t,
        size_t, size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t,
        bool, bool,
        bool, bool,
        bool, bool,
        uint8_t);

  if (dataflow == OUTPUT_STATIONARY) {
    inner = &sp_tiled_matmul_os;
  } else {
    inner = &sp_tiled_matmul_ws;
  }

  for (size_t i0 = 0; i0 < I0; i0++)
    //for (size_t j0 = 0; j0 < J0; j0++)
    for(size_t j1 = 0; j1 < J0; j1++){
      size_t j0 = ((int)(i0) % 2 == 0) ? j1 : J0 - j1 - 1;
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

        const elem_t * a = a_transpose ? (A + k0*tile_K*DIM*stride_A + i0*tile_I*DIM)
          : (A + i0*tile_I*DIM*stride_A + k0*tile_K*DIM);

        const elem_t * b = b_transpose ? (B + j0*tile_J*DIM*stride_B + k0*tile_K*DIM)
          : (B + k0*tile_K*DIM*stride_B + j0*tile_J*DIM);

        (*inner)(a, b, pre, out,
            A_scale_factor, B_scale_factor, D_scale_factor,
            I, J, K,
            pad_I, pad_J, pad_K,
            stride_A, stride_B, stride_D, stride_C,
            a_transpose, b_transpose,
            full_C, low_D,
            no_bias, repeating_bias,
            weightA);
      }
    }
  gemmini_fence();
}

static elem_t scale_and_sat(acc_t x, int act, acc_scale_t scale, size_t relu6_shift) {
  // Scale value down and round it
  x = ACC_SCALE(x, scale);
  // Clip result
  x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
  // Apply activation function
  if (act == RELU) {
    x = x < 0 ? 0 : x;
  }
  // TODO add another define to check if relu6_shift is actually used or not
  else if (act == RELU6) {
    int max = 6 << relu6_shift;
    x = x < 0 ? 0 : (x > max ? max : x);
  }
  return x;
}

#ifdef HAS_MVIN_SCALE
#define GEMMINI_SCALE(x, scale) MVIN_SCALE((x), (scale))
#else
#define GEMMINI_SCALE(x, scale) (x)
#endif

#ifdef HAS_MVIN_ACC_SCALE
#define GEMMINI_ACC_SCALE(x, scale) MVIN_SCALE_ACC((x), (scale))
#else
#define GEMMINI_ACC_SCALE(x, scale) (x)
#endif

static void matmul_cpu(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
        const elem_t* A, const elem_t* B, const acc_t * D,
        elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias) {

  const int no_bias = D == NULL;
  if (!transA && !transB && DIM_I % 4 == 0 && DIM_J % 4 == 0) {
    for (size_t i = 0; i < DIM_I; i += 4) {
      for (size_t j = 0; j < DIM_J; j += 4) {

        acc_t result[4][4]; // = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        for (size_t ii = 0; ii < 4; ii++)
          for (size_t jj = 0; jj < 4; jj++) {
            const size_t bias_row = repeating_bias ? 0 : i + ii;
            result[ii][jj] = no_bias ? 0 :
              GEMMINI_ACC_SCALE(*(D + bias_row*stride_D + j + jj), D_scale_factor);
          }

        for (size_t k = 0; k < DIM_K; k++) {
          result[0][0] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[0][1] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[0][2] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[0][3] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[1][0] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[1][1] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[1][2] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[1][3] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[2][0] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[2][1] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[2][2] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[2][3] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[3][0] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[3][1] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[3][2] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[3][3] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
        }

        *(C + i*stride_C + j) =
             scale_and_sat(result[0][0], act, scale, relu6_shift);
        *(C + i*stride_C + j+1) =
             scale_and_sat(result[0][1], act, scale, relu6_shift);
        *(C + i*stride_C + j+2) =
             scale_and_sat(result[0][2], act, scale, relu6_shift);
        *(C + i*stride_C + j+3) =
             scale_and_sat(result[0][3], act, scale, relu6_shift);
        *(C + (i+1)*stride_C + j) =
             scale_and_sat(result[1][0], act, scale, relu6_shift);
        *(C + (i+1)*stride_C + j+1) =
             scale_and_sat(result[1][1], act, scale, relu6_shift);
        *(C + (i+1)*stride_C + j+2) =
             scale_and_sat(result[1][2], act, scale, relu6_shift);
        *(C + (i+1)*stride_C + j+3) =
             scale_and_sat(result[1][3], act, scale, relu6_shift);
        *(C + (i+2)*stride_C + j) =
             scale_and_sat(result[2][0], act, scale, relu6_shift);
        *(C + (i+2)*stride_C + j+1) =
             scale_and_sat(result[2][1], act, scale, relu6_shift);
        *(C + (i+2)*stride_C + j+2) =
             scale_and_sat(result[2][2], act, scale, relu6_shift);
        *(C + (i+2)*stride_C + j+3) =
             scale_and_sat(result[2][3], act, scale, relu6_shift);
        *(C + (i+3)*stride_C + j) =
             scale_and_sat(result[3][0], act, scale, relu6_shift);
        *(C + (i+3)*stride_C + j+1) =
             scale_and_sat(result[3][1], act, scale, relu6_shift);
        *(C + (i+3)*stride_C + j+2) =
             scale_and_sat(result[3][2], act, scale, relu6_shift);
        *(C + (i+3)*stride_C + j+3) =
             scale_and_sat(result[3][3], act, scale, relu6_shift);
      }
    }
  } else {
    size_t A_dim_strides[2] = {!transA ? stride_A : 1, !transA ? 1 : stride_A}; // i, j stride
    size_t B_dim_strides[2] = {!transB ? 1 : stride_B, !transB ? stride_B : 1}; // j, k stride
    for (size_t i = 0; i < DIM_I; i++) {
      for (size_t j = 0; j < DIM_J; j++) {
        elem_t* c = C + (i * stride_C) + j;

        const size_t bias_row = repeating_bias ? 0 : i;
        acc_t sum = no_bias ? 0 : GEMMINI_ACC_SCALE(*(D + bias_row * stride_D + j), D_scale_factor);

        for (size_t k = 0; k < DIM_K; k++) {
          const elem_t* a = A + i * A_dim_strides[0] + k * A_dim_strides[1];
          const elem_t* b = B + j * B_dim_strides[0] + k * B_dim_strides[1];
          sum += (GEMMINI_SCALE(*a, A_scale_factor) * GEMMINI_SCALE(*b, B_scale_factor));
        }
        *c = scale_and_sat(sum, act, scale, relu6_shift);
      }
    }
  }
}

#undef GEMMINI_SCALE

// General matmul which can be run with different dataflows, or on the CPU
enum tiled_matmul_type_t {OS, WS, CPU}; // TODO rename this so it's name also applies to convs

// This function runs a tiled matrix multiplication, with hardcoded tiling
// factors
static void tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type,
        int window, int target_load) {
#ifdef GEMMINI_ASSERTIONS
  // Make sure that the tiling factors make sense
  if (tile_I <= 0) {
    printf("tile_I is non-positive\n");
    exit(1);
  } else if (tile_J <= 0) {
    printf("tile_J is non-positive\n");
    exit(1);
  } else if (tile_K <= 0) {
    printf("tile_K is non-positive\n");
    exit(1);
  }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  if (tile_I * DIM > dim_I_padded) {
    printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
    exit(1);
  } else if (tile_J * DIM > dim_J_padded) {
    printf("tile_J is too large (tile_J * DIM > dim_J_padded)\n");
    exit(1);
  } else if (tile_K * DIM > dim_K_padded) {
    printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
    exit(1);
  }

  const bool double_buffered = tiled_matmul_type == WS;

  const size_t total_spad_size = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
      BANK_NUM * BANK_ROWS;
  const size_t total_acc_size = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

  const size_t total_spad_rows =
      (tile_I * tile_K * DIM) +   // Rows to store A
      (tile_K * tile_J * DIM);    // Rows to store B

  if (total_spad_rows > total_spad_size) {
    printf("Not enough space in scratchpad to store A and B matrices\n");
    exit(1);
  }

  const size_t total_acc_rows =
      tile_I * tile_J * DIM;      // Rows to store C

  if (total_acc_rows > total_acc_size) {
    printf("Not enough space in accumulator to store C\n");
    exit(1);
  }

  if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535) {
    printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
    exit(1);
  }

  char matmul_type_str[][4] = {"OS", "WS", "CPU"};

  // Check if transpose options are correct
  if (((tiled_matmul_type == OS) && (transpose_A || transpose_B)) ||
    (tiled_matmul_type == WS && transpose_A && transpose_B)) {
    printf("Not implemented: %s matmul, a_transpose=%d, b_transpose=%d\n", matmul_type_str[tiled_matmul_type], transpose_A, transpose_B);
    exit(1);
  }

  // Check if full_C options are correct
  if ((tiled_matmul_type == CPU && (full_C || low_D)) ||
      (tiled_matmul_type == OS && low_D)) {
    printf("Not implemented: %s matmul, full_C=%d, low_D=%d\n", matmul_type_str[tiled_matmul_type], full_C, low_D);
  }
#endif

  // Run a tiled matrix multiplication on either Gemmini or the CPU
  if (tiled_matmul_type == OS || tiled_matmul_type == WS) {
    tiled_matmul_outer(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        tile_I, tile_J, tile_K,
        act, scale, relu6_shift, repeating_bias,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        (int)tiled_matmul_type,
        window, target_load);
  } else /*if (tiled_matmul_type == CPU)*/ {
    matmul_cpu(transpose_A, transpose_B, dim_I, dim_J, dim_K,
            A, B, (const acc_t*) D, (elem_t*)C,
            stride_A, stride_B, stride_D, stride_C,
            A_scale_factor, B_scale_factor, D_scale_factor,
            act, scale, relu6_shift, repeating_bias);
  }
}

// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {

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
    */

    tiled_matmul(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        act, scale, relu6_shift, repeating_bias,
        tile_I, tile_J, tile_K,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        tiled_matmul_type,
        0, 0);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}

static void sp_tiled_conv_A_stride(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim, int pool_out_dim,
        int in_stride, int out_stride, int weight_stride,

        int stride, int padding, int kernel_dim, int kernel_dilation,

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

        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        bool no_bias, bool no_pool, bool downsample, bool input_dilated) {

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

  gemmini_loop_conv_ws(batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, in_stride, weight_stride, out_stride);

  /*
  // mvin bias
  if (bias != NULL) {
    // TODO we probably don't need quite this many nested loops for this part

    const int max_ochs_per_mvin = ochs < MAX_BLOCK_LEN_ACC * DIM ? ochs :
        MAX_BLOCK_LEN_ACC * DIM;

    gemmini_extended4_config_ld(0, MVIN_SCALE_IDENTITY, false, batches * orows * ocols, 2);

    for (int b = 0; b < batches; b++)
      for (int orow = 0; orow < orows; orow++)
        for (int ocol = 0; ocol < ocols; ocol += DIM) {
          const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

          for (int och = 0; och < ochs; och += max_ochs_per_mvin) {
            const int J = ochs - och > max_ochs_per_mvin ? max_ochs_per_mvin : ochs - och;

            const uint32_t D_sp_addr = D_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

            const acc_t * bias_dram_addr = no_bias ? NULL : bias + och;

            gemmini_extended_mvin3(bias_dram_addr,
                    D_sp_addr,
                    J, I);
          }
        }
  }

  // mvin input
  {
    int max_chs_per_mvin = ichs < MAX_BLOCK_LEN * DIM ? ichs :
      MAX_BLOCK_LEN * DIM;
    if (trans_input_3120) {
      max_chs_per_mvin = batches < MAX_BLOCK_LEN * DIM ? batches :
        MAX_BLOCK_LEN * DIM;
    }

    const int dram_stride = trans_input_3120 ?
      batch_size * sizeof(elem_t) :
      in_channels * sizeof(elem_t);

    const int spad_stride = trans_input_3120 ?
      ichs * (irows >> downsample) * (icols >> downsample) :
      batches * (irows >> downsample) * (icols >> downsample);

    gemmini_extended4_config_ld(dram_stride << downsample, MVIN_SCALE_IDENTITY, false, spad_stride, 0);

    const int b_it = trans_input_3120 ? max_chs_per_mvin : 1;
    const int ich_it = trans_input_3120 ? 1 : max_chs_per_mvin;

    for (int b = 0; b < batches; b += b_it)
      for (int irow = -UNDILATED(upad); irow < irows_unpadded + UNDILATED(dpad); irow += 1 + downsample) {
        const int irow_padded = irow + UNDILATED(upad);

        for (int icol = -UNDILATED(lpad); icol < icols_unpadded + UNDILATED(rpad);) {
          // TODO There might be some unnecessary mvins here at the edge of the image

          int I = icols_unpadded - icol > (DIM << downsample) ?
            (DIM << downsample) : icols_unpadded - icol;

          if (icol < 0) {
            I = -icol > DIM ? DIM : -icol;
          } else if (icol >= icols_unpadded) {
            I = icols_unpadded + UNDILATED(rpad) - icol > DIM ? DIM : icols_unpadded + UNDILATED(rpad) - icol;
          }

          const int icol_padded = icol + UNDILATED(lpad);

          for (int ich = 0; ich < ichs; ich += ich_it) {
            int K = ichs - ich > max_chs_per_mvin ?
              max_chs_per_mvin : ichs - ich;
            if (trans_input_3120) {
              K = batches - b > max_chs_per_mvin ?
                max_chs_per_mvin : batches - b;
            }

#define DS(x) ((x) >> (downsample))

            uint32_t A_sp_addr = A_sp_addr_start + (ich / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
            if (trans_input_3120) {
              A_sp_addr = A_sp_addr_start + (b / DIM) * ichs * DS(irows) * DS(icols) + ich * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
            }

            const bool is_zeros = irow < 0 || irow >= irows_unpadded || icol < 0 || icol >= icols_unpadded;

            const elem_t * in = input + (b*in_dim*in_dim + irow*in_dim + icol) * in_channels + ich;
            if (is_zeros) {
              in = NULL;
            } else if (trans_input_3120) {
              in = input + (ich*in_dim*in_dim + irow*in_dim + icol) * batch_size + b;
            }

            gemmini_extended_mvin(in,
                A_sp_addr,
                K, I >> downsample);
          }

          icol += I;
        }
      }
  }

  // mvin weights
  {
    int max_chs_per_mvin = ochs < MAX_BLOCK_LEN * DIM ? ochs :
        MAX_BLOCK_LEN * DIM;
    if (trans_weight_0132) {
      max_chs_per_mvin = kchs < MAX_BLOCK_LEN * DIM ? kchs :
          MAX_BLOCK_LEN * DIM;
    }

    size_t dram_stride = out_channels * sizeof(elem_t);
    if (trans_weight_1203) {
      dram_stride = kernel_dim * kernel_dim * out_channels * sizeof(elem_t);
    } else if (trans_weight_0132) {
      dram_stride = in_channels * sizeof(elem_t);
    }

    const size_t spad_block_stride = trans_weight_0132 ?
      krows * kcols * ochs : krows * kcols * kchs;

    gemmini_extended4_config_ld(dram_stride, MVIN_SCALE_IDENTITY, false, spad_block_stride, 1);

    const size_t och_it = trans_weight_0132 ? DIM : max_chs_per_mvin;
    const size_t kch_it = trans_weight_0132 ? max_chs_per_mvin : DIM;

    for (int och = 0; och < ochs; och += och_it) {
      for (int krow = 0; krow < krows; krow++)
        for (int kcol = 0; kcol < kcols; kcol++)
          for (int kch = 0; kch < kchs; kch += kch_it) {
            int K = kchs - kch > DIM ? DIM : kchs - kch;
            int J = ochs - och > max_chs_per_mvin ? max_chs_per_mvin : ochs - och;
            if (trans_weight_0132) {
              K = ochs - och > DIM ? DIM : ochs - och;
              J = kchs - kch > max_chs_per_mvin ? max_chs_per_mvin : kchs - kch;
            }

            uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow * kcols * kchs + kcol * kchs + kch;
            if (trans_weight_0132) {
              B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow * kcols * ochs + kcol * ochs + och;
            }

            const elem_t * w = weights + (krow*kernel_dim*in_channels + kcol*in_channels + kch) * out_channels + och;
            if (trans_weight_1203) {
              w = weights + (kch * kernel_dim * kernel_dim + krow * kernel_dim + kcol) * out_channels + och;
            } else if (trans_weight_0132) {
              w = weights + (krow * kernel_dim * out_channels + kcol * out_channels + och) * in_channels + kch;
            }

            gemmini_extended_mvin2(w, B_sp_addr, J, K);
          }
    }
  }

  // Compute
  {
    const int b_it = trans_input_3120 ? DIM : 1;
    const int ocol_it = trans_input_3120 ? 1 : (DIM << input_dilated);

    if (trans_input_3120) {
      gemmini_extended3_config_ex(0, 0, 0, 0, 0, orows * ocols, irows * icols, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, true);
    }

    for (int och = 0; och < ochs; och += DIM) {
      for (int krow = 0; krow < krows; krow++) {
        for (int kcol = 0; kcol < kcols; kcol++) {
          for (int kch = 0; kch < kchs; kch += DIM) {
            bool new_weights = true;

            for (int b = 0; b < batches; b += b_it) {
              for (int orow = 0; orow < orows; orow++) {
                // Skip some kernel rows due to input-dilation
                if (input_dilated && ((krow * kernel_dilation + orow * stride - upad) % 2 != 0)) {
                  continue;
                }

                for (int ocol = 0; ocol < ocols;) {
                  // Skip some cols dimensions due to input-dilation
                  if (input_dilated && ((kcol + ocol * stride - lpad) % 2 != 0)) {
                    ocol++;
                    continue;
                  }

                  int irow = orow * stride + krow * kernel_dilation;
                  int icol = ocol * stride + kcol * kernel_dilation;

                  if (input_dilated) {
                    irow = (irow + 1) / 2;
                    icol = (icol + 1) / 2;
                  }

                  const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

                  // Over here, construct a new matrix
                  //
                  // Let us assume that we only ever operate on
                  // one pixel in one row.
                  // Thus, krows == kcols == 1
                  //
                  // Then, for every set of I, J, and K values
                  //     - I = ocols
                  //     - J = ochs
                  //     - K = kchs

                  int I = UNDILATED(ocols - ocol > (DIM << input_dilated) ? (DIM << input_dilated) : ocols - ocol);
                  const int J = ochs - och > DIM ? DIM : ochs - och;
                  const int K = kchs - kch > DIM ? DIM : kchs - kch;

                  if (trans_input_3120) {
                    I = batches - b > DIM ? DIM : batches - b;
                  }

                  uint32_t A_sp_addr = A_sp_addr_start + (kch / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
                  if (trans_input_3120) {
                    A_sp_addr = A_sp_addr_start + (b / DIM) * kchs * DS(irows) * DS(icols) + kch * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
                  }

                  const int krow_ = wrot180 ? krows - krow - 1 : krow;
                  const int kcol_ = wrot180 ? kcols - kcol - 1 : kcol;

                  uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow_ * kcols * kchs + kcol_ * kchs + kch;
                  if (trans_weight_0132) {
                    B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow_ * kcols * ochs + kcol_ * ochs + och;
                  }

                  const uint32_t pre_sp_addr = new_weights ?
                    B_sp_addr : GARBAGE_ADDR;

                  // perform matmul
                  gemmini_extended_preload(pre_sp_addr, C_sp_addr, J, K, J, I);

                  if (new_weights) {
                    gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, K, I, J, I);
                  } else {
                    gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, K, I, J, I);
                  }

                  ocol += ocol_it;
                  new_weights = false;
                }
              }
            }
          }
        }
      }
    }
  }

#undef DS
#undef UNDILATED

  // mvout output
  if (output != NULL) {
    if (no_pool) {
      for (int b = 0; b < batches; b++)
        for (int orow = 0; orow < orows; orow++)
          for (int ocol = 0; ocol < ocols; ocol += DIM) {
            const int I = ocols - ocol > DIM ? DIM : ocols - ocol;
  
            for (int och = 0; och < ochs; och += DIM) {
              const int J = ochs - och > DIM ? DIM : ochs - och;
  
              const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;
  
              elem_t * out = output + (b*out_dim*out_dim + orow*out_dim + ocol) * out_channels + och;
              if (trans_output_1203) {
                out = output + (orow*out_dim*batch_size + ocol*batch_size + b) * out_channels + och;
              }
  
              gemmini_extended_mvout(out,
                  C_sp_addr,
                  J, I);
            }
          }
    } else {
      gemmini_extended_config_st(out_channels * sizeof(elem_t), pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, pupad, plpad);
  
      for (int b = 0; b < batches; b++) {
        for (int poch = 0; poch < pochs; poch += DIM) {
          const int channels = poch + DIM >= pochs ? pochs - poch : DIM;
  
          elem_t * pout = output + (b * pool_out_dim * pool_out_dim)*out_channels + poch;
  
          const uint32_t C_sp_addr = C_sp_addr_start + (poch / DIM) * batches * orows * ocols + b * orows * ocols;
  
          gemmini_extended_mvout(pout,
              C_sp_addr,
              channels, 0);
        }
      }
  
      gemmini_config_st(out_channels * sizeof(elem_t));
    }
  }
  */
}

static int tiled_conv_total_spad_rows(bool acc, bool weight,
        int stride,
        int batches,
        int porows, int pocols, int ochs,
        int krows, int kcols, int kchs,
        int pool_size, int pool_stride) {

    const int orows = porows * pool_stride + pool_size - 1;
    const int ocols = pocols * pool_stride + pool_size - 1;

    const int irows = orows * stride + krows - 1; // - 2 * padding;
    const int icols = ocols * stride + kcols - 1; // - 2 * padding;
    const int ichs = kchs;

    const int in_channels_per_bank = ichs / DIM + (ichs % DIM != 0);
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);

    const int A_rows = in_channels_per_bank * batches * irows * icols;
    const int B_rows = out_channels_per_bank * kcols * krows * kchs;
    const int C_rows = out_channels_per_bank * batches * orows * ocols;

    if (acc)
        return C_rows;
    else if(weight)
        return B_rows;
    else
        return A_rows;
}

static int tiled_conv_total_spad_rows_A_stride(bool acc,
        int stride,
        int input_dilation,
        int kernel_dilation,
        bool downsample,
        bool trans_weight_0132,
        bool trans_input_3120,
        int batches,
        int porows, int pocols, int ochs,
        int krows, int kcols, int kchs,
        int pool_size, int pool_stride) {

    const int orows = porows * pool_stride + pool_size - 1;
    const int ocols = pocols * pool_stride + pool_size - 1;

    const int krows_dilated = krows + (kernel_dilation - 1)*(krows - 1);
    const int kcols_dilated = kcols + (kernel_dilation - 1)*(kcols - 1);

    int irows = orows * stride + krows_dilated - 1; // - 2 * padding;
    int icols = ocols * stride + kcols_dilated - 1; // - 2 * padding;
    const int ichs = kchs;

    irows = irows / input_dilation + (irows % input_dilation != 0);
    icols = icols / input_dilation + (icols % input_dilation != 0);

    const int in_channels_per_bank = ichs / DIM + (ichs % DIM != 0);
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
    const int batches_per_bank = batches / DIM + (batches % DIM != 0);

    const int A_rows = trans_input_3120 ?
        (batches_per_bank * ichs * (irows >> downsample) * (icols >> downsample)) :
        (in_channels_per_bank * batches * (irows >> downsample) * (icols >> downsample));

    const int B_rows = trans_weight_0132 ?
      in_channels_per_bank * kcols * krows * ochs :
      out_channels_per_bank * kcols * krows * kchs;

    const int C_rows = out_channels_per_bank * batches * orows * ocols;

    return acc ? C_rows : A_rows + B_rows;
}

int* tiling_factor_calculate(int args[], int stride, int pool_size, int pool_stride, int kernel_dilation, int padding){
  int batch_size = args[0];
  int pool_out_row = args[1];
  int pool_out_dim = args[2];
  int out_channels = args[3];
  int kernel_dim = args[4];
  int in_channels = args[6];
  const int max_args[] = {batch_size, pool_out_row, pool_out_dim, out_channels, kernel_dim, kernel_dim, in_channels};
/*
    printf("batches = %d\n", args[0]);
    printf("orows   = %d\n", args[1]);
    printf("ocols   = %d\n", args[2]);
    printf("ochs    = %d\n", args[3]);
    printf("krows   = %d\n", args[4]);
    printf("kcols   = %d\n", args[5]);
    printf("kchs    = %d\n\n", args[6]);
*/
  const int orows_idx = 1;
  const int ocols_idx = 2;
  const int out_channels_idx = 3;
  const int in_channels_idx = 6;
 
  const int input_dilation = 1;
  // We divide by 2 for the sake of double-buffering
  const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
  const int max_acc_rows = (ACC_ROWS / 2);
  int spad_rows = tiled_conv_total_spad_rows_A_stride(false,
    stride, input_dilation, kernel_dilation, false, false, false, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  int acc_rows = tiled_conv_total_spad_rows_A_stride(true,
    stride, input_dilation, kernel_dilation, false, false, false, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

  while (spad_rows > max_spad_rows || acc_rows > max_acc_rows) {
    int max_val = -1;
    int max_idx = -1;

    for (size_t j = 0; j < 7; j++) {
      // We avoid reducing ocols when possible to keep the spatial array fully utilized
      size_t i = 0;
      if(j == 0) i = 0;
      else if (j == 4) i = orows_idx;
      else if(j == 1) i = ocols_idx;
      else if (j == 2) i = 4;
      else if(j == 3) i = 5;
      else if(j == 5) i = out_channels_idx;
      else if(j == 6) i = in_channels_idx;

      if(i == 0 && args[0] > 1){ // batch first
        max_val = args[0];
        max_idx = 0;
        break;
      } else if(((pool_stride > 1 && args[in_channels_idx] >= DIM) || args[in_channels_idx] == MAX_BLOCK_LEN * DIM) && args[out_channels_idx] <= MAX_BLOCK_LEN * DIM){
        if(i == orows_idx && args[orows_idx] > 1 && (args[ocols_idx] <= DIM || (args[in_channels_idx] <= DIM * MAX_BLOCK_LEN && args[out_channels_idx] == MAX_BLOCK_LEN*DIM))){// && (args[orows_idx] >= args[ocols_idx] || args[ocols_idx] <= DIM)){ //decrease orows as much as possible 
          max_val = args[orows_idx];
          max_idx = orows_idx;
          break;
        }else if(i == ocols_idx && (args[i]) > DIM){
          max_val = args[ocols_idx];
          max_idx = ocols_idx;
          break;
        }else if((i==4 || i == 5) && args[i] > 1){
          max_val = args[i];
          max_idx = i;
          break;
        }else if(args[i] > DIM && pool_stride > 1 && (i == in_channels_idx || i == out_channels_idx)){
          max_val = args[i];
          max_idx = i;
        }
      }else if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1) && args[i] > max_val) { // and then move on to channels
        if(!((i==out_channels_idx || i==in_channels_idx) && args[i] <= DIM)){
            max_val = args[i];
            max_idx = i;
        }
      }
    }
    if (max_idx == out_channels_idx || max_idx == in_channels_idx) {
      if(max_val > MAX_BLOCK_LEN * DIM || pool_stride > 1){
         // For input and output channels, there's no point in subtracting by just one
        if (args[max_idx] > MAX_BLOCK_LEN*DIM && args[max_idx] % (MAX_BLOCK_LEN * DIM) != 0) {
          args[max_idx] = (args[max_idx] / (MAX_BLOCK_LEN * DIM)) * (MAX_BLOCK_LEN * DIM);
        } else {
          if(args[max_idx] % (2*DIM) == 0) args[max_idx] = args[max_idx] / 2;
          else args[max_idx] = ((args[max_idx]-1) / DIM) * DIM;
        }
        args[max_idx] = args[max_idx] == 0 ? 1 : args[max_idx];
      }
      else if (args[4] > 1 || args[5] > 1){
        if(args[4] > 1) args[4] = 1;//args[4]--;
        else if(args[5] > 1) args[5]--;
      }
      else if(args[in_channels_idx] < DIM){//for first layer
        args[max_idx] = args[max_idx] / 2;
      }
      else if (args[orows_idx] > 4){
        args[orows_idx] = args[orows_idx] / 2;
      }
      else if(args[ocols_idx] > DIM){
        args[ocols_idx] = DIM;
      }
    } else {
      if(max_idx == ocols_idx){
        if(args[max_idx] % DIM != 0) args[max_idx] = (args[max_idx]/DIM)*DIM;
        else args[max_idx] -= (DIM/pool_stride);
      }else{
        if(max_idx == 4 || max_idx == 5) args[max_idx] = 1;
        else args[max_idx]--;
      }
    }
    
    if(in_channels == 3 && padding == 0 && kernel_dim == 3){
      int prop = ceil_divide_int(out_channels, args[3]);
      args[3] = out_channels;
      args[2] = args[2] / prop;
    }
    //printf("max_val: %d, max_idx: %d \n", max_val, max_idx);

    spad_rows = tiled_conv_total_spad_rows_A_stride(false,
      stride, input_dilation, kernel_dilation, false, false, false, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows_A_stride(true,
      stride, input_dilation, kernel_dilation, false, false, false,  args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  }

/*
    printf("batches = %d\n", args[0]);
    printf("orows   = %d\n", args[1]);
    printf("ocols   = %d\n", args[2]);
    printf("ochs    = %d\n", args[3]);
    printf("krows   = %d\n", args[4]);
    printf("kcols   = %d\n", args[5]);
    printf("kchs    = %d\n\n", args[6]);
*/

  // Check if we can increase ocols
  bool not_increased = false;

  // Check if there are any parameters that we can currently still increase
  bool nothing_increased = false;
  bool kdim_increase = true;
  while (!nothing_increased) {
    nothing_increased = true;
    //kdim_increase = true;

    for (size_t j = 0; j < 7; j++) {
       //size_t i =j;//  down_sample ? j : 6-j;
      size_t i = j;
      if(j == 0) i = 5;//in_channels_idx;
      else if (j == 1) i = in_channels_idx;
      else if(j == 2) i = 4;//in_channels_idx;
      else if (j == 3) i = out_channels_idx;
      else if(j == 4) i = ocols_idx;
      else if(j == 5) i = orows_idx;
      else if(j == 6) i = 0; 
      int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
      if(i == out_channels_idx || i == in_channels_idx) args_candidate[i] *= 2;//+= MAX_BLOCK_LEN * DIM;//!down_sample ? MAX_BLOCK_LEN * DIM : DIM;
      else if(i == ocols_idx && (args[i] % DIM == 0)) args_candidate[i] += DIM;
      else args_candidate[i]+= kdim_increase && (i == 4 || i == 5) ? 2 : 1;
      if (args_candidate[i] > max_args[i])
        continue;

      spad_rows = tiled_conv_total_spad_rows_A_stride(false,
         stride, input_dilation, kernel_dilation, false, false, false,  args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
      acc_rows = tiled_conv_total_spad_rows_A_stride(true,
         stride, input_dilation, kernel_dilation, false, false, false,  args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

      if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
        args[i] = args_candidate[i];
        nothing_increased = false;
        kdim_increase = false;
      }
    }
  }
/*
    printf("batches = %d\n", args[0]);
    printf("orows   = %d\n", args[1]);
    printf("ocols   = %d\n", args[2]);
    printf("ochs    = %d\n", args[3]);
    printf("krows   = %d\n", args[4]);
    printf("kcols   = %d\n", args[5]);
    printf("kchs    = %d\n\n", args[6]);

*/

  return args;
  
}

// division by row dimension
int* tiled_conv_A_stride_bubble_calculate( // for sw padding
    int args[], //target_util
    //  int tile_args[],
    int batch_size, int in_dim, int in_channels,
    int out_channels, int out_dim,
    int stride, int dilation, int padding, int kernel_dim,
    int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,

    size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id){

  const bool no_pool = pool_stride == 0;
  if (no_pool) { 
    pool_size = 1;
    pool_stride = 1;
    pool_padding = 0;
  }

  int num_core = orow_divide > batch_divide ? orow_divide : batch_divide;
  if (orow_divide > 1 && batch_divide > 1)
    num_core = orow_divide + batch_divide;
#ifdef GEMMINI_ASSERTIONS
  if(batch_size == 1 && batch_divide > 1){
    printf("batch_divide set wrong \n");
    exit(1);
  }
  /*
  if(orow_divide > 1 && batch_divide > 1){
    printf("Allowed to divide in single dimension only \n");
    exit(1);
  }
  */
#endif
  int dram_util = args[0];

  int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;
  if (pool_ceil_dim)
    pool_out_dim += (out_dim + 2*pool_padding - pool_size) % pool_stride != 0;

  int original_batch_size = batch_size;
  // divide in batch dimension
  batch_size = batch_size / batch_divide;
  // divide in row dimension (single batch)
	bool row_divisible = (orow_divide > 1) && ((pool_out_dim % orow_divide == 0) || (in_channels == 3 && padding == 0)) && (dilation <= 2);
  if (orow_divide > 1 && padding == 0) row_divisible = true;
//  bool row_divisible = (orow_divide > 1) && ((pool_out_dim % orow_divide == 0) || (in_channels == 3 && padding == 0));
  int pool_out_row = (row_divisible) ? (pool_out_dim / orow_divide) : pool_out_dim;
  if(pool_out_dim % orow_divide != 0) {
    if(cid != orow_divide - 1) pool_out_row += 1;
    //pool_out_row += 1;
  }
  const size_t och_divide = (row_divisible) ? 1 : orow_divide; //if row isn't divisible, divide channel instead
    out_channels = out_channels / och_divide;
 
 
  int args_in[] = {batch_size, pool_out_row, pool_out_dim, out_channels, kernel_dim, kernel_dim, in_channels};
  int* tile_args;
  tile_args = tiling_factor_calculate(args_in, stride, pool_size, pool_stride, dilation, padding);

  int batches = tile_args[0];
  int porows = tile_args[1];
  int pocols = tile_args[2];
  int pochs = tile_args[3];
  int krows = tile_args[4];
  int kcols = tile_args[5];
  int kchs = tile_args[6];
 
  if(pool_out_dim < porows){
    porows = pool_out_dim;
  }
  const int input_dilation = 1;
  int spad_rows = tiled_conv_total_spad_rows_A_stride(false,
       stride, input_dilation, dilation, false, false, false,  batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);
  int acc_rows = tiled_conv_total_spad_rows_A_stride(true,
       stride, input_dilation, dilation, false, false, false,  batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);

   // We divide by 2 for the sake of double-buffering
  const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
  const int max_acc_rows = (ACC_ROWS / 2);
  const int spad_util = (spad_rows*100)/max_spad_rows;
  const int acc_util = (acc_rows*100)/max_acc_rows;
/*
  printf("total spad_rows reserved: %d\n", spad_rows);
  printf("total acc_rows reserved: %d\n\n", acc_rows);

  printf("scratchpad row utilization: %d%%\n", (spad_rows*100) / max_spad_rows);
  printf("accumulator row utilization: %d%%\n\n", (acc_rows*100) / max_acc_rows);
*/

  // for layer pre-compilation
  int weight_load = 0;
  int input_load = 0;
  int bias_load = 0;
  int weight_size = (ceil_divide_int)(out_channels*och_divide, DIM) * in_channels * kernel_dim * kernel_dim;
  int input_size = (ceil_divide_int)(in_channels, DIM) * in_dim * in_dim * original_batch_size; 
  int output_size = (ceil_divide_int)(out_channels*och_divide, DIM) * pool_out_dim * pool_out_dim * original_batch_size;
  size_t num_tiles_store = 1;
  size_t num_tiles = 1;

  int added_image = 0;
  uint64_t total_from_dram = (weight_size + output_size);
  if(input_size > CACHE_SIZE || total_from_dram + input_size > CACHE_SIZE) {
    added_image = 1;
    total_from_dram += input_size; // add for first layer
  }
  //int window = 0;
  //int target_load = 0;
  //printf("tiling factors: %d %d %d %d %d %d %d \n", batches, porows, pocols, pochs, krows, kcols, kchs);
  size_t out_row = (row_divisible) ? (pool_out_row - 1) * pool_stride + pool_size - 2 * pool_padding : out_dim;
  const uint64_t total_macs = out_channels * batch_size * out_dim * out_row * kernel_dim * kernel_dim * in_channels;
  uint64_t ideal_runtime = ((uint64_t)(total_macs / (DIM*DIM)));

  int bias_from_dram = ceil_divide_int(out_channels * och_divide, DIM) * 4 * ceil_divide_int(original_batch_size, batches) * ceil_divide_int(pool_out_dim, porows) * ceil_divide_int(pool_out_dim, pocols);
  bias_from_dram = bias_from_dram / num_core;

  if(in_channels < DIM) ideal_runtime = ideal_runtime * DIM / in_channels;
  int inner_tile_A = in_dim * in_dim * ceil_divide_int(in_channels, DIM) * original_batch_size;
  int inner_tile_B = kernel_dim * kernel_dim * in_channels * ceil_divide_int(pochs, DIM);
  int outer_loop_iter_A = ceil_divide_int(out_channels * och_divide, pochs);
  int outer_loop_iter_B = 1;
  if(inner_tile_A + inner_tile_B + (output_size / outer_loop_iter_A) > CACHE_SIZE || in_channels < DIM)
    total_from_dram += ((outer_loop_iter_A - added_image) * inner_tile_A);

  if(row_divisible){
    inner_tile_A = inner_tile_A / orow_divide;
    const int porow_start = 0;//pool_out_row * cid;
    const int porow_end = pool_out_row;//(cid == orow_divide - 1) ? pool_out_dim : pool_out_row * (cid + 1);

    //const size_t out_row = (pool_out_row - 1) * pool_stride + pool_size - 2 * pool_padding;
    num_tiles = round_divide_int(out_channels, pochs) * round_divide_int(batch_size, batches) * round_divide_int(porow_end - porow_start, porows) * round_divide_int(pool_out_dim, pocols) * round_divide_int(kernel_dim, krows) * round_divide_int(kernel_dim, kcols) * round_divide_int(in_channels, kchs);
    num_tiles_store = num_tiles;
    //const uint64_t target_runtime = target_util <= 100 ? (uint64_t)(ideal_runtime * 100 / target_util) : target_util;
    //uint64_t target_tile_runtime = target_runtime / num_tiles;
    //printf("total macs: %llu, num tiles: %d, porow: %d, pool out rows: %d, out_row: %d, pocol: %d, pool out col: %d, krow: %d, kcol: %d, kchs: %d, pochs: %d \n", total_macs, num_tiles, porows, porow_end - porow_start, out_row, pocols, pool_out_dim, krows, kcols, kchs, pochs);
    //printf("ideal runtime: %d, target_tile_runtime: %d\n", ideal_runtime, target_tile_runtime);
    bool full_power = false; // when mesh utilization is too low
    if(pochs < DIM || kchs < DIM){
      int eff_poch = pochs >= DIM ? DIM : pochs;
      int eff_koch = kchs >= DIM ? DIM : kchs;
      int ideal_tuned_runtime = ((int)(total_macs / (eff_poch * eff_koch)));
      //full_power = (ideal_tuned_runtime >= target_runtime);
    }

    for (int poch = 0; poch < out_channels; poch += pochs) {
      int eff_poch = poch + pochs > out_channels ? out_channels - poch : pochs;
      int poch_unit = eff_poch < DIM ? eff_poch : DIM;
      for (int b = 0; b < batch_size; b += batches) {
        const int batches_ = batch_size - b > batches ? batches : batch_size - b;
        for (int porow = porow_start; porow < porow_end; porow += porows) {
          int eff_porow = porow + porows > porow_end ? porow_end - porow : porows;
          int orow_position = porow * pool_stride - pool_padding;
          const int pupad = orow_position < 0 ? -orow_position : 0;
          const int orow = eff_porow * pool_stride + pool_size - 1;// eff_porow * pool_stride - pool_padding;
          const int pdpad = orow_position + orow > out_dim ? orow + orow_position - out_dim : 0;
          for (int pocol = 0; pocol < pool_out_dim; pocol += pocols) {
            int eff_pocol = pocol + pocols > pool_out_dim ? pool_out_dim - pocol : pocols;
            int ocol_position = pocol * pool_stride - pool_padding;
            const int plpad = ocol_position < 0 ? -ocol_position : 0;
            const int ocol = eff_pocol * pool_stride + pool_size - 1;//eff_pocol * pool_stride - pool_padding;
            const int prpad = ocol_position + ocol > out_dim ? ocol + ocol_position - out_dim : 0;
            int ocol_unit = ocol < DIM ? ocol : DIM;
            bias_load += batches_ * orow * ceil_divide_int(ocol, ocol_unit) * ceil_divide_int(eff_poch * 4, poch_unit);// (int)(orow * ocol * eff_poch / DIM);
            for (int krow = 0; krow < kernel_dim; krow += krows) {
              int eff_krow = krow + krows > kernel_dim ? kernel_dim - krow : krows;
              int dilated_krows = eff_krow + (dilation - 1) * (eff_krow - 1);
              const int irow = (orow - pupad - pdpad) * stride + dilated_krows - 1;//orow * stride + krow*dilation - padding;
              for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                int eff_kcol = kcol + kcols > kernel_dim ? kernel_dim - kcol : kcols;
                int dilated_kcols = eff_kcol + (dilation - 1) * (eff_kcol - 1);

                const int icol = (ocol - plpad - prpad) * stride + dilated_kcols - 1;//(ocol * stride + kcol*dilation - padding;

                for (int kch = 0; kch < in_channels; kch += kchs) {
                  int eff_kch = kch + kchs > in_channels ? in_channels - kch : kchs;
                  int kch_unit = eff_kch < DIM ? eff_kch : DIM;
                  input_load += batches_ * ceil_divide_int(eff_kch, kch_unit) * irow * icol;
                  weight_load += ceil_divide_int(eff_kch, kch_unit) * eff_poch * eff_krow * eff_kcol;
                }
              }
            }
          }
        }
      }
    }
        
    int num_K_tile = (int)((in_channels*kernel_dim*kernel_dim)/(kchs*krows*kcols));
    //int store_time = acc_rows / num_K_tile;
    //target_tile_runtime -= store_time;
    //int weight_time = krows * kcols * pochs * (ceil_divide_int)(kchs, DIM);
    //int input_time = input_load / num_tiles;
   // int bias_time = acc_rows / num_K_tile * 4;//bias_load / num_tiles;
    //int ideal_tile_cycle = (int)(ideal_runtime / num_tiles);
    //int fresh_weight_load = (kernel_dim * kernel_dim * out_channels / DIM * in_channels);
    //printf("ideal runtime: %d, target_runtime: %d, weight_time: %d, input_time: %d, bias_time: %d \n", ideal_runtime, target_runtime, weight_time, input_time, bias_time);
    //printf("fresh weight load count: %d\n", fresh_weight_load);
  /*
    if(target_util == 0 || full_power || weight_time + input_time + bias_time >= target_tile_runtime || ((ideal_tile_cycle + bias_time) > target_tile_runtime)){
      window = 0;
      target_load = 0; // full power
    }
    
    else{
      window = target_tile_runtime;
      target_load = (int)((weight_load + input_load + bias_load)/num_tiles);
      if(spad_util < acc_util){
        target_load = (int)((target_load * acc_util) / spad_util);
        window = (int)((window * acc_util) / spad_util);
        num_tiles = (size_t)((num_tiles * spad_util) / acc_util);
        if(num_tiles <= 4){
          target_load = 0;
          window = 0;
        }
        //printf("number of tiles after adjustment: %d \n", num_tiles);
      }
    }
    */
    if(spad_util < acc_util){
       num_tiles = (size_t)((num_tiles * spad_util) / acc_util);
    }    
    //printf("weight load: %d, input load: %d, bias_load: %d \n", weight_load, input_load, bias_load);
  }
  //if not row divisible
  else{
    num_tiles = (round_divide_int(out_channels, pochs)) * ((int)(batch_size / batches)) * round_divide_int(pool_out_dim, porows) * round_divide_int(pool_out_dim, pocols) * round_divide_int(kernel_dim, krows) * round_divide_int(kernel_dim, kcols) * round_divide_int(in_channels, kchs);
    num_tiles_store = num_tiles;
    /*
    const uint64_t target_runtime = target_util > 100 ? target_util : (uint64_t)(ideal_runtime * 100 / target_util);
    uint64_t target_tile_runtime = target_runtime / num_tiles;
*/
//       printf("total macs: %d, num tiles: %d, porow: %d, pool out rows: %d, pocol: %d, pool out col: %d, krow: %d, kcol: %d, kchs: %d, pochs: %d \n", total_macs, num_tiles, porows, pool_out_dim, pocols, pool_out_dim, krows, kcols, kchs, pochs);
//       printf("ideal runtime: %d, target_tile_runtime: %d\n", ideal_runtime, target_tile_runtime);
    bool full_power = false; // when mesh utilization is too low
    if(pochs < DIM || kchs < DIM){
      int eff_poch = pochs >= DIM ? DIM : pochs;
      int eff_koch = kchs >= DIM ? DIM : kchs;
      uint64_t ideal_tuned_runtime = ((uint64_t)(total_macs / (eff_poch * eff_koch)));
      //full_power = (ideal_tuned_runtime >= target_runtime);
    }

    for (int poch = 0; poch < out_channels; poch += pochs) {
      int eff_poch = poch + pochs > out_channels ? out_channels - poch : pochs;
      int poch_unit = eff_poch < DIM ? eff_poch : DIM;
      for (int b = 0; b < batch_size; b += batches) {
        const int batches_ = batch_size - b > batches ? batches : batch_size - b;
        for (int porow = 0; porow < pool_out_dim; porow += porows) {
          int eff_porow = porow + porows > pool_out_dim ? pool_out_dim - porow : porows;
          int orow_position = porow * pool_stride - pool_padding;
          const int pupad = orow_position < 0 ? -orow_position : 0;
          const int orow = eff_porow * pool_stride + pool_size - 1;// eff_porow * pool_stride - pool_padding;
          const int pdpad = orow_position + orow > out_dim ? orow + orow_position - out_dim : 0;
          for (int pocol = 0; pocol < pool_out_dim; pocol += pocols) {
            int eff_pocol = pocol + pocols > pool_out_dim ? pool_out_dim - pocol : pocols;
            int ocol_position = pocol * pool_stride - pool_padding;
            const int plpad = ocol_position < 0 ? -ocol_position : 0;
            const int ocol = eff_pocol * pool_stride + pool_size - 1;//eff_pocol * pool_stride - pool_padding;
            const int prpad = ocol_position + ocol > out_dim ? ocol + ocol_position - out_dim : 0;
            int ocol_unit = ocol < DIM ? ocol : DIM;
            bias_load += batches_ * orow * ceil_divide_int(ocol, ocol_unit) * ceil_divide_int(eff_poch * 4, poch_unit);// (int)(orow * ocol * eff_poch / DIM);
            //bias_load += (int)(orow * ocol * eff_poch / DIM);
            for (int krow = 0; krow < kernel_dim; krow += krows) {
              int eff_krow = krow + krows > kernel_dim ? kernel_dim - krow : krows;
              const int irow = orow * stride + krow*dilation - padding;
              for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                int eff_kcol = kcol + kcols > kernel_dim ? kernel_dim - kcol : kcols;
                int dilated_kcols = eff_kcol + (dilation - 1) * (eff_kcol - 1);

                const int icol = (ocol - plpad - prpad) * stride + dilated_kcols - 1;//(ocol * stride + kcol*dilation - padding;
                for (int kch = 0; kch < in_channels; kch += kchs) {
                  int eff_kch = kch + kchs > in_channels ? in_channels - kch : kchs;
                  int kch_unit = eff_kch < DIM ? eff_kch : DIM;
                  input_load += batches_ * ceil_divide_int(eff_kch, kch_unit) * irow * icol;
                  weight_load += ceil_divide_int(eff_kch, kch_unit) * eff_poch * eff_krow * eff_kcol;
                }
              }
            }
          }
        }
      }
    }
    int num_K_tile = (int)((in_channels*kernel_dim*kernel_dim)/(kchs*krows*kcols));
    /*
    uint64_t ideal_tile_cycle = (uint64_t)(ideal_runtime / num_tiles);
    uint64_t weight_time = krows * kcols * pochs* (ceil_divide_int)(kchs, DIM);
    uint64_t input_time = input_load / num_tiles;
    uint64_t bias_time = acc_rows / num_K_tile * 4;//bias_load / num_tiles;
    int fresh_weight_load = (kernel_dim * kernel_dim * out_channels / DIM * in_channels);
    */
    //printf("ideal runtime: %d, target_runtime: %d, weight_time: %d, input_time: %d, bias_time: %d \n", ideal_runtime, target_runtime, weight_time, input_time, bias_time);
    //printf("fresh weight load count: %d\n", fresh_weight_load);
    //printf("target_tile_runtime: %d, weight_time: %d, input_time: %d, bias_time: %d \n", target_tile_runtime, weight_time, input_time, bias_time);
    //if(target_util == 0 || full_power || weight_time + input_time + bias_time >= target_tile_runtime || ((ideal_tile_cycle + bias_time) > target_tile_runtime)){
    //  window = 0;
    //  target_load = 0; // full power
    /*}
    else{
      target_load = (int)((bias_load + weight_load + input_load)/num_tiles);
    //  window = target_tile_runtime;
      if(spad_util < acc_util){
        target_load = (int)((target_load * acc_util) / spad_util);
        window = (int)((window * acc_util) / spad_util);
        num_tiles = (size_t)((num_tiles * spad_util) / acc_util);
        //printf("number of tiles after adjustment: %d \n", num_tiles);
      }
      //printf("weight load: %d, input load: %d \n", weight_load, input_load);
      if(num_tiles <= 4){
        target_load = 0;
        window = 0;
      }
    }*/
    if(spad_util < acc_util){
       num_tiles = (size_t)((num_tiles * spad_util) / acc_util);
    } 
  }
  total_from_dram += bias_from_dram; //bias_load * och_divide;

  if(in_channels == 3) dram_util = -1;//just disable for first layer 
  uint64_t total_mem = input_load + weight_load + bias_load + (ceil_divide_int)(out_channels, DIM) * pool_out_row * pool_out_dim * batch_size;
  uint64_t mem_ideal = total_from_dram / DRAM_BW + (total_mem-total_from_dram/num_core);
  uint64_t ideal_prediction = MAX(mem_ideal, ideal_runtime) + MIN(mem_ideal, ideal_runtime) * 0.5;
  int workload_type = total_queue_type[gemmini_queue_id[group_id]];
  int queue_id = gemmini_queue_id[group_id];
#if PRINT_MOCA != 1
  // replace with pre-compiled data
  total_from_dram = from_dram[workload_type-1][total_queue_conv[queue_id]];
  ideal_prediction = conv_prediction_cycles[workload_type-1][total_queue_conv[queue_id]]; 
#endif
  int ideal_dram_bw_exp = (100 * total_from_dram) / ideal_prediction;
  int ideal_dram_util = (ideal_dram_bw_exp / DRAM_BW);

  if(cid == 0 && dram_util == -1) gemmini_dram_util[group_id] = 0;
  uint64_t dispatch_cycle = total_queue_dispatch[queue_id];
  uint64_t end = read_cycles();
  uint64_t this_cycles = end - gemmini_start_time[group_id];
  uint64_t slack = (this_cycles > dispatch_cycle) ? this_cycles - dispatch_cycle : total_queue_target[queue_id];
  int priority = total_queue_priority[queue_id];
  // MOCA runtime dynamic priority score
  int this_score = (1+priority)/4 + round_divide_int(10*(total_queue_togo[queue_id]), slack);//max(1, (int)((10*total_queue_togo[queue_id])/slack));
  // update for the next conv layer
  if(cid == 0){ 
    //gemmini_estimate_togo[group_id] -= conv_prediction_cycles[workload_type][gemmini_num_conv[group_id]];
    //gemmini_num_conv[group_id] ++;
    gemmini_score[group_id] = this_score;
  }

  int other_dram_util = 0;
  int other_score = 0;
//  int this_score = gemmini_score[group_id];
  int other_weight_sum = 0;
  for(int i = 0; i < NUM_SUB_GROUP; i++)
    if(i != group_id) {
      other_score += this_score;//gemmini_score[i];
      other_dram_util += gemmini_dram_util[i];
      other_weight_sum += this_score * gemmini_dram_util[i];
    }

  if(dram_util == 0){
    if(ideal_dram_util + other_dram_util > DRAM_MAX_UTIL){
      int excess = ideal_dram_util + other_dram_util - DRAM_MAX_UTIL;
      dram_util = ideal_dram_util - (int)((excess * other_weight_sum) / (this_score * ideal_dram_util + other_weight_sum));
      //dram_util = ideal_dram_util - (int)((excess * ideal_dram_util * other_score) / (this_score * ideal_dram_util + other_score * other_dram_util)); 
      dram_util = MAX(25, dram_util); 
      //dram_util = (int)((DRAM_MAX_UTIL * ideal_dram_util * this_score) / (this_score * ideal_dram_util + other_score * other_dram_util));
      //printf("conv ideal dram util: %d, other dram util: %d, dram util: %d, this score: %d, other score: %d\n", ideal_dram_util, other_dram_util, dram_util, this_score, other_score);
    }
    else{
      dram_util = -1; // don't really have to use memory modulation
    }
    if(cid == 0) gemmini_dram_util[group_id] = ideal_dram_util;//(dram_util != -1) ? dram_util : ideal_dram_util;//ideal_dram_util;
    //if(cid == 0) gemmini_dram_util[group_id] = ideal_dram_util;
  }

  uint64_t prediction = (100 * total_from_dram) / (DRAM_BW * dram_util);
  int window = prediction / num_tiles;
  int target_load = (int)((weight_load + input_load + bias_load)/num_tiles);
  //if(prediction <= ideal_prediction || num_tiles < 4){
  if(dram_util >= ideal_dram_util || num_tiles < 4){
    window = 0;
    target_load = 0;
  } // computation dominant

  window = (dram_util == -1) ? 0 : window;
  target_load = (dram_util == -1) ? 0 : target_load;

#if PRINT_MOCA == 1 
  printf("window: %d, target load: %d, prediction cycles: %llu, num tiles: %d \n", window, target_load, prediction, num_tiles);
  // for pre-compilation
  printf("compute_ideal: %llu, mem_ideal: %llu, ideal prediction cycles: %llu, ideal dram bw usage: %d, ideal dram bw util: %d, result dram bw util: %d\n", ideal_runtime, mem_ideal, ideal_prediction, ideal_dram_bw_exp, ideal_dram_util, dram_util);
 
  // for pre-compilation
  printf("total A load: %d, total B load: %d, total D load: %d, raw D: %d \n", input_load, weight_load, bias_load, bias_from_dram);
  printf("A size: %d, B size: %d, C size: %d \n", input_size, weight_size, output_size);
  printf("inner tile A: %d, inner tile B: %d, outer loop iteration A: %d, outer loop iteration B: %d \n", inner_tile_A, inner_tile_B, outer_loop_iter_A, outer_loop_iter_B);
  printf("number of tile: %d, target load per tile: %d, ideal runtime: %llu\n\n", num_tiles_store, (input_load + weight_load + bias_load) / num_tiles_store, ideal_runtime);
#endif

  args[0] = tile_args[0];
  args[1] = tile_args[1];
  args[2] = tile_args[2];
  args[3] = tile_args[3];
  args[4] = tile_args[4];
  args[5] = tile_args[5];
  args[6] = tile_args[6];
  //return:  gemmini_config_calm(window, target_load);
  args[7] = window;
  args[8] = target_load;
  args[9] = ideal_prediction;
  return args;
}

static void conv_cpu_without_pool(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift) {

  bool no_bias = bias == NULL;

  for (int b = 0; b < batch_size; b++) {
    for (int orow = 0; orow < out_dim; orow++) {
      for (int ocol = 0; ocol < out_dim; ocol++) {
        for (int och = 0; och < out_channels; och++) {

          acc_t opixel = no_bias ? 0 : bias[och];

          for (int krow = 0; krow < kernel_dim; krow++) {
            if ((orow * stride + krow * kernel_dilation - padding) % input_dilation != 0)
              continue;

            const int irow = (orow * stride + krow * kernel_dilation - padding) / input_dilation;

            for (int kcol = 0; kcol < kernel_dim; kcol++) {
              if ((ocol * stride + kcol * kernel_dilation - padding) % input_dilation != 0)
                continue;

              const int icol = (ocol * stride + kcol * kernel_dilation - padding) / input_dilation;

              for (int kch = 0; kch < in_channels; kch++) {
                const elem_t * in = input + (b * in_dim * in_dim + irow * in_dim + icol) * in_channels + kch;
                if (trans_input_3120) {
                  // NHWC to CHWN
                  in = input + (kch * in_dim * in_dim + irow * in_dim + icol) * batch_size + b;
                }

                elem_t ipixel = irow < 0 || irow >= in_dim || icol < 0 || icol >= in_dim ?
                    0 : *in;

                const int krow_ = wrot180 ? kernel_dim - krow - 1 : krow;
                const int kcol_ = wrot180 ? kernel_dim - kcol - 1 : kcol;

                elem_t weight = *(weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * out_channels + och);
                if (trans_weight_1203) {
                  // HWIO to WIHO
                  weight = *(weights + (kch * kernel_dim * kernel_dim  + krow_ * kernel_dim + kcol_) * out_channels + och);
                } else if (trans_weight_0132) {
                  // HWIO to HWOI
                  weight = *(weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + och) * in_channels + kch);
                }

                opixel += weight * ipixel;
              }
            }
          }

          elem_t * out = output+(b*out_dim*out_dim+orow*out_dim+ocol)*out_channels + och;
          if (trans_output_1203) {
            // NHWC to HWNC
            out = output+(orow*out_dim*batch_size+ocol*batch_size+b)*out_channels + och;
          }

          *out = scale_and_sat(opixel, act, scale, relu6_shift);
        }
      }
    }
  }
}

static void conv_cpu(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding) {

  const bool no_pool = pool_stride == 0;
  if (no_pool) {
    conv_cpu_without_pool(
        batch_size, in_dim, in_channels,
        out_channels, out_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale, relu6_shift);
    return;
  }

  const bool no_bias = bias == NULL;
  const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;

  for (int b = 0; b < batch_size; b++) {
    for (int porow = 0; porow < pool_out_dim; porow++) {
      for (int pocol = 0; pocol < pool_out_dim; pocol++) {
        for (int poch = 0; poch < out_channels; poch++) {

          elem_t running_max = 0;
          bool running_max_initialized = false;

          for (int pwrow = 0; pwrow < pool_size; pwrow++) {
            const int orow = porow * pool_stride + pwrow - pool_padding;

            for (int pwcol = 0; pwcol < pool_size; pwcol++) {
              const int ocol = pocol * pool_stride + pwcol - pool_padding;

              if (orow < 0 || orow >= out_dim || ocol < 0 || ocol >= out_dim) {
                if (!running_max_initialized || running_max < 0) {
                  running_max = 0;
                  running_max_initialized = true;
                }
              } else {

                acc_t opixel = no_bias ? 0 : bias[poch];

                for (int krow = 0; krow < kernel_dim; krow++) {
                  if ((orow * stride + krow * kernel_dilation - padding) % input_dilation != 0)
                    continue;

                  const int irow = (orow * stride + krow * kernel_dilation - padding) / input_dilation;

                  for (int kcol = 0; kcol < kernel_dim; kcol++) {
                    if ((ocol * stride + kcol * kernel_dilation - padding) % input_dilation != 0)
                      continue;

                    const int icol = (ocol * stride + kcol * kernel_dilation - padding) / input_dilation;

                    for (int kch = 0; kch < in_channels; kch++) {
                      const elem_t * in = input + (b * in_dim * in_dim + irow * in_dim + icol) * in_channels + kch;
                      if (trans_input_3120) {
                        // NHWC to CHWN
                        in = input + (kch * in_dim * in_dim + irow * in_dim + icol) * batch_size + b;
                      }

                      elem_t ipixel = irow < 0 || irow >= in_dim || icol < 0 || icol >= in_dim ?
                          0 : *in;

                      const int krow_ = wrot180 ? kernel_dim - krow - 1 : krow;
                      const int kcol_ = wrot180 ? kernel_dim - kcol - 1 : kcol;

                      elem_t weight = *(weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * out_channels + poch);
                      if (trans_weight_1203) {
                        // HWIO to WIHO
                        weight = *(weights + (kch * kernel_dim * kernel_dim  + krow_ * kernel_dim + kcol_) * out_channels + poch);
                      } else if (trans_weight_0132) {
                        // HWIO to HWOI
                        weight = *(weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + poch) * in_channels + kch);
                      }

                      opixel += weight * ipixel;
                    }
                  }
                }

                opixel = scale_and_sat(opixel, act, scale, relu6_shift);
                if (!running_max_initialized || opixel > running_max) {
                  running_max = opixel;
                  running_max_initialized = true;
                }
              }

              if (pwrow == pool_size - 1 && pwcol == pool_size - 1) {
                elem_t * out = output + (b*pool_out_dim*pool_out_dim + porow*pool_out_dim + pocol)*out_channels + poch;
                if (trans_output_1203) {
                  // NHWC to HWNC
                  out = output + (porow*pool_out_dim*batch_size + pocol*batch_size + b)*out_channels + poch;
                }

                *out = running_max;
              }
            }
          }
        }
      }
    }
  }
}

static void tiled_conv_A_stride(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding,

        enum tiled_matmul_type_t tiled_conv_type) {

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
        const int spad_rows = tiled_conv_total_spad_rows_A_stride(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);
        const int acc_rows = tiled_conv_total_spad_rows_A_stride(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);

        if (spad_rows > BANK_NUM * BANK_ROWS / 2) {
            printf("not enough scratchpad space to store inputs and weights, %d\n", spad_rows);
            exit(1);
        }
        if (acc_rows > ACC_ROWS / 2) {
            printf("not enough accumulator space to store outputs\n");
            exit(1);
        }/*
        if (kernel_dim <= padding) {
            printf("kernel_dim must be larger than padding\n");
            exit(1);
        }*/
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
        out_channels * sizeof(elem_t);
    gemmini_extended_config_st(st_dram_stride, act, scale);
    gemmini_config_calm(0, 0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, relu6_shift, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);

    const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;
    const int dilated_in_dim = in_dim + (input_dilation-1)*(in_dim-1);

    for (int b = 0; b < batch_size; b += batches) {
        for (int porow = 0; porow < pool_out_dim; porow += porows) {
            const int orow = porow * pool_stride - pool_padding;

            for (int pocol = 0; pocol < pool_out_dim; pocol += pocols) {
                const int ocol = pocol * pool_stride - pool_padding;

                for (int poch = 0; poch < out_channels; poch += pochs) {
                    for (int krow = 0; krow < kernel_dim; krow += krows) {
                        const int orow_floored = orow < 0 ? 0 : orow;
                        int irow = orow_floored * stride + krow * kernel_dilation - padding;

                        for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                            const int ocol_floored = ocol < 0 ? 0 : ocol;
                            int icol = ocol_floored * stride + kcol * kernel_dilation - padding;

                            for (int kch = 0; kch < in_channels; kch += kchs) {
                                elem_t * out = output + (b*pool_out_dim*pool_out_dim + porow*pool_out_dim + pocol) * out_channels + poch;
                                if (trans_output_1203) {
                                    out = output + (porow*pool_out_dim*batch_size + pocol*batch_size + b) * out_channels + poch;
                                }

                                if (krow + krows < kernel_dim ||
                                        kcol + kcols < kernel_dim ||
                                        kch + kchs < in_channels) {
                                    out = NULL;
                                }

                                const acc_t * bias_ = bias + poch;
                                if (krow > 0 ||
                                        kcol > 0 ||
                                        kch > 0) {
                                    bias_ = NULL;
                                }

                                const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                                const int porows_ = pool_out_dim - porow > porows ? porows : pool_out_dim - porow;
                                const int pocols_ = pool_out_dim - pocol > pocols ? pocols : pool_out_dim - pocol;
                                const int pochs_ = out_channels - poch > pochs ? pochs : out_channels - poch;
                                const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                                const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;
                                const int kchs_ = in_channels - kch > kchs ? kchs : in_channels - kch;

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

                                const elem_t * weights_slice = weights + (krow_*kernel_dim*in_channels + kcol_*in_channels + kch) * out_channels + poch;
                                if (trans_weight_1203) {
                                  weights_slice = weights + (kch*kernel_dim*kernel_dim + krow_*kernel_dim+kcol_) * out_channels + poch;
                                } else if (trans_weight_0132) {
                                  weights_slice = weights + (krow_*kernel_dim*out_channels + kcol_*out_channels + poch) * in_channels + kch;
                                }

                                const elem_t * in = input + (b*in_dim*in_dim + ((irow+upad)>>input_dilated)*in_dim + ((icol+lpad)>>input_dilated)) * in_channels + kch;
                                if (trans_input_3120) {
                                  in = input + (kch*in_dim*in_dim + ((irow+upad)>>input_dilated)*in_dim + ((icol+lpad)>>input_dilated)) * batch_size + b;
                                }

                                sp_tiled_conv_A_stride(
                                    batch_size, in_dim, in_channels,
                                    out_channels, out_dim, pool_out_dim,
                                    in_channels, out_channels, out_channels, // default strides
                                    //in_stride, out_stride, weight_stride,

                                    stride, padding, kernel_dim, kernel_dilation,

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

                                    wrot180, trans_output_1203, trans_input_3120,
                                    trans_weight_1203, trans_weight_0132,

                                    no_bias, no_pool, downsample, input_dilated);
                            }
                        }
                    }
                }
            }
        }
    }
}

static void tiled_conv_A_stride_cid(
    int batch_size, int in_dim, int in_channels,
    int out_channels, int out_dim,
    int stride, int kernel_dilation, int padding, int kernel_dim,
    int in_stride, int out_stride, int weight_stride,

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
    size_t och_divide, size_t orow_divide, size_t cid, size_t group_id,
    int window, int target_load){

  int input_dilation = 1;
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

  const bool downsample = false ;//stride == 2 && kernel_dim == 1 && in_dim % 2 == 0 && padding == 0 && no_pool && input_dilation == 1;

  const int input_dilated = 0;

#ifdef GEMMINI_ASSERTIONS
  {
      // const int orows = porows * pool_stride + pool_size - 1;
      // const int ocols = pocols * pool_stride + pool_size - 1;

      // Check that data will fit in scratchpad
      const int spad_rows = tiled_conv_total_spad_rows_A_stride(false,
          stride, input_dilation, kernel_dilation, downsample, false, false,// trans_weight_0132, trans_input_3120,
          batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);
      const int acc_rows = tiled_conv_total_spad_rows_A_stride(true,
          stride, input_dilation, kernel_dilation, downsample, false, false, // trans_weight_0132, trans_input_3120,
          batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);

      if (spad_rows > BANK_NUM * BANK_ROWS / 2) {
          printf("not enough scratchpad space to store inputs and weights, %d\n", spad_rows);
          exit(1);
      }
      if (acc_rows > ACC_ROWS / 2) {
          printf("not enough accumulator space to store outputs\n");
          exit(1);
      }/*
      if (kernel_dim <= padding) {
          printf("kernel_dim must be larger than padding\n");
          exit(1);
      }*/
  }
#endif

  gemmini_extended_config_st(out_stride * sizeof(elem_t), act, scale);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, relu6_shift, input_dilation, stride >> downsample, false, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);

  int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;
  const int dilated_in_dim = in_dim + (input_dilation-1)*(in_dim-1);

  if (pool_ceil_dim)
    pool_out_dim += (out_dim + 2*pool_padding - pool_size) % pool_stride != 0;
    
  int pool_out_row = (pool_out_dim % orow_divide == 0) ? pool_out_dim / orow_divide : ((int)(pool_out_dim/orow_divide)) + 1;
  int porow_start = (orow_divide == 1) ? 0 : pool_out_row * cid;
  int porow_end = (orow_divide == 1) ? pool_out_dim : ((cid == orow_divide - 1) ? pool_out_dim : pool_out_row * (cid + 1));
/*
  int tile_args[] = {batches, porows, pocols, pochs, krows, kcols, kchs};
  int args_in[] = {target_util, 0};
  int * args;
  args = tiled_conv_A_stride_bubble_calculate(args_in, tile_args, batch_size, in_dim, in_channels, out_channels, out_dim, stride, kernel_dilation, padding, kernel_dim, pool_size, pool_stride, pool_padding, pool_ceil_dim, orow_divide, 1, cid);
  int window = args[0];
  int target_load = args[1];
*/
  gemmini_config_calm(window, target_load);
  for (int poch = 0; poch < out_channels; poch += pochs) {
    for (int b = 0; b < batch_size; b += batches) {
      for (int porow = porow_start; porow < porow_end; porow += porows) {
        //printf("porow_start: %d, porow_end: %d, porow: %d \n", porow_start, porow_end, porow);
        const int orow = porow * pool_stride - pool_padding;
        for (int pocol = 0; pocol < pool_out_dim; pocol += pocols) {
          const int ocol = pocol * pool_stride - pool_padding;
          for (int krow = 0; krow < kernel_dim; krow += krows) {
            const int orow_floored = orow < 0 ? 0 : orow;
            const int irow = orow_floored * stride + krow*kernel_dilation - padding;
            for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
              const int ocol_floored = ocol < 0 ? 0 : ocol;
              const int icol = ocol_floored * stride + kcol*kernel_dilation - padding;
              for (int kch = 0; kch < in_channels; kch += kchs) {
                elem_t * out = output + (b*pool_out_dim*pool_out_dim + porow*pool_out_dim + pocol) * out_stride + poch;
                if (krow + krows < kernel_dim ||
                  kcol + kcols < kernel_dim ||
                  kch + kchs < in_channels) {
                  out = NULL;
                }

                acc_t * bias_ = bias + poch;
                if (krow > 0 ||
                    kcol > 0 ||
                    kch > 0) {
                  bias_ = NULL;
                }
                const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                const int porows_ = porow_end - porow > porows ? porows : porow_end - porow;
                const int pocols_ = pool_out_dim - pocol > pocols ? pocols : pool_out_dim - pocol;
                const int pochs_ = out_channels - poch > pochs ? pochs : out_channels - poch;
                const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;
                const int kchs_ = in_channels - kch > kchs ? kchs : in_channels - kch;

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

                const int lpad = icol < 0 ? -icol : 0;
                const int rpad = icol + icols_ > in_dim ? icol + icols_ - in_dim : 0;
                const int upad = irow < 0 ? -irow : 0;
                const int dpad = irow + irows_ > in_dim ? irow + irows_ - in_dim : 0;
              
                sp_tiled_conv_A_stride(
                    batch_size, in_dim, in_channels,
                    out_channels, out_dim, pool_out_dim,
                    in_stride, out_stride, weight_stride,

                    stride, padding, kernel_dim, kernel_dilation,
                    pool_size, pool_stride, pool_padding,

                    batches_,
                    porows_, pocols_, pochs_,
                    krows_, kcols_, kchs_,

                    lpad, rpad, upad, dpad,
                    plpad, prpad, pupad, pdpad,

                    input + (b*in_dim*in_dim + (irow+upad)*in_dim + (icol+lpad)) * in_stride + kch,
                    weights + (krow*kernel_dim*in_channels + kcol*in_channels + kch) * weight_stride + poch,
                    out,
                    bias_,

                    false, false, false, false, false,
                    no_bias, no_pool, downsample, input_dilated);
              }
            }
          }
        }
      }
    }
  }
}

// division by row dimension
static void tiled_conv_A_stride_auto_stride( // for sw padding
    int batch_size, int in_dim, int in_channels,
    int out_channels, int out_dim,
    int stride, int dilation, int padding, int kernel_dim,
    int out_stride, int in_stride, int weight_stride,

    const elem_t * input,
    const elem_t * weights,
    const acc_t * bias,
    elem_t * output,

    int act, acc_scale_t scale, size_t relu6_shift,
    int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,

    enum tiled_matmul_type_t tiled_conv_type,
    size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id, 
    int target_util){

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

   // tiling, calm configure
   int args_in[10] = {target_util, 0, 0, 0, 0, 0, 0, 0, 0, 0};
   int * args = tiled_conv_A_stride_bubble_calculate(args_in, batch_size, in_dim, in_channels, out_channels, out_dim, stride, dilation, padding, kernel_dim, pool_size, pool_stride, pool_padding, pool_ceil_dim, orow_divide, batch_divide, cid, group_id);

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
	bool row_divisible = (orow_divide > 1) && ((pool_out_dim % orow_divide == 0) || (in_channels == 3 && padding == 0)) && (dilation <= 2);
  //bool row_divisible = (orow_divide > 1) && (pool_out_dim % orow_divide == 0) && (dilation <= 2);
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
  args = tiling_factor_calculate(args_in, stride, pool_size, pool_stride, dilation, padding);
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

  if(row_divisible){
      tiled_conv_A_stride_cid(
          batch_size, in_dim, in_channels,
          out_channels, out_dim,
          stride, dilation, padding, kernel_dim,
          in_stride, out_stride, weight_stride,

          batches,
          orows, ocols, ochs,
          krows, kcols, kchs,

          (elem_t*) input + batch_in_offset,
          (elem_t*) weights,
          (acc_t*) bias,
          output + batch_out_offset,

          act, scale, relu6_shift,
          pool_size, no_pool ? 0 : pool_stride, pool_padding, pool_ceil_dim,

          tiled_conv_type, och_divide, orow_divide, orow_cid, group_id,
          window, target_load);

  }else{
    bool no_bias = (bias == NULL);
    tiled_conv_A_stride_cid(
        batch_size, in_dim, in_channels,
        out_channels, out_dim,
        stride, dilation, padding, kernel_dim,
        in_stride, out_stride, weight_stride,

        batches,
        orows, ocols, ochs,
        krows, kcols, kchs,

        (elem_t*) input + batch_in_offset,
        (elem_t*) weights + out_offset,
        no_bias ? NULL : (acc_t*) bias + out_offset,
        output + out_offset + batch_out_offset,

        act, scale, relu6_shift,
        pool_size, no_pool ? 0 : pool_stride, pool_padding, pool_ceil_dim,

        tiled_conv_type, och_divide, 1, orow_cid, group_id,
        window, target_load);

  }
 
  //update for the next layer
  if(cid == 0){ 
    int workload_type = total_queue_type[gemmini_queue_id[group_id]];
    int queue_id = gemmini_queue_id[group_id];
    total_queue_togo[queue_id] -= conv_prediction_cycles[workload_type-1][queue_id];
    total_queue_conv[queue_id] ++;
//    gemmini_score[group_id] = this_score;
  }
}

// for convert
static void tiled_conv_A_stride_auto_cid(
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
    size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id,
    int target_util){

  int in_stride = (in_channels % 128 == 0) ? in_channels + 64 : in_channels;
  int weight_stride = (out_channels % 128 == 0) ? out_channels + 64 : out_channels;
#ifdef GEMMINI_ASSERTIONS
  if(out_stride % 128 == 0){
    printf("need padding\n");
    exit(1);
  }
#endif
  tiled_conv_A_stride_auto_stride(
     batch_size, in_dim, in_channels,
     out_channels, out_dim,
     stride, dilation, padding, kernel_dim,
     out_stride, in_stride, weight_stride,

     input, weights, bias, output,

     act, scale, relu6_shift,
     pool_size, pool_stride, pool_padding, pool_ceil_dim,
     tiled_conv_type,
     orow_divide, batch_divide, cid, group_id,

     target_util);
}



static void tiled_conv_A_stride_auto(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding,

        enum tiled_matmul_type_t tiled_conv_type) {

    const bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;

    const bool downsample = stride == 2 && kernel_dim == 1 && padding == 0 && no_pool && in_dim % 2 == 0;

    // Tile convolution params

    // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
    int args[] = {batch_size, pool_out_dim, pool_out_dim, out_channels, kernel_dim, kernel_dim, in_channels};
    const int max_args[] = {batch_size, pool_out_dim, pool_out_dim, out_channels, kernel_dim, kernel_dim, in_channels};

    const int orows_idx = 1;
    const int ocols_idx = 2;
    const int out_channels_idx = 3;
    const int in_channels_idx = 6;

    // We divide by 2 for the sake of double-buffering
    const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
    const int max_acc_rows = (ACC_ROWS / 2);

    int spad_rows = tiled_conv_total_spad_rows_A_stride(false,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    int acc_rows = tiled_conv_total_spad_rows_A_stride(true,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    while (spad_rows > max_spad_rows || acc_rows > max_acc_rows) {
        int max_val = -1;
        int max_idx = -1;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            // We avoid reducing ocols when possible to keep the spatial array fully utilized
            if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1)
                    && args[i] > max_val) {
                max_val = args[i];
                max_idx = i;
            }
        }

        if (max_idx == out_channels_idx || max_idx == in_channels_idx) {
            // For input and output channels, there's no point in subtracting by just one
            if (args[max_idx] % DIM != 0) {
                args[max_idx] = (args[max_idx] / DIM) * DIM;
            } else {
                args[max_idx] -= DIM;
            }
            args[max_idx] = args[max_idx] == 0 ? 1 : args[max_idx];
        } else {
            args[max_idx]--;
        }

        spad_rows = tiled_conv_total_spad_rows_A_stride(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows_A_stride(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    }

    // Check if we can increase ocols
    bool not_increased = false;
    while (!not_increased) {
        not_increased = true;

        int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
        args_candidate[ocols_idx]++;

        if (args_candidate[ocols_idx] > max_args[ocols_idx])
            continue;

        spad_rows = tiled_conv_total_spad_rows_A_stride(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows_A_stride(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

        if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
            args[ocols_idx] = args_candidate[ocols_idx];
            not_increased = false;
        }
    }

    // Check if there are any parameters that we can currently still increase
    bool nothing_increased = false;
    while (!nothing_increased) {
        nothing_increased = true;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
            args_candidate[i]++;

            if (args_candidate[i] > max_args[i])
                continue;

            spad_rows = tiled_conv_total_spad_rows_A_stride(false,
                stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
            acc_rows = tiled_conv_total_spad_rows_A_stride(true,
                stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

            if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
                args[i] = args_candidate[i];
                nothing_increased = false;
            }
        }
    }

    const int batches = args[0];
    const int orows = args[1];
    const int ocols = args[2];
    const int ochs = args[3];
    const int krows = args[4];
    const int kcols = args[5];
    const int kchs = args[6];

    /*
    spad_rows = tiled_conv_total_spad_rows_A_stride(false,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows_A_stride(true,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    printf("batches = %d\n", batches);
    printf("orows   = %d\n", orows);
    printf("ocols   = %d\n", ocols);
    printf("ochs    = %d\n", ochs);
    printf("krows   = %d\n", krows);
    printf("kcols   = %d\n", kcols);
    printf("kchs    = %d\n\n", kchs);

    printf("total spad_rows reserved: %d\n", spad_rows);
    printf("total acc_rows reserved: %d\n\n", acc_rows);

    printf("scratchpad row utilization: %d%%\n", (spad_rows*100) / max_spad_rows);
    printf("accumulator row utilization: %d%%\n\n", (acc_rows*100) / max_acc_rows);

    printf("inner matmul size: i=%d, j=%d, k=%d\n\n", ocols, ochs, kchs);
    */

    tiled_conv_A_stride(
        batch_size, in_dim, in_channels,
        out_channels, out_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,

        batches,
        orows, ocols, ochs,
        krows, kcols, kchs,

        input,
        weights,
        bias,
        output,

        act, scale, relu6_shift,
        pool_size, no_pool ? 0 : pool_stride, pool_padding,

        tiled_conv_type);
}

// This function is for a convolution with kernel_dim=1, stride==2, padding=0, and no pooling
static void tiled_conv_downsample(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,

        enum tiled_matmul_type_t tiled_conv_type) {

    const int stride = 2;

    for (int b = 0; b < batch_size; b++) {
        for (int irow = 0; irow < in_dim; irow += stride) {
            const int orow = irow / stride;

            const int I = in_dim / stride; // number of columns in row
            const int J = out_channels;
            const int K = in_channels;

            const elem_t * A = input + (b*in_dim + irow)*in_dim*in_channels;
            const elem_t * B = weights;
            const acc_t * D = bias;
            elem_t * C = output + (b*out_dim + orow)*out_dim*out_channels;

            const int A_stride = in_channels * 2;
            const int B_stride = out_channels;
            const int D_stride = out_channels;
            const int C_stride = out_channels;

            tiled_matmul_auto(I, J, K, A, B, (void*)D, (void*)C,
                    A_stride, B_stride, D_stride, C_stride,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    MVIN_SCALE_IDENTITY, act, scale, relu6_shift,
                    true, false, false, false, false, 3, tiled_conv_type);
        }
    }
}
static void resadd_cpu(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu) {

	const int minimum = relu ? 0 : elem_t_min;

    for (size_t i = 0; i < I; i++) {
        for (size_t j = 0; j < J; j++) {
            const elem_t * a = A + i * J + j;
            const elem_t * b = B + i * J + j;
            elem_t * c = C + i * J + j;

            acc_t result = MVIN_SCALE(*a, A_scale) + MVIN_SCALE(*b, B_scale);
            result = ACC_SCALE(result, C_scale);
            result = result > elem_t_max ? elem_t_max :
                (result < minimum ? minimum : result);

            *c = result;
        }
    }
}

static void sp_tiled_resadd(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const elem_t * A, const elem_t * B, elem_t * C,
        size_t A_row_stride, size_t B_row_stride, size_t C_row_stride,
        bool relu) {

    // Use the new mvin2 command to overlap mvin A, mvin B, and mvout C

    size_t blocks = (J/DIM + (J % DIM != 0));
    if (blocks > MAX_BLOCK_LEN) blocks = MAX_BLOCK_LEN;

    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
    const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

    const size_t rounded_up_J = (J / DIM + (J % DIM != 0)) * DIM;

    // Mvin A
    // printf("Mving A\n");
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            const elem_t * const A_dram_addr = A + i * A_row_stride + j;
            const uint32_t A_sp_addr = D_sp_addr_start + i * (rounded_up_J/DIM) + j;

            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
        }
    }

    // Mvin B
    // printf("Mving B\n");
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            const elem_t * const B_dram_addr = B + i * B_row_stride + j;
            const uint32_t B_sp_addr = C_sp_addr_start + i * (rounded_up_J/DIM) + j;
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
        }
    }

    // Mvout C from accumulator
    // printf("Mvout C from accumulator\n");
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            elem_t * const C_dram_addr = C + i * C_row_stride + j;
            const uint32_t C_sp_addr = D_sp_addr_start + i * (rounded_up_J/DIM) + j;
            gemmini_extended_mvout(C_dram_addr, C_sp_addr, cols, rows);
        }
    }
}

// Compute MVIN_SCALE(A, A_scale) + MVIN_SCALE(B, B_scale) = C
static void tiled_resadd(const size_t I, const size_t J,
        const size_t tile_I, const size_t tile_J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const size_t J_stride,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type,
        int window, int target_load) {
        //size_t och_divide, int target_util) {
 /* 
  int window = 0;
  int target_load = 0;
	int total_mems = I * J * 3;
	int num_tile = round_divide_int(I, tile_I) * round_divide_int(J, tile_J);
	//printf("total macs: %d, number of tile: %d, tile_I: %d, tile_J: %d \n", total_mems, num_tile, tile_I, tile_J);
  
	int macs_per_tile = (int)(total_mems / num_tile);
	int ideal_cycles = (int)(total_mems / (DIM));// * DIM));
	ideal_cycles -= ((tile_I * tile_J) / DIM);
  int target_cycles = ideal_cycles * 100 / target_util;
	int total_load = (int)(I*J*2 / DIM);
  target_load = (int)(total_load / num_tile);
	window = (int)(target_cycles / num_tile);
	//printf("ideal cycle: %d, target_cycle: %d, \n", ideal_cycles, target_cycles);
  //printf("priority: %d, window: %d, target_load: %d \n", priority, window, target_load);
 */
  gemmini_extended_config_st(J_stride * sizeof(elem_t), relu ? RELU : NO_ACTIVATION, C_scale);
  gemmini_config_ex(WS, 0, 0, 0);
  gemmini_config_calm(window, target_load);
  gemmini_extended4_config_ld(J_stride * sizeof(elem_t), A_scale, true, DIM, 0);
  gemmini_extended4_config_ld(J_stride * sizeof(elem_t), B_scale, true, DIM, 1);

  for (size_t i = 0; i < I; i += tile_I) {
    for (size_t j = 0; j < J; j += tile_J) {
      const size_t I_tile = i + tile_I <= I ? tile_I : I - i;
      const size_t J_tile = j + tile_J <= J ? tile_J : J - j;

      const elem_t * a = A + i * J_stride + j;
      const elem_t * b = B + i * J_stride + j;
      elem_t * c = C + i * J_stride + j;

      sp_tiled_resadd(I_tile, J_tile,
          A_scale, B_scale, a, b, c,
          J_stride, J_stride, J_stride,
          relu);
    }
  }

  gemmini_fence();
}

int* tiled_resadd_bubble_calculate(
    int out_args[], // window, bubble, ideal cycles, tiling factors
    size_t I, size_t J, 
    size_t orow_divide, size_t batch_divide,
    size_t group_id, int dram_util, int cid){

  uint64_t total_from_dram = I * (ceil_divide_int(J, DIM)) * 3;
  //if (total_from_dram > CACHE_SIZE) total_from_dram += I * (ceil_divide_int(J, DIM));

  size_t batch_size = I / batch_divide;
  I = batch_size;

  bool row_divisible = orow_divide > 1 && (I % orow_divide == 0);
  I = (row_divisible) ? I / orow_divide : I;
  size_t och_divide = (row_divisible) ? 1 : orow_divide; // if row is divisible, no need to divide channel


  size_t tile_I = I;
  J = J / och_divide;
  size_t tile_J = J;

	if(I < MAX_BLOCK_LEN * DIM){
    tile_I = I;
	}
	else{
    tile_I = MAX_BLOCK_LEN * DIM;
	}

	 
	size_t total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));

  // TODO this is a very inefficient way of doing this...
  while (total_acc_rows > ACC_ROWS) {
    //   if (tile_I >= tile_J)
    //       tile_I--;
    //   else
           tile_J--;

    total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));
  }

  // computing window, target load
  //int window = 0;
 // int target_load = 0;
  int total_mems = I * J * 3;
  int total_load = (int)(I*J*2 / DIM);
  int num_tile = ceil_divide_int(I, tile_I) * ceil_divide_int(J, tile_J);
	//printf("total macs: %d, number of tile: %d, tile_I: %d, tile_J: %d \n", total_mems, num_tile, tile_I, tile_J);
/*
  int macs_per_tile = (int)(total_mems / num_tile);

  int ideal_cycles = (int)(total_mems / (DIM));// * DIM));
	ideal_cycles -= ((tile_I * tile_J) / DIM);

  if (target_util != 0){
    int target_cycles = ideal_cycles * 100 / target_util;
    target_load = (int)(total_load / num_tile);
    window = (int)(target_cycles / num_tile);
  }
  */
  uint64_t ideal_prediction = total_from_dram / DRAM_BW + ceil_divide_int(total_mems, DIM);
  // for pre-compilation
  int A_size = I * ceil_divide_int(J, DIM);

  int ideal_dram_bw_exp = (100 * total_from_dram) / ideal_prediction;
  int ideal_dram_util = (ideal_dram_bw_exp / DRAM_BW);

  int other_dram_util = 0;
  int other_score = 0;
  int other_weight_sum = 0;
  // just use previous score 
  int this_score = gemmini_score[group_id];
  for(int i = 0; i < NUM_SUB_GROUP; i++)
    if(i != group_id) {
      other_score += gemmini_score[i];
      other_dram_util += gemmini_dram_util[i];
      other_weight_sum += gemmini_score[i] * gemmini_dram_util[i];
    }
  
  if(dram_util == 0){
    if(ideal_dram_util + other_dram_util > DRAM_MAX_UTIL){
      int excess = ideal_dram_util + other_dram_util - DRAM_MAX_UTIL;
      dram_util = ideal_dram_util - (int)((excess * other_weight_sum) / (this_score * ideal_dram_util + other_weight_sum));
      //dram_util = ideal_dram_util - (int)((excess * ideal_dram_util * other_score) / (this_score * ideal_dram_util + other_score * other_dram_util)); 
      dram_util = MAX(25, dram_util); 
      //dram_util = (int)((DRAM_MAX_UTIL * ideal_dram_util * this_score) / (this_score * ideal_dram_util + other_score * other_dram_util));
      //printf("resadd ideal dram util: %d, other dram util: %d, dram util: %d, this score: %d, other score: %d\n", ideal_dram_util, other_dram_util, dram_util, this_score, other_score);
    }
    else{
      dram_util = -1; // don't really have to use memory modulation
    }
    // skip for resadd
    //if(cid == 0) gemmini_dram_util[group_id] = ideal_dram_bw_exp;
  }
  // but still udpate predicted to_go cycle
  if(cid == 0)
    total_queue_togo[gemmini_queue_id[group_id]] -= ideal_prediction; 

  uint64_t prediction = (100 * total_from_dram) / (DRAM_BW * dram_util);
  int window = prediction / num_tile;
  int target_load = (int)(total_load /num_tile);
  //if(prediction <= ideal_prediction){
  if(dram_util >= ideal_dram_util){
    window = 0;
    target_load = 0;
  } // computation dominant


#if PRINT_MOCA == 1
  printf("window: %d, target load: %d, prediction cycles: %llu \n", window, target_load, prediction);
  // for pre-compilation 
  printf("ideal prediction cycles: %llu, expected dram bw x 100: %d, ideal dram bw util: %d, real dram util: %d \n", ideal_prediction, ideal_dram_bw_exp, ideal_dram_util, dram_util);
  printf("total from dram resadd: %d\n", total_from_dram);
  printf("resadd A size: %d, B size: %d, C size: %d, number of tile: %d, target load per tile: %d\n\n", A_size, A_size, A_size, num_tile, target_load);
#endif

  out_args[0] = (dram_util == -1) ? 0 : window;
  out_args[1] = (dram_util == -1) ? 0 : target_load;
  out_args[2] = ideal_prediction;
  out_args[3] = tile_I;
  out_args[4] = tile_J;

  return out_args;
}

static void tiled_resadd_auto_stride(size_t I, size_t J,
    const scale_t A_scale,
    const scale_t B_scale,
    const acc_scale_t C_scale,
    const size_t J_stride,
    const elem_t * A,
    const elem_t * B,
    elem_t * C,
    bool relu,
    enum tiled_matmul_type_t matadd_type,
    size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id,
    int target_util) {
  if (matadd_type == CPU) {
    resadd_cpu(I, J,
    A_scale, B_scale, C_scale, A, B, C,
    relu);
    return;
  }
  size_t batch_cid = (size_t)(cid / orow_divide);
  size_t orow_cid = (size_t)(cid % orow_divide);

  int args_in[] = {0, 0, 0, 0, 0};
  int * args = tiled_resadd_bubble_calculate(args_in, I, J, orow_divide, batch_divide, group_id, target_util, cid);

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
  /*
	if(I < MAX_BLOCK_LEN * DIM){
    tile_I = I;
	}
	else{
    tile_I = MAX_BLOCK_LEN * DIM;
	}

	 
	size_t total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));

  // TODO this is a very inefficient way of doing this...
  while (total_acc_rows > ACC_ROWS) {
    //   if (tile_I >= tile_J)
    //       tile_I--;
    //   else
           tile_J--;

    total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));
  }

  // computing window, target load
  int window = 0;
  int target_load = 0;
	int total_mems = I * J * 3;
	int num_tile = round_divide_int(I, tile_I) * round_divide_int(J, tile_J);
	//printf("total macs: %d, number of tile: %d, tile_I: %d, tile_J: %d \n", total_mems, num_tile, tile_I, tile_J);
  
	int macs_per_tile = (int)(total_mems / num_tile);
	int ideal_cycles = (int)(total_mems / (DIM));// * DIM));
	ideal_cycles -= ((tile_I * tile_J) / DIM);
  if (target_util != 0){
    int target_cycles = ideal_cycles * 100 / target_util;
    int total_load = (int)(I*J*2 / DIM);
    target_load = (int)(total_load / num_tile);
    window = (int)(target_cycles / num_tile);
  }
  */
	//printf("ideal cycle: %d, target_cycle: %d, \n", ideal_cycles, target_cycles);
  //printf("priority: %d, window: %d, target_load: %d \n", priority, window, target_load);
 
    // printf("tile_I: %llu\n", tile_I);
    // printf("tile_J: %llu\n", tile_J);

  if (matadd_type == WS) {
    tiled_resadd(I, J, tile_I, tile_J,
        A_scale, B_scale, C_scale, J_stride, A + batch_offset + out_offset + orow_offset, B + batch_offset + orow_offset + out_offset, C + batch_offset + orow_offset + out_offset,
        relu, matadd_type, window, target_load);
  } else if(matadd_type == CPU){
    resadd_cpu(I, J, A_scale, B_scale, C_scale,
        A, B, C, relu);
  }
  else {
    printf("Unsupported type\n");
    exit(1);
  }
}

static void tiled_resadd_auto_cid(size_t I, size_t J,
    const scale_t A_scale,
    const scale_t B_scale,
    const acc_scale_t C_scale,
    const elem_t * A,
    const elem_t * B,
    elem_t * C,
    bool relu,
    enum tiled_matmul_type_t matadd_type,
    size_t orow_divide, size_t batch_divide, size_t cid, size_t group_id,
    int target_util) {
  
  size_t J_stride = (J % 128 == 0) ? J + 64 : J;
	tiled_resadd_auto_stride(I, J, A_scale, B_scale, C_scale,
      J_stride,
      A, B, C,
      relu, matadd_type,
      orow_divide, batch_divide, cid, group_id,
		  target_util);

}


// Compute (A >> A_shift) + B = C
static void tiled_resadd_auto(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type) {

    if (matadd_type == CPU) {
        resadd_cpu(I, J,
            A_scale, B_scale, C_scale, A, B, C,
            relu);
        return;
    }

    size_t tile_I = I, tile_J = J;

    // size_t total_spad_rows = 2 * (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));
    size_t total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));

    // TODO this is a very inefficient way of doing this...
    while (total_acc_rows > ACC_ROWS) {
        if (tile_I >= tile_J)
            tile_I--;
        else
            tile_J--;

        total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));
    }

    // printf("tile_I: %llu\n", tile_I);
    // printf("tile_J: %llu\n", tile_J);

    if (matadd_type == WS) {
      tiled_resadd(I, J, tile_I, tile_J,
            A_scale, B_scale, C_scale, J, A, B, C,
            relu, matadd_type,
            0, 0);
    } else if(matadd_type == CPU){
	    resadd_cpu(I, J, A_scale, B_scale, C_scale,
          A, B, C, relu);
    }
    else {
      printf("Unsupported type\n");
      exit(1);
    }
}

static void sp_tiled_pool(
    int batch_size, int in_dim, int channels,
		int pool_out_dim, 
    int pool_size, int pool_stride, int pool_padding,
		int stride,

    int batches,
    int porows, int pocols, int pochs,
    int plpad, int prpad, int pupad, int pdpad,

    const elem_t * input,
    elem_t * output)
{
    const int orows = porows * pool_stride + pool_size - 1 - pupad - pdpad;
    const int ocols = pocols * pool_stride + pool_size - 1 - plpad - prpad;
    const int ochs = pochs;

    int D_sp_addr_row = (D_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
    int C_sp_addr_row = (C_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;

    const uint32_t D_sp_addr_start = (1 << (ADDR_LEN - 1)) + D_sp_addr_row;
    const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) + C_sp_addr_row;
    gemmini_extended2_config_st(stride * sizeof(elem_t), 0, 1, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, pupad, plpad);
    gemmini_extended4_config_ld(stride * sizeof(elem_t), MVIN_SCALE_IDENTITY, true, batches * orows * ocols, 2);

  //  gemmini_extended4_config_ld(J_stride * sizeof(elem_t), B_scale, true, DIM, 1);


    const int max_ochs_per_mvin = ochs < MAX_BLOCK_LEN_ACC * DIM ? ochs : MAX_BLOCK_LEN_ACC * DIM;

	  for (int b = 0; b < batches; b++)
			for (int orow = 0; orow < orows; orow++)
				 for (int ocol = 0; ocol < ocols; ocol += DIM) {
					  const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

					  for (int och = 0; och < ochs; och += max_ochs_per_mvin) {
							const int J = ochs - och > max_ochs_per_mvin ? max_ochs_per_mvin : ochs - och;

							const uint32_t D_sp_addr = D_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

							gemmini_extended_mvin3(input + (b*in_dim*in_dim + orow*in_dim + ocol) * stride + och,
									  D_sp_addr,
									  J, I);
					  }
				 }

		for (int b = 0; b < batches; b++) {
			 for (int poch = 0; poch < pochs; poch += DIM) {
				  const int out_channels = poch + DIM >= pochs ? pochs - poch : DIM;

				  elem_t * const pout = output + (b * pool_out_dim * pool_out_dim)*stride + poch;

				  const uint32_t C_sp_addr = C_sp_addr_start + (poch / DIM) * batches * orows * ocols + b * orows * ocols;

				  gemmini_extended_mvout(pout,
							 C_sp_addr,
							 out_channels, 0);
			 }
		}

}

static void tiled_pool(
    int batch_size, int in_dim, int channels,
		int pool_out_dim,
		int batches,
    int porows, int pocols, int pochs,
    int out_stride,

		const elem_t * input,
    elem_t * pool_output,
		  
    int act, acc_scale_t scale, size_t relu6_shift,
    int pool_size, int pool_stride, int pool_padding,

		size_t orow_divide, size_t cid, size_t group_id, int window, int target_load) {

	 //int out_stride = channels * och_divide;

    gemmini_config_calm(window, target_load);	 
    gemmini_extended_config_st(out_stride * sizeof(elem_t), RELU, MVIN_SCALE_IDENTITY);
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 1, false, false);
//	 int stride = channels*och_divide;
//    gemmini_extended4_config_ld(stride * sizeof(elem_t), MVIN_SCALE_IDENTITY, true, DIM, 0);

    bool row_divide = (orow_divide > 1);
    int out_row = (row_divide) ? pool_out_dim / orow_divide : pool_out_dim;
    size_t och_cid = (size_t)(cid % orow_divide);
    int porow_start = row_divide ? out_row * och_cid : 0;
    int porow_end = row_divide ? out_row * (och_cid + 1) : pool_out_dim;
 
    for (int poch = 0; poch < channels; poch += pochs) {
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
                  const int pochs_ = channels - poch > pochs ? pochs : channels - poch;
                  const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                  const int orows_ = porows_ * pool_stride + pool_size - 1;

                  const int plpad = ocol < 0 ? -ocol : 0;
                  const int prpad = ocol + ocols_ > in_dim ? ocol + ocols_ - in_dim : 0;
                  const int pupad = orow < 0 ? -orow : 0;
                  const int pdpad = orow + orows_ > in_dim ? orow + orows_ - in_dim : 0;

                 sp_tiled_pool(
                  batch_size, in_dim, channels,
                  pool_out_dim,
                  pool_size, pool_stride, pool_padding,
                  out_stride,

                  batches_,
                  porows_, pocols_, pochs_,
                  plpad, prpad, pupad, pdpad,

                  in,
                  out);
               }
            }
        }
    }
    gemmini_fence();
}

int* tiled_pool_bubble_calculate(
    int out_args[], // window, bubble, ideal cycles, tiling factors
    int batch_size, int in_dim, int channels,
    int out_dim,
    int pool_size, int pool_stride, int pool_padding,
    bool row_divide, size_t och_divide, size_t batch_divide, size_t cid, size_t group_id,
    int target_util){
  
  batch_size = batch_size/batch_divide;
  channels = (row_divide) ? channels : channels / och_divide;

  int out_row = (row_divide) ? out_dim / och_divide : out_dim;
  int args[] = {batch_size, out_row, out_dim, channels, 1, 1, DIM};
  const int max_args[] = {batch_size, out_row, out_dim, channels, 1, 1, DIM};

  const int orows_idx = 1;
  const int ocols_idx = 2;
  const int channels_idx = 3;
  // We divide by 2 for the sake of double-buffering
  const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
  const int max_acc_rows = (ACC_ROWS / 2);
	const int dilation = 1;
  int acc_rows = tiled_conv_total_spad_rows_A_stride(true,
        1, dilation, dilation, false, false, false, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    while (acc_rows > max_acc_rows) {
      int max_val = -1;
      int max_idx = -1;

      for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
          // We avoid reducing ocols when possible to keep the spatial array fully utilized
      if(i == channels_idx && args[i] > MAX_BLOCK_LEN * DIM){
        args[i] = (args[i] - 1) / (MAX_BLOCK_LEN*DIM) * DIM;
        break;
      }
      else if(i == 0 && args[i] > 1){
        args[i] = 1;
        break;
      } // for batch
         else if ((i!=channels_idx) &&  args[i] > max_val) {
              max_val = args[i];
              max_idx = i;
         }
      }
		  args[max_idx]--;
      acc_rows = tiled_conv_total_spad_rows_A_stride(true,
          1, dilation, dilation, false, false, false, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    }

    const int batches = args[0];
    const int porows = args[1];
    const int pocols = args[2];
    const int pochs = args[3];
  
    int window = 0;
    int target_load = 0;
    int num_tiles = 0;
    int total_load = 0;
    int ideal_cycle = 0;

    size_t och_cid = (size_t)(cid % och_divide);
    int porow_start = row_divide ? out_row * och_cid : 0;
    int porow_end = row_divide ? out_row * (och_cid + 1) : out_dim;
    for (int poch = 0; poch < channels; poch += pochs) {
       for (int b = 0; b < batch_size; b += batches) {
           for (int porow = porow_start; porow < porow_end; porow += porows) {
               const int orow = porow * pool_stride - pool_padding;
               const int orow_floored = orow < 0 ? 0 : orow;        
               for (int pocol = 0; pocol < out_dim; pocol += pocols) {
                  num_tiles += 1;
                  const int ocol = pocol * pool_stride - pool_padding;
                  const int ocol_floored = ocol < 0 ? 0 : ocol;
                  const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                  const int porows_ = porow_end - porow > porows ? porows : porow_end - porow;
                  const int pocols_ = out_dim - pocol > pocols ? pocols : out_dim - pocol;
                  const int pochs_ = channels - poch > pochs ? pochs : channels - poch;
                  
                  int ocols_ = pocols_ * pool_stride + pool_size - 1;
                  int orows_ = porows_ * pool_stride + pool_size - 1;

                  const int plpad = ocol < 0 ? -ocol : 0;
                  const int prpad = ocol + ocols_ > in_dim ? ocol + ocols_ - in_dim : 0;
                  const int pupad = orow < 0 ? -orow : 0;
                  const int pdpad = orow + orows_ > in_dim ? orow + orows_ - in_dim : 0;

                  ocols_ -= (pupad + pdpad);
                  orows_ -= (plpad + prpad);

                  total_load += (int)(batches_ * ocols_ * orows_ * pochs_ / DIM);
                  ideal_cycle += (int)(batches_ * ocols_ * orows_ * pochs_ / DIM) + (int)(batches_ * porows_ * pocols_ * pochs_ / DIM) * (pool_size * pool_size + 1);
                  //printf("total macs: %d, number of tile: %d, tile_I: %d, tile_J: %d \n", total_mems, num_tile, tile_I, tile_J);
 
               }
           }
       }
    }
  if (target_util != 0){
    int target_cycles = ideal_cycle * 100 / target_util;
    target_load = (int)(total_load / num_tiles) ;
    window = (int)(target_cycles / num_tiles) ;
  }
#if PRINT_MOCA == 1
  // for pre-compilation
  int C_size = batch_size * out_dim * out_dim * (int)(channels / DIM);
  printf("pool total load: %d, C size: %d, number of tile: %d, target load per tile: %d\n\n", total_load, C_size, num_tiles, target_load);
#endif

  out_args[0] = window;
  out_args[1] = target_load;
  out_args[2] = ideal_cycle;
  out_args[3] = args[0];
  out_args[4] = args[1];
  out_args[5] = args[2];
  out_args[6] = args[3];

  return out_args;
}
// pooling using Gemmini DMA
static void tiled_pool_auto_cid(int batch_size, int channels, int in_dim,
    int pool_out_dim, int stride,
    int pool_size, int pool_stride, int pool_padding,
    const elem_t * A,
    elem_t * C,
    size_t och_divide, size_t batch_divide, size_t cid, size_t group_id,
    int target_util) {
  
  bool relu = true;
	//int stride = channels;

  bool row_divide = (och_divide > 1 && channels < 64);
  int * args;
  int args_in[] = {0, 0, 0, 0};
  args = tiled_pool_bubble_calculate(args_in, batch_size, in_dim, channels, pool_out_dim, pool_size, pool_stride, pool_padding, row_divide, och_divide, batch_divide, cid, group_id, target_util);
  
  size_t batch_cid = (size_t)(cid / och_divide);
  size_t och_cid = (size_t)(cid % och_divide);


  batch_size = batch_size/batch_divide;
  channels = (row_divide) ? channels : channels / och_divide;
  //int pool_out_dim = (in_dim + 2*pool_padding - pool_size) / pool_stride + 1;
	int batch_in_offset = (batch_divide > 1) ? batch_size*in_dim*in_dim*stride*batch_cid : 0;
	int batch_out_offset = (batch_divide > 1) ? batch_size*pool_out_dim*pool_out_dim*stride*batch_cid : 0; // not dividing in out_channel dimension
 	const int out_offset = (och_divide > 1 && !row_divide) ? channels * och_cid : 0;
  if(!row_divide) och_divide = 1;
	 /*
	 // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
  int args[] = {batch_size, pool_out_dim, pool_out_dim, channels, 1, 1, DIM};
  const int max_args[] = {batch_size, pool_out_dim, pool_out_dim, channels, 1, 1, DIM};

  const int orows_idx = 1;
  const int ocols_idx = 2;
  const int channels_idx = 3;
  // We divide by 2 for the sake of double-buffering
  const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
  const int max_acc_rows = (ACC_ROWS / 2);
	const int dilation = 1;
  int acc_rows = tiled_conv_total_spad_rows_A_stride(true,
        stride, dilation, dilation, false, false, false, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    while (acc_rows > max_acc_rows) {
      int max_val = -1;
      int max_idx = -1;

      for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
          // We avoid reducing ocols when possible to keep the spatial array fully utilized
      if(i == channels_idx && args[i] > MAX_BLOCK_LEN * DIM){
        args[i] = (args[i] - 1) / (MAX_BLOCK_LEN*DIM) * DIM;
        break;
      }
      else if(i == 0 && args[i] > 1){
        args[i] = 1;
        break;
      } // for batch
         else if ((i!=channels_idx) &&  args[i] > max_val) {
              max_val = args[i];
              max_idx = i;
         }
      }
		  args[max_idx]--;
      acc_rows = tiled_conv_total_spad_rows_A_stride(true,
          stride, dilation, dilation, false, false, false, args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    }

    const int batches = args[0];
    const int porows = args[1];
    const int pocols = args[2];
    const int pochs = args[3];
    // printf("tile_I: %llu\n", tile_I);
    // printf("tile_J: %llu\n", tile_J);
    // printf("in offset: %d, out offset: %d \n", batch_in_offset+out_offset, batch_out_offset+out_offset);

    int window = 0;
    int target_load = 0;
    int num_tiles = 0;
    int total_load = 0;
    int ideal_cycle = 0;
    for (int poch = 0; poch < channels; poch += pochs) {
       for (int b = 0; b < batch_size; b += batches) {
           for (int porow = 0; porow < pool_out_dim; porow += porows) {
               const int orow = porow * pool_stride - pool_padding;
               const int orow_floored = orow < 0 ? 0 : orow;        
               for (int pocol = 0; pocol < pool_out_dim; pocol += pocols) {
                 num_tiles += 1;
                  const int ocol = pocol * pool_stride - pool_padding;
                  const int ocol_floored = ocol < 0 ? 0 : ocol;
                  const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                  const int porows_ = pool_out_dim - porow > porows ? porows : pool_out_dim - porow;
                  const int pocols_ = pool_out_dim - pocol > pocols ? pocols : pool_out_dim - pocol;
                  const int pochs_ = channels - poch > pochs ? pochs : channels - poch;
                  const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                  const int orows_ = porows_ * pool_stride + pool_size - 1;

                  total_load += (int)(batches_ * ocols_ * orows_ * pochs_ / DIM);
                  ideal_cycle += (int)(batches_ * ocols_ * orows_ * pochs_ / DIM) + (int)(batches_ * porows_ * pocols_ * pochs_ / DIM) * (pool_size * pool_size + 1);
                  //printf("total macs: %d, number of tile: %d, tile_I: %d, tile_J: %d \n", total_mems, num_tile, tile_I, tile_J);
 
               }
           }
       }
    }
  if (target_util != 0){
    int target_cycles = ideal_cycle * 100 / target_util;
    target_load = (int)(total_load / num_tiles);
    window = (int)(target_cycles / num_tiles);
  }
  */
  int window = args[0];
  int target_load = args[1]; 
  const int batches = args[3];
  const int porows = args[4];
  const int pocols = args[5];
  const int pochs = args[6];
  //printf("window: %d, target_load: %d \n", window, target_load);

  window = 0;
  target_load = 0; // for now, disable MOCA on pooling
  //printf("C dram addr before pool: 0x%08lx\n", C);
  tiled_pool(batch_size, in_dim, channels, pool_out_dim,
				batches, porows, pocols, pochs,
        stride,
        A + batch_in_offset + out_offset, C + batch_out_offset + out_offset,	
				RELU, MVIN_SCALE_IDENTITY, 0,
				pool_size, pool_stride, pool_padding,
				och_divide, cid, group_id, window, target_load);
  
  //printf("C dram addr after pool: 0x%08lx\n", C);
}
static void global_average_cpu(const elem_t * input, elem_t * output,
    int batches, int channels, int dim) {
  const int count = dim * dim;

  for (int batch = 0; batch < batches; batch++) {
    for (int channel = 0; channel < channels; channel++) {
      acc_t sum = 0;
      for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
          size_t pixel = batch * dim * dim + row * dim + col;

          sum += input[pixel * channels + channel];
        }
      }

      output[batch * channels + channel] = (sum + count/2) / count;
    }
  }
}

static void sp_tiled_global_average(const elem_t * input, elem_t * output,
    int batches, int channels, int dim, int channel_tile_size) {
  const uint32_t C_acc_addr_start = ((uint32_t)1 << 31);

  size_t blocks = channel_tile_size/DIM + (channel_tile_size % DIM != 0);
  if (blocks > MAX_BLOCK_LEN) blocks = MAX_BLOCK_LEN;

  for (int channel = 0; channel < channel_tile_size; channel += blocks*DIM) {
    for (int row = 0; row < dim; row++) {
      for (int col = 0; col < dim; col++) {
        const elem_t * in = input +
          (row * dim + col) * channels +
          channel;

        const uint32_t acc_addr_start = C_acc_addr_start |
          ((row != 0 || col != 0) << 30);

        const uint32_t acc_addr = acc_addr_start + channel / DIM;

        const size_t cols = channel + blocks*DIM <= channel_tile_size ?
          blocks*DIM : channel_tile_size - channel;

        const size_t rows = 1;

        gemmini_extended_mvin(in, acc_addr, cols, rows);
      }
    }
  }

  for (int channel = 0; channel < channel_tile_size; channel += DIM) {
    elem_t * out = output + channel;

    const uint32_t acc_addr = C_acc_addr_start + channel / DIM;

    const size_t cols = channel + DIM <= channel_tile_size ?
      DIM : channel_tile_size - channel;

    const size_t rows = 1; // TODO we should move out more than just one row here

    gemmini_extended_mvout(out, acc_addr, cols, rows);
  }
}

static void tiled_global_average(const elem_t * input, elem_t * output,
    int batches, int channels, int dim,
    int channel_tile_size) {

  gemmini_extended4_config_ld(DIM*sizeof(elem_t), MVIN_SCALE_IDENTITY, true, 1, 0);
  gemmini_config_ex(0, NO_ACTIVATION, 0, 0);
  gemmini_extended_config_st(0, NO_ACTIVATION, 1.0 / (dim*dim));

  for (int batch = 0; batch < batches; batch++) {
    for (int channel = 0; channel < channels; channel += channel_tile_size) {
      const int tile_size = channel + channel_tile_size <= channels ?
        channel_tile_size : channels - channel;

      sp_tiled_global_average(input + batch * dim * dim * channels + channel,
          output + batch * channels + channel,
          batches, channels, dim, tile_size);
    }
  }
}

static void tiled_global_average_auto(const elem_t * input, elem_t * output,
    int batches, int channels, int dim,
    enum tiled_matmul_type_t type) {
  if (type == CPU) {
    return global_average_cpu(input, output, batches, channels, dim);
  }

  int channel_tile_size = channels;

  int acc_rows = channel_tile_size / DIM + (channel_tile_size % DIM != 0);
  while (acc_rows > ACC_ROWS) {
    channel_tile_size--;
    acc_rows = channel_tile_size / DIM + (channel_tile_size % DIM != 0);
  }

  tiled_global_average(input, output, batches, channels, dim,
      channel_tile_size);
}

#undef abs

#endif // SRC_MAIN_C_GEMMINI_H

