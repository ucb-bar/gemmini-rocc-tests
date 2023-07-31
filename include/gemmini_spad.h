#ifndef NUM_THREAD
#define NUM_THREAD 1
#endif

#define DEBUG_PRINT 1

#include "include/gemmini.h"
#include "include/dma.h"

static uint64_t sp_capacity_alloc[NUM_THREAD] = {0};
static uint64_t sp_base_addr[NUM_THREAD] = {0};
static int dma_channel_alloc[NUM_THREAD][4] = {0}; // need to initialize -1
static int64_t sp_input_base_addr[NUM_THREAD] = {0}; // need to initialize -1

size_t* matmul_tile_factor(size_t dim_I, size_t dim_J, size_t dim_K, size_t outer_tile_I, size_t outer_tile_J, size_t outer_tile_K){
   // should not exceed outer_tile
   // better to be divisible
   static size_t inner_tile[3]; // inner_tile_I, inner_tile_J, inner_tile_K
   size_t inner_tile_I, inner_tile_J, inner_tile_K;

   const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
   const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
   const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

   bool double_buffered = true;

   const size_t max_spad_rows = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
     BANK_NUM * BANK_ROWS;
   const size_t max_acc_rows = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;
    
   // "db_" means "double-buffered"
#define db_partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define db_mats_in_partition (db_partition_rows / DIM)
#define db_mats_in_acc ((ACC_ROWS / 2) / DIM)
#define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))
#define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)
   // assume double buffered
   inner_tile_I = dim_I_padded/DIM < db_max_tile_i_j ? dim_I_padded/DIM : db_max_tile_i_j;
   inner_tile_J = dim_J_padded/DIM < db_max_tile_i_j ? dim_J_padded/DIM : db_max_tile_i_j;
   inner_tile_K = dim_K_padded/DIM < db_max_tile_k ? dim_K_padded/DIM : db_max_tile_k;

   size_t factor_I = ceil_divide_int(outer_tile_I, inner_tile_I);
   size_t factor_J = ceil_divide_int(outer_tile_J, inner_tile_J);
   size_t factor_K = ceil_divide_int(outer_tile_K, inner_tile_K);
   
   inner_tile_I = outer_tile_I / factor_I;
   inner_tile_J = outer_tile_J / factor_J;
   inner_tile_K = outer_tile_K / factor_K;

   while(true) {
      bool increased = false;
      if(factor_J > 1){
         int increased_tile_J = outer_tile_J / (factor_J - 1);
         if(tiled_matmul_total_spad_rows(inner_tile_I, increased_tile_J, inner_tile_K) <= max_spad_rows &&
            tiled_matmul_total_acc_rows(inner_tile_I, increased_tile_J) <= max_acc_rows &&
            increased_tile_J * DIM <= dim_J_padded){
             inner_tile_J = increased_tile_J;
             factor_J -= 1;
             increased = true;
         }
      }
      if(factor_I > 1){
         int increased_tile_I = outer_tile_I / (factor_I - 1);
         if(tiled_matmul_total_spad_rows(increased_tile_I, inner_tile_J, inner_tile_K) <= max_spad_rows &&
            tiled_matmul_total_acc_rows(increased_tile_I, inner_tile_J) <= max_acc_rows &&
            increased_tile_I * DIM <= dim_I_padded){
             inner_tile_I = increased_tile_I;
             factor_I -= 1;
             increased = true;
         }
      }
      if(factor_K > 1){
         int increased_tile_K = outer_tile_K / (factor_K - 1);
         if(tiled_matmul_total_spad_rows(inner_tile_I, inner_tile_J, increased_tile_K) <= max_spad_rows &&
            increased_tile_K * DIM <= dim_K_padded){
             inner_tile_K = increased_tile_K;
             factor_K -= 1;
             increased = true;
         }
      }
      if (!increased)
          break;
   }

#if DEBUG_PRINT == 1 
   const int spad_rows = tiled_matmul_total_spad_rows(inner_tile_I, inner_tile_J, inner_tile_K);
   const int acc_rows = tiled_matmul_total_acc_rows(inner_tile_I, inner_tile_J);

   printf("inner_tile_I: %d\n", inner_tile_I);
   printf("inner_tile_J: %d\n", inner_tile_J);
   printf("inner_tile_K: %d\n\n", inner_tile_K);

//   printf("spad_rows: %d\n", spad_rows);
//   printf("acc_rows: %d\n\n", acc_rows);

//   printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
//   printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

   //exit(EXIT_SUCCESS);
#endif

   inner_tile[0] = inner_tile_I;
   inner_tile[1] = inner_tile_J;
   inner_tile[2] = inner_tile_K;
   return inner_tile;
}

static void double_tiled_matmul_outer(size_t dim_I, size_t dim_J, size_t dim_K,
        elem_t* A, elem_t* B,
        void * D, void * C,
        uint64_t A_sp_addr, uint64_t B_sp_addr,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C, 
        int A_channel, int B_channel,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        size_t outer_tile_I, size_t outer_tile_J, size_t outer_tile_K,
        size_t inner_tile_I, size_t inner_tile_J, size_t inner_tile_K,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        uint8_t weightA) {

  int dataflow = WS;
  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  //const size_t outer_I0 = dim_I_padded / (outer_tile_I*DIM) + (dim_I_padded % (outer_tile_I*DIM) != 0);
  //const size_t outer_J0 = dim_J_padded / (outer_tile_J*DIM) + (dim_J_padded % (outer_tile_J*DIM) != 0);
  const size_t K0 = dim_K_padded / (inner_tile_K*DIM) + (dim_K_padded % (inner_tile_K*DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t outer_last_I = dim_I_padded % (outer_tile_I*DIM) == 0 ? outer_tile_I : (dim_I_padded/DIM) % outer_tile_I;
  const size_t outer_last_J = dim_J_padded % (outer_tile_J*DIM) == 0 ? outer_tile_J : (dim_J_padded/DIM) % outer_tile_J;
  const size_t last_K = dim_K_padded % (inner_tile_K*DIM) == 0 ? inner_tile_K : (dim_K_padded/DIM) % inner_tile_K;

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

  gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose, b_transpose);
  gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
  gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
  int A_spad_stride = a_transpose ? outer_tile_I * DIM : outer_tile_K * DIM; // or A_stride?
  int B_spad_stride = b_transpose ? outer_tile_K * DIM : outer_tile_J * DIM; // or B_stride?
  printf("A_channel: %d, B_channel: %d\n", A_channel, B_channel);
  if(A_channel == -1){
    A_spad_stride = stride_A;
    gemmini_extended3_config_ld(stride_A * sizeof(elem_t), A_scale_factor, false, 0);
  }
  else
    dma_config(A_channel, LOAD, A, A_sp_addr, stride_A, A_spad_stride);
  // configure DMA
  dma_config(B_channel, LOAD, B, B_sp_addr, stride_B, B_spad_stride);

  for(size_t o_j0 = 0; o_j0 < dim_J_padded; o_j0+=(outer_tile_J*DIM)){
    size_t outer_J_dim = (o_j0 + (outer_tile_J*DIM)) >= dim_J_padded ? dim_J_padded - o_j0 : outer_tile_J*DIM;
    const size_t last_J = outer_J_dim % (inner_tile_J*DIM) == 0 ? inner_tile_J : (outer_J_dim/DIM) % inner_tile_J;
    /*
    uint64_t B_dram_offset = b_transpose ? (o_j0*stride_B) : (o_j0); 
    // weight has to stored scratchpad
    //int B_spad_stride = b_transpose ? dim_K : outer_J_dim;
    int B_row = b_transpose? outer_J_dim : dim_K;
    int B_col = b_transpose ? dim_K : outer_J_dim;
    int B_col_bytes = B_col * sizeof(elem_t);
    int B_inner_row = b_transpose? inner_tile_J * DIM : inner_tile_K * DIM;
    int B_inner_col_bytes = b_transpose ? inner_tile_K * DIM * sizeof(elem_t) :  inner_tile_J * DIM * sizeof(elem_t);
    int num_B_row_tile = b_transpose ? ceil_divide_int(B_row, inner_tile_J*DIM) : ceil_divide_int(B_row, inner_tile_K*DIM);
    int num_B_col_tile = b_transpose ? ceil_divide_int(B_col, inner_tile_K*DIM) : ceil_divide_int(B_col, inner_tile_J*DIM);
    printf("B_row: %d, B_col: %d, B_inner_row: %d, B_inner_col_bytes: %d\n", B_row, B_col, B_inner_row, B_inner_col_bytes);
    dma_memcpy_matrix(B_channel, B_dram_offset, num_B_row_tile, num_B_col_tile, B_row, B_col_bytes, B_inner_row, B_inner_col_bytes); 
    */
    size_t J0 = outer_J_dim / (inner_tile_J*DIM) + (outer_J_dim % (inner_tile_J*DIM) != 0);

#if DEBUG_PRINT == 1
    printf("outer_tile_J: %d, dim_J_padded: %d, J0: %d\n", outer_tile_J, dim_J_padded, J0);
#endif
    // memcpy B
    for(size_t spad_j0 = 0; spad_j0 < J0; spad_j0 ++){
       for(size_t spad_k0 = 0; spad_k0 < K0; spad_k0 ++){ 
          int B_tile_index = b_transpose ? spad_j0*K0 + spad_k0 : spad_k0*J0 + spad_j0;
          int B_row = spad_k0 == K0-1 ? dim_K - spad_k0 * inner_tile_K * DIM : inner_tile_K * DIM;
          int B_col = spad_j0 == J0-1 ? outer_J_dim - spad_j0 * inner_tile_J * DIM : inner_tile_J * DIM; 
          if(b_transpose){
             int B_col_save = B_col;
             B_col = B_row;
             B_row = B_col_save;
          }
          bool granted = false;
          uint64_t B_spad_offset = b_transpose ? (spad_j0*inner_tile_J*DIM*B_spad_stride + spad_k0*inner_tile_K*DIM) : (spad_k0*inner_tile_K*DIM*B_spad_stride + spad_j0*inner_tile_J*DIM);
          uint64_t B_dram_offset = b_transpose ? (o_j0*stride_B + spad_j0*inner_tile_J*DIM*stride_B + spad_k0*inner_tile_K*DIM) : (o_j0 + spad_j0*inner_tile_J*DIM + spad_k0*inner_tile_K*DIM*stride_B);
          dma_memcpy_tile(B_channel, granted, B_dram_offset, B_spad_offset, B_tile_index, B_row, B_col * sizeof(elem_t));
#if DEBUG_PRING == 1
          printf("granted: %d, B_tile_index: %d, B_row: %d, B_col: %d, B_dram_offset: 0x%08lx, B_spad_offset: 0x%08lx\n", granted, B_tile_index, B_row, B_col, B_dram_offset, B_spad_offset);
#endif
       }
    }
    for (size_t o_i0 = 0; o_i0 < dim_I_padded; o_i0+=(outer_tile_I*DIM)){
      size_t outer_I_dim = (o_i0 + (outer_tile_I*DIM)) >= dim_I_padded ? dim_I_padded - o_i0 : outer_tile_I*DIM; 
      const size_t last_I = outer_I_dim % (inner_tile_I*DIM) == 0 ? inner_tile_I : (outer_I_dim/DIM) % inner_tile_I;
      size_t I0 = outer_I_dim / (inner_tile_I*DIM) + (outer_I_dim % (inner_tile_I*DIM) != 0);
#if DEBUG_PRINT == 1
      printf("outer_tile_I: %d, dim_I_padded: %d, I0: %d\n", outer_tile_I, dim_I_padded, I0);
#endif
      //int A_spad_stride = stride_A;
      // -1: already at scratchpad
      if(A_channel != -1){
       /* 
        // ToDo: stride for sw padding
        //A_spad_stride = a_transpose ? outer_I_dim : dim_K; // no K outer tile
        uint64_t A_dram_offset = a_transpose ? o_i0 : (o_i0*stride_A);
      
        int A_row = a_transpose? dim_K : outer_I_dim;
        int A_col = a_transpose ? outer_I_dim : dim_K;
        int A_col_bytes = A_col * sizeof(elem_t);
        int A_inner_row = a_transpose? inner_tile_K * DIM : inner_tile_I * DIM;
        int A_inner_col_bytes = a_transpose ? inner_tile_I * DIM * sizeof(elem_t) :  inner_tile_K * DIM * sizeof(elem_t);
        int num_A_row_tile = a_transpose ? ceil_divide_int(A_row, inner_tile_K*DIM) : ceil_divide_int(A_row, inner_tile_I*DIM);
        int num_A_col_tile = a_transpose ? ceil_divide_int(A_col, inner_tile_I*DIM) : ceil_divide_int(A_col, inner_tile_K*DIM);

        dma_memcpy_matrix(A_channel, A_dram_offset, num_A_row_tile, num_A_col_tile, A_row, A_col_bytes, A_inner_row, A_inner_col_bytes); 
        */
        gemmini_extended3_config_ld(A_spad_stride * sizeof(elem_t), A_scale_factor, false, 0);
 
        // memcpy A
        for(size_t spad_i0 = 0; spad_i0 < I0; spad_i0 ++){
           for(size_t spad_k0 = 0; spad_k0 < K0; spad_k0 ++){ 
              int A_tile_index = a_transpose ? spad_k0*I0 + spad_i0 : spad_i0*K0 + spad_k0;
              int A_col = spad_k0 == K0-1 ? dim_K - spad_k0 * inner_tile_K * DIM : inner_tile_K * DIM;
              int A_row = spad_i0 == I0-1 ? outer_I_dim - spad_i0 * inner_tile_I * DIM : inner_tile_I * DIM; 
              if(a_transpose){
                 int A_col_save = A_col;
                 A_col = A_row;
                 A_row = A_col_save;
              }
              bool granted = false;
              uint64_t A_spad_offset = a_transpose ? (spad_k0*inner_tile_K*DIM*A_spad_stride + spad_i0*inner_tile_I*DIM) : (spad_i0*inner_tile_I*DIM*A_spad_stride + spad_k0*inner_tile_K*DIM);
              uint64_t A_dram_offset = a_transpose ? (o_i0 + spad_k0*inner_tile_K*DIM*stride_A + spad_i0*inner_tile_I*DIM) : (o_i0*stride_A + spad_k0*inner_tile_K*DIM + spad_i0*inner_tile_I*DIM*stride_A);
              dma_memcpy_tile(A_channel, granted, A_dram_offset, A_spad_offset, A_tile_index, A_row, A_col * sizeof(elem_t));
#if DEBUG_PRING == 1
              printf("granted: %d, A_tile_index: %d, A_row: %d, A_col: %d, A_dram_offset: 0x%08lx, A_spad_offset: 0x%08lx\n", granted, A_tile_index, A_row, A_col, A_dram_offset, A_spad_offset);
#endif
           }
        }
      } 

      // start inner loop
      // reuse operand if it fits scratchpad
      int a_spad_id = 0;
      int b_spad_id = 0;
      bool b_reuse = (J0 * K0 <= 2) && (dataflow == WEIGHT_STATIONARY);
      bool a_reuse = (I0 * K0 <= 2) && (dataflow == WEIGHT_STATIONARY);
      gemmini_extended3_config_ld(B_spad_stride * sizeof(elem_t), B_scale_factor, false, 1);
     
      for (size_t i0 = 0; i0 < I0; i0++)
        for (size_t j0 = 0; j0 < J0; j0++)
          for (size_t k0 = 0; k0 < K0; k0++) {
            int A_tile_index = a_transpose ? k0*I0 + i0 : i0*K0 + k0; 
            int B_tile_index = b_transpose ? j0*K0 + k0 : k0*J0 + j0;
            // wait until inner tile is ready
            while(A_channel != -1){
              uint64_t curr_tile;
              dma_probe_state(curr_tile, A_channel);
              if(curr_tile > A_tile_index)
                break;
            }
            while(true){
              uint64_t curr_tile;
              dma_probe_state(curr_tile, B_channel);
              if(curr_tile > B_tile_index)
                break;
            }
            if(a_reuse)
              a_spad_id = ((i0+k0) == 0) ? 1 : 2;
            if(b_reuse)
              b_spad_id = ((j0+k0) == 0) ? 1 : 2;

            const void * pre;
            // ToDo: bias
            if (k0 != 0) {
              pre = NULL;
            } else {
              size_t bias_row = repeating_bias ? 0 : o_i0 + i0*inner_tile_I*DIM;
              // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * inner_tile_J * DIM]);
              pre = (int8_t*)D + (bias_row * stride_D + o_j0 + j0 * inner_tile_J * DIM)*sizeof_D;
            }

            // does not need DMA configuration
            void * out = k0 == K0-1 ? (int8_t*)C + (o_i0*stride_C + o_j0 + i0*inner_tile_I*DIM*stride_C + j0*inner_tile_J*DIM)*sizeof_C : NULL;

            const size_t I = ((o_i0 + outer_tile_I*DIM >= dim_I_padded) && (i0 < I0-1)) ? inner_tile_I : last_I;
            const size_t J = ((o_j0 + outer_tile_J*DIM >= dim_J_padded) && (j0 < J0-1)) ? inner_tile_J : last_J;
            const size_t K = k0 < K0-1 ? inner_tile_K : last_K;

            const size_t pad_I = i0 == I0-1 ? padding_I : 0;
            const size_t pad_J = j0 == J0-1 ? padding_J : 0;
            const size_t pad_K = k0 == K0-1 ? padding_K : 0;

            // ensure A is not outer-tiled when all pre-loaded
            const elem_t * a = a_transpose ? (elem_t*) (BASE_ADDR + A_sp_addr + k0*inner_tile_K*DIM*A_spad_stride + i0*inner_tile_I*DIM)
              : (elem_t*) (BASE_ADDR + A_sp_addr + i0*inner_tile_I*DIM*A_spad_stride + k0*inner_tile_K*DIM);

            const elem_t * b = b_transpose ? (elem_t*) (BASE_ADDR + B_sp_addr + j0*inner_tile_J*DIM*B_spad_stride + k0*inner_tile_K*DIM)
              : (elem_t*) (BASE_ADDR + B_sp_addr + k0*inner_tile_K*DIM*B_spad_stride + j0*inner_tile_J*DIM);

            if(a_reuse && j0 >= 1) a = NULL;
            if(b_reuse && i0 >= 1) b = NULL;
#if DEBUG_PRINT == 1
            printf("a_reuse: %d, b_reuse: %d, A_spad_stride: %d, B_spad_stride: %d, a: 0x%08lx, b: 0x%08lx C: 0x%08lx, out: 0x%08lx\n", a_reuse, b_reuse, A_spad_stride, B_spad_stride, a, b, C, out);
#endif
            sp_tiled_matmul_ws(a, b, pre, out,
                A_scale_factor, B_scale_factor, D_scale_factor,
                I, J, K,
                pad_I, pad_J, pad_K,
                A_spad_stride, B_spad_stride, stride_D, stride_C,
                a_transpose, b_transpose,
                full_C, low_D,
                no_bias, repeating_bias,
                act, a_spad_id, b_spad_id);
                
          }
    }
  }

  gemmini_fence();
}



// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
// use A or C only when it is not in global spad (else, ignore)
static void double_tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        elem_t* A, elem_t* B,
        void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        int thread_id) {

   uint64_t global_spad_capacity = sp_capacity_alloc[thread_id];
   int size_input = dim_I * dim_K;
   int size_output = dim_I * dim_J;
   int size_weight = dim_J * dim_K;
   // ToDo: bias (for now, skip bias)
   bool input_from_dram = (sp_input_base_addr[thread_id] == -1);
   bool output_to_dram = false;
   uint64_t total_inout_requirement = input_from_dram ? size_input+size_output : size_output;
   if (total_inout_requirement > global_spad_capacity)
       output_to_dram = true;
   
   // outer tile factor
   const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
   const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
   const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;
   // currently, just assume entire K = outer tile dimension (don't want to move-in/out partial sum)
   int outer_tile_K = dim_K_padded / DIM;
   int outer_tile_I = dim_I_padded / DIM;
   int outer_tile_J = dim_J_padded / DIM;
   if(!output_to_dram){
     global_spad_capacity -= size_output;
   }
   if(!input_from_dram){
     // outer_tile_I: don't care
     // need to outer tile weight 
     int factor_J = 1;
     global_spad_capacity -= size_input;
     uint64_t outer_size = size_output/factor_J;
     while(outer_size > global_spad_capacity){
        factor_J ++;
        if(global_spad_capacity >= (size_output / (factor_J-1))){
            factor_J --;
        }
        outer_size = (size_output / factor_J);
     }
     outer_tile_J /= factor_J;
   }
   else{
     // need to outer tile both input and wieght
     int factor_I = 1;
     int factor_J = 1;
     // shall be a better way to do this
     // currently, assume almost equal division factor of input and weight
     uint64_t outer_size = size_input/factor_I + size_output/factor_J;
     while(outer_size > global_spad_capacity){
        factor_I ++;
        factor_J ++;
        if(global_spad_capacity >= (size_input / (factor_I-1)) + (size_output / factor_J)){
            factor_I --;
        }
        if(global_spad_capacity >= (size_input / (factor_I)) + (size_output / (factor_J-1))){
            factor_J --;
        }
        outer_size = (size_input / factor_I) + (size_output / factor_J);
     }
     outer_tile_I /= factor_I;
     outer_tile_J /= factor_J;
   }
  
#if DEBUG_PRINT == 1 
   printf("outer_tile_I: %d\n", outer_tile_I);
   printf("outer_tile_J: %d\n", outer_tile_J);
   printf("outer_tile_K: %d\n\n", outer_tile_K);
#endif 

   size_t* inner_tile;
   inner_tile = matmul_tile_factor(dim_I, dim_J, dim_K, outer_tile_I, outer_tile_J, outer_tile_K);
   size_t inner_tile_I = inner_tile[0];
   size_t inner_tile_J = inner_tile[1];
   size_t inner_tile_K = inner_tile[2];

   int A_channel = dma_channel_alloc[thread_id][0];
   int B_channel = dma_channel_alloc[thread_id][1];
   int D_channel = dma_channel_alloc[thread_id][2]; // ToDo
   
   uint64_t A_sp_addr = 0;
   uint64_t B_sp_addr = 0;
   uint64_t C_sp_addr = 0;
   if(!input_from_dram){
      A_sp_addr = sp_input_base_addr[thread_id];
      B_sp_addr = A_sp_addr + size_input;
      A_channel = -1;
   }
   else{
      A_sp_addr = sp_base_addr[thread_id];
      //A = NULL;
      B_sp_addr = A_sp_addr + (outer_tile_K*DIM * outer_tile_I*DIM);
   }
   if(!output_to_dram){
      C_sp_addr = B_sp_addr + (outer_tile_K*DIM * outer_tile_J*DIM);
   }
#if DEBUG_PRINT == 1
   printf("A_sp_addr: 0x%08lx, B_sp_addr: 0x%081x, C_sp_addr: 0x%081x, A_channel: %d, B_channel: %d \n", A_sp_addr, B_sp_addr, C_sp_addr, A_channel, B_channel);
#endif
   double_tiled_matmul_outer(dim_I, dim_J, dim_K,
        A, B, D, output_to_dram ? C : (void*) (C_sp_addr+BASE_ADDR),
        A_sp_addr, B_sp_addr, // todo: bias
        stride_A, stride_B, stride_D, output_to_dram ? stride_C : dim_J,
        A_channel, B_channel,
        A_scale_factor, B_scale_factor, D_scale_factor,
        act, scale, bert_scale, repeating_bias,
        outer_tile_I, outer_tile_J, outer_tile_K,
        inner_tile_I, inner_tile_J, inner_tile_K,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA);

    if(!output_to_dram){
        sp_input_base_addr[thread_id] = C_sp_addr;
    }
    else{
        sp_input_base_addr[thread_id] = -1;
    }
#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}

int* tiling_factor_conv(int args[], int stride, int input_dilation, int kernel_dilation, bool downsample, bool trans_weight_0132, bool trans_input_3120, int pool_size, int pool_padding, int pool_stride){
  int batch_size = args[0];
  int pool_out_row_dim = args[1];
  int pool_out_col_dim = args[2];
  int out_channels = args[3];
  int kernel_dim = args[4];
  int in_channels = args[6];
  const int max_args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels, kernel_dim, kernel_dim, in_channels};

  const int orows_idx = 1;
  const int ocols_idx = 2;
  const int out_channels_idx = 3;
  const int in_channels_idx = 6;

  // We divide by 2 for the sake of double-buffering
  const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
  const int max_acc_rows = (ACC_ROWS / 2);

  int spad_rows = tiled_conv_total_spad_rows(false,
      stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  int acc_rows = tiled_conv_total_spad_rows(true,
      stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

  while (spad_rows > max_spad_rows || acc_rows > max_acc_rows) {
      int max_val = -1;
      int max_idx = -1;

      for (size_t i = 0; i < 7; i++) {
          if(i == out_channels_idx || i == in_channels_idx){
              if(args[i] <= DIM)
                  continue;
          }
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

      spad_rows = tiled_conv_total_spad_rows(false,
          stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
          args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
      acc_rows = tiled_conv_total_spad_rows(true,
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

      spad_rows = tiled_conv_total_spad_rows(false,
          stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
          args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
      acc_rows = tiled_conv_total_spad_rows(true,
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

      for (size_t i = 0; i < 7; i++) {
          int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
        
          if (i == out_channels_idx || i == in_channels_idx) 
              args_candidate[i] += DIM;
          else
              args_candidate[i] ++;

          
          if (args_candidate[i] > max_args[i])
              continue;

          spad_rows = tiled_conv_total_spad_rows(false,
              stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
              args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
          acc_rows = tiled_conv_total_spad_rows(true,
              stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
              args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

          if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
              args[i] = args_candidate[i];
              nothing_increased = false;
          }
      }
  }

#if DEBUG_PRINT == 1
  const int batches = args[0];
  const int orows = args[1];
  const int ocols = args[2];
  const int ochs = args[3];
  const int krows = args[4];
  const int kcols = args[5];
  const int kchs = args[6];

  /*
  spad_rows = tiled_conv_total_spad_rows(false,
      stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  acc_rows = tiled_conv_total_spad_rows(true,
      stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  */

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
#endif
    return args;
}

static void double_tiled_conv(
        int batch_size,
        int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        int in_stride, int weight_stride, int out_stride,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs,
        int och_outer_factor, int kch_outer_factor,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int A_channel, int B_channel,
        elem_t* A_sp_addr, elem_t* B_sp_addr,

        int act, acc_scale_t scale,
        int pool_size, int pool_stride, int pool_padding) {

#ifdef GEMMINI_ASSERTIONS
  if (trans_weight_1203 && trans_weight_0132) {
    printf("Only one weight transformation can be applied at a time\n");
    exit(1);
  }
#endif

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

    const bool downsample = stride == 2 && kernel_dim == 1 && in_row_dim % 2 == 0 && in_col_dim % 2 == 0
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
    gemmini_extended_config_st(st_dram_stride, act, scale);

    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);

    int out_channels_tile = pochs * och_outer_factor;
    int in_channels_tile = kchs * kch_outer_factor;
    
    const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int dilated_in_row_dim = in_row_dim + (input_dilation - 1) * (in_row_dim- 1);
    const int dilated_in_col_dim = in_col_dim + (input_dilation - 1) * (in_col_dim- 1);

    size_t a_spad_id = 0;
    size_t b_spad_id = 0;

    int porow_end = pool_out_row_dim;
	int porow_start = 0;
    bool a_reuse = false;
    bool b_reuse = false;
    size_t num_kch = ceil_divide_int(in_channels, kchs);
    size_t num_poch = ceil_divide_int(out_channels_tile, pochs);
    size_t num_b = ceil_divide_int(batch_size, batches);
    size_t num_porow = ceil_divide_int((porow_end - porow_start), porows);
    size_t num_pocol = ceil_divide_int(pool_out_col_dim, pocols);
    size_t num_krow = ceil_divide_int(kernel_dim, krows);
    size_t num_kcol = ceil_divide_int(kernel_dim, kcols);

    int A_spad_stride = kchs;
    int B_spad_stride = pochs;
  if(A_channel == -1){
    A_spad_stride = in_stride;
    //gemmini_extended3_config_ld(A_spad_stride * sizeof(elem_t), A_scale_factor, false, 0);
  }
  else
    dma_config(A_channel, LOAD, input, A_sp_addr, in_stride, A_spad_stride);
  // configure DMA
  dma_config(B_channel, LOAD, weights, B_sp_addr, weight_stride, B_spad_stride);

//    printf("num_kch: %d, num_poch: %d, num_b: %d, num_porow: %d, num_pocol: %d, num_krow: %d, num_kcol: %d\n", num_kch, num_poch, num_b, num_porow, num_pocol, num_krow, num_kcol);
/*
    if(num_kch * num_poch * num_krow * num_kcol <= 2) 
      b_reuse = true;
    if(num_kch * num_krow * num_kcol * num_b * num_porow * num_pocol <= 2)
      a_reuse = true;
*/
#if DEBUG_PRINT == 1
    printf("a reuse: %d, b reuse: %d\n", a_reuse, b_reuse);
#endif

    for (int b = 0; b < batch_size; b += batches) {
      for(int poch_outer = 0; poch_outer < out_channels; poch_outer += out_channels_tile){
        int this_tile_out_channel = poch_outer + out_channels_tile > out_channels ? out_channels - poch_outer : out_channels_tile;
        for (int porow = porow_start; porow < porow_end; porow += porows) {
            const int orow = porow * pool_stride - pool_padding;
            const int orow_floored = orow < 0 ? 0 : orow;
            const int pupad = orow < 0 ? -orow : 0;
            const int porows_ = pool_out_row_dim - porow > porows ? porows : pool_out_row_dim - porow;
            const int orows_ = porows_ * pool_stride + pool_size - 1;
            const int pdpad = orow + orows_ > out_row_dim ? orow + orows_ - out_row_dim : 0;
            
            int irows_outer = (orows_ - pupad - pdpad) * stride + (kernel_dim + (kernel_dilation - 1) * (kernel_dim - 1)) - 1;
            const int irow_out = orow_floored * stride - padding;
            int upad_out = irow_out < 0 ? -irow_out : 0;
            int dpad_out = irow_out + irows_outer > dilated_in_row_dim ? irow_out + irows_outer - dilated_in_row_dim : 0;
            irows_outer -= (upad_out + dpad_out);
            for (int pocol = 0; pocol < pool_out_col_dim; pocol += pocols) {
                const int ocol = pocol * pool_stride - pool_padding;
                const int ocol_floored = ocol < 0 ? 0 : ocol;
                const int plpad = ocol < 0 ? -ocol : 0;
                const int pocols_ = pool_out_col_dim - pocol > pocols ? pocols : pool_out_col_dim - pocol;
                const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                const int prpad = ocol + ocols_ > out_col_dim ? ocol + ocols_ - out_col_dim : 0;

                int icols_outer = (ocols_ - pupad - pdpad) * stride + (kernel_dim + (kernel_dilation - 1) * (kernel_dim - 1)) - 1;
                const int icol_out = ocol_floored * stride - padding;
                int lpad_out = icol_out < 0 ? -icol_out : 0;
                int rpad_out = icol_out + icols_outer > dilated_in_col_dim ? icol_out + icols_outer - dilated_in_col_dim : 0;
                icols_outer -= (lpad_out + rpad_out);
                for (int poch = 0; poch < this_tile_out_channel; poch += pochs) {
                  for(int kch_outer = 0; kch_outer < in_channels; kch_outer += in_channels_tile){
                    int this_tile_in_channel = kch_outer + in_channels_tile > in_channels ? in_channels - kch_outer : in_channels_tile;
                    if(poch == 0){
                      for(int poch_spad = 0; poch_spad < this_tile_out_channel; poch_spad += pochs){
                        // prefetch input / weight
                        for(int kch_spad = 0; kch_spad < this_tile_in_channel; kch_spad += kchs){
                          if(A_channel != -1){
                             // fetch input
                             uint64_t in_dram_offset = (b * in_row_dim * in_col_dim + ((irow_out+upad_out)>>input_dilated) * in_col_dim + ((icol_out+lpad_out)>>input_dilated))*in_stride + kch_outer + kch_spad;
                             int in_tile_row_dram_offset = in_col_dim;
                             int in_num_tile = irows_outer;//irows_outer - upad_out - dpad_out;
                             int in_tile_rows = icols_outer;// icols_outer - lpad_out - rpad_out;
                             int in_tile_bytes_per_row = (kch_spad + kchs > this_tile_in_channel) ? this_tile_in_channel - kch_spad : kchs;
                             uint64_t in_spad_offset = irows_outer * icols_outer * kchs * (int)(kch_spad/kchs) * sizeof(elem_t);
#if DEBUG_PRINT == 1
                             printf("irow_out: %d, upad_out: %d, dpad_out: %d\n", irow_out, upad_out, dpad_out);
                             printf("icol_out: %d, lpad_out: %d, rpad_out: %d\n", icol_out, lpad_out, rpad_out);
                             printf("irows outer: %d, icols outer: %d, dram offset: 0x%08lx, spad offset: 0x%08lx\n", irows_outer, icols_outer, in_dram_offset, in_spad_offset);
#endif
                             bool granted = false;
                             dma_memcpy_subtile(A_channel, granted, in_dram_offset, in_spad_offset, (int)(kch_spad / kchs), in_tile_row_dram_offset, icols_outer, in_num_tile, in_tile_rows, in_tile_bytes_per_row); 
                          }
                          // later move this to outside and create poch/koch loop (under if poch == 0)
                          uint64_t weight_dram_offset = (kch_outer + kch_spad) * weight_stride + (poch_outer + poch_spad);
                          int weight_outer_index = (int)(poch_spad/pochs) * ceil_divide_int(this_tile_in_channel, kchs) + (int)(kch_spad/kchs);
                          int weight_spad_offset = kernel_dim * kernel_dim * kchs * pochs * weight_outer_index * sizeof(elem_t); 
                          int weight_tile_rows = (kch_spad + kchs > this_tile_in_channel) ? this_tile_in_channel - kch_spad : kchs;
                          int weight_tile_bytes_per_row = poch_spad + pochs > this_tile_out_channel ? this_tile_out_channel - poch_spad : pochs;
                          bool granted = false;
                          dma_memcpy_subtile(B_channel, granted, weight_dram_offset, weight_spad_offset, weight_outer_index, in_channels, kchs, kernel_dim * kernel_dim, weight_tile_rows, weight_tile_bytes_per_row);
                        }
                      }
                    }
                    for (int krow = 0; krow < kernel_dim; krow += krows) {
                        int irow = orow_floored * stride + krow * kernel_dilation - padding;

                        for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                            int icol = ocol_floored * stride + kcol * kernel_dilation - padding;

                            for (int kch = 0; kch < this_tile_in_channel; kch += kchs) {
#if DEBUG_PRINT == 1
                                printf("b: %d, poch_outer: %d, porow: %d, pocol: %d, poch: %d, kch_outer: %d, kch: %d, krow: %d, kcol: %d\n", b, poch_outer, porow, pocol, poch, kch_outer, kch, krow, kcol);
#endif
                                if(a_reuse)
						           a_spad_id = (kch + krow + kcol + b + (porow - porow_start) + pocol) == 0 ? 1 : 2;
					            if(b_reuse)
						           b_spad_id = (kch + poch + krow + kcol) == 0 ? 1 : 2;
                                elem_t * out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * out_stride + poch_outer + poch;
                                /*
                                if (trans_output_1203) {
                                    out = output + (porow * pool_out_col_dim * batch_size + pocol * batch_size + b) * out_channels + poch;
                                }
                                */

                                if (krow + krows < kernel_dim ||
                                        kcol + kcols < kernel_dim ||
                                        kch + kchs + kch_outer < in_channels) {
                                    out = NULL;
                                }

                                const acc_t * bias_ = bias + poch_outer + poch;
                                if (krow > 0 ||
                                        kcol > 0 ||
                                        (kch+kch_outer) > 0) {
                                    bias_ = NULL;
                                }

                                const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                                const int pochs_ = this_tile_out_channel - poch > pochs ? pochs : this_tile_out_channel - poch;
                                const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                                const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;
                                const int kchs_ = this_tile_in_channel - kch > kchs ? kchs : this_tile_in_channel - kch;
                                const int dilated_krows_ = krows_ + (kernel_dilation - 1)*(krows_ - 1);
                                const int dilated_kcols_ = kcols_ + (kernel_dilation - 1)*(kcols_ - 1);

                                const int icols_ = (ocols_ - plpad - prpad) * stride + dilated_kcols_ - 1;
                                const int irows_ = (orows_ - pupad - pdpad) * stride + dilated_krows_ - 1;

                                int lpad = icol < 0 ? -icol : 0;
                                int rpad = icol + icols_ > dilated_in_col_dim ? icol + icols_ - dilated_in_col_dim : 0;
                                int upad = irow < 0 ? -irow : 0;
                                int dpad = irow + irows_ > dilated_in_row_dim ? irow + irows_ - dilated_in_row_dim : 0;

                                if (input_dilated) {
                                  lpad += lpad == 0 && icol % 2 != 0;
                                  rpad += rpad == 0 && (icol + icols_) % 2 != 1;
                                  upad += upad == 0 && irow % 2 != 0;
                                  dpad += dpad == 0 && (irow + irows_) % 2 != 1;
                                }

                                int krow_ = krow;
                                int kcol_ = kcol;
                                /*
                                // ToDo
                                if (wrot180) {
                                  krow_ = kernel_dim - krow - krows_;
                                  kcol_ = kernel_dim - kcol - kcols_;
                                }
                                */

                                int weight_index = (int)(poch/pochs)*ceil_divide_int(this_tile_in_channel, kchs) + (int)(kch/kchs);
                                const elem_t* weights_slice = (elem_t*) (BASE_ADDR+B_sp_addr+kernel_dim*kernel_dim*kchs*pochs*weight_index) + (krow_*kernel_dim*kchs + kcol_*kchs)*pochs;
                                //const elem_t * weights_slice = BASE_ADDR + weights + (krow_*kernel_dim*in_channels + kcol_*in_channels + kch) * weight_stride + poch;
                                /*
                                // ToDo
                                if (trans_weight_1203) {
                                  weights_slice = weights + (kch*kernel_dim*kernel_dim + krow_*kernel_dim+kcol_) * out_channels + poch;
                                } else if (trans_weight_0132) {
                                  weights_slice = weights + (krow_*kernel_dim*out_channels + kcol_*out_channels + poch) * in_channels + kch;
                                }
                                */

                                const elem_t * in = (A_channel != -1) ? (elem_t*) (BASE_ADDR+A_sp_addr) + irows_outer*icols_outer*kchs * (int)(kch/kchs) : input + (b *in_row_dim * in_col_dim + ((irow+upad)>>input_dilated) * in_col_dim + ((icol+lpad)>>input_dilated)) * in_stride + kch_outer + kch;
                                //const elem_t * in = input + (b *in_row_dim * in_col_dim + ((irow+upad)>>input_dilated) * in_col_dim + ((icol+lpad)>>input_dilated)) * in_stride + kch;
                                /*
                                 // ToDo
                                if (trans_input_3120) {
                                  in = input + (kch * in_row_dim * in_col_dim + ((irow+upad)>>input_dilated) * in_col_dim + ((icol+lpad)>>input_dilated)) * batch_size + b;
                                }
                                */
                                if(b_reuse && (pocol + (porow - porow_start) + b > 0)) weights_slice = NULL;
							    if(a_reuse && (poch > 0)) in = NULL;
                                //printf("a_reuse: %d, b_reuse: %d, a_spad_id: %d, b_spad_id: %d, in: %llu, weight: %llu \n", a_reuse, b_reuse, a_spad_id, b_spad_id, in, weights_slice);
#if DEBUG_PRINT == 1
                                printf("input address: 0x%08lx, weight address: 0x%08lx, output address: 0x%08lx\n", in, weights_slice, out);
#endif
                                // for now, no krow/kcol outer tiling
                                // do this after probing done
                                while(A_channel != -1 && in != NULL){
                                  uint64_t curr_loaded_tile;
                                  dma_probe_state(curr_loaded_tile, A_channel);
                                  if(curr_loaded_tile > (int)(kch/kchs))
                                    break;
                                }
                                int curr_weight_tile = (int)(poch/pochs) * ceil_divide_int(this_tile_in_channel, kchs) + (int)(kch/kchs);
                                while(weights_slice != NULL){
                                  uint64_t curr_loaded_tile;
                                  dma_probe_state(curr_loaded_tile, B_channel);
                                  if(curr_loaded_tile > curr_weight_tile)
                                    break;
                                }
                                sp_tiled_conv(
                                    batch_size, A_channel == -1 ? in_row_dim : irows_outer, A_channel == -1 ? in_col_dim : icols_outer, kchs,// A_channel == -1 ? in_channels : kchs,
                                    out_channels, out_row_dim, out_col_dim,
                                    pool_out_row_dim, pool_out_col_dim,

                                    stride, padding, kernel_dim, kernel_dilation,
                                    A_spad_stride, B_spad_stride, out_stride,

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
                                    false, a_spad_id, b_spad_id);

                            }
                        }
                    }
                  }
                }
            }
        }
      }
    }
}


// need to specify each operand/output's stride
// stride only for trans == false, wrot == false
static void double_tiled_conv_auto(
      int batch_size, int in_row_dim, int in_col_dim, int in_channels,
      int out_channels, int out_row_dim, int out_col_dim,
      int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
      int in_stride, int weight_stride, int out_stride, // specify in/output's stride
      bool wrot180, bool trans_output_1203, bool trans_input_3120,
      bool trans_weight_1203, bool trans_weight_0132,

      const elem_t * input,
      const elem_t * weights,
      const acc_t * bias,
      elem_t * output,

      int act, acc_scale_t scale,
      int pool_size, int pool_stride, int pool_padding,

      int thread_id) {

  const bool no_pool = pool_stride == 0;
  if (no_pool) {
      pool_size = 1;
      pool_stride = 1;
      pool_padding = 0;
  }

  const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

  const bool downsample = stride == 2 && kernel_dim == 1 && padding == 0 && no_pool && in_row_dim % 2 == 0 && in_col_dim % 2 == 0;

  int A_channel = dma_channel_alloc[thread_id][0];
  int B_channel = dma_channel_alloc[thread_id][1];
  int D_channel = dma_channel_alloc[thread_id][2]; // ToDo

  uint64_t global_spad_capacity = sp_capacity_alloc[thread_id];
  int size_input = batch_size * in_channels * in_row_dim * in_col_dim;
  int size_output = batch_size * out_channels * pool_out_row_dim * pool_out_col_dim;
  int size_weight = in_channels * out_channels * kernel_dim * kernel_dim;

  // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
  int args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels, kernel_dim, kernel_dim, in_channels};

  int* args_out;
  args_out = tiling_factor_conv(args, stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120, pool_size, pool_padding, pool_stride);

  const int batches = args_out[0];
  const int orows = args_out[1];
  const int ocols = args_out[2];
  const int ochs = args_out[3];
  const int krows = args_out[4];
  const int kcols = args_out[5];
  const int kchs = args_out[6];
 
  // no outer tile icols
  // channel moves with weight kch
  int irows = orows * stride + krows + (kernel_dilation - 1) * (krows - 1) - 1;
  int icols = ocols * stride + kcols + (kernel_dilation - 1) * (kcols - 1) - 1;

  bool input_from_dram = (sp_input_base_addr[thread_id] == -1);
  bool output_to_dram = false;
  // ToDo: add this to tiled_matmul
  //int size_weight_tile = kchs * ochs * krows * kcols * (kernel_dim / krows) * (kernel_dim / kcols);
  int size_input_tile = input_from_dram ? batches * kchs * irows * icols : size_input;
  uint64_t total_inout_requirement = input_from_dram ? size_output : size_input + size_output;
  //uint64_t total_inout_requirement = input_from_dram ? 0 : size_input;
  // number of inner tile per outer tile
  // -1 : entire data fits as outer tile
  //int outer_input_tile = -1;
  // for now, don't outer tile kernel_row/co
  int size_weight_tile = kchs * ochs * kernel_dim * kernel_dim;
  //int outer_weight_tile = ceil_divide_int(kernel_dim, krows) * ceil_divide_int(kernel_dim, kcols);
   
  if(total_inout_requirement + 2 * size_weight_tile > global_spad_capacity){
    output_to_dram = true;
    total_inout_requirement -= size_output;
  }
  
  int kch_outer = 1;
  int och_outer = 1;
  // ToDo: do irow, icol
  int64_t spad_capacity = global_spad_capacity - total_inout_requirement; 
  if(input_from_dram){ 
    int max_kch_outer = ceil_divide_int(in_channels , kchs);
    int max_och_outer = ceil_divide_int(out_channels, ochs);
    int outer_tile_size = size_input_tile * kch_outer + size_weight_tile * kch_outer * och_outer;
    while(spad_capacity > outer_tile_size && kch_outer < max_kch_outer){
      outer_tile_size = (size_input_tile + size_weight_tile * och_outer) * (kch_outer + 1);
      if(spad_capacity >= outer_tile_size)
        kch_outer ++;
    }
    while(spad_capacity > outer_tile_size && och_outer < max_och_outer){
      outer_tile_size = (size_input_tile + size_weight_tile * (och_outer + 1)) * kch_outer;
      if(spad_capacity >= outer_tile_size)
        och_outer ++;
    }
    //kch_outer = 1;
    size_input_tile = size_input_tile * kch_outer;
#if DEBUG_PRINT == 1
    printf("kch outer: %d, och outer: %d\n", kch_outer, och_outer);
#endif
  }
  else{
    A_channel = -1; // no need to configure channel for memcpy
    int max_kch_outer = ceil_divide_int(in_channels , kchs);
    int max_och_outer = ceil_divide_int(out_channels, ochs);
    int outer_tile_size = size_weight_tile * kch_outer * och_outer;
    while(spad_capacity > outer_tile_size && kch_outer < max_kch_outer){
      outer_tile_size = (size_weight_tile * och_outer) * (kch_outer + 1);
      if(spad_capacity >= outer_tile_size)
        kch_outer ++;
    }
    while(spad_capacity > outer_tile_size && och_outer < max_och_outer){
      outer_tile_size = (size_weight_tile * (och_outer + 1)) * kch_outer;
      if(spad_capacity >= outer_tile_size)
        och_outer ++;
    }
#if DEBUG_PRINT == 1
    printf("kch outer: %d, och outer: %d\n", kch_outer, och_outer);
#endif
  }

  // input address comes first, out sp address from end
  bool loop_id = sp_base_addr[thread_id] == sp_input_base_addr[thread_id];
  uint64_t C_sp_addr = (loop_id || input_from_dram) ? sp_base_addr[thread_id] +  sp_capacity_alloc[thread_id] - size_output : sp_base_addr[thread_id];
  uint64_t A_sp_addr = input_from_dram ? sp_base_addr[thread_id] : sp_input_base_addr[thread_id]; 
  uint64_t B_sp_addr = loop_id || input_from_dram ? sp_base_addr[thread_id] + size_input_tile : sp_base_addr[thread_id] + size_output;
#if DEBUG_PRINT == 1
  printf("input spad address: 0x%08lx, weight spad address: 0x%08lx, output spad address: 0x%08lx\n", A_sp_addr, B_sp_addr, C_sp_addr);
  printf("input from dram: %d, output to dram: %d\n", input_from_dram, output_to_dram);
#endif
  
  double_tiled_conv(
      batch_size, in_row_dim, in_col_dim, in_channels,
      out_channels, out_row_dim, out_col_dim,
      stride, input_dilation, kernel_dilation, padding, kernel_dim,
      in_stride, weight_stride, out_stride,
      wrot180, trans_output_1203, trans_input_3120,
      trans_weight_1203, trans_weight_0132,

      batches,
      orows, ocols, ochs,
      krows, kcols, kchs,
      och_outer, kch_outer,

      input,
      weights,
      bias,
      output_to_dram ? output : (elem_t*) (C_sp_addr+BASE_ADDR),

      A_channel, B_channel,
      (elem_t*) A_sp_addr, (elem_t*) B_sp_addr,

      act, scale,
      pool_size, no_pool ? 0 : pool_stride, pool_padding);

  if(!output_to_dram){
    sp_input_base_addr[thread_id] = C_sp_addr;
  }
  else
    sp_input_base_addr[thread_id] = -1;
}
