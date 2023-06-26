#ifndef NUM_THREAD
#define NUM_THREAD 1
#endif

#define DEBUG_PRINT 1

#include "include/gemmini.h"
#include "include/dma.h"

static uint64_t sp_capacity_alloc[NUM_THREAD] = {0};
static uint64_t sp_base_addr[NUM_THREAD] = {0};
static int dma_channel_alloc[NUM_THREAD][4] = {0}; // need to initialize -1
static int sp_input_base_addr[NUM_THREAD] = {0}; // need to initialize -1

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

   printf("inner_tile_I: %d\n", inner_tile_I);
   printf("inner_tile_J: %d\n", inner_tile_J);
   printf("inner_tile_K: %d\n\n", inner_tile_K);


   size_t factor_I = ceil_divide_int(outer_tile_I, inner_tile_I);
   size_t factor_J = ceil_divide_int(outer_tile_J, inner_tile_J);
   size_t factor_K = ceil_divide_int(outer_tile_K, inner_tile_K);
   
   inner_tile_I = outer_tile_I / factor_I;
   inner_tile_J = outer_tile_J / factor_J;
   inner_tile_K = outer_tile_K / factor_K;

   printf("factor_I: %d\n", factor_I);
   printf("factor_J: %d\n", factor_J);
   printf("factor_K: %d\n\n", factor_K);
   printf("inner_tile_I: %d\n", inner_tile_I);
   printf("inner_tile_J: %d\n", inner_tile_J);
   printf("inner_tile_K: %d\n\n", inner_tile_K);

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
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        elem_t* A_sp_addr, elem_t* B_sp_addr,
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
  int A_spad_stride = a_transpose ? dim_I : dim_K; // or A_stride?
  int B_spad_stride = b_transpose ? dim_K : dim_J; // or B_stride?
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
    printf("outer_tile_J: %d, dim_J_padded: %d\n", outer_tile_J, dim_J_padded);
    uint64_t B_dram_offset = b_transpose ? (o_j0*stride_B) : (o_j0); 
    const size_t last_J = outer_J_dim % (inner_tile_J*DIM) == 0 ? inner_tile_J : (outer_J_dim/DIM) % inner_tile_J;
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
    dma_memcpy(B_channel, B_dram_offset, num_B_row_tile, num_B_col_tile, B_row, B_col_bytes, B_inner_row, B_inner_col_bytes); 
    
    size_t J0 = outer_J_dim / (inner_tile_J*DIM) + (outer_J_dim % (inner_tile_J*DIM) != 0);
    
    for (size_t o_i0 = 0; o_i0 < dim_I_padded; o_i0+=(outer_tile_I*DIM)){
      size_t outer_I_dim = (o_i0 + (outer_tile_I*DIM)) >= dim_I_padded ? dim_I_padded - o_i0 : outer_tile_I*DIM; 
      const size_t last_I = outer_I_dim % (inner_tile_I*DIM) == 0 ? inner_tile_I : (outer_I_dim/DIM) % inner_tile_I;
      //int A_spad_stride = stride_A;
      // -1: already at scratchpad
      if(A_channel != -1){
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

        dma_memcpy(A_channel, A_dram_offset, num_A_row_tile, num_A_col_tile, A_row, A_col_bytes, A_inner_row, A_inner_col_bytes); 
        gemmini_extended3_config_ld(A_spad_stride * sizeof(elem_t), A_scale_factor, false, 0);
 
      } 
      size_t I0 = outer_I_dim / (inner_tile_I*DIM) + (outer_I_dim % (inner_tile_I*DIM) != 0);

      // start inner loop
      // reuse operand if it fits scratchpad
      int a_spad_id = 0;
      int b_spad_id = 0;
      bool b_reuse = (J0 * K0 <= 2) && (dataflow == WEIGHT_STATIONARY);
      bool a_reuse = (I0 * K0 <= 2) && (dataflow == WEIGHT_STATIONARY);
      gemmini_extended3_config_ld(B_spad_stride * sizeof(elem_t), B_scale_factor, false, 1)
     
      for (size_t i0 = 0; i0 < I0; i0++)
        for (size_t j0 = 0; j0 < J0; j0++)
          for (size_t k0 = 0; k0 < K0; k0++) {
            int A_tile_index = a_transpose ? k0*I0 + i0 : i0*K0 + k0; 
            int B_tile_index = b_transpose ? j0*K0 + k0 : k0*J0 + j0;
            // wait until inner tile is ready
            while(A_channel != -1){
              uint64_t curr_tile;
              dma_probe(curr_tile, A_channel);
              if(curr_tile > A_tile_index)
                break;
            }
            while(true){
              uint64_t curr_tile;
              dma_probe(curr_tile, B_channel);
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
              size_t bias_row = repeating_bias ? 0 : i0*inner_tile_I*DIM;
              // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * inner_tile_J * DIM]);
              pre = (int8_t*)D + (bias_row * stride_D + j0 * inner_tile_J * DIM)*sizeof_D;
            }

            // does not need DMA configuration
            void * out = k0 == K0-1 ? (int8_t*)C + (i0*inner_tile_I*DIM*stride_C + j0*inner_tile_J*DIM)*sizeof_C : NULL;

            const size_t I = ((o_i0 + outer_tile_I*DIM >= dim_I_padded) && (i0 < I0-1)) ? inner_tile_I : last_I;
            const size_t J = ((o_j0 + outer_tile_J*DIM >= dim_J_padded) && (j0 < J0-1)) ? inner_tile_J : last_J;
            const size_t K = k0 < K0-1 ? inner_tile_K : last_K;

            const size_t pad_I = i0 == I0-1 ? padding_I : 0;
            const size_t pad_J = j0 == J0-1 ? padding_J : 0;
            const size_t pad_K = k0 == K0-1 ? padding_K : 0;

            // ensure A is not outer-tiled when all pre-loaded
            const elem_t * a = a_transpose ? (BASE_ADDR + A_sp_addr + k0*inner_tile_K*DIM*A_spad_stride + i0*inner_tile_I*DIM)
              : (BASE_ADDR + A_sp_addr + i0*inner_tile_I*DIM*A_spad_stride + k0*inner_tile_K*DIM);

            const elem_t * b = b_transpose ? (BASE_ADDR + B_sp_addr + j0*inner_tile_J*DIM*B_spad_stride + k0*inner_tile_K*DIM)
              : (BASE_ADDR + B_sp_addr + k0*inner_tile_K*DIM*B_spad_stride + j0*inner_tile_J*DIM);

            if(a_reuse && j0 >= 1) a = NULL;
            if(b_reuse && i0 >= 1) b = NULL;
            printf("a_reuse: %d, b_reuse: %d, a_spad_id: %d, b_spad_id: %d, a: 0x%08lx, b: 0x%08lx C: 0x%08lx, out: 0x%08lx\n", a_reuse, b_reuse, a_spad_id, b_spad_id, a, b, C, out);
            
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
   bool output_to_dram = true;
   uint64_t total_inout_requirement = input_from_dram ? size_input+size_output : size_output;
   // for now, test load only
   /*
   if (total_inout_requirement > global_spad_capacity)
       output_to_dram = true;
   */
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
        A, B, D, output_to_dram ? C : (void*) C_sp_addr,
        (elem_t*) A_sp_addr, (elem_t*) B_sp_addr, // todo: bias
        stride_A, stride_B, stride_D, output_to_dram ? stride_C : dim_J,
        A_channel, B_channel,
        A_scale_factor, B_scale_factor, D_scale_factor,
        act, scale, bert_scale, repeating_bias,
        outer_tile_I, outer_tile_J, outer_tile_K,
        inner_tile_I, inner_tile_J, inner_tile_K,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA);

    if(output_to_dram){
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

