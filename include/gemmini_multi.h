
static void multi_tiled_matmul_outer(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t tile_I, size_t tile_J, size_t tile_K,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        uint8_t weightA,
        int num_accel) {

  dim_J = ceil_divide_int(dim_J, num_accel);
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

  //printf("dim_J: %d, J0: %d, padding_J: %d, stride_B: %d, sizeof D: %d, sizeof C: %d\n", dim_J, J0, padding_J, stride_B, sizeof_D, sizeof_C);
  
  for(int i = 0; i < num_accel; i++){
    rr_set_opc(XCUSTOM_ACC, i);
    gemmini_extended_config_ex(WS, act & 3, 0, 1, a_transpose, b_transpose);
    gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
    gemmini_extended3_config_ld(stride_A * sizeof(elem_t), A_scale_factor, false, 0);
    gemmini_extended3_config_ld(stride_B * sizeof(elem_t), B_scale_factor, false, 1);
    gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
    //rr_fence(i);
  }

  void (*inner)(const elem_t *, const elem_t *, const void *, void *,
        scale_t, scale_t, scale_acc_t,
        size_t, size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t,
        bool, bool,
        bool, bool,
        bool, bool,
        int, int, int);

  //if (dataflow == OUTPUT_STATIONARY) {
  //  inner = &sp_tiled_matmul_os;
  //} else /* if (dataflow == WEIGHT_STATIONARY) */ {
    inner = &sp_tiled_matmul_ws;
  //}

  // reuse operand if it fits scratchpad
  int a_spad_id = 0;
  int b_spad_id = 0;
  bool b_reuse = (J0 * K0 <= 2);// && (dataflow == WEIGHT_STATIONARY);
  bool a_reuse = (I0 * K0 <= 2);// && (dataflow == WEIGHT_STATIONARY);

  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++) {
        if(a_reuse)
          a_spad_id = ((i0+k0) == 0) ? 1 : 2;
        if(b_reuse)
          b_spad_id = ((j0+k0) == 0) ? 1 : 2;

        const size_t I = i0 < I0-1 ? tile_I : last_I;
        const size_t J = j0 < J0-1 ? tile_J : last_J;
        const size_t K = k0 < K0-1 ? tile_K : last_K;

        const size_t pad_I = i0 == I0-1 ? padding_I : 0;
        const size_t pad_J = j0 == J0-1 ? padding_J : 0;
        const size_t pad_K = k0 == K0-1 ? padding_K : 0;
        
        for (size_t n = 0; n < num_accel; n++){
          const void * pre;
          int J_offset = j0 < J0 - 1? (num_accel*j0+n)*tile_J*DIM : (num_accel*j0)*tile_J*DIM + n*last_J*DIM;
          if (k0 != 0) {
            pre = NULL;
          } else {
            size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
            // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
            pre = (int8_t*)D + (bias_row * stride_D + J_offset)*sizeof_D;
          }
          rr_set_opc(XCUSTOM_ACC, n);

          void * out = k0 == K0-1 ? (int8_t*)C + (i0*tile_I*DIM*stride_C + J_offset)*sizeof_C : NULL;


          const elem_t * a = a_transpose ? (A + k0*tile_K*DIM*stride_A + i0*tile_I*DIM)
            : (A + i0*tile_I*DIM*stride_A + k0*tile_K*DIM);

          const elem_t * b = b_transpose ? (B + J_offset*stride_B + k0*tile_K*DIM)
            : (B + k0*tile_K*DIM*stride_B + J_offset);
          
          if(a_reuse && j0 >= 1) a = NULL;
          if(b_reuse && i0 >= 1) b = NULL;
          //printf("a_reuse: %d, b_reuse: %d, a_spad_id: %d, b_spad_id: %d, a: %llu, b: %llu, c: %llu, d: %llu, J: %d\n", a_reuse, b_reuse, a_spad_id, b_spad_id, a, b, out, pre, J);
          (*inner)(A == NULL ? NULL : a, B == NULL ? NULL : b, D == NULL ? NULL : pre, C == NULL ? NULL : out,
              A_scale_factor, B_scale_factor, D_scale_factor,
              I, J, K,
              pad_I, pad_J, pad_K,
              stride_A, stride_B, stride_D, stride_C,
              a_transpose, b_transpose,
              full_C, low_D,
              no_bias, repeating_bias,
              act, a_spad_id, b_spad_id);
         
        }
      }
  for(int n = 0; n < num_accel; n++)
      rr_fence(n);
  //gemmini_fence();
}


static void multi_tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        int num_accel) {

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

    // for now, just assume all divide in J axis
    const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
    const size_t dim_J_padded = (ceil_divide_int(dim_J, num_accel) / DIM + (ceil_divide_int(dim_J, num_accel) % DIM != 0)) * DIM;
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

#ifdef PRINT_TILE
#if PRINT_TILE
    const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
    const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);

    printf("tile_I: %d\n", tile_I);
    printf("tile_J: %d\n", tile_J);
    printf("tile_K: %d\n\n", tile_K);

    printf("spad_rows: %d\n", spad_rows);
    printf("acc_rows: %d\n\n", acc_rows);

    printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
    printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

    exit(EXIT_SUCCESS);
#endif
#endif

    //printf("DIM: %d\n", DIM);
    //printf("tile_I: %d\n", tile_I);
    //printf("tile_J: %d\n", tile_J);
    //printf("tile_K: %d\n\n", tile_K);
    /*
    tiled_matmul(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        act, scale, bert_scale, repeating_bias,
        tile_I, tile_J, tile_K,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        tiled_matmul_type);
        */
    multi_tiled_matmul_outer(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        tile_I, tile_J, tile_K,
        act, scale, bert_scale, repeating_bias,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        num_accel);
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

static void multi_tiled_conv(
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

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,
        int pool_size, int pool_stride, int pool_padding,

        int num_accel){
        //enum tiled_matmul_type_t tiled_conv_type) {

#ifdef GEMMINI_ASSERTIONS
  if (trans_weight_1203 && trans_weight_0132) {
    printf("Only one weight transformation can be applied at a time\n");
    exit(1);
  }
#endif
/*
    if (tiled_conv_type == CPU) {
      if (pool_size == 1 && pool_stride == 1 && pool_padding == 0) {
        pool_stride = 0;
      }

      // assume in_dim_rows = in_dim_cols
      // and out_dim_rows = out_dim_cols for now
      conv_cpu(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        in_stride, weight_stride, out_stride,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale,
        pool_size, pool_stride, pool_padding);
      return;
    } else if (tiled_conv_type == OS) {
      printf("Gemmini convs do not currently support OS\n");
      exit(1);
    }
*/
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
    for(int n = 0; n < num_accel; n++){
        rr_set_opc(XCUSTOM_ACC, n);
        gemmini_extended_config_st(st_dram_stride, act, scale);

        gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
    }
    const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int dilated_in_row_dim = in_row_dim + (input_dilation - 1) * (in_row_dim- 1);
    const int dilated_in_col_dim = in_col_dim + (input_dilation - 1) * (in_col_dim- 1);

    out_channels = ceil_divide_int(out_channels, num_accel);
    size_t a_spad_id = 0;
    size_t b_spad_id = 0;

    int porow_end = pool_out_row_dim;
	int porow_start = 0;
    bool a_reuse = false;
    bool b_reuse = false;
    size_t num_kch = ceil_divide_int(in_channels, kchs);
    size_t num_poch = ceil_divide_int(out_channels, pochs);
    size_t num_b = ceil_divide_int(batch_size, batches);
    size_t num_porow = ceil_divide_int((porow_end - porow_start), porows);
    size_t num_pocol = ceil_divide_int(pool_out_col_dim, pocols);
    size_t num_krow = ceil_divide_int(kernel_dim, krows);
    size_t num_kcol = ceil_divide_int(kernel_dim, kcols);


//    printf("num_kch: %d, num_poch: %d, num_b: %d, num_porow: %d, num_pocol: %d, num_krow: %d, num_kcol: %d\n", num_kch, num_poch, num_b, num_porow, num_pocol, num_krow, num_kcol);

    if(num_kch * num_poch * num_krow * num_kcol <= 2) 
      b_reuse = true;
    if(num_kch * num_krow * num_kcol * num_b * num_porow * num_pocol <= 2)
      a_reuse = true;

    for (int b = 0; b < batch_size; b += batches) {
        for (int porow = porow_start; porow < porow_end; porow += porows) {
            const int orow = porow * pool_stride - pool_padding;

            for (int pocol = 0; pocol < pool_out_col_dim; pocol += pocols) {
                const int ocol = pocol * pool_stride - pool_padding;

                for (int poch = 0; poch < out_channels; poch += pochs) {
                    for (int krow = 0; krow < kernel_dim; krow += krows) {
                        const int orow_floored = orow < 0 ? 0 : orow;
                        int irow = orow_floored * stride + krow * kernel_dilation - padding;

                        for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                            const int ocol_floored = ocol < 0 ? 0 : ocol;
                            int icol = ocol_floored * stride + kcol * kernel_dilation - padding;

                            for (int kch = 0; kch < in_channels; kch += kchs) {
                                if(a_reuse)
						           a_spad_id = (kch + krow + kcol + b + (porow - porow_start) + pocol) == 0 ? 1 : 2;
					            if(b_reuse)
						           b_spad_id = (kch + poch + krow + kcol) == 0 ? 1 : 2;
                                
                                const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                                const int porows_ = pool_out_row_dim - porow > porows ? porows : pool_out_row_dim - porow;
                                const int pocols_ = pool_out_col_dim - pocol > pocols ? pocols : pool_out_col_dim - pocol;
                                const int pochs_ = out_channels - poch > pochs ? pochs : out_channels - poch;
                                const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                                const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;
                                const int kchs_ = in_channels - kch > kchs ? kchs : in_channels - kch;

                                const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                                const int orows_ = porows_ * pool_stride + pool_size - 1;

                                const int plpad = ocol < 0 ? -ocol : 0;
                                const int prpad = ocol + ocols_ > out_col_dim ? ocol + ocols_ - out_col_dim : 0;
                                const int pupad = orow < 0 ? -orow : 0;
                                const int pdpad = orow + orows_ > out_row_dim ? orow + orows_ - out_row_dim : 0;

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
                                if (wrot180) {
                                  krow_ = kernel_dim - krow - krows_;
                                  kcol_ = kernel_dim - kcol - kcols_;
                                }

                                for(int n = 0; n < num_accel; n ++){
                                  int channel_accel_offset = poch * num_accel + pochs_ * n;
                                  elem_t * out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * out_stride + channel_accel_offset;//poch; 
                                  const acc_t * bias_ = bias + channel_accel_offset;//poch;
                                  if (krow > 0 ||
                                        kcol > 0 ||
                                        kch > 0) {
                                      bias_ = NULL;
                                  }
                                  if (krow + krows < kernel_dim ||
                                        kcol + kcols < kernel_dim ||
                                        kch + kchs < in_channels) {
                                      out = NULL;
                                  }

                                  rr_set_opc(XCUSTOM_ACC, n);

                                  const elem_t * weights_slice = weights + (krow_*kernel_dim*in_channels + kcol_*in_channels + kch) * weight_stride + channel_accel_offset;//poch;

                                  const elem_t * in = input + (b *in_row_dim * in_col_dim + ((irow+upad)>>input_dilated) * in_col_dim + ((icol+lpad)>>input_dilated)) * in_stride + kch;
                                
                                  if(b_reuse && (pocol + (porow - porow_start) + b > 0)) weights_slice = NULL;
							      if(a_reuse && (poch > 0)) in = NULL;
                                  //printf("a_reuse: %d, b_reuse: %d, a_spad_id: %d, b_spad_id: %d, in: %llu, weight: %llu \n", a_reuse, b_reuse, a_spad_id, b_spad_id, in, weights_slice);
 
                                  sp_tiled_conv(
                                      batch_size, in_row_dim, in_col_dim, in_channels,
                                      out_channels, out_row_dim, out_col_dim,
                                      pool_out_row_dim, pool_out_col_dim,

                                      stride, padding, kernel_dim, kernel_dilation,
                                      in_stride, weight_stride, out_stride,

                                      pool_size, pool_stride, pool_padding,

                                      batches_,
                                      porows_, pocols_, pochs_,
                                      krows_, kcols_, kchs_,

                                      lpad, rpad, upad, dpad,
                                      plpad, prpad, pupad, pdpad,

                                      input == NULL ? NULL : in,
                                      weights == NULL ? NULL : weights_slice,
                                      output == NULL ? NULL : out,
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
  for(int n = 0; n < num_accel; n++)
      rr_fence(n);
}


static void multi_tiled_conv_auto(
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

        int num_accel) {

    const bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

    const bool downsample = stride == 2 && kernel_dim == 1 && padding == 0 && no_pool && in_row_dim % 2 == 0 && in_col_dim % 2 == 0;

    // Tile convolution params

    // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs}
    // for now, assume everything divide in output channel dimension
    int out_channels_accel = ceil_divide_int(out_channels, num_accel);
    int args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels_accel, kernel_dim, kernel_dim, in_channels};
    const int max_args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels_accel, kernel_dim, kernel_dim, in_channels};

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

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    }
    args[1]=args[1]/2;
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

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
            if(i == out_channels_idx || i == in_channels_idx)
               args_candidate[i] += DIM;
            else
               args_candidate[i]++;

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
                //printf("i: %d, spad_rows: %d, args: %d\n", i, spad_rows, args[i]);
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

 
#ifdef PRINT_TILE
#if PRINT_TILE   
    spad_rows = tiled_conv_total_spad_rows(false,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
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
#endif
#endif

    multi_tiled_conv(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        in_stride, weight_stride, out_stride,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,

        batches,
        orows, ocols, ochs,
        krows, kcols, kchs,

        input,
        weights,
        bias,
        output,

        act, scale,
        pool_size, no_pool ? 0 : pool_stride, pool_padding,

        num_accel);
}
