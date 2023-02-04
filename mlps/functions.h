
void full_is_equal(int dim_r, int dim_c, elem_t x[dim_r][dim_c], elem_t y[dim_r][dim_c]) {
  for (size_t i = 0; i < dim_r; ++i)
    for (size_t j = 0; j < dim_c; ++j)
      if (abs_diff((int)(x[i][j]*1000), (int)(y[i][j]*1000)) > 1){
          printf("i: %d, j: %d, x: %d, y: %d\n", i, j, (int)(x[i][j]*1000), (int)(y[i][j]*1000));
      }
}

// there should be better way to store matrix
void cpu_A_inv(int mat_dim, int stride, elem_t in[mat_dim][stride], elem_t out[mat_dim][stride], int block_dim){

    int num_block = ceil_divide_int(mat_dim, block_dim);
    for (int i = 0; i < num_block; i++){
      int start_index = i * block_dim;
      if(i!=num_block - 1|| mat_dim%block_dim == 0){
        elem_t mul1122 = in[start_index+1][start_index+1]*in[start_index+2][start_index+2];
        elem_t mul2112 = in[start_index+2][start_index+1]*in[start_index+1][start_index+2];
        elem_t mul1022 = in[start_index+1][start_index]*in[start_index+2][start_index+2];
        elem_t mul1220 = in[start_index+1][start_index+2]*in[start_index+2][start_index];
        elem_t mul1021 = in[start_index+1][start_index]*in[start_index+2][start_index+1];
        elem_t mul1120 = in[start_index+1][start_index+1]*in[start_index+2][start_index];
        elem_t det = in[start_index][start_index]*(mul1122-mul2112) - in[start_index][start_index+1]*(mul1022-mul1220) + in[start_index][start_index+2]*(mul1021-mul1120);
        elem_t invdet = 1/det;

        out[start_index][start_index] = (mul1122 - mul2112) * invdet;
        out[start_index][start_index+1] = (in[start_index][start_index+2]*in[start_index+2][start_index+1]-in[start_index][start_index+1]*in[start_index+2][start_index+2])*invdet;
        out[start_index][start_index+2] = (in[start_index][start_index+1]*in[start_index+1][start_index+2]-in[start_index][start_index+2]*in[start_index+1][start_index+1])*invdet;
        out[start_index+1][start_index] = (mul1220 - mul1022) * invdet;
        out[start_index+1][start_index+1] = (in[start_index][start_index]*in[start_index+2][start_index+2]-in[start_index][start_index+2]*in[start_index+2][start_index])*invdet;
        out[start_index+1][start_index+2] = (in[start_index+1][start_index]*in[start_index][start_index+2]-in[start_index][start_index]*in[start_index+1][start_index+2])*invdet;
        out[start_index+2][start_index] = (mul1021 - mul1120) * invdet;
        out[start_index+2][start_index+1] = (in[start_index+2][start_index]*in[start_index][start_index+1]-in[start_index][start_index]*in[start_index+2][start_index+1])*invdet;
        out[start_index+2][start_index+2] = (in[start_index][start_index]*in[start_index+1][start_index+1]-in[start_index+1][start_index]*in[start_index][start_index+1])*invdet;
      }
      else if(mat_dim%block_dim == 2){
        elem_t d = in[start_index][start_index]*in[start_index+1][start_index+1] - in[start_index][start_index+1]*in[start_index+1][start_index];
        out[start_index][start_index] = in[start_index+1][start_index+1]/d;
        out[start_index+1][start_index] = in[start_index][start_index]/d;
        out[start_index][start_index+1] = (-1)*in[start_index][start_index+1]/d;
        out[start_index+1][start_index] = (-1)*in[start_index+1][start_index]/d;
      }
      else{
        out[start_index][start_index] = 1/in[start_index][start_index];
      }
    }
}

void cpu_matmul(int out_row, int out_col, int a_stride, int b_row, int b_stride, int d_stride, int out_stride, elem_t in_a[out_row][a_stride], elem_t in_b[b_row][b_stride], elem_t bias[out_row][d_stride], elem_t out[out_row][out_stride]) {
  for (size_t r = 0; r < out_row; r++)
    for (size_t c = 0; c < out_col; c++) {
      out[r][c] = d_stride > 0 ? bias[r][c] : 0;
      for (size_t k = 0; k < b_row; k++)
        out[r][c] += in_a[r][k]*in_b[k][c];
    }
}
void cpu_matmul_transpose(int out_row, int out_col, int a_stride, int K, int b_stride, int d_stride, int out_stride, elem_t in_a[out_row][a_stride], elem_t in_b[out_col][b_stride], elem_t bias[out_row][d_stride], elem_t out[out_row][out_stride]) {
  for (size_t r = 0; r < out_row; r++)
    for (size_t c = 0; c < out_col; c++) {
      out[r][c] = d_stride > 0 ? bias[r][c] : 0;
      for (size_t k = 0; k < K; k++)
        out[r][c] += in_a[r][k]*in_b[c][k];
    }
}

// D-CA^-1B = D-CA^-1C^T
// for now, input already inversed
void schur(int a_dim, int a_stride, int d_dim, int d_stride, int c_stride, elem_t* A_inv_in, elem_t* in_C, elem_t* in_D, elem_t* temp_out, int block_dim){
    int num_block = ceil_divide_int(a_dim, block_dim);
    //elem_t temp_out[d_dim][a_stride] = {0};
    //elem_t* temp_out_pt = (elem_t*) temp_out;
    
    for (int i = 0; i < num_block; i++){
      int start_index = i * block_dim;
      int eff_block_dim = i != num_block - 1 ? block_dim : (a_dim - start_index);
      elem_t* in_A = in_C + start_index;
      elem_t* in_B = A_inv_in + a_stride * start_index + start_index;
      elem_t* out = temp_out + start_index;
      tiled_matmul_auto(d_dim, eff_block_dim, eff_block_dim, in_A, in_B, NULL, out,
            c_stride, a_stride, d_stride, a_stride,
            false, false, false, false,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false, false, 0,
            0, WS);
    }
    //printf("finished CA^-1\n");
    
    tiled_matmul_auto(d_dim, d_dim, a_dim, temp_out,  (elem_t*) in_C, (elem_t*) in_D, (elem_t*) in_D,
        a_stride, c_stride, d_stride, d_stride,
        false, false, false, false,
        MVIN_SCALE_IDENTITY, -1, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        false, true, false, 1,
        0, WS);
}

 
void full_printMatrix(int row, int col, elem_t m[row][col]) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j)
		 printf("%d.%d ", (int)m[i][j], ((int)(m[i][j]*1000))%1000);
    printf("\n");
  }
}

bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i])
            //if (((int)(a[i]*100)) != ((int)(b[i]*100)))
            if(abs_diff((int)(a[i]*100), (int)(b[i]*100)) > 1)
			    printf("i: %d, value: %d.%d, %d.%d \n", i, (int)a[i], ((int)(a[i]*1000))%1000 , (int)b[i], ((int)(b[i]*1000))%1000);
            //return false;
    return true;
}

// Lx = b (solve x)
// for partial matrix, update is whether to update the below part of b (togo_dimension)
void gaussian_elimination(elem_t* L, elem_t* x, elem_t* b, int L_stride, int dimension, bool update, int togo_dimension){
    for(int i = 0; i < dimension; i++){
        elem_t sum = 0;
        for(int j = 0; j <= i; j++){
            if(j < i)
                sum += *(L+(L_stride*i+j)) * (*(x+j));
            else {
                elem_t Lx = *(b+j) - sum;
                //printf("i: %d, Lx x 1000: %d, L x 1000: %d\n", i, (int)(1000*Lx), (int)(1000 * (*(L+(L_stride*i+j)))));
                *(x+j) = Lx / (*(L+(L_stride*i+j)));
            }
        }
    }
    
    if(update && togo_dimension > 0){
       tiled_matmul_auto(1, togo_dimension, dimension, x, L+L_stride*dimension, b+dimension, b+dimension,
               L_stride, L_stride, L_stride, L_stride,
               false, false, false, false,
               -1, 1, 1,
               0, 1, 0, false,
               false, true, // transpose
               false, false,
               3, WS);
    }
    /*
    if(update && togo_dimension > 0){
       tiled_matmul_auto(togo_dimension, 1, dimension, L+L_stride*dimension, x, b+dimension, b+dimension,
               L_stride, 1, 1, 1,
               false, false, false, false,
               1, -1, 1,
               0, 1, 0, false,
               false, false, // transpose
               false, false,
               3, WS);
    }
    */
}

void lower_triangle_inverse(elem_t* A, int stride, int dim, elem_t M[dim][dim]){
	for(int i = 0; i < dim; i++)
		for(int j = 0; j < dim; j++)
			M[i][j] = 0;

	for(int i = 0; i < dim; i++){
		M[i][i] = 1/(*(A+i*stride+i));
		for(int j = i+1; j < dim; j++){
			elem_t sum = 0;
            int jstride = j * stride;
			for(int k = i; k < j; k++)
				sum += (*(A+jstride+k))*M[k][i];///(*(A+j*stride+j));
			M[j][i] = M[j][i] - (sum)/(*(A+jstride+j));
		}
	}
}

void full_transposed_matmul(int I_block, int J_block, int K_block, int block_dim, int A_stride, int B_stride, int C_stride, elem_t* A, elem_t* B, elem_t* C, bool sub, int num_array) {
/*	
	tiled_opcode_matmul_auto_multi(block_dim*I_block, block_dim*J_block, block_dim*K_block,
			A_stride, B_stride, C_stride, C_stride,
            false, false, false, false,
			A, B, sub ? C : NULL, C,
			MVIN_SCALE_IDENTITY, sub ? (-1) :MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            false, true,
			NO_ACTIVATION, ACC_SCALE_IDENTITY, false, false, false, 
			WS,
            num_array, 0);	
*/
    tiled_matmul_auto(block_dim*I_block, block_dim*J_block, block_dim*K_block,
			A, B, sub ? C : NULL, C,
			A_stride, B_stride, C_stride, C_stride,
            false, false, false, false,
			MVIN_SCALE_IDENTITY, sub ? (-1) :MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
			NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, true, false, 0,
			0, WS);	


}
void full_left_chol(elem_t* L, int block_dim, int stride){
	for(int j=0; j < block_dim; j++){
        int jstride = j*stride;
		for(int k=0; k < j; k++){
			*(L+jstride+j) -= (*(L+jstride+k))*(*(L+jstride+k));
			for(int i = j+1; i < block_dim; i++)
				*(L+i*stride+j) -= (*(L+i*stride+k))*(*(L+jstride+k));
		}
		*(L+jstride+j) = (sqrt(*(L+jstride+j)));
		for(int i = j+1; i < block_dim; i++)
			*(L+i*stride+j) = ((*(L+i*stride+j))/(*(L+jstride+j)));
		for(int i = 0; i < j; i++)
			*(L+i*stride+j) = 0;
	}
}
elem_t temp_inv[BLOCK_DIM][BLOCK_DIM] row_align(MAX_BLOCK_LEN) = {0};
void block_left_chol(elem_t* L, int dimension, int stride, int block_size){
	int num_block = dimension / block_size;
    int num_array = 1;
    for(int k = 0; k < num_block; k++){
		//printf("k: %d\n", k);
        if(k > 0) full_transposed_matmul(1, 1, k, block_size, stride, stride, stride, L+block_size*(k*stride), L+block_size*(k*stride), L+block_size*(k*stride+k), true, num_array);	
		//printf("left looking chol \n");
        full_left_chol(L+block_size*(k*stride+k), block_size, stride); 
        //printf("cpu inversion\n");
		lower_triangle_inverse(L+(k*stride+k)*block_size, stride, block_size, temp_inv);

        //printf("transposed matmul\n");
        if(k > 0 && k < num_block - 1) full_transposed_matmul(num_block-k-1, 1, k, block_size, stride, stride, stride, L+block_size*((k+1)*stride), L+block_size*(k*stride), L+block_size*((k+1)*stride+k), true, num_array);

        //printf("transposed matmul\n");
        if (k < num_block - 1) 
          full_transposed_matmul(num_block-k-1, 1, 1, block_size, stride, block_size, stride, L+block_size*((k+1)*stride+k), (elem_t*) temp_inv, L+block_size*((k+1)*stride+k), false, num_array);
	}
}
