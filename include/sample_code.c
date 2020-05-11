


//compute
for (int b = 0; b < batches; b++)
    for(int orow = 0; orow < orows; orow++) //orows: tile size of output matrix P dimension
        for(int ocol = 0; ocol < ocols; ocol += DIM) //ocols: tile size of output matrix Q dimension
            int I = ocols - ocol > DIM ? DIM : ocols - ocol;
            for(int och = 0; och < ochs; och += DIM) //ochs: tile size of output channel dimension
                int J = ochs - och > DIM ? DIM : ochs - och;
                int C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols 
				+ orow * ocols + ocol; //output scratchpad address
                for(int krow = 0; krow < krows; krow++) //krows: tile size of kernel R dimension
                    int irow = orow * stride + krow;
                    for(int kcol = 0; kcol < kcols; kcol++) //kcols: tile size of kernel S dimension
		        int icol = ocol * stride + kcol;
                        for(int kch = 0; kch < kchs; kch += DIM) //kchs: tile size of kernel channel dimension
                            int K = kchs - kch > DIM ? DIM : kchs - kch;
			    //input activation scratchpad address
                            const A_sp_addr = A_sp_addr_start + (kch / DIM) * batches * irows * icols 
					    + b * irows * icols + irow * icols + icol;                            
			    //kernel scratchpad address
			    const B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs 
					    + krow * kcols * kchs + kcol * kchs + kch;


// Compute
int odims = orows * ocols;
int kdims = krows * kcols; //merge row and col dimension of output and kernel
for (int b = 0; b < batches; b++)
    for (int och = 0; och < ochs; och += DIM)
        int J = ochs - och > DIM ? DIM : ochs - och;
 	for (int kch = 0; kch < kchs; kch += DIM)
	    int K = kchs - kch > DIM ? DIM : kchs - kch;
	    const A_sp_addr = A_sp_addr_start + (kch / DIM)*batches*irows*icols + b*irows*icols;           
                for(int odim = 0; odim < odims; odim += DIM)//iterate merged dimension of output at once
		    int I = odims - odim > DIM ? DIM : odims - odim;
     	       	    const C_sp_addr = C_sp_addr_start + (och / DIM) * batches * odims + b * odims + odim;
			//iterate CRS dimension of flattened weight tile at once
			for(int kkdim = 0; kkdim < K*kdims; kkdim += DIM)
			    int kk = K*kdims - kkdim > DIM ? DIM : K*kdims - kkdim;
			    const B_sp_addr = B_sp_addr_start + (och / DIM) * kdims * kchs + kch*kdims + kkdim;


for (int och = 0; och < ochs; och += DIM)
    int J = ochs - och > DIM ? DIM : ochs - och;
    for (int krow = 0; krow < krows; krow++)
        for (int kcol = 0; kcol < kcols; kcol++)
            for (int kch = 0; kch < kchs; kch += DIM)
                int K = kchs - kch > DIM ? DIM : kchs - kch;
		//computation of scratchpad address for weight to move-in
		const B_sp_addr = B_sp_addr_start + (och/DIM)*krows*kcols*kchs + krow*kcols*kchs + kcol*kchs + kch;
                gemmini_extended_mvin(weights + ochs*(krow*kcols*kchs + kcol*kchs + kch) + och,
                        	      B_sp_addr, J, K);
               




for (int och = 0; och < ochs; och += DIM) {
     int J = ochs - och > DIM ? DIM : ochs - och;
	for (int kch = 0; kch < kchs; kch += DIM) {
             int K = kchs - kch > DIM ? DIM : kchs - kch;
	    for(int kkdim = 0; kkdim < K*kdims; kkdim += DIM){
		int KK = K*kdims - kkdim > DIM ? DIM : K*kdims - kkdim;
		const B_sp_addr = B_sp_addr_start + (och/DIM)*kdims*kchs + kch*kdims + kkdim;
		gemmini_extended_mvin(weights + ochs*(kch*kdims + kkdim) + och, B_sp_addr, J, KK);



