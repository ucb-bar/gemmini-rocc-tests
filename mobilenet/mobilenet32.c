#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <sys/mman.h>
#include "include/gemmini.h"
#include "kernels32.h"
#define img_dim1 224
#define img_dim2 224
#define rgb 3

#define verbose(layer_num,old_C,filter,C) printf("layer %d: operand %d %d filter %d %d result %d %d\n", layer_num, LEN(old_C),LEN(old_C[0]),LEN(filter),LEN(filter[0]),LEN(C),LEN(C[0]));

int compute_index(int c,int i, int j, int dim1, int dim2){
    return c*dim1*dim2+dim2*i+j;

}
/*
void reshape(elem_t* img, int channels,int dim1,int dim2,int kdim,elem_t* A){
    int w1,w2,channel,i,j;
    int k = 0;
    int row = 0;
    for (w1=0;w1<dim1;w1++){
        for(w2=0;w2<dim2;w2++){
            k=0;
            for(channel = 0; channel<channels; channel++){
                for(i=-kdim/2;i<=kdim/2;i++){
                    for(j=-kdim/2;j<=kdim/2;j++){
                        if(i+w1<0 || i+w1>=dim1||j+w2<0 ||j+w2>=dim2)
                            A[compute_index(0,row,k,dim1,dim2)]=0;
                        else
                            A[compute_index(0,row,k,dim1,dim2)]=img[compute_index(channel,i+w1,j+w2,dim1,dim2)];
                        k++;
                    }
                }
            }
            row++; 
        }
    }
}
    void zeropad(elem_t* img,int channels,int dim1,int dim2,int kdim1,int kdim2,elem_t* new_img){
        int i,j,c;
        for(c=0;c<channels;c++)
        for(i = 0; i<dim1;i++){
            for(j=0;j<dim2;j++){
                new_img[c][i+kdim1/2][j+kdim2/2]=img[c][i][j];
            }
        }

    }
    */
    void zeropad_array(elem_t array[][32],int dim1,int dim2, elem_t* new_array){
    //    if(dim1%DIM ==0 && dim2%DIM==0){
    //        new_array=array;
    //        return;
    //    }

        int i,j; 
        for(i = 0; i<dim1;i++){
            for(j=0;j<dim2;j++){
                new_array[i*dim2+j]=array[i][j];
            }
        }
    }
    

    void check_dimensions(elem_t* a, elem_t* b){
        printf("a %d b %d ", LEN(a),LEN(b));
    }


void dwconv(int num_imgs, elem_t C[][num_imgs],elem_t old_C[][num_imgs],elem_t filter[][num_imgs],int dim1,int dim2,int kdim,int stride){
    int id,comp_pixel,res_pixel,fidx,w1,w2,i,j;
    for(id=0;id<num_imgs;id++){
        for(w1=0;w1<dim1;w1+=stride){
            for(w2=0;w2<dim2;w2+=stride){
                 for(i=-kdim/2;i<=kdim/2;i++){
                    for(j=-kdim/2;j<=kdim/2;j++){
                        if(i+w1<0 || i+w1>=dim1||j+w2<0 ||j+w2>=dim2){
                            continue;
                        }
                        else{
                            comp_pixel = (w1+i)*dim2+w2+j;
                            res_pixel = (w1*dim2+w2)/stride;
                            fidx = (i+kdim/2)*kdim+j+kdim/2;
                            C[res_pixel][id]+=old_C[comp_pixel][id]*filter[fidx][id];
                        }
                
                    }
                }
    
            }
        }
    }
}

void pool7(int len, elem_t in[][len],elem_t out[][len]){
    int i, j;
    for(i=0;i<len;i++){
        for(j=0;j<7*7;j++){
            out[0][i] += in[j][i];
        }
        out[0][i]=out[0][i]/49;
    }
}

static void tiled_matmul_compare(size_t DIM_I, size_t DIM_J, size_t DIM_K,
        // elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], acc_t D[DIM_I][DIM_J],
        elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], void * D,
        elem_t C[DIM_I][DIM_J],
        int act, int shift, int relu6_shift, int full_bias_width,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool compare, char * layer_name)
{
    if (compare)
        printf("%s: gemmini\n", layer_name);
    tiled_matmul_option(DIM_I, DIM_J, DIM_K,
        A, B, D, C, act, shift, relu6_shift, full_bias_width,
        tiled_matmul_type);

    if (compare) {
        printf("%s: CPU\n", layer_name);
        elem_t gold[DIM_I][DIM_J];
        tiled_matmul_option(DIM_I, DIM_J, DIM_K,
            A, B, D, gold, act, shift, relu6_shift, full_bias_width,
            CPU);

        if (!MAT_IS_EQUAL(DIM_I, DIM_J, C, gold)) {
            printf("Layer calculated incorrectly: %s\n", layer_name);
            exit(1);
        }
    }
}

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    matmul_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type;
    if (argc < 2) {
        // printf("usage: %s matmul_option\n  matmul_option may be 'os', 'ws', or cpu'\n");
        // exit(0);
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    }

    bool compare;
    if (argc < 3) {
        compare = false;
    } else if (strcmp(argv[2], "compare") == 0) {
        compare = true;
    } else {
        printf("Unknown command-line argument\n");
        exit(1);
    }

    unsigned long cycles[29]={0};
    unsigned long start,end;

    ///////// first layer - sys array///////////
    //stride = 2
    elem_t img[rgb][img_dim1][img_dim2] row_align(1) = {0};
    // TODO initialize img with random values
    /*for (size_t i = 0; i < img_dim1; i++) {
        for (size_t j = 0; j < img_dim2; j++) {
            for (size_t k = 0; k < rgb; k++) {
                img[k][i][j] = (rand()%3)-1;
            }
        }
    }*/
    static elem_t A[img_dim1*img_dim2/2/2][32] row_align(1) = {0};//it should be 27 but for zeropad
    int kdim = 3;
    //reshape(img,rgb,img_dim1,img_dim2,kdim,A);// reshape img and store it in A
    int w1,w2,channel,i,j;
    int k = 0;
    int row = 0;
    for (w1=0;w1<img_dim1;w1+=2){
        for(w2=0;w2<img_dim2;w2+=2){
            k=0;
            for(channel = 0; channel<rgb; channel++){
                for(i=-kdim/2;i<=kdim/2;i++){
                    for(j=-kdim/2;j<=kdim/2;j++){
                        if(i+w1<0 || i+w1>=img_dim1||j+w2<0 ||j+w2>=img_dim2)
                            A[row][k]=0;
                        else
                            A[row][k]=img[channel][i+w1][j+w2];
                        k++;
                    }
                }
            }
            row++; 
        }
    }
    /////// this could be replaced to immediately store filters zeropadded
    /*int fdim1 = LEN(filter0);
    int fdim2 = LEN(filter0[0]);
    int new_fdim1 = fdim1-fdim1%DIM+DIM;//32
    int new_fdim2 = fdim2-fdim2%DIM+DIM;//32
    
    elem_t zpf0[32][32]={0}; //new_fdim1*new_fdim2
    int dim1=27;
    int dim2=32;
    for(i = 0; i<dim1;i++){
        for(j=0;j<dim2;j++){
            zpf0[i][j]=filter0[i][j];
        }
    }
    
    // test
    // 
    //for(i =0; i< 32; i++){
    //    for(j=0;j<32;j++){
    //        printf("%d ",zpf0[i][j]);
    //    }
    //    printf("\n");
    //}
    
    
    */
    ////// replace upper part by immediate generation of zeropadded filters ///// 
    
    start = read_cycles();
    static elem_t C0[112*112][32] row_align(1) = {0};
    /* TODO: call systolic array C0 = A*filter0 */
    // I = 112*112, J = 32, K = 32
    tiled_matmul_compare(112*112, 32, 32,    // dimensions
            A, filter0, NULL, C0,       // addresses
            RELU, 0, 0, 0,           // act, shift, r6_shift
            tiled_matmul_type, compare, "layer_1");
    /* end of first layer */
    
    end = read_cycles();
    cycles[0] = end-start;
    start = end;
    // verbose(0,A,filter0,C0) 
    
    /* second layer, depthwise conv, Ameer decided to put it on  on rocket*/
    int num_imgs = 32;
    static elem_t C1[112*112][32] row_align(1) = {0};
    dwconv(num_imgs, C1,C0,filter1,112,112,3,1);
    // verbose(1,C0,filter1,C1) 
    /* end of second layer*/
    
    end = read_cycles();
    cycles[1] = end-start;
    start = end;
    
    /* third layer, directly matmul because it is 1x1 conv, hell yeah!!*/

    static elem_t C2[112*112][64] row_align(1) = {0};
    //TODO: call systolic array C2 = C1*filter2
    // I = 112*112, J = 64, K = 32
    tiled_matmul_compare(112*112, 64, 32,    // dimensions
            C1, filter2, NULL, C2,      // addresses
            RELU, 0, 0, 0,           // act, shift, r6_shift
            tiled_matmul_type, compare, "layer_2");
    // verbose(2,C1,filter2,C2) 
    /* end of third layer */

    end = read_cycles();
    cycles[2] = end-start;
    start = end;

    /* fourth layer, depthwise conv, on rocket*/

    num_imgs = 64;
    static elem_t C3[56*56][64] row_align(1) = {0};
    dwconv(num_imgs, C3,C2,filter3,112,112,3,2);
    // verbose(3,C2,filter3,C3) 
    /* end of fourth layer*/

    end = read_cycles();
    cycles[3] = end-start;
    start = end;
    
    /* fifth layer, directly matmul because it is 1x1 conv, hell yeah!!*/

    static elem_t C4[56*56][128] row_align(1) = {0};
    //TODO: call systolic array C4 = C3*filter4
    // I = 56*56, J = 128, K = 64
    tiled_matmul_compare(56*56, 128, 64,     // dimensions
            C3, filter4, NULL, C4,      // addresses
            RELU, 0, 0, 0,           // act, shift, r6_shift
            tiled_matmul_type, compare, "layer_5");
    // verbose(4,C3,filter4,C4) 
    /* end of fifth layer */
      
    end = read_cycles();
    cycles[4] = end-start;
    start = end;
     
    /* sixth layer, depthwise conv, on rocket*/
    num_imgs = 128;
    static elem_t C5[56*56][128] row_align(1) = {0};
    dwconv(num_imgs, C5,C4,filter5,56,56,3,1);
    // verbose(5,C4,filter5,C5) 
    /* end of sixth layer*/

    end = read_cycles();
    cycles[5] = end-start;
    start = end;

    /* seventh layer, directly matmul because it is 1x1 conv, hell yeah!!*/

    static elem_t C6[56*56][128] row_align(1) = {0};
    //TODO: call systolic array C6 = C5*filter6
    tiled_matmul_compare(56*56, 128, 128,    // dimensions
            C5, filter6, NULL, C6,      // addresses
            RELU, 0, 0, 0,           // act, shift, r6_shift
            tiled_matmul_type, compare, "layer_7");
    // verbose(6,C5,filter6,C6) 
    /* end of seventh layer */


    end = read_cycles();
    cycles[6] = end-start;
    start = end;

    /* 8th layer, depthwise conv, on rocket*/
    num_imgs = 128;
    static elem_t C7[800][128] row_align(1) = {0};
    dwconv(num_imgs, C7,C6,filter7,56,56,3,2);
    // verbose(7,C6,filter7,C7) 
    /* end of 8th layer*/

    end = read_cycles();
    cycles[7] = end-start;
    start = end;

    /* 9th layer, directly matmul because it is 1x1 conv, hell yeah!!*/

    static elem_t C8[800][256] row_align(1) = {0};
    //TODO: call systolic array C8 = C7*filter8
    tiled_matmul_compare(800, 256, 128,    // dimensions
            C7, filter8, NULL, C8,      // addresses
            RELU, 0, 0, 0,           // act, shift, r6_shift
            tiled_matmul_type, compare, "layer_9");
    // verbose(8,C7,filter8,C8) 
    /* end of 9th layer */

    end = read_cycles();
    cycles[8] = end-start;
    start = end;
        
    /* 10th layer, depthwise conv, on rocket*/
    num_imgs = 256;
    static elem_t C9[800][256] row_align(1) = {0};
    dwconv(num_imgs, C9,C8,filter9,28,28,3,1);
    // verbose(9,C8,filter9,C9) 
    /* end of 10th layer*/

    end = read_cycles();
    cycles[9] = end-start;
    start = end;

    /* 11th layer, directly matmul because it is 1x1 conv, hell yeah!!*/

    static elem_t C10[800][256] row_align(1) = {0};
    //TODO: call systolic array C10 = C9*filter10
    tiled_matmul_compare(800, 256, 256,    // dimensions
            C9, filter10, NULL, C10,    // addresses
            RELU, 0, 0, 0,           // act, shift, r6_shift
            tiled_matmul_type, compare, "layer_11");
    // verbose(10,C9,filter10,C10) 
    /* end of 11th layer */
       
    end = read_cycles();
    cycles[10] = end-start;
    start = end;

    /* 12th layer, depthwise conv, on rocket*/
    num_imgs = 256;
    // it should be 14*14 but it doesn't divide by 16
    static elem_t C11[224][256] row_align(1) = {0};
    dwconv(num_imgs, C11,C10,filter11,28,28,3,2);
    // verbose(11,C10,filter11,C11) 
    /* end of 12th layer*/

    end = read_cycles();
    cycles[11] = end-start;
    start = end;

    /* 13th layer, directly matmul because it is 1x1 conv, hell yeah!!*/
    // it should be 14*14 but it doesn't divide by 16
    static elem_t C12[224][512] row_align(1) = {0};
    //TODO: call systolic array C12 = C11*filter12
    tiled_matmul_compare(224, 512, 256,    // dimensions
            C11, filter12, NULL, C12,   // addresses
            RELU, 0, 0, 0,           // act, shift, r6_shift
            tiled_matmul_type, compare, "layer_13");
    // verbose(12,C11,filter12,C12) 
    /* end of 13th layer */
   
    end = read_cycles();
    cycles[12] = end-start;
    start = end;

    /* layers 14-23 that combine 1x1 conv and dw convs */
    //C13 and C12 reused for are savings
    num_imgs = 512;
    
    // it should be 14*14 but it doesn't divide by 16
    static elem_t C13[224][512] row_align(1) = {0};
    //static elem_t C14[14*14][512] row_align(1) = {0};

/* 5 repeated depthwise and 1x1 layers */
    /* dw conv */
    dwconv(num_imgs, C13,C12,filter13,14,14,3,1);
    // verbose(13,C12,filter13,C13);
    end = read_cycles();
    cycles[13] = end-start;
    start = end;

    /* 1x1 conv */
    //TODO: call systolic array C12 = C13*filter14
    tiled_matmul_compare(224, 512, 512,      // dimensions
                C13, filter14, NULL, C12, // addresses
                RELU, 0, 0, 0,         // act, shift, r6_shift
            tiled_matmul_type, compare, "dw_1");
    // verbose(14,C13,filter14,C12) 

    end = read_cycles();
    cycles[14] = end-start;
    start = end;

    /* dw conv */
    dwconv(num_imgs, C13,C12,filter15,14,14,3,1);
    
    end = read_cycles();
    cycles[15] = end-start;
    start = end;
    
    // verbose(15,C12,filter15,C13);
    /* 1x1 conv */
    //TODO: call systolic array C12 = C13*filter16
    tiled_matmul_compare(224, 512, 512,        // dimensions
                C13, filter16, NULL, C12,   // addresses
                RELU, 0, 0, 0,           // act, shift, r6_shift
            tiled_matmul_type, compare, "dw_2");
    // verbose(16,C13,filter16,C12) 

    end = read_cycles();
    cycles[16] = end-start;
    start = end;
        
    /* dw conv */
    dwconv(num_imgs, C13,C12,filter17,14,14,3,1);
    
    end = read_cycles();
    cycles[17] = end-start;
    start = end;
    // verbose(17,C12,filter17,C13);
    /* 1x1 conv */
    //TODO: call systolic array C12 = C13*filter18
    tiled_matmul_compare(224, 512, 512,        // dimensions
                C13, filter18, NULL, C12,   // addresses
                RELU, 0, 0, 0,           // act, shift, r6_shift
            tiled_matmul_type, compare, "dw_3");
    // verbose(18,C13,filter18,C12) 

    end = read_cycles();
    cycles[18] = end-start;
    start = end;
        
    /* dw conv */
    dwconv(num_imgs, C13,C12,filter19,14,14,3,1);
    
    end = read_cycles();
    cycles[19] = end-start;
    start = end;
    // verbose(19,C12,filter19,C13);
    /* 1x1 conv */
    //TODO: call systolic array C12 = C13*filter20
    tiled_matmul_compare(224, 512, 512,        // dimensions
                C13, filter20, NULL, C12,   // addresses
                RELU, 0, 0, 0,           // act, shift, r6_shift
            tiled_matmul_type, compare, "dw_4");
    // verbose(20,C13,filter20,C12) 
    
    end = read_cycles();
    cycles[20] = end-start;
    start = end;

    /* dw conv */
    dwconv(num_imgs, C13,C12,filter21,14,14,3,1);
    // verbose(21,C12,filter21,C13);
    
    end = read_cycles();
    cycles[21] = end-start;
    start = end;

    /* 1x1 conv */
    //TODO: call systolic array C12 = C13*filter22
    tiled_matmul_compare(224, 512, 512,        // dimensions
                C13, filter22, NULL, C12,   // addresses
                RELU, 0, 0, 0,           // act, shift, r6_shift
            tiled_matmul_type, compare, "dw_5");
    // verbose(22,C13,filter22,C12) 
    
    end = read_cycles();
    cycles[22] = end-start;
    start = end;
        
/* end of 5 repeated dw and 1x1 layers*/    

    /* 24th layer, depthwise conv, on rocket*/
    num_imgs = 512;
    //it should 7*7 but replaced with 64 to divide 16
    static elem_t C14[64][512] row_align(1) = {0};
    dwconv(num_imgs, C14,C13,filter23,14,14,3,2);
    // verbose(23,C13,filter23,C14) 
    /* end of 24th layer*/

    end = read_cycles();
    cycles[23] = end-start;
    start = end;

    /* 25th layer, directly matmul because it is 1x1 conv, hell yeah!!*/
    //it should 7*7 but replaced with 64 to divide 16
    static elem_t C15[64][1024] row_align(1) = {0};
    //TODO: call systolic array C15 = C14*filter24
    tiled_matmul_compare(64, 1024, 512,      // dimensions
            C14, filter24, NULL, C15,   // addresses
            RELU, 0, 0, 0,           // act, shift, r6_shift
            tiled_matmul_type, compare, "layer_25");
    // verbose(24,C14,filter24,C15)

    end = read_cycles();
    cycles[24] = end-start;
    start = end;

    /* end of 25th layer */

    /* 26th layer, depthwise conv, on rocket*/
    //it should 7*7 but replaced with 64 to divide 16
    num_imgs = 1024;
    static elem_t C16[64][1024] row_align(1) = {0};
    dwconv(num_imgs, C16,C15,filter25,7,7,3,1);
    // verbose(25,C15,filter25,C16)
    /* end of 26th layer*/

    end = read_cycles();
    cycles[25] = end-start;
    start = end;

    /* 27th layer, directly matmul because it is 1x1 conv, hell yeah!!*/
    //it should 7*7 but replaced with 64 to divide 16

    static elem_t C17[64][1024] row_align(1) = {0};
    //TODO: call systolic array C17 = C16*filter26
    tiled_matmul_compare(64, 1024, 1024,     // dimensions
            C16, filter26, NULL, C17,   // addresses
            RELU, 0, 0, 0,          // act, shift, r6_shift
            tiled_matmul_type, compare, "layer_27");
    // verbose(26,C16,filter26,C17) 
    /* end of 27th layer */
    
    end = read_cycles();
    cycles[26] = end-start;
    start = end;
    
    /* 28th layer, pooling, on rocket, can be done on the array with fixed point divison*/
    //elem_t C18[1*1][1024] row_align(1) = {0}; this is replaced with 16 for zeropadding for the FC...
    static elem_t C18[32][1024] row_align(1) = {0};
    pool7(1024,C17,C18);
    //elem_t garbage[7][7] row_align(1) = {0};
    // verbose(27,C17,garbage,C18) 
    /*end of 28th layer */    

    end = read_cycles();
    cycles[27] = end-start;
    start = end;
    
    /* 29th and last layer, FC, originally its 1024x1000, zeropadded to 1024x1024 */
    static elem_t C19[32][1024] row_align(1) = {0};
    //TODO: call systolic array C19 = C18*fc27
    tiled_matmul_compare(32, 1024, 1024,    // dimensions
            C18, fc27, NULL, C19,      // addresses
            RELU, 0, 0, 0,          // act, shift, r6_shift
            tiled_matmul_type, compare, "layer_29");
    // verbose(28,C18,fc27,C19) 

    end = read_cycles();
    cycles[28] = end-start;
    
    /* end of 29th layer */
    
    unsigned long overall_cycles = 0;
    for(int cyc = 0; cyc < 29 ; cyc++){
        overall_cycles += cycles[cyc];
        printf("Cycles taken in layer %d: %lu\n", cyc,cycles[cyc]);
    }
    printf("Overall cycles taken: %lu\n",overall_cycles);

    /*for (int i = 0; i < LEN(C19); i++) {
        for (int j = 0; j < LEN(C19[0]); j++) {
            printf ("%d,", C19[i][j]);
        }
        printf("\n");
    }*/

    return 0;
}


