

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <sys/mman.h>
#include "include/gemmini.h"

#define verbose(layer_num,old_C,filter,C) printf("layer %d: operand %d %d filter %d %d result %d %d\n", layer_num, LEN(old_C),LEN(old_C[0]),LEN(filter),LEN(filter[0]),LEN(C),LEN(C[0]));
#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))
#define N 224


static void tensor_reshape(int channels, elem_t in_tensor[][channels],int dim1,int dim2,int kdim, int stride, elem_t out_tensor[][channels]){
    int w1,w2,channel,i,j;
    int k = 0;
    int row = 0;
    for (w1=0;w1<dim1;w1+=stride){
        for(w2=0;w2<dim2;w2+=stride){
            k=0;
            for(channel = 0; channel<channels; channel++){
                for(i=-kdim/2;i<=kdim/2;i++){
                    for(j=-kdim/2;j<=kdim/2;j++){
                        if(i+w1<0 || i+w1>=dim1||j+w2<0 ||j+w2>=dim2)
                            out_tensor[row][k]=0;
                        else
                            out_tensor[row][k]=in_tensor[dim2*(i+w1)+j+w2][channel];
                        k++;
                    }
                }
            }
            row++; 
        }
    }
}    
static void avg_pool7(int len, elem_t in[][len],elem_t out[][len]){
    int i, j;
    for(i=0;i<len;i++){
        for(j=0;j<7*7;j++){
            out[0][i] += in[j][i];
        }
        out[0][i]=out[0][i]/49;
    }
}

static void rocket_fix_strided_dimension(int img_dim, int len2, elem_t in[][len2], int len3, elem_t out[][len3]){
        
    for(int i =0;i<img_dim;i+=2){
        for(int k = 0; k<img_dim;k+=2)
            for(int j=0;j<len2; j+=1){
                out[(i/2)*img_dim+k/2][j]=in[i*img_dim+k][j];

            }
    }
}

static void rocket_zeropad(int dim1,int dim2, elem_t in[][dim2], int dim3, elem_t out[][dim3]){
    for(int i = 0; i<dim1;i++){
        for(int j =0; j<dim2;j++){
            out[i][j] = in[i][j];
        }
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


    static elem_t img[N*N][3] = {0};
    unsigned long cycles[52]={0};
    unsigned long start,end;
    start = read_cycles();
    static elem_t kernel0[192][64] row_align(1)= {0};
    static elem_t tensor0[12544][192] row_align(1)= {0};
    static elem_t result0[12544][64] row_align(1)= {0};
    tensor_reshape(3,img,224, 224, 7, 2, tensor0);


    /* matmul number: 0 */

    tiled_matmul_compare(12544, 64, 192,    // dimensions
    tensor0, kernel0, NULL, result0,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_0");
    // verbose(0,tensor0,kernel0,result0)
    /* end of matmul number: 0 */

    end = read_cycles();
    cycles[0] = end-start;
    start = end;
    static elem_t kernel1[576][64] row_align(1)= {0};
    static elem_t tensor1[3136][576] row_align(1)= {0};
    static elem_t result1[3136][64] row_align(1)= {0};
    tensor_reshape(64,result0,112, 112, 3, 2, tensor1);


    /* matmul number: 1 */

    tiled_matmul_compare(3136, 64, 576,    // dimensions
    tensor1, kernel1, NULL, result1,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_1");
    // verbose(1,tensor1,kernel1,result1)
    /* end of matmul number: 1 */

    end = read_cycles();
    cycles[1] = end-start;
    start = end;
    static elem_t kernel2[64][64] row_align(1)= {0};
    static elem_t tensor2[3136][64] row_align(1)= {0};
    static elem_t result2[3136][64] row_align(1)= {0};


    /* matmul number: 2 */

    tiled_matmul_compare(3136, 64, 64,    // dimensions
    tensor2, kernel2, NULL, result2,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_2");
    // verbose(2,tensor2,kernel2,result2)
    /* end of matmul number: 2 */

    end = read_cycles();
    cycles[2] = end-start;
    start = end;
    static elem_t kernel3[576][64] row_align(1)= {0};
    static elem_t tensor3[3136][576] row_align(1)= {0};
    static elem_t result3[3136][64] row_align(1)= {0};
    tensor_reshape(64,result2,56, 56, 3, 1, tensor3);


    /* matmul number: 3 */

    tiled_matmul_compare(3136, 64, 576,    // dimensions
    tensor3, kernel3, NULL, result3,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_3");
    // verbose(3,tensor3,kernel3,result3)
    /* end of matmul number: 3 */

    end = read_cycles();
    cycles[3] = end-start;
    start = end;
    static elem_t kernel4[64][256] row_align(1)= {0};
    static elem_t tensor4[3136][64] row_align(1)= {0};
    static elem_t result4[3136][256] row_align(1)= {0};
    static elem_t inter_result4[3136][256] row_align(1)= {0};
    rocket_zeropad(3136,64,result2,256,  inter_result4);


    /* matmul number: 4 */

    tiled_matmul_compare(3136, 256, 64,    // dimensions
    tensor4, kernel4, inter_result4, result4,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_4");
    // verbose(4,tensor4,kernel4,result4)
    /* end of matmul number: 4 */

    end = read_cycles();
    cycles[4] = end-start;
    start = end;
    static elem_t kernel5[64][64] row_align(1)= {0};
    static elem_t tensor5[3136][64] row_align(1)= {0};
    static elem_t result5[3136][64] row_align(1)= {0};


    /* matmul number: 5 */

    tiled_matmul_compare(3136, 64, 64,    // dimensions
    tensor5, kernel5, NULL, result5,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_5");
    // verbose(5,tensor5,kernel5,result5)
    /* end of matmul number: 5 */

    end = read_cycles();
    cycles[5] = end-start;
    start = end;
    static elem_t kernel6[576][64] row_align(1)= {0};
    static elem_t tensor6[3136][576] row_align(1)= {0};
    static elem_t result6[3136][64] row_align(1)= {0};
    tensor_reshape(64,result5,56, 56, 3, 1, tensor6);


    /* matmul number: 6 */

    tiled_matmul_compare(3136, 64, 576,    // dimensions
    tensor6, kernel6, NULL, result6,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_6");
    // verbose(6,tensor6,kernel6,result6)
    /* end of matmul number: 6 */

    end = read_cycles();
    cycles[6] = end-start;
    start = end;
    static elem_t kernel7[64][256] row_align(1)= {0};
    static elem_t tensor7[3136][64] row_align(1)= {0};
    static elem_t result7[3136][256] row_align(1)= {0};
    static elem_t inter_result7[3136][256] row_align(1)= {0};
    rocket_zeropad(3136,64,result5,256,  inter_result7);


    /* matmul number: 7 */

    tiled_matmul_compare(3136, 256, 64,    // dimensions
    tensor7, kernel7, inter_result7, result7,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_7");
    // verbose(7,tensor7,kernel7,result7)
    /* end of matmul number: 7 */

    end = read_cycles();
    cycles[7] = end-start;
    start = end;
    static elem_t kernel8[64][64] row_align(1)= {0};
    static elem_t tensor8[3136][64] row_align(1)= {0};
    static elem_t result8[3136][64] row_align(1)= {0};


    /* matmul number: 8 */

    tiled_matmul_compare(3136, 64, 64,    // dimensions
    tensor8, kernel8, NULL, result8,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_8");
    // verbose(8,tensor8,kernel8,result8)
    /* end of matmul number: 8 */

    end = read_cycles();
    cycles[8] = end-start;
    start = end;
    static elem_t kernel9[576][64] row_align(1)= {0};
    static elem_t tensor9[3136][576] row_align(1)= {0};
    static elem_t result9[3136][64] row_align(1)= {0};
    tensor_reshape(64,result8,56, 56, 3, 1, tensor9);


    /* matmul number: 9 */

    tiled_matmul_compare(3136, 64, 576,    // dimensions
    tensor9, kernel9, NULL, result9,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_9");
    // verbose(9,tensor9,kernel9,result9)
    /* end of matmul number: 9 */

    end = read_cycles();
    cycles[9] = end-start;
    start = end;
    static elem_t kernel10[64][256] row_align(1)= {0};
    static elem_t tensor10[3136][64] row_align(1)= {0};
    static elem_t result10[3136][256] row_align(1)= {0};
    static elem_t inter_result10[3136][256] row_align(1)= {0};
    rocket_zeropad(3136,64,result8,256,  inter_result10);


    /* matmul number: 10 */

    tiled_matmul_compare(3136, 256, 64,    // dimensions
    tensor10, kernel10, inter_result10, result10,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_10");
    // verbose(10,tensor10,kernel10,result10)
    /* end of matmul number: 10 */

    end = read_cycles();
    cycles[10] = end-start;
    start = end;
    static elem_t kernel11[256][128] row_align(1)= {0};
    static elem_t tensor11[3136][256] row_align(1)= {0};
    static elem_t result11[3136][128] row_align(1)= {0};


    /* matmul number: 11 */

    tiled_matmul_compare(3136, 128, 256,    // dimensions
    tensor11, kernel11, NULL, result11,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_11");
    // verbose(11,tensor11,kernel11,result11)
    /* end of matmul number: 11 */

    end = read_cycles();
    cycles[11] = end-start;
    start = end;
    static elem_t kernel12[1152][128] row_align(1)= {0};
    static elem_t tensor12[832][1152] row_align(1)= {0};
    static elem_t result12[832][128] row_align(1)= {0};
    tensor_reshape(128,result11,56, 56, 3, 2, tensor12);


    /* matmul number: 12 */

    tiled_matmul_compare(832, 128, 1152,    // dimensions
    tensor12, kernel12, NULL, result12,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_12");
    // verbose(12,tensor12,kernel12,result12)
    /* end of matmul number: 12 */

    end = read_cycles();
    cycles[12] = end-start;
    start = end;
    static elem_t kernel13[128][512] row_align(1)= {0};
    static elem_t tensor13[832][128] row_align(1)= {0};
    static elem_t result13[832][512] row_align(1)= {0};
    static elem_t inter_result13[832][512] row_align(1)= {0};
    rocket_fix_strided_dimension(56,128,result11,512,  inter_result13);


    /* matmul number: 13 */

    tiled_matmul_compare(832, 512, 128,    // dimensions
    tensor13, kernel13, inter_result13, result13,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_13");
    // verbose(13,tensor13,kernel13,result13)
    /* end of matmul number: 13 */

    end = read_cycles();
    cycles[13] = end-start;
    start = end;
    static elem_t kernel14[256][128] row_align(1)= {0};
    static elem_t tensor14[832][256] row_align(1)= {0};
    static elem_t result14[832][128] row_align(1)= {0};


    /* matmul number: 14 */

    tiled_matmul_compare(832, 128, 256,    // dimensions
    tensor14, kernel14, NULL, result14,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_14");
    // verbose(14,tensor14,kernel14,result14)
    /* end of matmul number: 14 */

    end = read_cycles();
    cycles[14] = end-start;
    start = end;
    static elem_t kernel15[1152][128] row_align(1)= {0};
    static elem_t tensor15[832][1152] row_align(1)= {0};
    static elem_t result15[832][128] row_align(1)= {0};
    tensor_reshape(128,result14,28, 28, 3, 1, tensor15);


    /* matmul number: 15 */

    tiled_matmul_compare(832, 128, 1152,    // dimensions
    tensor15, kernel15, NULL, result15,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_15");
    // verbose(15,tensor15,kernel15,result15)
    /* end of matmul number: 15 */

    end = read_cycles();
    cycles[15] = end-start;
    start = end;
    static elem_t kernel16[128][512] row_align(1)= {0};
    static elem_t tensor16[832][128] row_align(1)= {0};
    static elem_t result16[832][512] row_align(1)= {0};
    static elem_t inter_result16[832][512] row_align(1)= {0};
    rocket_zeropad(832,128,result14,512,  inter_result16);


    /* matmul number: 16 */

    tiled_matmul_compare(832, 512, 128,    // dimensions
    tensor16, kernel16, inter_result16, result16,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_16");
    // verbose(16,tensor16,kernel16,result16)
    /* end of matmul number: 16 */

    end = read_cycles();
    cycles[16] = end-start;
    start = end;
    static elem_t kernel17[256][128] row_align(1)= {0};
    static elem_t tensor17[832][256] row_align(1)= {0};
    static elem_t result17[832][128] row_align(1)= {0};


    /* matmul number: 17 */

    tiled_matmul_compare(832, 128, 256,    // dimensions
    tensor17, kernel17, NULL, result17,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_17");
    // verbose(17,tensor17,kernel17,result17)
    /* end of matmul number: 17 */

    end = read_cycles();
    cycles[17] = end-start;
    start = end;
    static elem_t kernel18[1152][128] row_align(1)= {0};
    static elem_t tensor18[832][1152] row_align(1)= {0};
    static elem_t result18[832][128] row_align(1)= {0};
    tensor_reshape(128,result17,28, 28, 3, 1, tensor18);


    /* matmul number: 18 */

    tiled_matmul_compare(832, 128, 1152,    // dimensions
    tensor18, kernel18, NULL, result18,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_18");
    // verbose(18,tensor18,kernel18,result18)
    /* end of matmul number: 18 */

    end = read_cycles();
    cycles[18] = end-start;
    start = end;
    static elem_t kernel19[128][512] row_align(1)= {0};
    static elem_t tensor19[832][128] row_align(1)= {0};
    static elem_t result19[832][512] row_align(1)= {0};
    static elem_t inter_result19[832][512] row_align(1)= {0};
    rocket_zeropad(832,128,result17,512,  inter_result19);


    /* matmul number: 19 */

    tiled_matmul_compare(832, 512, 128,    // dimensions
    tensor19, kernel19, inter_result19, result19,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_19");
    // verbose(19,tensor19,kernel19,result19)
    /* end of matmul number: 19 */

    end = read_cycles();
    cycles[19] = end-start;
    start = end;
    static elem_t kernel20[256][128] row_align(1)= {0};
    static elem_t tensor20[832][256] row_align(1)= {0};
    static elem_t result20[832][128] row_align(1)= {0};


    /* matmul number: 20 */

    tiled_matmul_compare(832, 128, 256,    // dimensions
    tensor20, kernel20, NULL, result20,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_20");
    // verbose(20,tensor20,kernel20,result20)
    /* end of matmul number: 20 */

    end = read_cycles();
    cycles[20] = end-start;
    start = end;
    static elem_t kernel21[1152][128] row_align(1)= {0};
    static elem_t tensor21[832][1152] row_align(1)= {0};
    static elem_t result21[832][128] row_align(1)= {0};
    tensor_reshape(128,result20,28, 28, 3, 1, tensor21);


    /* matmul number: 21 */

    tiled_matmul_compare(832, 128, 1152,    // dimensions
    tensor21, kernel21, NULL, result21,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_21");
    // verbose(21,tensor21,kernel21,result21)
    /* end of matmul number: 21 */

    end = read_cycles();
    cycles[21] = end-start;
    start = end;
    static elem_t kernel22[128][512] row_align(1)= {0};
    static elem_t tensor22[832][128] row_align(1)= {0};
    static elem_t result22[832][512] row_align(1)= {0};
    static elem_t inter_result22[832][512] row_align(1)= {0};
    rocket_zeropad(832,128,result20,512,  inter_result22);


    /* matmul number: 22 */

    tiled_matmul_compare(832, 512, 128,    // dimensions
    tensor22, kernel22, inter_result22, result22,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_22");
    // verbose(22,tensor22,kernel22,result22)
    /* end of matmul number: 22 */

    end = read_cycles();
    cycles[22] = end-start;
    start = end;
    static elem_t kernel23[512][256] row_align(1)= {0};
    static elem_t tensor23[832][512] row_align(1)= {0};
    static elem_t result23[832][256] row_align(1)= {0};


    /* matmul number: 23 */

    tiled_matmul_compare(832, 256, 512,    // dimensions
    tensor23, kernel23, NULL, result23,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_23");
    // verbose(23,tensor23,kernel23,result23)
    /* end of matmul number: 23 */

    end = read_cycles();
    cycles[23] = end-start;
    start = end;
    static elem_t kernel24[2304][256] row_align(1)= {0};
    static elem_t tensor24[256][2304] row_align(1)= {0};
    static elem_t result24[256][256] row_align(1)= {0};
    tensor_reshape(256,result23,28, 28, 3, 2, tensor24);


    /* matmul number: 24 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor24, kernel24, NULL, result24,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_24");
    // verbose(24,tensor24,kernel24,result24)
    /* end of matmul number: 24 */

    end = read_cycles();
    cycles[24] = end-start;
    start = end;
    static elem_t kernel25[256][1024] row_align(1)= {0};
    static elem_t tensor25[256][256] row_align(1)= {0};
    static elem_t result25[256][1024] row_align(1)= {0};
    static elem_t inter_result25[256][1024] row_align(1)= {0};
    rocket_fix_strided_dimension(28,256,result23,1024,  inter_result25);


    /* matmul number: 25 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor25, kernel25, inter_result25, result25,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_25");
    // verbose(25,tensor25,kernel25,result25)
    /* end of matmul number: 25 */

    end = read_cycles();
    cycles[25] = end-start;
    start = end;
    static elem_t kernel26[512][256] row_align(1)= {0};
    static elem_t tensor26[256][512] row_align(1)= {0};
    static elem_t result26[256][256] row_align(1)= {0};


    /* matmul number: 26 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor26, kernel26, NULL, result26,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_26");
    // verbose(26,tensor26,kernel26,result26)
    /* end of matmul number: 26 */

    end = read_cycles();
    cycles[26] = end-start;
    start = end;
    static elem_t kernel27[2304][256] row_align(1)= {0};
    static elem_t tensor27[256][2304] row_align(1)= {0};
    static elem_t result27[256][256] row_align(1)= {0};
    tensor_reshape(256,result26,14, 14, 3, 1, tensor27);


    /* matmul number: 27 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor27, kernel27, NULL, result27,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_27");
    // verbose(27,tensor27,kernel27,result27)
    /* end of matmul number: 27 */

    end = read_cycles();
    cycles[27] = end-start;
    start = end;
    static elem_t kernel28[256][1024] row_align(1)= {0};
    static elem_t tensor28[256][256] row_align(1)= {0};
    static elem_t result28[256][1024] row_align(1)= {0};
    static elem_t inter_result28[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result26,1024,  inter_result28);


    /* matmul number: 28 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor28, kernel28, inter_result28, result28,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_28");
    // verbose(28,tensor28,kernel28,result28)
    /* end of matmul number: 28 */

    end = read_cycles();
    cycles[28] = end-start;
    start = end;
    static elem_t kernel29[512][256] row_align(1)= {0};
    static elem_t tensor29[256][512] row_align(1)= {0};
    static elem_t result29[256][256] row_align(1)= {0};


    /* matmul number: 29 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor29, kernel29, NULL, result29,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_29");
    // verbose(29,tensor29,kernel29,result29)
    /* end of matmul number: 29 */

    end = read_cycles();
    cycles[29] = end-start;
    start = end;
    static elem_t kernel30[2304][256] row_align(1)= {0};
    static elem_t tensor30[256][2304] row_align(1)= {0};
    static elem_t result30[256][256] row_align(1)= {0};
    tensor_reshape(256,result29,14, 14, 3, 1, tensor30);


    /* matmul number: 30 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor30, kernel30, NULL, result30,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_30");
    // verbose(30,tensor30,kernel30,result30)
    /* end of matmul number: 30 */

    end = read_cycles();
    cycles[30] = end-start;
    start = end;
    static elem_t kernel31[256][1024] row_align(1)= {0};
    static elem_t tensor31[256][256] row_align(1)= {0};
    static elem_t result31[256][1024] row_align(1)= {0};
    static elem_t inter_result31[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result29,1024,  inter_result31);


    /* matmul number: 31 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor31, kernel31, inter_result31, result31,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_31");
    // verbose(31,tensor31,kernel31,result31)
    /* end of matmul number: 31 */

    end = read_cycles();
    cycles[31] = end-start;
    start = end;
    static elem_t kernel32[512][256] row_align(1)= {0};
    static elem_t tensor32[256][512] row_align(1)= {0};
    static elem_t result32[256][256] row_align(1)= {0};


    /* matmul number: 32 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor32, kernel32, NULL, result32,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_32");
    // verbose(32,tensor32,kernel32,result32)
    /* end of matmul number: 32 */

    end = read_cycles();
    cycles[32] = end-start;
    start = end;
    static elem_t kernel33[2304][256] row_align(1)= {0};
    static elem_t tensor33[256][2304] row_align(1)= {0};
    static elem_t result33[256][256] row_align(1)= {0};
    tensor_reshape(256,result32,14, 14, 3, 1, tensor33);


    /* matmul number: 33 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor33, kernel33, NULL, result33,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_33");
    // verbose(33,tensor33,kernel33,result33)
    /* end of matmul number: 33 */

    end = read_cycles();
    cycles[33] = end-start;
    start = end;
    static elem_t kernel34[256][1024] row_align(1)= {0};
    static elem_t tensor34[256][256] row_align(1)= {0};
    static elem_t result34[256][1024] row_align(1)= {0};
    static elem_t inter_result34[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result32,1024,  inter_result34);


    /* matmul number: 34 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor34, kernel34, inter_result34, result34,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_34");
    // verbose(34,tensor34,kernel34,result34)
    /* end of matmul number: 34 */

    end = read_cycles();
    cycles[34] = end-start;
    start = end;
    static elem_t kernel35[512][256] row_align(1)= {0};
    static elem_t tensor35[256][512] row_align(1)= {0};
    static elem_t result35[256][256] row_align(1)= {0};


    /* matmul number: 35 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor35, kernel35, NULL, result35,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_35");
    // verbose(35,tensor35,kernel35,result35)
    /* end of matmul number: 35 */

    end = read_cycles();
    cycles[35] = end-start;
    start = end;
    static elem_t kernel36[2304][256] row_align(1)= {0};
    static elem_t tensor36[256][2304] row_align(1)= {0};
    static elem_t result36[256][256] row_align(1)= {0};
    tensor_reshape(256,result35,14, 14, 3, 1, tensor36);


    /* matmul number: 36 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor36, kernel36, NULL, result36,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_36");
    // verbose(36,tensor36,kernel36,result36)
    /* end of matmul number: 36 */

    end = read_cycles();
    cycles[36] = end-start;
    start = end;
    static elem_t kernel37[256][1024] row_align(1)= {0};
    static elem_t tensor37[256][256] row_align(1)= {0};
    static elem_t result37[256][1024] row_align(1)= {0};
    static elem_t inter_result37[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result35,1024,  inter_result37);


    /* matmul number: 37 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor37, kernel37, inter_result37, result37,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_37");
    // verbose(37,tensor37,kernel37,result37)
    /* end of matmul number: 37 */

    end = read_cycles();
    cycles[37] = end-start;
    start = end;
    static elem_t kernel38[512][256] row_align(1)= {0};
    static elem_t tensor38[256][512] row_align(1)= {0};
    static elem_t result38[256][256] row_align(1)= {0};


    /* matmul number: 38 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor38, kernel38, NULL, result38,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_38");
    // verbose(38,tensor38,kernel38,result38)
    /* end of matmul number: 38 */

    end = read_cycles();
    cycles[38] = end-start;
    start = end;
    static elem_t kernel39[2304][256] row_align(1)= {0};
    static elem_t tensor39[256][2304] row_align(1)= {0};
    static elem_t result39[256][256] row_align(1)= {0};
    tensor_reshape(256,result38,14, 14, 3, 1, tensor39);


    /* matmul number: 39 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor39, kernel39, NULL, result39,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_39");
    // verbose(39,tensor39,kernel39,result39)
    /* end of matmul number: 39 */

    end = read_cycles();
    cycles[39] = end-start;
    start = end;
    static elem_t kernel40[256][1024] row_align(1)= {0};
    static elem_t tensor40[256][256] row_align(1)= {0};
    static elem_t result40[256][1024] row_align(1)= {0};
    static elem_t inter_result40[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result38,1024,  inter_result40);


    /* matmul number: 40 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor40, kernel40, inter_result40, result40,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_40");
    // verbose(40,tensor40,kernel40,result40)
    /* end of matmul number: 40 */

    end = read_cycles();
    cycles[40] = end-start;
    start = end;
    static elem_t kernel41[1024][512] row_align(1)= {0};
    static elem_t tensor41[256][1024] row_align(1)= {0};
    static elem_t result41[256][512] row_align(1)= {0};


    /* matmul number: 41 */

    tiled_matmul_compare(256, 512, 1024,    // dimensions
    tensor41, kernel41, NULL, result41,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_41");
    // verbose(41,tensor41,kernel41,result41)
    /* end of matmul number: 41 */

    end = read_cycles();
    cycles[41] = end-start;
    start = end;
    static elem_t kernel42[4608][512] row_align(1)= {0};
    static elem_t tensor42[64][4608] row_align(1)= {0};
    static elem_t result42[64][512] row_align(1)= {0};
    tensor_reshape(512,result41,14, 14, 3, 2, tensor42);


    /* matmul number: 42 */

    tiled_matmul_compare(64, 512, 4608,    // dimensions
    tensor42, kernel42, NULL, result42,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_42");
    // verbose(42,tensor42,kernel42,result42)
    /* end of matmul number: 42 */

    end = read_cycles();
    cycles[42] = end-start;
    start = end;
    static elem_t kernel43[512][2048] row_align(1)= {0};
    static elem_t tensor43[64][512] row_align(1)= {0};
    static elem_t result43[64][2048] row_align(1)= {0};
    static elem_t inter_result43[64][2048] row_align(1)= {0};
    rocket_fix_strided_dimension(16,512,result41,2048,  inter_result43);


    /* matmul number: 43 */

    tiled_matmul_compare(64, 2048, 512,    // dimensions
    tensor43, kernel43, inter_result43, result43,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_43");
    // verbose(43,tensor43,kernel43,result43)
    /* end of matmul number: 43 */

    end = read_cycles();
    cycles[43] = end-start;
    start = end;
    static elem_t kernel44[1024][512] row_align(1)= {0};
    static elem_t tensor44[64][1024] row_align(1)= {0};
    static elem_t result44[64][512] row_align(1)= {0};


    /* matmul number: 44 */

    tiled_matmul_compare(64, 512, 1024,    // dimensions
    tensor44, kernel44, NULL, result44,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_44");
    // verbose(44,tensor44,kernel44,result44)
    /* end of matmul number: 44 */

    end = read_cycles();
    cycles[44] = end-start;
    start = end;
    static elem_t kernel45[4608][512] row_align(1)= {0};
    static elem_t tensor45[64][4608] row_align(1)= {0};
    static elem_t result45[64][512] row_align(1)= {0};
    tensor_reshape(512,result44,7, 7, 3, 1, tensor45);


    /* matmul number: 45 */

    tiled_matmul_compare(64, 512, 4608,    // dimensions
    tensor45, kernel45, NULL, result45,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_45");
    // verbose(45,tensor45,kernel45,result45)
    /* end of matmul number: 45 */

    end = read_cycles();
    cycles[45] = end-start;
    start = end;
    static elem_t kernel46[512][2048] row_align(1)= {0};
    static elem_t tensor46[64][512] row_align(1)= {0};
    static elem_t result46[64][2048] row_align(1)= {0};
    static elem_t inter_result46[64][2048] row_align(1)= {0};
    rocket_zeropad(64,512,result44,2048,  inter_result46);


    /* matmul number: 46 */

    tiled_matmul_compare(64, 2048, 512,    // dimensions
    tensor46, kernel46, inter_result46, result46,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_46");
    // verbose(46,tensor46,kernel46,result46)
    /* end of matmul number: 46 */

    end = read_cycles();
    cycles[46] = end-start;
    start = end;
    static elem_t kernel47[1024][512] row_align(1)= {0};
    static elem_t tensor47[64][1024] row_align(1)= {0};
    static elem_t result47[64][512] row_align(1)= {0};


    /* matmul number: 47 */

    tiled_matmul_compare(64, 512, 1024,    // dimensions
    tensor47, kernel47, NULL, result47,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_47");
    // verbose(47,tensor47,kernel47,result47)
    /* end of matmul number: 47 */

    end = read_cycles();
    cycles[47] = end-start;
    start = end;
    static elem_t kernel48[4608][512] row_align(1)= {0};
    static elem_t tensor48[64][4608] row_align(1)= {0};
    static elem_t result48[64][512] row_align(1)= {0};
    tensor_reshape(512,result47,7, 7, 3, 1, tensor48);


    /* matmul number: 48 */

    tiled_matmul_compare(64, 512, 4608,    // dimensions
    tensor48, kernel48, NULL, result48,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_48");
    // verbose(48,tensor48,kernel48,result48)
    /* end of matmul number: 48 */

    end = read_cycles();
    cycles[48] = end-start;
    start = end;
    static elem_t kernel49[512][2048] row_align(1)= {0};
    static elem_t tensor49[64][512] row_align(1)= {0};
    static elem_t result49[64][2048] row_align(1)= {0};
    static elem_t inter_result49[64][2048] row_align(1)= {0};
    rocket_zeropad(64,512,result47,2048,  inter_result49);


    /* matmul number: 49 */

    tiled_matmul_compare(64, 2048, 512,    // dimensions
    tensor49, kernel49, inter_result49, result49,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_49");
    // verbose(49,tensor49,kernel49,result49)
    /* end of matmul number: 49 */

    end = read_cycles();
    cycles[49] = end-start;
    start = end;


    /* AVG Pool: 50 */

    static elem_t tensor51[64][2048] row_align(1)= {0};
    avg_pool7(2048,result49,tensor51);
    end = read_cycles();
    cycles[50] = end-start;
    start = end;


    static elem_t kernel51[2048][1024] row_align(1)= {0};
    static elem_t result51[64][1024] row_align(1)= {0};


    /* matmul number: 51 */

    tiled_matmul_compare(64, 1024, 2048,    // dimensions
    tensor51, kernel51, NULL, result51,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_51");
    // verbose(51,tensor51,kernel51,result51)
    /* end of matmul number: 51 */

    end = read_cycles();
    cycles[51] = end-start;
    start = end;

    unsigned long overall_cycles = 0;
    for(int cyc = 0; cyc < 52 ; cyc++){
        overall_cycles += cycles[cyc];
    }
    for(int cyc = 0; cyc < 52 ; cyc++){
        printf("Cycles taken in layer %d: %lu, %lf\n", cyc,cycles[cyc],cycles[cyc]*100.0/(1.0*overall_cycles));
    }
    printf("Overall cycles taken: %lu\n",overall_cycles);


    return 0;
}

