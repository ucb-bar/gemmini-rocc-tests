

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
    unsigned long cycles[154]={0};
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
    static elem_t kernel23[256][128] row_align(1)= {0};
    static elem_t tensor23[832][256] row_align(1)= {0};
    static elem_t result23[832][128] row_align(1)= {0};


    /* matmul number: 23 */

    tiled_matmul_compare(832, 128, 256,    // dimensions
    tensor23, kernel23, NULL, result23,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_23");
    // verbose(23,tensor23,kernel23,result23)
    /* end of matmul number: 23 */

    end = read_cycles();
    cycles[23] = end-start;
    start = end;
    static elem_t kernel24[1152][128] row_align(1)= {0};
    static elem_t tensor24[832][1152] row_align(1)= {0};
    static elem_t result24[832][128] row_align(1)= {0};
    tensor_reshape(128,result23,28, 28, 3, 1, tensor24);


    /* matmul number: 24 */

    tiled_matmul_compare(832, 128, 1152,    // dimensions
    tensor24, kernel24, NULL, result24,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_24");
    // verbose(24,tensor24,kernel24,result24)
    /* end of matmul number: 24 */

    end = read_cycles();
    cycles[24] = end-start;
    start = end;
    static elem_t kernel25[128][512] row_align(1)= {0};
    static elem_t tensor25[832][128] row_align(1)= {0};
    static elem_t result25[832][512] row_align(1)= {0};
    static elem_t inter_result25[832][512] row_align(1)= {0};
    rocket_zeropad(832,128,result23,512,  inter_result25);


    /* matmul number: 25 */

    tiled_matmul_compare(832, 512, 128,    // dimensions
    tensor25, kernel25, inter_result25, result25,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_25");
    // verbose(25,tensor25,kernel25,result25)
    /* end of matmul number: 25 */

    end = read_cycles();
    cycles[25] = end-start;
    start = end;
    static elem_t kernel26[256][128] row_align(1)= {0};
    static elem_t tensor26[832][256] row_align(1)= {0};
    static elem_t result26[832][128] row_align(1)= {0};


    /* matmul number: 26 */

    tiled_matmul_compare(832, 128, 256,    // dimensions
    tensor26, kernel26, NULL, result26,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_26");
    // verbose(26,tensor26,kernel26,result26)
    /* end of matmul number: 26 */

    end = read_cycles();
    cycles[26] = end-start;
    start = end;
    static elem_t kernel27[1152][128] row_align(1)= {0};
    static elem_t tensor27[832][1152] row_align(1)= {0};
    static elem_t result27[832][128] row_align(1)= {0};
    tensor_reshape(128,result26,28, 28, 3, 1, tensor27);


    /* matmul number: 27 */

    tiled_matmul_compare(832, 128, 1152,    // dimensions
    tensor27, kernel27, NULL, result27,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_27");
    // verbose(27,tensor27,kernel27,result27)
    /* end of matmul number: 27 */

    end = read_cycles();
    cycles[27] = end-start;
    start = end;
    static elem_t kernel28[128][512] row_align(1)= {0};
    static elem_t tensor28[832][128] row_align(1)= {0};
    static elem_t result28[832][512] row_align(1)= {0};
    static elem_t inter_result28[832][512] row_align(1)= {0};
    rocket_zeropad(832,128,result26,512,  inter_result28);


    /* matmul number: 28 */

    tiled_matmul_compare(832, 512, 128,    // dimensions
    tensor28, kernel28, inter_result28, result28,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_28");
    // verbose(28,tensor28,kernel28,result28)
    /* end of matmul number: 28 */

    end = read_cycles();
    cycles[28] = end-start;
    start = end;
    static elem_t kernel29[256][128] row_align(1)= {0};
    static elem_t tensor29[832][256] row_align(1)= {0};
    static elem_t result29[832][128] row_align(1)= {0};


    /* matmul number: 29 */

    tiled_matmul_compare(832, 128, 256,    // dimensions
    tensor29, kernel29, NULL, result29,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_29");
    // verbose(29,tensor29,kernel29,result29)
    /* end of matmul number: 29 */

    end = read_cycles();
    cycles[29] = end-start;
    start = end;
    static elem_t kernel30[1152][128] row_align(1)= {0};
    static elem_t tensor30[832][1152] row_align(1)= {0};
    static elem_t result30[832][128] row_align(1)= {0};
    tensor_reshape(128,result29,28, 28, 3, 1, tensor30);


    /* matmul number: 30 */

    tiled_matmul_compare(832, 128, 1152,    // dimensions
    tensor30, kernel30, NULL, result30,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_30");
    // verbose(30,tensor30,kernel30,result30)
    /* end of matmul number: 30 */

    end = read_cycles();
    cycles[30] = end-start;
    start = end;
    static elem_t kernel31[128][512] row_align(1)= {0};
    static elem_t tensor31[832][128] row_align(1)= {0};
    static elem_t result31[832][512] row_align(1)= {0};
    static elem_t inter_result31[832][512] row_align(1)= {0};
    rocket_zeropad(832,128,result29,512,  inter_result31);


    /* matmul number: 31 */

    tiled_matmul_compare(832, 512, 128,    // dimensions
    tensor31, kernel31, inter_result31, result31,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_31");
    // verbose(31,tensor31,kernel31,result31)
    /* end of matmul number: 31 */

    end = read_cycles();
    cycles[31] = end-start;
    start = end;
    static elem_t kernel32[256][128] row_align(1)= {0};
    static elem_t tensor32[832][256] row_align(1)= {0};
    static elem_t result32[832][128] row_align(1)= {0};


    /* matmul number: 32 */

    tiled_matmul_compare(832, 128, 256,    // dimensions
    tensor32, kernel32, NULL, result32,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_32");
    // verbose(32,tensor32,kernel32,result32)
    /* end of matmul number: 32 */

    end = read_cycles();
    cycles[32] = end-start;
    start = end;
    static elem_t kernel33[1152][128] row_align(1)= {0};
    static elem_t tensor33[832][1152] row_align(1)= {0};
    static elem_t result33[832][128] row_align(1)= {0};
    tensor_reshape(128,result32,28, 28, 3, 1, tensor33);


    /* matmul number: 33 */

    tiled_matmul_compare(832, 128, 1152,    // dimensions
    tensor33, kernel33, NULL, result33,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_33");
    // verbose(33,tensor33,kernel33,result33)
    /* end of matmul number: 33 */

    end = read_cycles();
    cycles[33] = end-start;
    start = end;
    static elem_t kernel34[128][512] row_align(1)= {0};
    static elem_t tensor34[832][128] row_align(1)= {0};
    static elem_t result34[832][512] row_align(1)= {0};
    static elem_t inter_result34[832][512] row_align(1)= {0};
    rocket_zeropad(832,128,result32,512,  inter_result34);


    /* matmul number: 34 */

    tiled_matmul_compare(832, 512, 128,    // dimensions
    tensor34, kernel34, inter_result34, result34,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_34");
    // verbose(34,tensor34,kernel34,result34)
    /* end of matmul number: 34 */

    end = read_cycles();
    cycles[34] = end-start;
    start = end;
    static elem_t kernel35[512][256] row_align(1)= {0};
    static elem_t tensor35[832][512] row_align(1)= {0};
    static elem_t result35[832][256] row_align(1)= {0};


    /* matmul number: 35 */

    tiled_matmul_compare(832, 256, 512,    // dimensions
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
    tensor_reshape(256,result35,28, 28, 3, 2, tensor36);


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
    rocket_fix_strided_dimension(28,256,result35,1024,  inter_result37);


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
    static elem_t kernel41[512][256] row_align(1)= {0};
    static elem_t tensor41[256][512] row_align(1)= {0};
    static elem_t result41[256][256] row_align(1)= {0};


    /* matmul number: 41 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor41, kernel41, NULL, result41,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_41");
    // verbose(41,tensor41,kernel41,result41)
    /* end of matmul number: 41 */

    end = read_cycles();
    cycles[41] = end-start;
    start = end;
    static elem_t kernel42[2304][256] row_align(1)= {0};
    static elem_t tensor42[256][2304] row_align(1)= {0};
    static elem_t result42[256][256] row_align(1)= {0};
    tensor_reshape(256,result41,14, 14, 3, 1, tensor42);


    /* matmul number: 42 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor42, kernel42, NULL, result42,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_42");
    // verbose(42,tensor42,kernel42,result42)
    /* end of matmul number: 42 */

    end = read_cycles();
    cycles[42] = end-start;
    start = end;
    static elem_t kernel43[256][1024] row_align(1)= {0};
    static elem_t tensor43[256][256] row_align(1)= {0};
    static elem_t result43[256][1024] row_align(1)= {0};
    static elem_t inter_result43[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result41,1024,  inter_result43);


    /* matmul number: 43 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor43, kernel43, inter_result43, result43,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_43");
    // verbose(43,tensor43,kernel43,result43)
    /* end of matmul number: 43 */

    end = read_cycles();
    cycles[43] = end-start;
    start = end;
    static elem_t kernel44[512][256] row_align(1)= {0};
    static elem_t tensor44[256][512] row_align(1)= {0};
    static elem_t result44[256][256] row_align(1)= {0};


    /* matmul number: 44 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor44, kernel44, NULL, result44,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_44");
    // verbose(44,tensor44,kernel44,result44)
    /* end of matmul number: 44 */

    end = read_cycles();
    cycles[44] = end-start;
    start = end;
    static elem_t kernel45[2304][256] row_align(1)= {0};
    static elem_t tensor45[256][2304] row_align(1)= {0};
    static elem_t result45[256][256] row_align(1)= {0};
    tensor_reshape(256,result44,14, 14, 3, 1, tensor45);


    /* matmul number: 45 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor45, kernel45, NULL, result45,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_45");
    // verbose(45,tensor45,kernel45,result45)
    /* end of matmul number: 45 */

    end = read_cycles();
    cycles[45] = end-start;
    start = end;
    static elem_t kernel46[256][1024] row_align(1)= {0};
    static elem_t tensor46[256][256] row_align(1)= {0};
    static elem_t result46[256][1024] row_align(1)= {0};
    static elem_t inter_result46[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result44,1024,  inter_result46);


    /* matmul number: 46 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor46, kernel46, inter_result46, result46,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_46");
    // verbose(46,tensor46,kernel46,result46)
    /* end of matmul number: 46 */

    end = read_cycles();
    cycles[46] = end-start;
    start = end;
    static elem_t kernel47[512][256] row_align(1)= {0};
    static elem_t tensor47[256][512] row_align(1)= {0};
    static elem_t result47[256][256] row_align(1)= {0};


    /* matmul number: 47 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor47, kernel47, NULL, result47,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_47");
    // verbose(47,tensor47,kernel47,result47)
    /* end of matmul number: 47 */

    end = read_cycles();
    cycles[47] = end-start;
    start = end;
    static elem_t kernel48[2304][256] row_align(1)= {0};
    static elem_t tensor48[256][2304] row_align(1)= {0};
    static elem_t result48[256][256] row_align(1)= {0};
    tensor_reshape(256,result47,14, 14, 3, 1, tensor48);


    /* matmul number: 48 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor48, kernel48, NULL, result48,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_48");
    // verbose(48,tensor48,kernel48,result48)
    /* end of matmul number: 48 */

    end = read_cycles();
    cycles[48] = end-start;
    start = end;
    static elem_t kernel49[256][1024] row_align(1)= {0};
    static elem_t tensor49[256][256] row_align(1)= {0};
    static elem_t result49[256][1024] row_align(1)= {0};
    static elem_t inter_result49[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result47,1024,  inter_result49);


    /* matmul number: 49 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor49, kernel49, inter_result49, result49,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_49");
    // verbose(49,tensor49,kernel49,result49)
    /* end of matmul number: 49 */

    end = read_cycles();
    cycles[49] = end-start;
    start = end;
    static elem_t kernel50[512][256] row_align(1)= {0};
    static elem_t tensor50[256][512] row_align(1)= {0};
    static elem_t result50[256][256] row_align(1)= {0};


    /* matmul number: 50 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor50, kernel50, NULL, result50,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_50");
    // verbose(50,tensor50,kernel50,result50)
    /* end of matmul number: 50 */

    end = read_cycles();
    cycles[50] = end-start;
    start = end;
    static elem_t kernel51[2304][256] row_align(1)= {0};
    static elem_t tensor51[256][2304] row_align(1)= {0};
    static elem_t result51[256][256] row_align(1)= {0};
    tensor_reshape(256,result50,14, 14, 3, 1, tensor51);


    /* matmul number: 51 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor51, kernel51, NULL, result51,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_51");
    // verbose(51,tensor51,kernel51,result51)
    /* end of matmul number: 51 */

    end = read_cycles();
    cycles[51] = end-start;
    start = end;
    static elem_t kernel52[256][1024] row_align(1)= {0};
    static elem_t tensor52[256][256] row_align(1)= {0};
    static elem_t result52[256][1024] row_align(1)= {0};
    static elem_t inter_result52[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result50,1024,  inter_result52);


    /* matmul number: 52 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor52, kernel52, inter_result52, result52,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_52");
    // verbose(52,tensor52,kernel52,result52)
    /* end of matmul number: 52 */

    end = read_cycles();
    cycles[52] = end-start;
    start = end;
    static elem_t kernel53[512][256] row_align(1)= {0};
    static elem_t tensor53[256][512] row_align(1)= {0};
    static elem_t result53[256][256] row_align(1)= {0};


    /* matmul number: 53 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor53, kernel53, NULL, result53,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_53");
    // verbose(53,tensor53,kernel53,result53)
    /* end of matmul number: 53 */

    end = read_cycles();
    cycles[53] = end-start;
    start = end;
    static elem_t kernel54[2304][256] row_align(1)= {0};
    static elem_t tensor54[256][2304] row_align(1)= {0};
    static elem_t result54[256][256] row_align(1)= {0};
    tensor_reshape(256,result53,14, 14, 3, 1, tensor54);


    /* matmul number: 54 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor54, kernel54, NULL, result54,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_54");
    // verbose(54,tensor54,kernel54,result54)
    /* end of matmul number: 54 */

    end = read_cycles();
    cycles[54] = end-start;
    start = end;
    static elem_t kernel55[256][1024] row_align(1)= {0};
    static elem_t tensor55[256][256] row_align(1)= {0};
    static elem_t result55[256][1024] row_align(1)= {0};
    static elem_t inter_result55[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result53,1024,  inter_result55);


    /* matmul number: 55 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor55, kernel55, inter_result55, result55,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_55");
    // verbose(55,tensor55,kernel55,result55)
    /* end of matmul number: 55 */

    end = read_cycles();
    cycles[55] = end-start;
    start = end;
    static elem_t kernel56[512][256] row_align(1)= {0};
    static elem_t tensor56[256][512] row_align(1)= {0};
    static elem_t result56[256][256] row_align(1)= {0};


    /* matmul number: 56 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor56, kernel56, NULL, result56,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_56");
    // verbose(56,tensor56,kernel56,result56)
    /* end of matmul number: 56 */

    end = read_cycles();
    cycles[56] = end-start;
    start = end;
    static elem_t kernel57[2304][256] row_align(1)= {0};
    static elem_t tensor57[256][2304] row_align(1)= {0};
    static elem_t result57[256][256] row_align(1)= {0};
    tensor_reshape(256,result56,14, 14, 3, 1, tensor57);


    /* matmul number: 57 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor57, kernel57, NULL, result57,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_57");
    // verbose(57,tensor57,kernel57,result57)
    /* end of matmul number: 57 */

    end = read_cycles();
    cycles[57] = end-start;
    start = end;
    static elem_t kernel58[256][1024] row_align(1)= {0};
    static elem_t tensor58[256][256] row_align(1)= {0};
    static elem_t result58[256][1024] row_align(1)= {0};
    static elem_t inter_result58[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result56,1024,  inter_result58);


    /* matmul number: 58 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor58, kernel58, inter_result58, result58,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_58");
    // verbose(58,tensor58,kernel58,result58)
    /* end of matmul number: 58 */

    end = read_cycles();
    cycles[58] = end-start;
    start = end;
    static elem_t kernel59[512][256] row_align(1)= {0};
    static elem_t tensor59[256][512] row_align(1)= {0};
    static elem_t result59[256][256] row_align(1)= {0};


    /* matmul number: 59 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor59, kernel59, NULL, result59,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_59");
    // verbose(59,tensor59,kernel59,result59)
    /* end of matmul number: 59 */

    end = read_cycles();
    cycles[59] = end-start;
    start = end;
    static elem_t kernel60[2304][256] row_align(1)= {0};
    static elem_t tensor60[256][2304] row_align(1)= {0};
    static elem_t result60[256][256] row_align(1)= {0};
    tensor_reshape(256,result59,14, 14, 3, 1, tensor60);


    /* matmul number: 60 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor60, kernel60, NULL, result60,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_60");
    // verbose(60,tensor60,kernel60,result60)
    /* end of matmul number: 60 */

    end = read_cycles();
    cycles[60] = end-start;
    start = end;
    static elem_t kernel61[256][1024] row_align(1)= {0};
    static elem_t tensor61[256][256] row_align(1)= {0};
    static elem_t result61[256][1024] row_align(1)= {0};
    static elem_t inter_result61[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result59,1024,  inter_result61);


    /* matmul number: 61 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor61, kernel61, inter_result61, result61,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_61");
    // verbose(61,tensor61,kernel61,result61)
    /* end of matmul number: 61 */

    end = read_cycles();
    cycles[61] = end-start;
    start = end;
    static elem_t kernel62[512][256] row_align(1)= {0};
    static elem_t tensor62[256][512] row_align(1)= {0};
    static elem_t result62[256][256] row_align(1)= {0};


    /* matmul number: 62 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor62, kernel62, NULL, result62,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_62");
    // verbose(62,tensor62,kernel62,result62)
    /* end of matmul number: 62 */

    end = read_cycles();
    cycles[62] = end-start;
    start = end;
    static elem_t kernel63[2304][256] row_align(1)= {0};
    static elem_t tensor63[256][2304] row_align(1)= {0};
    static elem_t result63[256][256] row_align(1)= {0};
    tensor_reshape(256,result62,14, 14, 3, 1, tensor63);


    /* matmul number: 63 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor63, kernel63, NULL, result63,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_63");
    // verbose(63,tensor63,kernel63,result63)
    /* end of matmul number: 63 */

    end = read_cycles();
    cycles[63] = end-start;
    start = end;
    static elem_t kernel64[256][1024] row_align(1)= {0};
    static elem_t tensor64[256][256] row_align(1)= {0};
    static elem_t result64[256][1024] row_align(1)= {0};
    static elem_t inter_result64[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result62,1024,  inter_result64);


    /* matmul number: 64 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor64, kernel64, inter_result64, result64,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_64");
    // verbose(64,tensor64,kernel64,result64)
    /* end of matmul number: 64 */

    end = read_cycles();
    cycles[64] = end-start;
    start = end;
    static elem_t kernel65[512][256] row_align(1)= {0};
    static elem_t tensor65[256][512] row_align(1)= {0};
    static elem_t result65[256][256] row_align(1)= {0};


    /* matmul number: 65 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor65, kernel65, NULL, result65,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_65");
    // verbose(65,tensor65,kernel65,result65)
    /* end of matmul number: 65 */

    end = read_cycles();
    cycles[65] = end-start;
    start = end;
    static elem_t kernel66[2304][256] row_align(1)= {0};
    static elem_t tensor66[256][2304] row_align(1)= {0};
    static elem_t result66[256][256] row_align(1)= {0};
    tensor_reshape(256,result65,14, 14, 3, 1, tensor66);


    /* matmul number: 66 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor66, kernel66, NULL, result66,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_66");
    // verbose(66,tensor66,kernel66,result66)
    /* end of matmul number: 66 */

    end = read_cycles();
    cycles[66] = end-start;
    start = end;
    static elem_t kernel67[256][1024] row_align(1)= {0};
    static elem_t tensor67[256][256] row_align(1)= {0};
    static elem_t result67[256][1024] row_align(1)= {0};
    static elem_t inter_result67[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result65,1024,  inter_result67);


    /* matmul number: 67 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor67, kernel67, inter_result67, result67,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_67");
    // verbose(67,tensor67,kernel67,result67)
    /* end of matmul number: 67 */

    end = read_cycles();
    cycles[67] = end-start;
    start = end;
    static elem_t kernel68[512][256] row_align(1)= {0};
    static elem_t tensor68[256][512] row_align(1)= {0};
    static elem_t result68[256][256] row_align(1)= {0};


    /* matmul number: 68 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor68, kernel68, NULL, result68,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_68");
    // verbose(68,tensor68,kernel68,result68)
    /* end of matmul number: 68 */

    end = read_cycles();
    cycles[68] = end-start;
    start = end;
    static elem_t kernel69[2304][256] row_align(1)= {0};
    static elem_t tensor69[256][2304] row_align(1)= {0};
    static elem_t result69[256][256] row_align(1)= {0};
    tensor_reshape(256,result68,14, 14, 3, 1, tensor69);


    /* matmul number: 69 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor69, kernel69, NULL, result69,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_69");
    // verbose(69,tensor69,kernel69,result69)
    /* end of matmul number: 69 */

    end = read_cycles();
    cycles[69] = end-start;
    start = end;
    static elem_t kernel70[256][1024] row_align(1)= {0};
    static elem_t tensor70[256][256] row_align(1)= {0};
    static elem_t result70[256][1024] row_align(1)= {0};
    static elem_t inter_result70[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result68,1024,  inter_result70);


    /* matmul number: 70 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor70, kernel70, inter_result70, result70,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_70");
    // verbose(70,tensor70,kernel70,result70)
    /* end of matmul number: 70 */

    end = read_cycles();
    cycles[70] = end-start;
    start = end;
    static elem_t kernel71[512][256] row_align(1)= {0};
    static elem_t tensor71[256][512] row_align(1)= {0};
    static elem_t result71[256][256] row_align(1)= {0};


    /* matmul number: 71 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor71, kernel71, NULL, result71,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_71");
    // verbose(71,tensor71,kernel71,result71)
    /* end of matmul number: 71 */

    end = read_cycles();
    cycles[71] = end-start;
    start = end;
    static elem_t kernel72[2304][256] row_align(1)= {0};
    static elem_t tensor72[256][2304] row_align(1)= {0};
    static elem_t result72[256][256] row_align(1)= {0};
    tensor_reshape(256,result71,14, 14, 3, 1, tensor72);


    /* matmul number: 72 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor72, kernel72, NULL, result72,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_72");
    // verbose(72,tensor72,kernel72,result72)
    /* end of matmul number: 72 */

    end = read_cycles();
    cycles[72] = end-start;
    start = end;
    static elem_t kernel73[256][1024] row_align(1)= {0};
    static elem_t tensor73[256][256] row_align(1)= {0};
    static elem_t result73[256][1024] row_align(1)= {0};
    static elem_t inter_result73[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result71,1024,  inter_result73);


    /* matmul number: 73 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor73, kernel73, inter_result73, result73,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_73");
    // verbose(73,tensor73,kernel73,result73)
    /* end of matmul number: 73 */

    end = read_cycles();
    cycles[73] = end-start;
    start = end;
    static elem_t kernel74[512][256] row_align(1)= {0};
    static elem_t tensor74[256][512] row_align(1)= {0};
    static elem_t result74[256][256] row_align(1)= {0};


    /* matmul number: 74 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor74, kernel74, NULL, result74,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_74");
    // verbose(74,tensor74,kernel74,result74)
    /* end of matmul number: 74 */

    end = read_cycles();
    cycles[74] = end-start;
    start = end;
    static elem_t kernel75[2304][256] row_align(1)= {0};
    static elem_t tensor75[256][2304] row_align(1)= {0};
    static elem_t result75[256][256] row_align(1)= {0};
    tensor_reshape(256,result74,14, 14, 3, 1, tensor75);


    /* matmul number: 75 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor75, kernel75, NULL, result75,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_75");
    // verbose(75,tensor75,kernel75,result75)
    /* end of matmul number: 75 */

    end = read_cycles();
    cycles[75] = end-start;
    start = end;
    static elem_t kernel76[256][1024] row_align(1)= {0};
    static elem_t tensor76[256][256] row_align(1)= {0};
    static elem_t result76[256][1024] row_align(1)= {0};
    static elem_t inter_result76[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result74,1024,  inter_result76);


    /* matmul number: 76 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor76, kernel76, inter_result76, result76,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_76");
    // verbose(76,tensor76,kernel76,result76)
    /* end of matmul number: 76 */

    end = read_cycles();
    cycles[76] = end-start;
    start = end;
    static elem_t kernel77[512][256] row_align(1)= {0};
    static elem_t tensor77[256][512] row_align(1)= {0};
    static elem_t result77[256][256] row_align(1)= {0};


    /* matmul number: 77 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor77, kernel77, NULL, result77,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_77");
    // verbose(77,tensor77,kernel77,result77)
    /* end of matmul number: 77 */

    end = read_cycles();
    cycles[77] = end-start;
    start = end;
    static elem_t kernel78[2304][256] row_align(1)= {0};
    static elem_t tensor78[256][2304] row_align(1)= {0};
    static elem_t result78[256][256] row_align(1)= {0};
    tensor_reshape(256,result77,14, 14, 3, 1, tensor78);


    /* matmul number: 78 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor78, kernel78, NULL, result78,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_78");
    // verbose(78,tensor78,kernel78,result78)
    /* end of matmul number: 78 */

    end = read_cycles();
    cycles[78] = end-start;
    start = end;
    static elem_t kernel79[256][1024] row_align(1)= {0};
    static elem_t tensor79[256][256] row_align(1)= {0};
    static elem_t result79[256][1024] row_align(1)= {0};
    static elem_t inter_result79[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result77,1024,  inter_result79);


    /* matmul number: 79 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor79, kernel79, inter_result79, result79,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_79");
    // verbose(79,tensor79,kernel79,result79)
    /* end of matmul number: 79 */

    end = read_cycles();
    cycles[79] = end-start;
    start = end;
    static elem_t kernel80[512][256] row_align(1)= {0};
    static elem_t tensor80[256][512] row_align(1)= {0};
    static elem_t result80[256][256] row_align(1)= {0};


    /* matmul number: 80 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor80, kernel80, NULL, result80,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_80");
    // verbose(80,tensor80,kernel80,result80)
    /* end of matmul number: 80 */

    end = read_cycles();
    cycles[80] = end-start;
    start = end;
    static elem_t kernel81[2304][256] row_align(1)= {0};
    static elem_t tensor81[256][2304] row_align(1)= {0};
    static elem_t result81[256][256] row_align(1)= {0};
    tensor_reshape(256,result80,14, 14, 3, 1, tensor81);


    /* matmul number: 81 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor81, kernel81, NULL, result81,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_81");
    // verbose(81,tensor81,kernel81,result81)
    /* end of matmul number: 81 */

    end = read_cycles();
    cycles[81] = end-start;
    start = end;
    static elem_t kernel82[256][1024] row_align(1)= {0};
    static elem_t tensor82[256][256] row_align(1)= {0};
    static elem_t result82[256][1024] row_align(1)= {0};
    static elem_t inter_result82[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result80,1024,  inter_result82);


    /* matmul number: 82 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor82, kernel82, inter_result82, result82,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_82");
    // verbose(82,tensor82,kernel82,result82)
    /* end of matmul number: 82 */

    end = read_cycles();
    cycles[82] = end-start;
    start = end;
    static elem_t kernel83[512][256] row_align(1)= {0};
    static elem_t tensor83[256][512] row_align(1)= {0};
    static elem_t result83[256][256] row_align(1)= {0};


    /* matmul number: 83 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor83, kernel83, NULL, result83,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_83");
    // verbose(83,tensor83,kernel83,result83)
    /* end of matmul number: 83 */

    end = read_cycles();
    cycles[83] = end-start;
    start = end;
    static elem_t kernel84[2304][256] row_align(1)= {0};
    static elem_t tensor84[256][2304] row_align(1)= {0};
    static elem_t result84[256][256] row_align(1)= {0};
    tensor_reshape(256,result83,14, 14, 3, 1, tensor84);


    /* matmul number: 84 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor84, kernel84, NULL, result84,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_84");
    // verbose(84,tensor84,kernel84,result84)
    /* end of matmul number: 84 */

    end = read_cycles();
    cycles[84] = end-start;
    start = end;
    static elem_t kernel85[256][1024] row_align(1)= {0};
    static elem_t tensor85[256][256] row_align(1)= {0};
    static elem_t result85[256][1024] row_align(1)= {0};
    static elem_t inter_result85[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result83,1024,  inter_result85);


    /* matmul number: 85 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor85, kernel85, inter_result85, result85,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_85");
    // verbose(85,tensor85,kernel85,result85)
    /* end of matmul number: 85 */

    end = read_cycles();
    cycles[85] = end-start;
    start = end;
    static elem_t kernel86[512][256] row_align(1)= {0};
    static elem_t tensor86[256][512] row_align(1)= {0};
    static elem_t result86[256][256] row_align(1)= {0};


    /* matmul number: 86 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor86, kernel86, NULL, result86,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_86");
    // verbose(86,tensor86,kernel86,result86)
    /* end of matmul number: 86 */

    end = read_cycles();
    cycles[86] = end-start;
    start = end;
    static elem_t kernel87[2304][256] row_align(1)= {0};
    static elem_t tensor87[256][2304] row_align(1)= {0};
    static elem_t result87[256][256] row_align(1)= {0};
    tensor_reshape(256,result86,14, 14, 3, 1, tensor87);


    /* matmul number: 87 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor87, kernel87, NULL, result87,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_87");
    // verbose(87,tensor87,kernel87,result87)
    /* end of matmul number: 87 */

    end = read_cycles();
    cycles[87] = end-start;
    start = end;
    static elem_t kernel88[256][1024] row_align(1)= {0};
    static elem_t tensor88[256][256] row_align(1)= {0};
    static elem_t result88[256][1024] row_align(1)= {0};
    static elem_t inter_result88[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result86,1024,  inter_result88);


    /* matmul number: 88 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor88, kernel88, inter_result88, result88,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_88");
    // verbose(88,tensor88,kernel88,result88)
    /* end of matmul number: 88 */

    end = read_cycles();
    cycles[88] = end-start;
    start = end;
    static elem_t kernel89[512][256] row_align(1)= {0};
    static elem_t tensor89[256][512] row_align(1)= {0};
    static elem_t result89[256][256] row_align(1)= {0};


    /* matmul number: 89 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor89, kernel89, NULL, result89,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_89");
    // verbose(89,tensor89,kernel89,result89)
    /* end of matmul number: 89 */

    end = read_cycles();
    cycles[89] = end-start;
    start = end;
    static elem_t kernel90[2304][256] row_align(1)= {0};
    static elem_t tensor90[256][2304] row_align(1)= {0};
    static elem_t result90[256][256] row_align(1)= {0};
    tensor_reshape(256,result89,14, 14, 3, 1, tensor90);


    /* matmul number: 90 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor90, kernel90, NULL, result90,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_90");
    // verbose(90,tensor90,kernel90,result90)
    /* end of matmul number: 90 */

    end = read_cycles();
    cycles[90] = end-start;
    start = end;
    static elem_t kernel91[256][1024] row_align(1)= {0};
    static elem_t tensor91[256][256] row_align(1)= {0};
    static elem_t result91[256][1024] row_align(1)= {0};
    static elem_t inter_result91[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result89,1024,  inter_result91);


    /* matmul number: 91 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor91, kernel91, inter_result91, result91,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_91");
    // verbose(91,tensor91,kernel91,result91)
    /* end of matmul number: 91 */

    end = read_cycles();
    cycles[91] = end-start;
    start = end;
    static elem_t kernel92[512][256] row_align(1)= {0};
    static elem_t tensor92[256][512] row_align(1)= {0};
    static elem_t result92[256][256] row_align(1)= {0};


    /* matmul number: 92 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor92, kernel92, NULL, result92,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_92");
    // verbose(92,tensor92,kernel92,result92)
    /* end of matmul number: 92 */

    end = read_cycles();
    cycles[92] = end-start;
    start = end;
    static elem_t kernel93[2304][256] row_align(1)= {0};
    static elem_t tensor93[256][2304] row_align(1)= {0};
    static elem_t result93[256][256] row_align(1)= {0};
    tensor_reshape(256,result92,14, 14, 3, 1, tensor93);


    /* matmul number: 93 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor93, kernel93, NULL, result93,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_93");
    // verbose(93,tensor93,kernel93,result93)
    /* end of matmul number: 93 */

    end = read_cycles();
    cycles[93] = end-start;
    start = end;
    static elem_t kernel94[256][1024] row_align(1)= {0};
    static elem_t tensor94[256][256] row_align(1)= {0};
    static elem_t result94[256][1024] row_align(1)= {0};
    static elem_t inter_result94[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result92,1024,  inter_result94);


    /* matmul number: 94 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor94, kernel94, inter_result94, result94,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_94");
    // verbose(94,tensor94,kernel94,result94)
    /* end of matmul number: 94 */

    end = read_cycles();
    cycles[94] = end-start;
    start = end;
    static elem_t kernel95[512][256] row_align(1)= {0};
    static elem_t tensor95[256][512] row_align(1)= {0};
    static elem_t result95[256][256] row_align(1)= {0};


    /* matmul number: 95 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor95, kernel95, NULL, result95,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_95");
    // verbose(95,tensor95,kernel95,result95)
    /* end of matmul number: 95 */

    end = read_cycles();
    cycles[95] = end-start;
    start = end;
    static elem_t kernel96[2304][256] row_align(1)= {0};
    static elem_t tensor96[256][2304] row_align(1)= {0};
    static elem_t result96[256][256] row_align(1)= {0};
    tensor_reshape(256,result95,14, 14, 3, 1, tensor96);


    /* matmul number: 96 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor96, kernel96, NULL, result96,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_96");
    // verbose(96,tensor96,kernel96,result96)
    /* end of matmul number: 96 */

    end = read_cycles();
    cycles[96] = end-start;
    start = end;
    static elem_t kernel97[256][1024] row_align(1)= {0};
    static elem_t tensor97[256][256] row_align(1)= {0};
    static elem_t result97[256][1024] row_align(1)= {0};
    static elem_t inter_result97[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result95,1024,  inter_result97);


    /* matmul number: 97 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor97, kernel97, inter_result97, result97,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_97");
    // verbose(97,tensor97,kernel97,result97)
    /* end of matmul number: 97 */

    end = read_cycles();
    cycles[97] = end-start;
    start = end;
    static elem_t kernel98[512][256] row_align(1)= {0};
    static elem_t tensor98[256][512] row_align(1)= {0};
    static elem_t result98[256][256] row_align(1)= {0};


    /* matmul number: 98 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor98, kernel98, NULL, result98,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_98");
    // verbose(98,tensor98,kernel98,result98)
    /* end of matmul number: 98 */

    end = read_cycles();
    cycles[98] = end-start;
    start = end;
    static elem_t kernel99[2304][256] row_align(1)= {0};
    static elem_t tensor99[256][2304] row_align(1)= {0};
    static elem_t result99[256][256] row_align(1)= {0};
    tensor_reshape(256,result98,14, 14, 3, 1, tensor99);


    /* matmul number: 99 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor99, kernel99, NULL, result99,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_99");
    // verbose(99,tensor99,kernel99,result99)
    /* end of matmul number: 99 */

    end = read_cycles();
    cycles[99] = end-start;
    start = end;
    static elem_t kernel100[256][1024] row_align(1)= {0};
    static elem_t tensor100[256][256] row_align(1)= {0};
    static elem_t result100[256][1024] row_align(1)= {0};
    static elem_t inter_result100[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result98,1024,  inter_result100);


    /* matmul number: 100 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor100, kernel100, inter_result100, result100,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_100");
    // verbose(100,tensor100,kernel100,result100)
    /* end of matmul number: 100 */

    end = read_cycles();
    cycles[100] = end-start;
    start = end;
    static elem_t kernel101[512][256] row_align(1)= {0};
    static elem_t tensor101[256][512] row_align(1)= {0};
    static elem_t result101[256][256] row_align(1)= {0};


    /* matmul number: 101 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor101, kernel101, NULL, result101,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_101");
    // verbose(101,tensor101,kernel101,result101)
    /* end of matmul number: 101 */

    end = read_cycles();
    cycles[101] = end-start;
    start = end;
    static elem_t kernel102[2304][256] row_align(1)= {0};
    static elem_t tensor102[256][2304] row_align(1)= {0};
    static elem_t result102[256][256] row_align(1)= {0};
    tensor_reshape(256,result101,14, 14, 3, 1, tensor102);


    /* matmul number: 102 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor102, kernel102, NULL, result102,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_102");
    // verbose(102,tensor102,kernel102,result102)
    /* end of matmul number: 102 */

    end = read_cycles();
    cycles[102] = end-start;
    start = end;
    static elem_t kernel103[256][1024] row_align(1)= {0};
    static elem_t tensor103[256][256] row_align(1)= {0};
    static elem_t result103[256][1024] row_align(1)= {0};
    static elem_t inter_result103[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result101,1024,  inter_result103);


    /* matmul number: 103 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor103, kernel103, inter_result103, result103,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_103");
    // verbose(103,tensor103,kernel103,result103)
    /* end of matmul number: 103 */

    end = read_cycles();
    cycles[103] = end-start;
    start = end;
    static elem_t kernel104[512][256] row_align(1)= {0};
    static elem_t tensor104[256][512] row_align(1)= {0};
    static elem_t result104[256][256] row_align(1)= {0};


    /* matmul number: 104 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor104, kernel104, NULL, result104,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_104");
    // verbose(104,tensor104,kernel104,result104)
    /* end of matmul number: 104 */

    end = read_cycles();
    cycles[104] = end-start;
    start = end;
    static elem_t kernel105[2304][256] row_align(1)= {0};
    static elem_t tensor105[256][2304] row_align(1)= {0};
    static elem_t result105[256][256] row_align(1)= {0};
    tensor_reshape(256,result104,14, 14, 3, 1, tensor105);


    /* matmul number: 105 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor105, kernel105, NULL, result105,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_105");
    // verbose(105,tensor105,kernel105,result105)
    /* end of matmul number: 105 */

    end = read_cycles();
    cycles[105] = end-start;
    start = end;
    static elem_t kernel106[256][1024] row_align(1)= {0};
    static elem_t tensor106[256][256] row_align(1)= {0};
    static elem_t result106[256][1024] row_align(1)= {0};
    static elem_t inter_result106[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result104,1024,  inter_result106);


    /* matmul number: 106 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor106, kernel106, inter_result106, result106,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_106");
    // verbose(106,tensor106,kernel106,result106)
    /* end of matmul number: 106 */

    end = read_cycles();
    cycles[106] = end-start;
    start = end;
    static elem_t kernel107[512][256] row_align(1)= {0};
    static elem_t tensor107[256][512] row_align(1)= {0};
    static elem_t result107[256][256] row_align(1)= {0};


    /* matmul number: 107 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor107, kernel107, NULL, result107,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_107");
    // verbose(107,tensor107,kernel107,result107)
    /* end of matmul number: 107 */

    end = read_cycles();
    cycles[107] = end-start;
    start = end;
    static elem_t kernel108[2304][256] row_align(1)= {0};
    static elem_t tensor108[256][2304] row_align(1)= {0};
    static elem_t result108[256][256] row_align(1)= {0};
    tensor_reshape(256,result107,14, 14, 3, 1, tensor108);


    /* matmul number: 108 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor108, kernel108, NULL, result108,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_108");
    // verbose(108,tensor108,kernel108,result108)
    /* end of matmul number: 108 */

    end = read_cycles();
    cycles[108] = end-start;
    start = end;
    static elem_t kernel109[256][1024] row_align(1)= {0};
    static elem_t tensor109[256][256] row_align(1)= {0};
    static elem_t result109[256][1024] row_align(1)= {0};
    static elem_t inter_result109[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result107,1024,  inter_result109);


    /* matmul number: 109 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor109, kernel109, inter_result109, result109,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_109");
    // verbose(109,tensor109,kernel109,result109)
    /* end of matmul number: 109 */

    end = read_cycles();
    cycles[109] = end-start;
    start = end;
    static elem_t kernel110[512][256] row_align(1)= {0};
    static elem_t tensor110[256][512] row_align(1)= {0};
    static elem_t result110[256][256] row_align(1)= {0};


    /* matmul number: 110 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor110, kernel110, NULL, result110,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_110");
    // verbose(110,tensor110,kernel110,result110)
    /* end of matmul number: 110 */

    end = read_cycles();
    cycles[110] = end-start;
    start = end;
    static elem_t kernel111[2304][256] row_align(1)= {0};
    static elem_t tensor111[256][2304] row_align(1)= {0};
    static elem_t result111[256][256] row_align(1)= {0};
    tensor_reshape(256,result110,14, 14, 3, 1, tensor111);


    /* matmul number: 111 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor111, kernel111, NULL, result111,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_111");
    // verbose(111,tensor111,kernel111,result111)
    /* end of matmul number: 111 */

    end = read_cycles();
    cycles[111] = end-start;
    start = end;
    static elem_t kernel112[256][1024] row_align(1)= {0};
    static elem_t tensor112[256][256] row_align(1)= {0};
    static elem_t result112[256][1024] row_align(1)= {0};
    static elem_t inter_result112[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result110,1024,  inter_result112);


    /* matmul number: 112 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor112, kernel112, inter_result112, result112,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_112");
    // verbose(112,tensor112,kernel112,result112)
    /* end of matmul number: 112 */

    end = read_cycles();
    cycles[112] = end-start;
    start = end;
    static elem_t kernel113[512][256] row_align(1)= {0};
    static elem_t tensor113[256][512] row_align(1)= {0};
    static elem_t result113[256][256] row_align(1)= {0};


    /* matmul number: 113 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor113, kernel113, NULL, result113,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_113");
    // verbose(113,tensor113,kernel113,result113)
    /* end of matmul number: 113 */

    end = read_cycles();
    cycles[113] = end-start;
    start = end;
    static elem_t kernel114[2304][256] row_align(1)= {0};
    static elem_t tensor114[256][2304] row_align(1)= {0};
    static elem_t result114[256][256] row_align(1)= {0};
    tensor_reshape(256,result113,14, 14, 3, 1, tensor114);


    /* matmul number: 114 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor114, kernel114, NULL, result114,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_114");
    // verbose(114,tensor114,kernel114,result114)
    /* end of matmul number: 114 */

    end = read_cycles();
    cycles[114] = end-start;
    start = end;
    static elem_t kernel115[256][1024] row_align(1)= {0};
    static elem_t tensor115[256][256] row_align(1)= {0};
    static elem_t result115[256][1024] row_align(1)= {0};
    static elem_t inter_result115[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result113,1024,  inter_result115);


    /* matmul number: 115 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor115, kernel115, inter_result115, result115,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_115");
    // verbose(115,tensor115,kernel115,result115)
    /* end of matmul number: 115 */

    end = read_cycles();
    cycles[115] = end-start;
    start = end;
    static elem_t kernel116[512][256] row_align(1)= {0};
    static elem_t tensor116[256][512] row_align(1)= {0};
    static elem_t result116[256][256] row_align(1)= {0};


    /* matmul number: 116 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor116, kernel116, NULL, result116,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_116");
    // verbose(116,tensor116,kernel116,result116)
    /* end of matmul number: 116 */

    end = read_cycles();
    cycles[116] = end-start;
    start = end;
    static elem_t kernel117[2304][256] row_align(1)= {0};
    static elem_t tensor117[256][2304] row_align(1)= {0};
    static elem_t result117[256][256] row_align(1)= {0};
    tensor_reshape(256,result116,14, 14, 3, 1, tensor117);


    /* matmul number: 117 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor117, kernel117, NULL, result117,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_117");
    // verbose(117,tensor117,kernel117,result117)
    /* end of matmul number: 117 */

    end = read_cycles();
    cycles[117] = end-start;
    start = end;
    static elem_t kernel118[256][1024] row_align(1)= {0};
    static elem_t tensor118[256][256] row_align(1)= {0};
    static elem_t result118[256][1024] row_align(1)= {0};
    static elem_t inter_result118[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result116,1024,  inter_result118);


    /* matmul number: 118 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor118, kernel118, inter_result118, result118,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_118");
    // verbose(118,tensor118,kernel118,result118)
    /* end of matmul number: 118 */

    end = read_cycles();
    cycles[118] = end-start;
    start = end;
    static elem_t kernel119[512][256] row_align(1)= {0};
    static elem_t tensor119[256][512] row_align(1)= {0};
    static elem_t result119[256][256] row_align(1)= {0};


    /* matmul number: 119 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor119, kernel119, NULL, result119,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_119");
    // verbose(119,tensor119,kernel119,result119)
    /* end of matmul number: 119 */

    end = read_cycles();
    cycles[119] = end-start;
    start = end;
    static elem_t kernel120[2304][256] row_align(1)= {0};
    static elem_t tensor120[256][2304] row_align(1)= {0};
    static elem_t result120[256][256] row_align(1)= {0};
    tensor_reshape(256,result119,14, 14, 3, 1, tensor120);


    /* matmul number: 120 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor120, kernel120, NULL, result120,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_120");
    // verbose(120,tensor120,kernel120,result120)
    /* end of matmul number: 120 */

    end = read_cycles();
    cycles[120] = end-start;
    start = end;
    static elem_t kernel121[256][1024] row_align(1)= {0};
    static elem_t tensor121[256][256] row_align(1)= {0};
    static elem_t result121[256][1024] row_align(1)= {0};
    static elem_t inter_result121[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result119,1024,  inter_result121);


    /* matmul number: 121 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor121, kernel121, inter_result121, result121,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_121");
    // verbose(121,tensor121,kernel121,result121)
    /* end of matmul number: 121 */

    end = read_cycles();
    cycles[121] = end-start;
    start = end;
    static elem_t kernel122[512][256] row_align(1)= {0};
    static elem_t tensor122[256][512] row_align(1)= {0};
    static elem_t result122[256][256] row_align(1)= {0};


    /* matmul number: 122 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor122, kernel122, NULL, result122,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_122");
    // verbose(122,tensor122,kernel122,result122)
    /* end of matmul number: 122 */

    end = read_cycles();
    cycles[122] = end-start;
    start = end;
    static elem_t kernel123[2304][256] row_align(1)= {0};
    static elem_t tensor123[256][2304] row_align(1)= {0};
    static elem_t result123[256][256] row_align(1)= {0};
    tensor_reshape(256,result122,14, 14, 3, 1, tensor123);


    /* matmul number: 123 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor123, kernel123, NULL, result123,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_123");
    // verbose(123,tensor123,kernel123,result123)
    /* end of matmul number: 123 */

    end = read_cycles();
    cycles[123] = end-start;
    start = end;
    static elem_t kernel124[256][1024] row_align(1)= {0};
    static elem_t tensor124[256][256] row_align(1)= {0};
    static elem_t result124[256][1024] row_align(1)= {0};
    static elem_t inter_result124[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result122,1024,  inter_result124);


    /* matmul number: 124 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor124, kernel124, inter_result124, result124,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_124");
    // verbose(124,tensor124,kernel124,result124)
    /* end of matmul number: 124 */

    end = read_cycles();
    cycles[124] = end-start;
    start = end;
    static elem_t kernel125[512][256] row_align(1)= {0};
    static elem_t tensor125[256][512] row_align(1)= {0};
    static elem_t result125[256][256] row_align(1)= {0};


    /* matmul number: 125 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor125, kernel125, NULL, result125,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_125");
    // verbose(125,tensor125,kernel125,result125)
    /* end of matmul number: 125 */

    end = read_cycles();
    cycles[125] = end-start;
    start = end;
    static elem_t kernel126[2304][256] row_align(1)= {0};
    static elem_t tensor126[256][2304] row_align(1)= {0};
    static elem_t result126[256][256] row_align(1)= {0};
    tensor_reshape(256,result125,14, 14, 3, 1, tensor126);


    /* matmul number: 126 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor126, kernel126, NULL, result126,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_126");
    // verbose(126,tensor126,kernel126,result126)
    /* end of matmul number: 126 */

    end = read_cycles();
    cycles[126] = end-start;
    start = end;
    static elem_t kernel127[256][1024] row_align(1)= {0};
    static elem_t tensor127[256][256] row_align(1)= {0};
    static elem_t result127[256][1024] row_align(1)= {0};
    static elem_t inter_result127[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result125,1024,  inter_result127);


    /* matmul number: 127 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor127, kernel127, inter_result127, result127,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_127");
    // verbose(127,tensor127,kernel127,result127)
    /* end of matmul number: 127 */

    end = read_cycles();
    cycles[127] = end-start;
    start = end;
    static elem_t kernel128[512][256] row_align(1)= {0};
    static elem_t tensor128[256][512] row_align(1)= {0};
    static elem_t result128[256][256] row_align(1)= {0};


    /* matmul number: 128 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor128, kernel128, NULL, result128,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_128");
    // verbose(128,tensor128,kernel128,result128)
    /* end of matmul number: 128 */

    end = read_cycles();
    cycles[128] = end-start;
    start = end;
    static elem_t kernel129[2304][256] row_align(1)= {0};
    static elem_t tensor129[256][2304] row_align(1)= {0};
    static elem_t result129[256][256] row_align(1)= {0};
    tensor_reshape(256,result128,14, 14, 3, 1, tensor129);


    /* matmul number: 129 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor129, kernel129, NULL, result129,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_129");
    // verbose(129,tensor129,kernel129,result129)
    /* end of matmul number: 129 */

    end = read_cycles();
    cycles[129] = end-start;
    start = end;
    static elem_t kernel130[256][1024] row_align(1)= {0};
    static elem_t tensor130[256][256] row_align(1)= {0};
    static elem_t result130[256][1024] row_align(1)= {0};
    static elem_t inter_result130[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result128,1024,  inter_result130);


    /* matmul number: 130 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor130, kernel130, inter_result130, result130,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_130");
    // verbose(130,tensor130,kernel130,result130)
    /* end of matmul number: 130 */

    end = read_cycles();
    cycles[130] = end-start;
    start = end;
    static elem_t kernel131[512][256] row_align(1)= {0};
    static elem_t tensor131[256][512] row_align(1)= {0};
    static elem_t result131[256][256] row_align(1)= {0};


    /* matmul number: 131 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor131, kernel131, NULL, result131,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_131");
    // verbose(131,tensor131,kernel131,result131)
    /* end of matmul number: 131 */

    end = read_cycles();
    cycles[131] = end-start;
    start = end;
    static elem_t kernel132[2304][256] row_align(1)= {0};
    static elem_t tensor132[256][2304] row_align(1)= {0};
    static elem_t result132[256][256] row_align(1)= {0};
    tensor_reshape(256,result131,14, 14, 3, 1, tensor132);


    /* matmul number: 132 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor132, kernel132, NULL, result132,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_132");
    // verbose(132,tensor132,kernel132,result132)
    /* end of matmul number: 132 */

    end = read_cycles();
    cycles[132] = end-start;
    start = end;
    static elem_t kernel133[256][1024] row_align(1)= {0};
    static elem_t tensor133[256][256] row_align(1)= {0};
    static elem_t result133[256][1024] row_align(1)= {0};
    static elem_t inter_result133[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result131,1024,  inter_result133);


    /* matmul number: 133 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor133, kernel133, inter_result133, result133,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_133");
    // verbose(133,tensor133,kernel133,result133)
    /* end of matmul number: 133 */

    end = read_cycles();
    cycles[133] = end-start;
    start = end;
    static elem_t kernel134[512][256] row_align(1)= {0};
    static elem_t tensor134[256][512] row_align(1)= {0};
    static elem_t result134[256][256] row_align(1)= {0};


    /* matmul number: 134 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor134, kernel134, NULL, result134,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_134");
    // verbose(134,tensor134,kernel134,result134)
    /* end of matmul number: 134 */

    end = read_cycles();
    cycles[134] = end-start;
    start = end;
    static elem_t kernel135[2304][256] row_align(1)= {0};
    static elem_t tensor135[256][2304] row_align(1)= {0};
    static elem_t result135[256][256] row_align(1)= {0};
    tensor_reshape(256,result134,14, 14, 3, 1, tensor135);


    /* matmul number: 135 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor135, kernel135, NULL, result135,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_135");
    // verbose(135,tensor135,kernel135,result135)
    /* end of matmul number: 135 */

    end = read_cycles();
    cycles[135] = end-start;
    start = end;
    static elem_t kernel136[256][1024] row_align(1)= {0};
    static elem_t tensor136[256][256] row_align(1)= {0};
    static elem_t result136[256][1024] row_align(1)= {0};
    static elem_t inter_result136[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result134,1024,  inter_result136);


    /* matmul number: 136 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor136, kernel136, inter_result136, result136,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_136");
    // verbose(136,tensor136,kernel136,result136)
    /* end of matmul number: 136 */

    end = read_cycles();
    cycles[136] = end-start;
    start = end;
    static elem_t kernel137[512][256] row_align(1)= {0};
    static elem_t tensor137[256][512] row_align(1)= {0};
    static elem_t result137[256][256] row_align(1)= {0};


    /* matmul number: 137 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor137, kernel137, NULL, result137,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_137");
    // verbose(137,tensor137,kernel137,result137)
    /* end of matmul number: 137 */

    end = read_cycles();
    cycles[137] = end-start;
    start = end;
    static elem_t kernel138[2304][256] row_align(1)= {0};
    static elem_t tensor138[256][2304] row_align(1)= {0};
    static elem_t result138[256][256] row_align(1)= {0};
    tensor_reshape(256,result137,14, 14, 3, 1, tensor138);


    /* matmul number: 138 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor138, kernel138, NULL, result138,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_138");
    // verbose(138,tensor138,kernel138,result138)
    /* end of matmul number: 138 */

    end = read_cycles();
    cycles[138] = end-start;
    start = end;
    static elem_t kernel139[256][1024] row_align(1)= {0};
    static elem_t tensor139[256][256] row_align(1)= {0};
    static elem_t result139[256][1024] row_align(1)= {0};
    static elem_t inter_result139[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result137,1024,  inter_result139);


    /* matmul number: 139 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor139, kernel139, inter_result139, result139,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_139");
    // verbose(139,tensor139,kernel139,result139)
    /* end of matmul number: 139 */

    end = read_cycles();
    cycles[139] = end-start;
    start = end;
    static elem_t kernel140[512][256] row_align(1)= {0};
    static elem_t tensor140[256][512] row_align(1)= {0};
    static elem_t result140[256][256] row_align(1)= {0};


    /* matmul number: 140 */

    tiled_matmul_compare(256, 256, 512,    // dimensions
    tensor140, kernel140, NULL, result140,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_140");
    // verbose(140,tensor140,kernel140,result140)
    /* end of matmul number: 140 */

    end = read_cycles();
    cycles[140] = end-start;
    start = end;
    static elem_t kernel141[2304][256] row_align(1)= {0};
    static elem_t tensor141[256][2304] row_align(1)= {0};
    static elem_t result141[256][256] row_align(1)= {0};
    tensor_reshape(256,result140,14, 14, 3, 1, tensor141);


    /* matmul number: 141 */

    tiled_matmul_compare(256, 256, 2304,    // dimensions
    tensor141, kernel141, NULL, result141,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_141");
    // verbose(141,tensor141,kernel141,result141)
    /* end of matmul number: 141 */

    end = read_cycles();
    cycles[141] = end-start;
    start = end;
    static elem_t kernel142[256][1024] row_align(1)= {0};
    static elem_t tensor142[256][256] row_align(1)= {0};
    static elem_t result142[256][1024] row_align(1)= {0};
    static elem_t inter_result142[256][1024] row_align(1)= {0};
    rocket_zeropad(256,256,result140,1024,  inter_result142);


    /* matmul number: 142 */

    tiled_matmul_compare(256, 1024, 256,    // dimensions
    tensor142, kernel142, inter_result142, result142,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_142");
    // verbose(142,tensor142,kernel142,result142)
    /* end of matmul number: 142 */

    end = read_cycles();
    cycles[142] = end-start;
    start = end;
    static elem_t kernel143[1024][512] row_align(1)= {0};
    static elem_t tensor143[256][1024] row_align(1)= {0};
    static elem_t result143[256][512] row_align(1)= {0};


    /* matmul number: 143 */

    tiled_matmul_compare(256, 512, 1024,    // dimensions
    tensor143, kernel143, NULL, result143,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_143");
    // verbose(143,tensor143,kernel143,result143)
    /* end of matmul number: 143 */

    end = read_cycles();
    cycles[143] = end-start;
    start = end;
    static elem_t kernel144[4608][512] row_align(1)= {0};
    static elem_t tensor144[64][4608] row_align(1)= {0};
    static elem_t result144[64][512] row_align(1)= {0};
    tensor_reshape(512,result143,14, 14, 3, 2, tensor144);


    /* matmul number: 144 */

    tiled_matmul_compare(64, 512, 4608,    // dimensions
    tensor144, kernel144, NULL, result144,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_144");
    // verbose(144,tensor144,kernel144,result144)
    /* end of matmul number: 144 */

    end = read_cycles();
    cycles[144] = end-start;
    start = end;
    static elem_t kernel145[512][2048] row_align(1)= {0};
    static elem_t tensor145[64][512] row_align(1)= {0};
    static elem_t result145[64][2048] row_align(1)= {0};
    static elem_t inter_result145[64][2048] row_align(1)= {0};
    rocket_fix_strided_dimension(16,512,result143,2048,  inter_result145);


    /* matmul number: 145 */

    tiled_matmul_compare(64, 2048, 512,    // dimensions
    tensor145, kernel145, inter_result145, result145,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_145");
    // verbose(145,tensor145,kernel145,result145)
    /* end of matmul number: 145 */

    end = read_cycles();
    cycles[145] = end-start;
    start = end;
    static elem_t kernel146[1024][512] row_align(1)= {0};
    static elem_t tensor146[64][1024] row_align(1)= {0};
    static elem_t result146[64][512] row_align(1)= {0};


    /* matmul number: 146 */

    tiled_matmul_compare(64, 512, 1024,    // dimensions
    tensor146, kernel146, NULL, result146,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_146");
    // verbose(146,tensor146,kernel146,result146)
    /* end of matmul number: 146 */

    end = read_cycles();
    cycles[146] = end-start;
    start = end;
    static elem_t kernel147[4608][512] row_align(1)= {0};
    static elem_t tensor147[64][4608] row_align(1)= {0};
    static elem_t result147[64][512] row_align(1)= {0};
    tensor_reshape(512,result146,7, 7, 3, 1, tensor147);


    /* matmul number: 147 */

    tiled_matmul_compare(64, 512, 4608,    // dimensions
    tensor147, kernel147, NULL, result147,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_147");
    // verbose(147,tensor147,kernel147,result147)
    /* end of matmul number: 147 */

    end = read_cycles();
    cycles[147] = end-start;
    start = end;
    static elem_t kernel148[512][2048] row_align(1)= {0};
    static elem_t tensor148[64][512] row_align(1)= {0};
    static elem_t result148[64][2048] row_align(1)= {0};
    static elem_t inter_result148[64][2048] row_align(1)= {0};
    rocket_zeropad(64,512,result146,2048,  inter_result148);


    /* matmul number: 148 */

    tiled_matmul_compare(64, 2048, 512,    // dimensions
    tensor148, kernel148, inter_result148, result148,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_148");
    // verbose(148,tensor148,kernel148,result148)
    /* end of matmul number: 148 */

    end = read_cycles();
    cycles[148] = end-start;
    start = end;
    static elem_t kernel149[1024][512] row_align(1)= {0};
    static elem_t tensor149[64][1024] row_align(1)= {0};
    static elem_t result149[64][512] row_align(1)= {0};


    /* matmul number: 149 */

    tiled_matmul_compare(64, 512, 1024,    // dimensions
    tensor149, kernel149, NULL, result149,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_149");
    // verbose(149,tensor149,kernel149,result149)
    /* end of matmul number: 149 */

    end = read_cycles();
    cycles[149] = end-start;
    start = end;
    static elem_t kernel150[4608][512] row_align(1)= {0};
    static elem_t tensor150[64][4608] row_align(1)= {0};
    static elem_t result150[64][512] row_align(1)= {0};
    tensor_reshape(512,result149,7, 7, 3, 1, tensor150);


    /* matmul number: 150 */

    tiled_matmul_compare(64, 512, 4608,    // dimensions
    tensor150, kernel150, NULL, result150,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_150");
    // verbose(150,tensor150,kernel150,result150)
    /* end of matmul number: 150 */

    end = read_cycles();
    cycles[150] = end-start;
    start = end;
    static elem_t kernel151[512][2048] row_align(1)= {0};
    static elem_t tensor151[64][512] row_align(1)= {0};
    static elem_t result151[64][2048] row_align(1)= {0};
    static elem_t inter_result151[64][2048] row_align(1)= {0};
    rocket_zeropad(64,512,result149,2048,  inter_result151);


    /* matmul number: 151 */

    tiled_matmul_compare(64, 2048, 512,    // dimensions
    tensor151, kernel151, inter_result151, result151,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_151");
    // verbose(151,tensor151,kernel151,result151)
    /* end of matmul number: 151 */

    end = read_cycles();
    cycles[151] = end-start;
    start = end;


    /* AVG Pool: 152 */

    static elem_t tensor153[64][2048] row_align(1)= {0};
    avg_pool7(2048,result151,tensor153);
    end = read_cycles();
    cycles[152] = end-start;
    start = end;


    static elem_t kernel153[2048][1024] row_align(1)= {0};
    static elem_t result153[64][1024] row_align(1)= {0};


    /* matmul number: 153 */

    tiled_matmul_compare(64, 1024, 2048,    // dimensions
    tensor153, kernel153, NULL, result153,      // addresses
    RELU, 0, 0, 0,              // activation, shift, r6_shift, full_width_bias
    tiled_matmul_type, compare, "layer_153");
    // verbose(153,tensor153,kernel153,result153)
    /* end of matmul number: 153 */

    end = read_cycles();
    cycles[153] = end-start;
    start = end;

    unsigned long overall_cycles = 0;
    for(int cyc = 0; cyc < 154 ; cyc++){
        overall_cycles += cycles[cyc];
    }
    for(int cyc = 0; cyc < 154 ; cyc++){
        printf("Cycles taken in layer %d: %lu, %lf\n", cyc,cycles[cyc],cycles[cyc]*100.0/(1.0*overall_cycles));
    }
    printf("Overall cycles taken: %lu\n",overall_cycles);


    return 0;
}

