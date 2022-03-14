#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "attention_params.h"

#define E_LUT_SIZE 256
#define E_LUT_ABS_RANGE 2
static const float e_lookup[257] = {0.13640,0.13854,0.14073,0.14294,0.14519,0.14748,0.14980,0.15216,0.15456,0.15699,0.15946,0.16198,0.16453,0.16712,0.16975,0.17242,0.17514,0.17789,0.18070,0.18354,0.18643,0.18937,0.19235,0.19538,0.19846,0.20158,0.20476,0.20798,0.21126,0.21458,0.21796,0.22139,0.22488,0.22842,0.23202,0.23567,0.23938,0.24315,0.24698,0.25087,0.25482,0.25884,0.26291,0.26705,0.27126,0.27553,0.27987,0.28428,0.28875,0.29330,0.29792,0.30261,0.30737,0.31222,0.31713,0.32213,0.32720,0.33235,0.33759,0.34290,0.34830,0.35379,0.35936,0.36502,0.37076,0.37660,0.38253,0.38856,0.39468,0.40089,0.40721,0.41362,0.42013,0.42675,0.43347,0.44029,0.44723,0.45427,0.46142,0.46869,0.47607,0.48357,0.49118,0.49892,0.50678,0.51476,0.52286,0.53110,0.53946,0.54795,0.55658,0.56535,0.57425,0.58329,0.59248,0.60181,0.61129,0.62091,0.63069,0.64062,0.65071,0.66096,0.67137,0.68194,0.69268,0.70359,0.71467,0.72592,0.73735,0.74897,0.76076,0.77274,0.78491,0.79727,0.80982,0.82258,0.83553,0.84869,0.86205,0.87563,0.88942,0.90342,0.91765,0.93210,0.94678,0.96169,0.97684,0.99222,1.00784,1.02371,1.03984,1.05621,1.07284,1.08974,1.10690,1.12433,1.14204,1.16002,1.17829,1.19684,1.21569,1.23484,1.25428,1.27403,1.29410,1.31448,1.33518,1.35620,1.37756,1.39925,1.42129,1.44367,1.46640,1.48950,1.51295,1.53678,1.56098,1.58556,1.61053,1.63589,1.66165,1.68782,1.71440,1.74140,1.76882,1.79667,1.82497,1.85371,1.88290,1.91255,1.94267,1.97326,2.00434,2.03590,2.06796,2.10053,2.13360,2.16720,2.20133,2.23600,2.27121,2.30698,2.34331,2.38021,2.41769,2.45576,2.49444,2.53372,2.57362,2.61415,2.65531,2.69713,2.73960,2.78274,2.82657,2.87108,2.91629,2.96222,3.00886,3.05625,3.10438,3.15326,3.20292,3.25336,3.30459,3.35663,3.40949,3.46318,3.51772,3.57311,3.62938,3.68654,3.74459,3.80356,3.86346,3.92430,3.98610,4.04887,4.11263,4.17739,4.24318,4.31000,4.37787,4.44681,4.51684,4.58797,4.66022,4.73361,4.80815,4.88387,4.96078,5.03890,5.11825,5.19885,5.28072,5.36388,5.44835,5.53415,5.62130,5.70982,5.79974,5.89107,5.98384,6.07807,6.17379,6.27101,6.36976,6.47007,6.57196,6.67546,6.78058,6.88736,6.99582,7.10599,7.21789,7.33155};

static inline void print_float(float x);

#define ABS(x) ((x) < 0 ? -(x) : (x))

static inline void print_float(float x) {
    printf(" %c%d.%05d ", x < 0 ? 45 : 32, (int) ABS(x), (int) ((ABS(x) - (float) ((int) ABS(x))) * 100000));
}

void single_head_attn(struct MultiheadAttnParams params,
    const elem_t *q, const elem_t *k, const elem_t *v, elem_t *result) {

    elem_t temp[SEQ_LEN][SEQ_LEN];
    int head_dim = params.embed_dim / params.num_heads;

    // Q @ K.T
    tiled_matmul_auto(SEQ_LEN, SEQ_LEN, head_dim,
        q, k, 0, temp, params.embed_dim * 3, params.embed_dim * 3, 1, SEQ_LEN,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 1.f / sqrt(head_dim), 0, true,
        false, true, false, false, 0, WS);

    // softmax
    for (int i = 0; i < SEQ_LEN; i++) {
        float e_x[SEQ_LEN];
        float *x = &temp[i];
        float e_x_sum = 0;
        for (int j = 0; j < SEQ_LEN; j++) {
            // lut:
            // float idx_float = (x[j] + 2) * 64;
            // if (idx_float < 0) {
            //     e_x[j] = e_lookup[0];
            // } else if (idx_float >= E_LUT_SIZE) {
            //     e_x[j] = e_lookup[E_LUT_SIZE];
            // } else {
            //     int idx_int = (int) idx_float;
            //     float idx_frac = idx_float - idx_int;
            //     e_x[j] = e_lookup[idx_int] * (1 - idx_frac) + e_lookup[idx_int + 1] * idx_frac;
            // }
            
            // taylor series
            float x2 = x[j] * x[j];
            e_x[j] = 1.f + x[j] + x2 / 2.f + x[j] * x2 / 6.f + x2 * x2 / 24.f + x2 * x2 * x[j] / 120.f;
            e_x_sum += e_x[j];
        }
        float e_x_factor = 1 / e_x_sum;

        for (int j = 0; j < SEQ_LEN; j++) {
            x[j] = e_x[j] * e_x_factor;
        }
    }

    #ifdef VERBOSE
    printf("softmax=\n");
    for (size_t i = 0; i < SEQ_LEN; i++) {
        for (size_t j = 0; j < SEQ_LEN; j++) print_float(temp[i][j]);
        printf("\n");
    }
    #endif

    // [sm(Q @ K.T)] @ V
    tiled_matmul_auto(SEQ_LEN, head_dim, SEQ_LEN,
        temp, v, 0, result, SEQ_LEN, params.embed_dim * 3, 1, params.embed_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, true,
        false, false, false, false, 0, WS);

}

void layer_norm(struct LayerNormLayer layer, elem_t *x_) {
    elem_t bias[SEQ_LEN];

    elem_t (*x)[layer.embed_dim] = (elem_t (*)[layer.embed_dim]) x_;

    elem_t sum_vec[layer.embed_dim];
    for (int i = 0; i < layer.embed_dim; i++) {
        sum_vec[i] = 1;
    }

    // calculate mean
    tiled_matmul_auto(SEQ_LEN, 1, layer.embed_dim,
        x, sum_vec, 0, bias, layer.embed_dim, 1, 1, 1,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 1.f / (float) layer.embed_dim, 0, true,
        false, false, false, false, 0, WS);

    for (int i = 0; i < SEQ_LEN; i++) {
        acc_t squared_sum = 0;
        // demean
        for (int j = 0; j < layer.embed_dim; j++) {
            x[i][j] -= bias[i];
            squared_sum += x[i][j] * x[i][j];
        }
        // normalize
        acc_t coef = 1.f / sqrt(squared_sum / ((float) layer.embed_dim) + layer.eps);
        for (int j = 0; j < layer.embed_dim; j++) {
            x[i][j] *= coef;
        }
    }

    #ifndef VERBOSE
    printf("\nnormalized:\n\n");
    for (size_t i = 0; i < SEQ_LEN; i++) {
        for (size_t j = 0; j < layer.embed_dim; j++) print_float(x[i][j]);
        printf("\n");
    }
    #endif

    // affine transformation
    elem_t scale_mat[layer.embed_dim][layer.embed_dim];
    memset(scale_mat, 0, layer.embed_dim * layer.embed_dim * sizeof(elem_t));
    for (int i = 0; i < layer.embed_dim; i++) {
        scale_mat[i][i] = layer.weight[i];
    }

    tiled_matmul_auto(SEQ_LEN, layer.embed_dim, layer.embed_dim,
        x, scale_mat, layer.bias, x, layer.embed_dim, layer.embed_dim, 1, layer.embed_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, true,
        false, false, false, false, 0, WS);

    #ifdef VERBOSE
    printf("\nlayer norm out:\n\n");
    for (size_t i = 0; i < SEQ_LEN; i++) {
        for (size_t j = 0; j < layer.embed_dim; j++) print_float(x[i][j]);
        printf("\n");
    }
    #endif
}

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type = WS;

    if (argc < 2) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s matmul_option\n  matmul_option may be 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    elem_t attn_out[SEQ_LEN][self_attn.embed_dim];
    elem_t in_projected[SEQ_LEN][self_attn.embed_dim * 3];

    tiled_matmul_auto(SEQ_LEN, self_attn.embed_dim * 3, self_attn.embed_dim,
        in_seq, in_proj_weight, in_proj_bias, in_projected,
        self_attn.embed_dim, self_attn.embed_dim, 1, self_attn.embed_dim * 3,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, true,
        false, true, false, false, 0, WS);
        
    printf("projected:\n");
    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < self_attn.embed_dim * 3; j++) print_float(in_projected[i][j]);
        printf("\n");
    }

    size_t weight_offset = self_attn.embed_dim;
    for (int i = 0; i < self_attn.num_heads; i++) {
        int head_dim = self_attn.embed_dim / self_attn.num_heads;
        size_t head_offset = head_dim * i;

        single_head_attn(self_attn,
            &in_projected[0][head_offset],
            &in_projected[0][self_attn.embed_dim + head_offset],
            &in_projected[0][self_attn.embed_dim * 2 + head_offset],
            &attn_out[0][head_offset]);
    }

    tiled_matmul_auto(SEQ_LEN, self_attn.embed_dim, self_attn.embed_dim,
        attn_out, out_proj_weight, out_proj_bias, attn_out,
        self_attn.embed_dim, self_attn.embed_dim, 1, self_attn.embed_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, true,
        false, true, false, false, 0, WS);
        
    printf("attention out:\n");

    for (size_t i = 0; i < SEQ_LEN; i++) {
        for (size_t j = 0; j < self_attn.embed_dim; j++) print_float(attn_out[i][j]);
        printf("\n");
    }

    // resadd1
    tiled_resadd_auto(SEQ_LEN, EMBED_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        in_seq, attn_out, in_seq, false, WS);

    elem_t (*ff_out)[EMBED_DIM] = (elem_t (*)[EMBED_DIM]) attn_out; // reuse same size
    layer_norm(norm1, in_seq);

    elem_t ff_hidden[SEQ_LEN][FF_DIM];
    
    // ff_linear_1
    tiled_matmul_auto(SEQ_LEN, FF_DIM, EMBED_DIM,
        in_seq, ff_linear1_weight, ff_linear1_bias, ff_hidden,
        EMBED_DIM, EMBED_DIM, 1, FF_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        RELU, ACC_SCALE_IDENTITY, 0, true,
        false, true, false, false, 0, WS);

    printf("linear1 out:\n");
    for (size_t i = 0; i < SEQ_LEN; i++) {
        for (size_t j = 0; j < FF_DIM; j++) print_float(ff_hidden[i][j]);
        printf("\n");
    }

    // ff_linear_2
    tiled_matmul_auto(SEQ_LEN, EMBED_DIM, FF_DIM,
        ff_hidden, ff_linear2_weight, ff_linear2_bias, ff_out,
        FF_DIM, FF_DIM, 1, EMBED_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, true,
        false, true, false, false, 0, WS);

    printf("linear2 out:\n");
    for (size_t i = 0; i < SEQ_LEN; i++) {
        for (size_t j = 0; j < EMBED_DIM; j++) print_float(ff_out[i][j]);
        printf("\n");
    }

    // resadd2
    tiled_resadd_auto(SEQ_LEN, EMBED_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        in_seq, ff_out, in_seq, false, WS);
    
    layer_norm(norm2, in_seq);
    
    printf("final out:\n");
    for (size_t i = 0; i < SEQ_LEN; i++) {
        for (size_t j = 0; j < self_attn.embed_dim; j++) print_float(in_seq[i][j]);
        printf("\n");
    }
}
