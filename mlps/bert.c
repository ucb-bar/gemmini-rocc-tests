#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// bert base
#define SEQ_LEN 128
#define HIDDEN_DIM 768
#define NUM_HEAD 12
#define EXPANSION 3072

#define EXPANSION_STRIDE (EXPANSION + (DIM*MAX_BLOCK_LEN))
#define HIDDEN_STRIDE (HIDDEN_DIM + (DIM*MAX_BLOCK_LEN))
#define SEQ_STRIDE (SEQ_LEN + (DIM*MAX_BLOCK_LEN))

// Note: For self-attention, "enc_out" should be the same as "input".
// Note: "compression_factor" should be 1 for most use cases.
void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        //int compression_factor,
        int hidden_stride, int expansion_stride, int seq_stride,

        elem_t * input, elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        elem_t * Wq, elem_t * Wk, elem_t * Wv, elem_t * Wo,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf) //, elem_t * out_buf)
{
    const int hidden_dim_compressed = hidden_dim; // / compression_factor;
    const int hidden_dim_per_head = hidden_dim_compressed / num_heads;

//    printf("starting attention \n");

    // Q = Wq * input
    // K = Wk * enc_out
    // V = Wv * enc_out
    const int qkv_matmuls_n = 3;
   
    for (int i = 0; i < qkv_matmuls_n; i++) {
        elem_t * qkv_weights[] = {Wq, Wk, Wv};
        elem_t * qkv_ins[] = {input, enc_out, enc_out};
        elem_t * qkv_outs[] = {Q_buf, K_buf, V_buf};

        elem_t * qkv_w = qkv_weights[i];
        elem_t * qkv_in = qkv_ins[i];
        elem_t * qkv_out = qkv_outs[i];

        tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
            /*A=*/ qkv_in, /*B=*/ qkv_w,
            /*D=*/ NULL, /*C=*/ qkv_out,
            /*stride_A=*/hidden_stride, /*stride_B=*/hidden_stride, /*stride_D=*/0, /*stride_C=*/hidden_stride,
            false, false, false, false,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);
    }

//    printf("done Q K V\n");
     // attn = Q * K
    // attn = softmax(attn)
    for (int head = 0; head < num_heads; head++) {
        elem_t * A = Q_buf + head * hidden_dim_per_head;
        elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf + head * seq_len * seq_stride;

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/hidden_stride, /*stride_B=*/hidden_stride, /*stride_D=*/0, /*stride_C=*/seq_stride,
            false, false, false, false,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            ///*SOFTMAX*/ LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*SOFTMAX*/ NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ true,
            false, false,
            0,
            WS);
//        printf("attention head %d\n", head);
    }

//    printf("done attn \n");


    // out = attn * V // V = out
    for (int head = 0; head < num_heads; head++) {
        elem_t * A = attn_buf + head * seq_len * seq_stride;
        elem_t * B = V_buf + head * hidden_dim_per_head;
        elem_t * C = out + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/seq_stride, /*stride_B=*/hidden_stride, /*stride_D=*/0, /*stride_C=*/hidden_stride,
            false, false, false, false,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);
    }

//    printf("done attn * V\n");

    // out = out * Wo
    // out = LN(out)
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        /*A=*/ out, /*B=*/ Wo,
        /*D=*/ NULL, /*C=*/ out,
        /*stride_A=*/hidden_stride, /*stride_B=*/hidden_stride, /*stride_D=*/0, /*stride_C=*/hidden_stride,
        false, false, false, false,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        //LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ false,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        WS);

//    printf("done out * Wo\n");

    // input = out + input
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        hidden_stride,
        false, false, false, 
        input,
        out,
        resadd_out,
        /*relu=*/ false,
        WS);

//    printf("done resadd\n");
}

void ffn(int hidden_dim, int expansion_dim, int seq_len, 
        int hidden_stride, int expansion_stride, 
        elem_t * input, elem_t * out,
        elem_t * ff1_w, elem_t * ff2_w,
        acc_t * ff1_b, acc_t * ff2_b,

        elem_t * out_buf)
{
    // out = FF1(input)
    // out = GELU(out)
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        /*A=*/ input, /*B=*/ ff1_w,
        /*D=*/ ff1_b, /*C=*/ out_buf,
        /*stride_A=*/hidden_stride, /*stride_B=*/expansion_stride, /*stride_D=*/expansion_stride, /*stride_C=*/expansion_stride,
        false, false, false, false,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        //IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        WS);

    gemmini_fence();

    // out = FF2(out)
    // out = LN(out)

    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim,
        /*A=*/ out_buf, /*B=*/ ff2_w,
        /*D=*/ ff2_b, /*C=*/ out,
        /*stride_A=*/expansion_stride, /*stride_B=*/hidden_stride, /*stride_D=*/expansion_stride, /*stride_C=*/expansion_stride,
        false, false, false, false,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0, /*repeating_bias=*/ true,
       // LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0, /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        WS);

    gemmini_fence();

    // out = out + input
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        hidden_stride,
        false, false, false,
        out,
        input,
        out,
        /*relu=*/ false,
        WS);

    gemmini_fence();
}

// Note: If "enc_out == NULL", then this will act as an encoder layer.
//   Otherwise, it will act as a decoder layer. If this is an encoder layer,
//   then "cross_num_heads" and all the "W*_cross" args are ignored.
uint64_t encoder_decoder(
        int hidden_dim, int expansion_dim, int num_heads, //int cross_num_heads,
        int seq_len, //int compression_factor, 
        int hidden_stride, int expansion_stride, int seq_stride,

        elem_t * input, elem_t * enc_out, elem_t * out,
        elem_t * Wq, elem_t * Wk, elem_t * Wv, elem_t * Wo,
        //elem_t * Wq_cross, elem_t * Wk_cross, elem_t * Wv_cross, elem_t * Wo_cross,
        elem_t * ff1_w, elem_t * ff2_w,
        acc_t * ff1_b, acc_t * ff2_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf,
        elem_t * resadd1_buf, elem_t * resadd2_buf)
{
    const bool is_encoder = enc_out == NULL;

    uint64_t start = read_cycles();

    attention(hidden_dim, expansion_dim, num_heads, seq_len,// compression_factor, 
        hidden_stride, expansion_stride, seq_stride,
        input, input,
        out, resadd1_buf,
        Wq, Wk, Wv, Wo,
        Q_buf, K_buf, V_buf,
        attn_buf);//, out_buf);
/*
    if (!is_encoder) {
        attention(hidden_dim, expansion_dim, cross_num_heads, seq_len, compression_factor,
            resadd1_buf, enc_out,
            out, resadd2_buf,
            Wq_cross, Wk_cross, Wv_cross, Wo_cross,
            Q_buf, K_buf, V_buf,
            attn_buf, out_buf);
    }
*/
    ffn(hidden_dim, expansion_dim, seq_len, 
        hidden_stride, expansion_stride, 
        is_encoder ? resadd1_buf : resadd2_buf,
        out,
        ff1_w, ff2_w,
        ff1_b, ff2_b,
        out_buf);

    uint64_t end = read_cycles();

    return end - start;
}

static elem_t input[SEQ_LEN][HIDDEN_STRIDE] row_align(MAX_BLOCK_LEN);
//static elem_t output[SEQ_LEN][HIDDEN_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t enc_out[SEQ_LEN][HIDDEN_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t Wqkvo[4][HIDDEN_DIM][HIDDEN_STRIDE] row_align(MAX_BLOCK_LEN); 
static elem_t ff_w[2][HIDDEN_DIM*EXPANSION] row_align(MAX_BLOCK_LEN); 
static acc_t ff1_b[EXPANSION] row_align(MAX_BLOCK_LEN_ACC); 
static acc_t ff2_b[HIDDEN_DIM] row_align(MAX_BLOCK_LEN_ACC);
static elem_t QKV_buf[2][SEQ_LEN][HIDDEN_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t attn_buf[NUM_HEAD][SEQ_LEN][SEQ_STRIDE] row_align(MAX_BLOCK_LEN);
static elem_t out_buf[SEQ_LEN][EXPANSION_STRIDE] row_align(MAX_BLOCK_LEN);
//static elem_t resadd1_buf[SEQ_LEN][HIDDEN_STRIDE] row_align(MAX_BLOCK_LEN);
//static elem_t resadd2_buf[SEQ_LEN][HIDDEN_STRIDE] row_align(MAX_BLOCK_LEN);


// for now, encoder only
// ToDo: decoder
void enc_dec_func(bool is_encoder, int hidden_dim, int expansion_dim, int num_heads, int seq_len, int hidden_stride, int expansion_stride, int seq_stride){
    //elem_t * output = input;
    encoder_decoder( 
            hidden_dim, expansion_dim, num_heads, seq_len, 
            hidden_stride, expansion_stride, seq_stride,
            (elem_t*) input, NULL, (elem_t*) input,  // is_encoder ? NULL : enc_out, output
            (elem_t*) Wqkvo[0], (elem_t*) Wqkvo[1], (elem_t*) Wqkvo[2], (elem_t*) Wqkvo[3],
            (elem_t*) ff_w[0], (elem_t*) ff_w[1], 
            (acc_t*) ff1_b, (acc_t*) ff2_b, 
            (elem_t*) QKV_buf[0], (elem_t*) QKV_buf[1], (elem_t*) input,// QKV_buf[2], 
            (elem_t*) attn_buf, (elem_t*) out_buf, 
            (elem_t*) input, (elem_t*) input
            //resadd1_buf, resadd2_buf 
    );
}   
int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enc_dec_func(true, HIDDEN_DIM, EXPANSION, NUM_HEAD, SEQ_LEN, HIDDEN_STRIDE, EXPANSION_STRIDE, SEQ_STRIDE);

    exit(0);
}

