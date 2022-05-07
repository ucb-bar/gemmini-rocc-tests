#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// Note: For self-attention, "enc_out" should be the same as "input"
void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf)
{
    const int hidden_dim_per_head = hidden_dim / num_heads;

    // Q = Wq * input
    // K = Wk * enc_out
    // V = Wv * enc_out
    const int qkv_matmuls_n = 3;
    for (int i = 0; i < qkv_matmuls_n; i++) {
        const elem_t * qkv_weights[] = {Wq, Wk, Wv};
        const elem_t * qkv_ins[] = {input, enc_out, enc_out};
        elem_t * qkv_outs[] = {Q_buf, K_buf, V_buf};

        const elem_t * qkv_w = qkv_weights[i];
        const elem_t * qkv_in = qkv_ins[i];
        elem_t * qkv_out = qkv_outs[i];

        tiled_matmul_auto(seq_len, hidden_dim, hidden_dim,
            /*A=*/ qkv_in, /*B=*/ qkv_w,
            /*D=*/ NULL, /*C=*/ qkv_out,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);
    }

    gemmini_fence();

    // attn = Q * K
    // attn = softmax(attn)
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = Q_buf + head * hidden_dim_per_head;
        const elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf + head * seq_len * seq_len;

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            /*SOFTMAX*/ LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ true,
            false, false,
            0,
            WS);
    }

    gemmini_fence();

    // out_buf = attn * V
    for (int head = 0; head < num_heads; head++) {
        const elem_t * A = attn_buf + head * seq_len * seq_len;
        const elem_t * B = V_buf + head * hidden_dim_per_head;
        elem_t * C = out_buf + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/seq_len, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            WS);
    }

    gemmini_fence();

    // out = out_buf * Wo
    // out = LN(out)
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim,
        /*A=*/ out_buf, /*B=*/ Wo,
        /*D=*/ NULL, /*C=*/ out,
        /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ false,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        WS);

    gemmini_fence();

    // input = out + input
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        input,
        out,
        resadd_out,
        /*relu=*/ false,
        WS);

    gemmini_fence();
}

void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf)
{
    // out = FF1(input)
    // out = GELU(out)
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        /*A=*/ input, /*B=*/ ff1_w,
        /*D=*/ ff1_b, /*C=*/ out_buf,
        /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        /*repeating_bias=*/ false,
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
        /*stride_A=*/expansion_dim, /*stride_B=*/hidden_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ false,
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
        out,
        input,
        out,
        /*relu=*/ false,
        WS);

    gemmini_fence();
}

// Note: If "enc_out == NULL || enc_out == input", then this will act as an
//   encoder layer. Otherwise, it will act as a decoder layer.
//   If this is an encoder layer, then "cross_num_heads" and all the "W*_cross"
//   args are ignored.
uint64_t encoder_decoder(int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads, int seq_len,
        const elem_t * input, const elem_t * enc_out, elem_t * out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf,
        elem_t * resadd1_buf, elem_t * resadd2_buf)
{
    const bool is_encoder = enc_out == NULL || enc_out == input;

    uint64_t start = read_cycles();

    attention(hidden_dim, expansion_dim, num_heads, seq_len,
        input, input,
        out, resadd1_buf,
        Wq, Wk, Wv, Wo,
        Q_buf, K_buf, V_buf,
        attn_buf, out_buf);

    if (!is_encoder) {
        attention(hidden_dim, expansion_dim, cross_num_heads, seq_len,
            resadd1_buf, enc_out,
            out, resadd2_buf,
            Wq_cross, Wk_cross, Wv_cross, Wo_cross,
            Q_buf, K_buf, V_buf,
            attn_buf, out_buf);
    }

    ffn(hidden_dim, expansion_dim, seq_len,
        is_encoder ? resadd1_buf : resadd2_buf,
        out,
        ff1_w, ff2_w,
        ff1_b, ff2_b,
        out_buf);

    uint64_t end = read_cycles();

    return end - start;
}

#define ENCODER(hidden_dim, expansion_dim, num_heads, seq_len, input, output) ({ \
    static const elem_t Wqkvo[4][hidden_dim][hidden_dim]; \
    static const elem_t ff_w[2][hidden_dim*expansion_dim]; \
    static const acc_t ff1_b[seq_len][expansion_dim]; \
    static const acc_t ff2_b[seq_len][hidden_dim]; \
    \
    static elem_t QKV_buf[3][seq_len][hidden_dim];\
    static elem_t attn_buf[num_heads][seq_len][seq_len];\
    static elem_t out_buf[seq_len][expansion_dim];\
    static elem_t resadd1_buf[seq_len][hidden_dim];\
    static elem_t resadd2_buf[seq_len][hidden_dim];\
    \
    uint64_t cycles = encoder_decoder(hidden_dim, expansion_dim, num_heads, num_heads, seq_len, \
            input, NULL, output, \
            Wqkvo[0], Wqkvo[1], Wqkvo[2], Wqkvo[3],\
            Wqkvo[0], Wqkvo[1], Wqkvo[2], Wqkvo[3],\
            ff_w[0], ff_w[1], \
            ff1_b, ff2_b, \
            \
            QKV_buf[0], QKV_buf[1], QKV_buf[2], \
            attn_buf, out_buf, \
            resadd1_buf, resadd2_buf \
    ); \
    \
    cycles; \
})

#define DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, input, enc_out, output) ({ \
    static const elem_t Wqkvo[4][hidden_dim][hidden_dim]; \
    static const elem_t Wqkvo_cross[4][hidden_dim][hidden_dim]; \
    static const elem_t ff_w[2][hidden_dim*expansion_dim]; \
    static const acc_t ff1_b[seq_len][expansion_dim]; \
    static const acc_t ff2_b[seq_len][hidden_dim]; \
    \
    static elem_t QKV_buf[3][seq_len][hidden_dim];\
    static elem_t attn_buf[num_heads][seq_len][seq_len];\
    static elem_t out_buf[seq_len][expansion_dim];\
    static elem_t resadd1_buf[seq_len][hidden_dim];\
    static elem_t resadd2_buf[seq_len][hidden_dim];\
    \
    uint64_t cycles = encoder_decoder(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, \
            input, enc_out, output, \
            Wqkvo[0], Wqkvo[1], Wqkvo[2], Wqkvo[3],\
            Wqkvo_cross[0], Wqkvo_cross[1], Wqkvo_cross[2], Wqkvo_cross[3],\
            ff_w[0], ff_w[1], \
            ff1_b, ff2_b, \
            \
            QKV_buf[0], QKV_buf[1], QKV_buf[2], \
            attn_buf, out_buf, \
            resadd1_buf, resadd2_buf \
    ); \
    \
    cycles; \
})

#define PRINT_ENCODER(name, hidden_dim, expansion_dim, num_heads, seq_len) { \
    static const elem_t input[seq_len][hidden_dim]; \
    static elem_t output[seq_len][hidden_dim]; \
    \
    uint64_t cycles = ENCODER(hidden_dim, expansion_dim, num_heads, seq_len, input, output); \
    \
    printf("%s stats: encoder, hidden_dim=%d, expansion_dim=%d, num_heads=%d, seq_len=%d\n", \
            name, hidden_dim, expansion_dim, num_heads, seq_len); \
    printf("%s cycles: %llu\n\n", name, cycles); \
}

#define PRINT_DECODER(name, hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len) { \
    static const elem_t input[seq_len][hidden_dim]; \
    static const elem_t enc_out[seq_len][hidden_dim]; \
    static elem_t output[seq_len][hidden_dim]; \
    \
    uint64_t cycles = DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, input, enc_out, output); \
    \
    printf("%s stats: decoder, hidden_dim=%d, expansion_dim=%d, num_heads=%d, cross_num_heads=%d, seq_len=%d\n", \
            name, hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len); \
    printf("%s cycles: %llu\n\n", name, cycles); \
}

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    PRINT_ENCODER("transformer-small",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/4, /*seq_len=*/128);

    PRINT_ENCODER("bert-base",
            /*hidden_dim=*/768, /*expansion_dim=*/3072, /*num_heads=*/12, /*seq_len=*/128);


    PRINT_ENCODER("sehoon-0-enc-0",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/4, /*seq_len=*/128);

    PRINT_DECODER("sehoon-0-dec-0",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/4, /*cross_num_heads=*/4, /*seq_len=*/128);


    PRINT_ENCODER("sehoon-1-enc-0",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/2, /*seq_len=*/128);

    PRINT_ENCODER("sehoon-1-enc-1",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/4, /*seq_len=*/128);

    PRINT_DECODER("sehoon-1-dec-0",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/2, /*cross_num_heads=*/4, /*seq_len=*/128);

    PRINT_DECODER("sehoon-1-dec-1",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/2, /*cross_num_heads=*/2, /*seq_len=*/128);

    PRINT_DECODER("sehoon-1-dec-2",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/2, /*cross_num_heads=*/2, /*seq_len=*/128);


    PRINT_ENCODER("sehoon-2-enc-0",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/2, /*seq_len=*/128);

    PRINT_ENCODER("sehoon-2-enc-1",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/2, /*seq_len=*/128);

    PRINT_ENCODER("sehoon-2-enc-2",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/4, /*seq_len=*/128);

    PRINT_DECODER("sehoon-2-dec-0",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/4, /*cross_num_heads=*/2, /*seq_len=*/128);

    PRINT_DECODER("sehoon-2-dec-1",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/2, /*cross_num_heads=*/4, /*seq_len=*/128);

    PRINT_DECODER("sehoon-2-dec-2",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/2, /*cross_num_heads=*/2, /*seq_len=*/128);

    PRINT_DECODER("sehoon-2-dec-3",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/2, /*cross_num_heads=*/2, /*seq_len=*/128);


    PRINT_ENCODER("sehoon-3-enc-0",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/2, /*seq_len=*/128);

    PRINT_ENCODER("sehoon-3-enc-1",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/2, /*seq_len=*/128);

    PRINT_ENCODER("sehoon-3-enc-2",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/4, /*seq_len=*/128);

    PRINT_DECODER("sehoon-3-dec-0",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/4, /*cross_num_heads=*/2, /*seq_len=*/128);

    PRINT_DECODER("sehoon-3-dec-1",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/2, /*cross_num_heads=*/4, /*seq_len=*/128);

    PRINT_DECODER("sehoon-3-dec-2",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/2, /*cross_num_heads=*/2, /*seq_len=*/128);


    PRINT_ENCODER("sehoon-4-enc-0",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/2, /*seq_len=*/128);

    PRINT_ENCODER("sehoon-4-enc-1",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/2, /*seq_len=*/128);

    PRINT_ENCODER("sehoon-4-enc-2",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/4, /*seq_len=*/128);

    PRINT_DECODER("sehoon-4-dec-0",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/4, /*cross_num_heads=*/4, /*seq_len=*/128);

    PRINT_DECODER("sehoon-4-dec-1",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/2, /*cross_num_heads=*/4, /*seq_len=*/128);

    PRINT_DECODER("sehoon-4-dec-2",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/2, /*cross_num_heads=*/2, /*seq_len=*/128);

    PRINT_DECODER("sehoon-4-dec-3",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/4, /*cross_num_heads=*/2, /*seq_len=*/128);


    PRINT_ENCODER("sehoon-5-enc-0",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/4, /*seq_len=*/128);

    PRINT_ENCODER("sehoon-5-enc-1",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/4, /*seq_len=*/128);

    PRINT_ENCODER("sehoon-5-enc-2",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/2, /*seq_len=*/128);

    PRINT_ENCODER("sehoon-5-enc-3",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/2, /*seq_len=*/128);

    PRINT_DECODER("sehoon-5-dec-0",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/4, /*cross_num_heads=*/4, /*seq_len=*/128);

    PRINT_DECODER("sehoon-5-dec-1",
            /*hidden_dim=*/512, /*expansion_dim=*/512, /*num_heads=*/4, /*cross_num_heads=*/4, /*seq_len=*/128);

    PRINT_DECODER("sehoon-5-dec-2",
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/2, /*cross_num_heads=*/2, /*seq_len=*/128);

    exit(0);
}

