#ifndef TRANSFORMERS_H
#define TRANSFORMERS_H

#include <include/gemmini_params.h>

struct MultiheadAttnLayer {
    const int embed_dim;
    const int ff_dim;
    const int num_heads;
    const elem_t *in_proj_weight;
    const elem_t *in_proj_bias;
    const elem_t *out_proj_weight;
    const elem_t *out_proj_bias;
};

struct LayerNormLayer {
    const int embed_dim;
    const float eps;
    const elem_t *weight;
    const elem_t *bias;
};

struct LinearLayer {
    const int input_dim;
    const int output_dim;
    const elem_t *weight;
    const elem_t *bias;
};

struct EncoderLayer {
    const int embed_dim;
    const int ff_dim;
    const int num_heads;
    struct MultiheadAttnLayer *attn;
    struct LayerNormLayer *norm1;
    struct LayerNormLayer *norm2;
    struct LinearLayer *linear1;
    struct LinearLayer *linear2;
};

#endif