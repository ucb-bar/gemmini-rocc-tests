#include <include/gemmini_params.h>
#include <stdbool.h>

// input

// conv_1
static const elem_t conv_1_w[256][144] row_align(1) = {0};
static const acc_t conv_1_b[144] row_align_acc(1) = {0};
static elem_t conv_1_in[1][256] row_align(1) = {0};
static elem_t conv_1_out[1][144] row_align(1) = {0};
static const struct ConvParams conv_1_params = {.batch_size=4, .in_dim=1, .kernel_size=3, .in_channels=256, .out_channels=144,
        .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=1, .n_patches=1, .patch_size=144, .pool_size=0,
        .pool_stride=0, .pool_padding=0, .out_dim_pooled=1, .output_scale=0.0011489674216136336, .I=1, .J=144, .K=256, .res_scale=(1.0 / (1 << 0))};

// conv_2
static const elem_t conv_2_w[144][96] row_align(1) = {0};
static const acc_t conv_2_b[96] row_align_acc(1) = {0};
static elem_t conv_2_out[1][96] row_align(1) = {0};
static const struct ConvParams conv_2_params = {.batch_size=4, .in_dim=1, .kernel_size=3, .in_channels=144, .out_channels=96,
        .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=1, .n_patches=1, .patch_size=96, .pool_size=0,
        .pool_stride=0, .pool_padding=0, .out_dim_pooled=1, .output_scale=0.001182529958896339, .I=1, .J=96, .K=144, .res_scale=(1.0 / (1 << 0))};

// conv_3
static const elem_t conv_3_w[96][96] row_align(1) = {0};
static const acc_t conv_3_b[96] row_align_acc(1) = {0};
static elem_t conv_3_out[1][96] row_align(1) = {0};
static const struct ConvParams conv_3_params = {.batch_size=4, .in_dim=1, .kernel_size=3, .in_channels=96, .out_channels=96,
        .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=1, .n_patches=1, .patch_size=96, .pool_size=0,
        .pool_stride=0, .pool_padding=0, .out_dim_pooled=1, .output_scale=0.0012923858594149351, .I=1, .J=96, .K=96, .res_scale=(1.0 / (1 << 0))};

// conv_4
static const elem_t conv_4_w[96][96] row_align(1) = {0};
static const acc_t conv_4_b[96] row_align_acc(1) = {0};
static elem_t conv_4_out[1][96] row_align(1) = {0};
static const struct ConvParams conv_4_params = {.batch_size=4, .in_dim=1, .kernel_size=3, .in_channels=96, .out_channels=96,
        .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=1, .n_patches=1, .patch_size=96, .pool_size=0,
        .pool_stride=0, .pool_padding=0, .out_dim_pooled=1, .output_scale=0.0008491309708915651, .I=1, .J=96, .K=96, .res_scale=(1.0 / (1 << 0))};

// conv_5 - clshad
static const elem_t conv_5_w[96][1] row_align(1) = {0};
static const acc_t conv_5_b[1] row_align_acc(1) = {0};
static elem_t conv_5_out[1][1] row_align(1) = {0};
static const struct ConvParams conv_5_params = {.batch_size=4, .in_dim=1, .kernel_size=3, .in_channels=96, .out_channels=1,
        .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=1, .n_patches=1, .patch_size=1, .pool_size=0,
        .pool_stride=0, .pool_padding=0, .out_dim_pooled=1, .output_scale=0.0033772671595215797, .I=1, .J=1, .K=96, .res_scale=(1.0 / (1 << 0))};

// conv_6 - reghead
static const elem_t conv_6_w[96][2] row_align(1) = {0};
static const acc_t conv_6_b[2] row_align_acc(1) = {0};
static elem_t conv_6_out[1][2] row_align(1) = {0};
static const struct ConvParams conv_6_params = {.batch_size=4, .in_dim=1, .kernel_size=3, .in_channels=96, .out_channels=2,
        .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=1, .n_patches=1, .patch_size=2, .pool_size=0,
        .pool_stride=0, .pool_padding=0, .out_dim_pooled=1, .output_scale=0.0004996534553356469, .I=1, .J=2, .K=96, .res_scale=(1.0 / (1 << 0))};

