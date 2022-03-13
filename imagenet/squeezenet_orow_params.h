#ifndef SQUEEZENET_MT_PARAMS_H
#define SQUEEZENET_MT_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_1_w[27][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_1_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_1_in[12321][27] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out[12321][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_pooled[1][55][55][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params = {.batch_size=1, .in_dim=224, .kernel_size=3, .in_channels=3, .out_channels=64, .out_stride=(64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=111, .n_patches=12321, .patch_size=27, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=55, .output_scale=9, .I=12321, .J=64, .K=27, .res_scale=0};


static const elem_t conv_2_w[64][16] row_align(MAX_BLOCK_LEN);
static const acc_t conv_2_b[16] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_2_in[3025][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out[3025][16] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params = {.batch_size=1, .in_dim=55, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=16, .out_stride=(16), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=55, .output_scale=7, .I=3025, .J=16, .K=64, .res_scale=0};


static const elem_t conv_3_w[16][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_3_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_3_in[3025][16] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out[3025][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params = {.batch_size=1, .in_dim=55, .kernel_size=1, .in_channels=16, .out_channels=64, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=55, .output_scale=6, .I=3025, .J=64, .K=16, .res_scale=0};


static const elem_t conv_4_w[144][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_4_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_4_in[3025][144] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out[3025][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params = {.batch_size=1, .in_dim=55, .kernel_size=3, .in_channels=16, .out_channels=64, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=144, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=55, .output_scale=6, .I=3025, .J=64, .K=144, .res_scale=0};


static const elem_t conv_5_w[128][16] row_align(MAX_BLOCK_LEN);
static const acc_t conv_5_b[16] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_5_in[3025][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out[3025][16] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params = {.batch_size=1, .in_dim=55, .kernel_size=1, .in_channels=128, .out_channels=16, .out_stride=(16), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=55, .output_scale=7, .I=3025, .J=16, .K=128, .res_scale=0};


static const elem_t conv_6_w[16][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_6_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_6_in[3025][16] row_align(MAX_BLOCK_LEN);
static elem_t conv_6_out[3025][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_6_out_pooled[1][27][27][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_6_params = {.batch_size=1, .in_dim=55, .kernel_size=1, .in_channels=16, .out_channels=64, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=16, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=27, .output_scale=6, .I=3025, .J=64, .K=16, .res_scale=0};


static const elem_t conv_7_w[144][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_7_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_7_in[3025][144] row_align(MAX_BLOCK_LEN);
static elem_t conv_7_out[3025][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_7_out_pooled[1][27][27][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_7_params = {.batch_size=1, .in_dim=55, .kernel_size=3, .in_channels=16, .out_channels=64, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=144, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=27, .output_scale=8, .I=3025, .J=64, .K=144, .res_scale=0};


static const elem_t conv_8_w[128][32] row_align(MAX_BLOCK_LEN);
static const acc_t conv_8_b[32] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_8_in[729][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_8_out[729][32] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_8_params = {.batch_size=1, .in_dim=27, .kernel_size=1, .in_channels=128, .out_channels=32, .out_stride=32, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=27, .output_scale=7, .I=729, .J=32, .K=128, .res_scale=0};


static const elem_t conv_9_w[32][(128+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_9_b[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_9_in[729][32] row_align(MAX_BLOCK_LEN);
static elem_t conv_9_out[729][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_9_params = {.batch_size=1, .in_dim=27, .kernel_size=1, .in_channels=32, .out_channels=128, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=27, .output_scale=7, .I=729, .J=128, .K=32, .res_scale=0};


static const elem_t conv_10_w[288][(128+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_10_b[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_10_in[729][288] row_align(MAX_BLOCK_LEN);
static elem_t conv_10_out[729][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_10_params = {.batch_size=1, .in_dim=27, .kernel_size=3, .in_channels=32, .out_channels=128, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=288, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=27, .output_scale=7, .I=729, .J=128, .K=288, .res_scale=0};


static const elem_t conv_11_w[256][32] row_align(MAX_BLOCK_LEN);
static const acc_t conv_11_b[32] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_11_in[729][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_11_out[729][32] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_11_params = {.batch_size=1, .in_dim=27, .kernel_size=1, .in_channels=256, .out_channels=32, .out_stride=(32), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=27, .output_scale=7, .I=729, .J=32, .K=256, .res_scale=0};


static const elem_t conv_12_w[32][(128+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_12_b[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_12_in[729][32] row_align(MAX_BLOCK_LEN);
static elem_t conv_12_out[729][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_12_out_pooled[1][13][13][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_12_params = {.batch_size=1, .in_dim=27, .kernel_size=1, .in_channels=32, .out_channels=128, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=32, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=729, .J=128, .K=32, .res_scale=0};


static const elem_t conv_13_w[288][(128+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_13_b[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_13_in[729][288] row_align(MAX_BLOCK_LEN);
static elem_t conv_13_out[729][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_13_out_pooled[1][13][13][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_13_params = {.batch_size=1, .in_dim=27, .kernel_size=3, .in_channels=32, .out_channels=128, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=288, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=729, .J=128, .K=288, .res_scale=0};


static const elem_t conv_14_w[256][48] row_align(MAX_BLOCK_LEN);
static const acc_t conv_14_b[48] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_14_in[169][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_14_out[169][48] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_14_params = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=256, .out_channels=48, .out_stride=48, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=9, .I=169, .J=48, .K=256, .res_scale=0};


static const elem_t conv_15_w[48][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_15_b[192] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_15_in[169][48] row_align(MAX_BLOCK_LEN);
static elem_t conv_15_out[169][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_15_params = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=48, .out_channels=192, .out_stride=(384+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=8, .I=169, .J=192, .K=48, .res_scale=0};


static const elem_t conv_16_w[432][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_16_b[192] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_16_in[169][432] row_align(MAX_BLOCK_LEN);
static elem_t conv_16_out[169][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_16_params = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=48, .out_channels=192, .out_stride=(384+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=432, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=192, .K=432, .res_scale=0};


static const elem_t conv_17_w[384][48] row_align(MAX_BLOCK_LEN);
static const acc_t conv_17_b[48] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_17_in[169][(384+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_17_out[169][48] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_17_params = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=384, .in_stride=(384+64), .out_channels=48, .out_stride=48, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=384, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=8, .I=169, .J=48, .K=384, .res_scale=0};


static const elem_t conv_18_w[48][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_18_b[192] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_18_in[169][48] row_align(MAX_BLOCK_LEN);
static elem_t conv_18_out[169][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_18_params = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=48, .out_channels=192, .out_stride=(384+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=192, .K=48, .res_scale=0};


static const elem_t conv_19_w[432][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_19_b[192] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_19_in[169][432] row_align(MAX_BLOCK_LEN);
static elem_t conv_19_out[169][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_19_params = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=48, .out_channels=192, .out_stride=(384+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=432, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=192, .K=432, .res_scale=0};


static const elem_t conv_20_w[384][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_20_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_20_in[169][(384+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_20_out[169][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_20_params = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=384, .in_stride=(384+64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=384, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=8, .I=169, .J=64, .K=384, .res_scale=0};


static const elem_t conv_21_w[64][(256+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_21_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_21_in[169][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_21_out[169][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_21_params = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=8, .I=169, .J=256, .K=64, .res_scale=0};


static const elem_t conv_22_w[576][(256+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_22_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_22_in[169][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_22_out[169][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_22_params = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=256, .K=576, .res_scale=0};


static const elem_t conv_23_w[512][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_23_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_23_in[169][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_23_out[169][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_23_params = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=64, .K=512, .res_scale=0};


static const elem_t conv_24_w[64][(256+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_24_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_24_in[169][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_24_out[169][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_24_params = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=256, .K=64, .res_scale=0};


static const elem_t conv_25_w[576][(256+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_25_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_25_in[169][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_25_out[169][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_25_params = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=8, .I=169, .J=256, .K=576, .res_scale=0};


static const elem_t conv_26_w[512][1000] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_26_b[1000] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_26_in[169][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_26_out[169][1000] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_26_params = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=1000, .out_stride = 1000, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=8, .I=169, .J=1000, .K=512, .res_scale=0};


#endif

