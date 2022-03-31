#ifndef SQUEEZENET_MT_PARAMS_H
#define SQUEEZENET_MT_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_1_w_squeeze1[27][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_1_b_squeeze1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_1_in_squeeze1[12321][27] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_squeeze1[12321][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_squeeze1_pooled[1][55][55][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_squeeze1 = {.batch_size=1, .in_dim=224, .kernel_size=3, .in_channels=3, .out_channels=64, .out_stride=(64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=111, .n_patches=12321, .patch_size=27, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=55, .output_scale=9, .I=12321, .J=64, .K=27, .res_scale=0};


static const elem_t conv_2_w_squeeze1[64][16] row_align(MAX_BLOCK_LEN);
static const acc_t conv_2_b_squeeze1[16] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_2_in_squeeze1[3025][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_squeeze1[3025][16] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_squeeze1 = {.batch_size=1, .in_dim=55, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=16, .out_stride=(16), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=55, .output_scale=7, .I=3025, .J=16, .K=64, .res_scale=0};


static const elem_t conv_3_w_squeeze1[16][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_3_b_squeeze1[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_3_in_squeeze1[3025][16] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_squeeze1[3025][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_squeeze1 = {.batch_size=1, .in_dim=55, .kernel_size=1, .in_channels=16, .out_channels=64, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=16, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=55, .output_scale=6, .I=3025, .J=64, .K=16, .res_scale=0};


static const elem_t conv_4_w_squeeze1[144][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_4_b_squeeze1[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_4_in_squeeze1[3025][144] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_squeeze1[3025][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_squeeze1 = {.batch_size=1, .in_dim=55, .kernel_size=3, .in_channels=16, .out_channels=64, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=144, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=55, .output_scale=6, .I=3025, .J=64, .K=144, .res_scale=0};


static const elem_t conv_5_w_squeeze1[128][16] row_align(MAX_BLOCK_LEN);
static const acc_t conv_5_b_squeeze1[16] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_5_in_squeeze1[3025][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_squeeze1[3025][16] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_squeeze1 = {.batch_size=1, .in_dim=55, .kernel_size=1, .in_channels=128, .out_channels=16, .out_stride=(16), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=55, .output_scale=7, .I=3025, .J=16, .K=128, .res_scale=0};


static const elem_t conv_6_w_squeeze1[16][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_6_b_squeeze1[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_6_in_squeeze1[3025][16] row_align(MAX_BLOCK_LEN);
static elem_t conv_6_out_squeeze1[3025][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_6_out_squeeze1_pooled[1][27][27][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_6_params_squeeze1 = {.batch_size=1, .in_dim=55, .kernel_size=1, .in_channels=16, .out_channels=64, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=16, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=27, .output_scale=6, .I=3025, .J=64, .K=16, .res_scale=0};


static const elem_t conv_7_w_squeeze1[144][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_7_b_squeeze1[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_7_in_squeeze1[3025][144] row_align(MAX_BLOCK_LEN);
static elem_t conv_7_out_squeeze1[3025][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_7_out_squeeze1_pooled[1][27][27][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_7_params_squeeze1 = {.batch_size=1, .in_dim=55, .kernel_size=3, .in_channels=16, .out_channels=64, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=144, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=27, .output_scale=8, .I=3025, .J=64, .K=144, .res_scale=0};


static const elem_t conv_8_w_squeeze1[128][32] row_align(MAX_BLOCK_LEN);
static const acc_t conv_8_b_squeeze1[32] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_8_in_squeeze1[729][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_8_out_squeeze1[729][32] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_8_params_squeeze1 = {.batch_size=1, .in_dim=27, .kernel_size=1, .in_channels=128, .out_channels=32, .out_stride=32, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=27, .output_scale=7, .I=729, .J=32, .K=128, .res_scale=0};


static const elem_t conv_9_w_squeeze1[32][(128+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_9_b_squeeze1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_9_in_squeeze1[729][32] row_align(MAX_BLOCK_LEN);
static elem_t conv_9_out_squeeze1[729][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_9_params_squeeze1 = {.batch_size=1, .in_dim=27, .kernel_size=1, .in_channels=32, .out_channels=128, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=32, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=27, .output_scale=7, .I=729, .J=128, .K=32, .res_scale=0};


static const elem_t conv_10_w_squeeze1[288][(128+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_10_b_squeeze1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_10_in_squeeze1[729][288] row_align(MAX_BLOCK_LEN);
static elem_t conv_10_out_squeeze1[729][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_10_params_squeeze1 = {.batch_size=1, .in_dim=27, .kernel_size=3, .in_channels=32, .out_channels=128, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=288, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=27, .output_scale=7, .I=729, .J=128, .K=288, .res_scale=0};


static const elem_t conv_11_w_squeeze1[256][32] row_align(MAX_BLOCK_LEN);
static const acc_t conv_11_b_squeeze1[32] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_11_in_squeeze1[729][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_11_out_squeeze1[729][32] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_11_params_squeeze1 = {.batch_size=1, .in_dim=27, .kernel_size=1, .in_channels=256, .out_channels=32, .out_stride=(32), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=27, .output_scale=7, .I=729, .J=32, .K=256, .res_scale=0};


static const elem_t conv_12_w_squeeze1[32][(128+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_12_b_squeeze1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_12_in_squeeze1[729][32] row_align(MAX_BLOCK_LEN);
static elem_t conv_12_out_squeeze1[729][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_12_out_squeeze1_pooled[1][13][13][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_12_params_squeeze1 = {.batch_size=1, .in_dim=27, .kernel_size=1, .in_channels=32, .out_channels=128, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=32, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=729, .J=128, .K=32, .res_scale=0};


static const elem_t conv_13_w_squeeze1[288][(128+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_13_b_squeeze1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_13_in_squeeze1[729][288] row_align(MAX_BLOCK_LEN);
static elem_t conv_13_out_squeeze1[729][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_13_out_squeeze1_pooled[1][13][13][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_13_params_squeeze1 = {.batch_size=1, .in_dim=27, .kernel_size=3, .in_channels=32, .out_channels=128, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=288, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=729, .J=128, .K=288, .res_scale=0};


static const elem_t conv_14_w_squeeze1[256][48] row_align(MAX_BLOCK_LEN);
static const acc_t conv_14_b_squeeze1[48] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_14_in_squeeze1[169][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_14_out_squeeze1[169][48] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_14_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=256, .out_channels=48, .out_stride=48, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=9, .I=169, .J=48, .K=256, .res_scale=0};


static const elem_t conv_15_w_squeeze1[48][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_15_b_squeeze1[192] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_15_in_squeeze1[169][48] row_align(MAX_BLOCK_LEN);
static elem_t conv_15_out_squeeze1[169][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_15_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=48, .out_channels=192, .out_stride=(384+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=8, .I=169, .J=192, .K=48, .res_scale=0};


static const elem_t conv_16_w_squeeze1[432][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_16_b_squeeze1[192] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_16_in_squeeze1[169][432] row_align(MAX_BLOCK_LEN);
static elem_t conv_16_out_squeeze1[169][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_16_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=48, .out_channels=192, .out_stride=(384+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=432, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=192, .K=432, .res_scale=0};


static const elem_t conv_17_w_squeeze1[384][48] row_align(MAX_BLOCK_LEN);
static const acc_t conv_17_b_squeeze1[48] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_17_in_squeeze1[169][(384+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_17_out_squeeze1[169][48] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_17_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=384, .in_stride=(384+64), .out_channels=48, .out_stride=48, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=384, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=8, .I=169, .J=48, .K=384, .res_scale=0};


static const elem_t conv_18_w_squeeze1[48][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_18_b_squeeze1[192] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_18_in_squeeze1[169][48] row_align(MAX_BLOCK_LEN);
static elem_t conv_18_out_squeeze1[169][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_18_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=48, .out_channels=192, .out_stride=(384+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=192, .K=48, .res_scale=0};


static const elem_t conv_19_w_squeeze1[432][192] row_align(MAX_BLOCK_LEN);
static const acc_t conv_19_b_squeeze1[192] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_19_in_squeeze1[169][432] row_align(MAX_BLOCK_LEN);
static elem_t conv_19_out_squeeze1[169][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_19_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=48, .out_channels=192, .out_stride=(384+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=432, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=192, .K=432, .res_scale=0};


static const elem_t conv_20_w_squeeze1[384][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_20_b_squeeze1[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_20_in_squeeze1[169][(384+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_20_out_squeeze1[169][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_20_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=384, .in_stride=(384+64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=384, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=8, .I=169, .J=64, .K=384, .res_scale=0};


static const elem_t conv_21_w_squeeze1[64][(256+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_21_b_squeeze1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_21_in_squeeze1[169][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_21_out_squeeze1[169][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_21_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=8, .I=169, .J=256, .K=64, .res_scale=0};


static const elem_t conv_22_w_squeeze1[576][(256+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_22_b_squeeze1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_22_in_squeeze1[169][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_22_out_squeeze1[169][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_22_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=256, .K=576, .res_scale=0};


static const elem_t conv_23_w_squeeze1[512][64] row_align(MAX_BLOCK_LEN);
static const acc_t conv_23_b_squeeze1[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_23_in_squeeze1[169][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_23_out_squeeze1[169][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_23_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=64, .K=512, .res_scale=0};


static const elem_t conv_24_w_squeeze1[64][(256+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_24_b_squeeze1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_24_in_squeeze1[169][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_24_out_squeeze1[169][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_24_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=256, .K=64, .res_scale=0};


static const elem_t conv_25_w_squeeze1[576][(256+64)] row_align(MAX_BLOCK_LEN);
static const acc_t conv_25_b_squeeze1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_25_in_squeeze1[169][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_25_out_squeeze1[169][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_25_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=8, .I=169, .J=256, .K=576, .res_scale=0};


static const elem_t conv_26_w_squeeze1[512][1000] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_26_b_squeeze1[1000] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_26_in_squeeze1[169][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_26_out_squeeze1[169][1000] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_26_params_squeeze1 = {.batch_size=1, .in_dim=13, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=1000, .out_stride = 1000, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=8, .I=169, .J=1000, .K=512, .res_scale=0};


#endif

