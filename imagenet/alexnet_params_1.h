#ifndef ALEXNET_MT_PARAMS_H
#define ALEXNET_MT_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_1_w_alex1[363][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_1_b_alex1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_1_in_alex1[3025][363] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_alex1[3025][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_alex1_pooled[1][27][27][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_alex1 = {.batch_size=1, .in_dim=224, .kernel_size=11, .in_channels=3, .out_channels=64, .out_stride=(64), .stride=4, .padding=2, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=363, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=27, .output_scale=10, .I=3025, .J=64, .K=363, .res_scale=0};


static const elem_t conv_2_w_alex1[1600][192] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_2_b_alex1[192] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_2_in_alex1[729][1600] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_alex1[729][192] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_alex1_pooled[1][13][13][192] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_alex1 = {.batch_size=1, .in_dim=27, .kernel_size=5, .in_channels=64, .out_channels=192, .out_stride=(192), .stride=1, .padding=2, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=1600, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=13, .output_scale=6, .I=729, .J=192, .K=1600, .res_scale=0};


static const elem_t conv_3_w_alex1[1728][(384+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_3_b_alex1[(384+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_3_in_alex1[169][1728] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_alex1[169][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_alex1 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=192, .out_channels=384, .out_stride=(384+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=1728, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=384, .K=1728, .res_scale=0};


static const elem_t conv_4_w_alex1[3456][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_4_b_alex1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_4_in_alex1[169][3456] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_alex1[169][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_alex1 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=384, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=3456, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=256, .K=3456, .res_scale=0};


static const elem_t conv_5_w_alex1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_5_b_alex1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_5_in_alex1[169][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_alex1[169][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_alex1_pooled[1][6][6][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_alex1 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=2304, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=6, .output_scale=9, .I=169, .J=256, .K=2304, .res_scale=0};


static const elem_t fc_6_w_alex1[9216][(4096+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_6_b_alex1[1][(4096+64)] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_6_out_alex1[1][(4096+64)] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_6_params_alex1 = {.batch_size=1, .in_features=9216, .out_features=4096, .out_stride=(4096+64), .bias=1, .output_scale=9, .J=4096, .I=1, .K=9216};


static const elem_t fc_7_w_alex1[4096][(4096+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_7_b_alex1[1][(4096+64)] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_7_out_alex1[1][(4096+64)] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_7_params_alex1 = {.batch_size=1, .in_features=4096, .out_features=4096, .out_stride=(4096+64), .bias=1, .output_scale=10, .J=4096, .I=1, .K=4096};


static const elem_t fc_8_w_alex1[4096][1024+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_8_b_alex1[1][1024] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_8_out_alex1[1][1024+64] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_8_params_alex1 = {.batch_size=1, .in_features=4096, .out_features=1024, .out_stride = (1024+64), .bias=1, .output_scale=10, .J=1024, .I=1, .K=4096};

//#if NUM_CORE == 8
static const elem_t conv_1_w_alex11[363][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_1_b_alex11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_1_in_alex11[3025][363] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_alex11[3025][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_alex11_pooled[1][27][27][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_alex11 = {.batch_size=1, .in_dim=224, .kernel_size=11, .in_channels=3, .out_channels=64, .out_stride=(64), .stride=4, .padding=2, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=363, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=27, .output_scale=10, .I=3025, .J=64, .K=363, .res_scale=0};


static const elem_t conv_2_w_alex11[1600][192] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_2_b_alex11[192] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_2_in_alex11[729][1600] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_alex11[729][192] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_alex11_pooled[1][13][13][192] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_alex11 = {.batch_size=1, .in_dim=27, .kernel_size=5, .in_channels=64, .out_channels=192, .out_stride=(192), .stride=1, .padding=2, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=1600, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=13, .output_scale=6, .I=729, .J=192, .K=1600, .res_scale=0};


static const elem_t conv_3_w_alex11[1728][(384+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_3_b_alex11[(384+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_3_in_alex11[169][1728] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_alex11[169][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_alex11 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=192, .out_channels=384, .out_stride=(384+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=1728, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=384, .K=1728, .res_scale=0};


static const elem_t conv_4_w_alex11[3456][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_4_b_alex11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_4_in_alex11[169][3456] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_alex11[169][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_alex11 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=384, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=3456, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=256, .K=3456, .res_scale=0};


static const elem_t conv_5_w_alex11[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_5_b_alex11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_5_in_alex11[169][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_alex11[169][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_alex11_pooled[1][6][6][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_alex11 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=2304, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=6, .output_scale=9, .I=169, .J=256, .K=2304, .res_scale=0};


static const elem_t fc_6_w_alex11[9216][(4096+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_6_b_alex11[1][(4096+64)] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_6_out_alex11[1][(4096+64)] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_6_params_alex11 = {.batch_size=1, .in_features=9216, .out_features=4096, .out_stride=(4096+64), .bias=1, .output_scale=9, .J=4096, .I=1, .K=9216};


static const elem_t fc_7_w_alex11[4096][(4096+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_7_b_alex11[1][(4096+64)] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_7_out_alex11[1][(4096+64)] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_7_params_alex11 = {.batch_size=1, .in_features=4096, .out_features=4096, .out_stride=(4096+64), .bias=1, .output_scale=10, .J=4096, .I=1, .K=4096};


static const elem_t fc_8_w_alex11[4096][1024+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_8_b_alex11[1][1024] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_8_out_alex11[1][1024+64] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_8_params_alex11 = {.batch_size=1, .in_features=4096, .out_features=1024, .out_stride = (1024+64), .bias=1, .output_scale=10, .J=1024, .I=1, .K=4096};


//#if NUM_CORE == 8
static const elem_t conv_1_w_alex111[363][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_1_b_alex111[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_1_in_alex111[3025][363] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_alex111[3025][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_alex111_pooled[1][27][27][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_alex111 = {.batch_size=1, .in_dim=224, .kernel_size=11, .in_channels=3, .out_channels=64, .out_stride=(64), .stride=4, .padding=2, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=363, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=27, .output_scale=10, .I=3025, .J=64, .K=363, .res_scale=0};


static const elem_t conv_2_w_alex111[1600][192] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_2_b_alex111[192] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_2_in_alex111[729][1600] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_alex111[729][192] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_alex111_pooled[1][13][13][192] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_alex111 = {.batch_size=1, .in_dim=27, .kernel_size=5, .in_channels=64, .out_channels=192, .out_stride=(192), .stride=1, .padding=2, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=1600, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=13, .output_scale=6, .I=729, .J=192, .K=1600, .res_scale=0};


static const elem_t conv_3_w_alex111[1728][(384+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_3_b_alex111[(384+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_3_in_alex111[169][1728] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_alex111[169][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_alex111 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=192, .out_channels=384, .out_stride=(384+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=1728, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=384, .K=1728, .res_scale=0};


static const elem_t conv_4_w_alex111[3456][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_4_b_alex111[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_4_in_alex111[169][3456] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_alex111[169][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_alex111 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=384, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=3456, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=256, .K=3456, .res_scale=0};


static const elem_t conv_5_w_alex111[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_5_b_alex111[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_5_in_alex111[169][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_alex111[169][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_alex111_pooled[1][6][6][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_alex111 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=2304, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=6, .output_scale=9, .I=169, .J=256, .K=2304, .res_scale=0};


static const elem_t fc_6_w_alex111[9216][(4096+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_6_b_alex111[1][(4096+64)] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_6_out_alex111[1][(4096+64)] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_6_params_alex111 = {.batch_size=1, .in_features=9216, .out_features=4096, .out_stride=(4096+64), .bias=1, .output_scale=9, .J=4096, .I=1, .K=9216};


static const elem_t fc_7_w_alex111[4096][(4096+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_7_b_alex111[1][(4096+64)] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_7_out_alex111[1][(4096+64)] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_7_params_alex111 = {.batch_size=1, .in_features=4096, .out_features=4096, .out_stride=(4096+64), .bias=1, .output_scale=10, .J=4096, .I=1, .K=4096};


static const elem_t fc_8_w_alex111[4096][1024+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_8_b_alex111[1][1024] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_8_out_alex111[1][1024+64] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_8_params_alex111 = {.batch_size=1, .in_features=4096, .out_features=1024, .out_stride = (1024+64), .bias=1, .output_scale=10, .J=1024, .I=1, .K=4096};


static const elem_t conv_1_w_alex1111[363][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_1_b_alex1111[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_1_in_alex1111[3025][363] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_alex1111[3025][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_alex1111_pooled[1][27][27][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_alex1111 = {.batch_size=1, .in_dim=224, .kernel_size=11, .in_channels=3, .out_channels=64, .out_stride=(64), .stride=4, .padding=2, .bias=1, .depthwise=0, .out_dim=55, .n_patches=3025, .patch_size=363, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=27, .output_scale=10, .I=3025, .J=64, .K=363, .res_scale=0};


static const elem_t conv_2_w_alex1111[1600][192] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_2_b_alex1111[192] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_2_in_alex1111[729][1600] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_alex1111[729][192] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_alex1111_pooled[1][13][13][192] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_alex1111 = {.batch_size=1, .in_dim=27, .kernel_size=5, .in_channels=64, .out_channels=192, .out_stride=(192), .stride=1, .padding=2, .bias=1, .depthwise=0, .out_dim=27, .n_patches=729, .patch_size=1600, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=13, .output_scale=6, .I=729, .J=192, .K=1600, .res_scale=0};


static const elem_t conv_3_w_alex1111[1728][(384+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_3_b_alex1111[(384+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_3_in_alex1111[169][1728] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_alex1111[169][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_alex1111 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=192, .out_channels=384, .out_stride=(384+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=1728, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=384, .K=1728, .res_scale=0};


static const elem_t conv_4_w_alex1111[3456][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_4_b_alex1111[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_4_in_alex1111[169][3456] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_alex1111[169][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_alex1111 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=384, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=3456, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=13, .output_scale=7, .I=169, .J=256, .K=3456, .res_scale=0};


static const elem_t conv_5_w_alex1111[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_5_b_alex1111[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_5_in_alex1111[169][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_alex1111[169][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_alex1111_pooled[1][6][6][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_alex1111 = {.batch_size=1, .in_dim=13, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=13, .n_patches=169, .patch_size=2304, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=6, .output_scale=9, .I=169, .J=256, .K=2304, .res_scale=0};


static const elem_t fc_6_w_alex1111[9216][(4096+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_6_b_alex1111[1][(4096+64)] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_6_out_alex1111[1][(4096+64)] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_6_params_alex1111 = {.batch_size=1, .in_features=9216, .out_features=4096, .out_stride=(4096+64), .bias=1, .output_scale=9, .J=4096, .I=1, .K=9216};


static const elem_t fc_7_w_alex1111[4096][(4096+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_7_b_alex1111[1][(4096+64)] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_7_out_alex1111[1][(4096+64)] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_7_params_alex1111 = {.batch_size=1, .in_features=4096, .out_features=4096, .out_stride=(4096+64), .bias=1, .output_scale=10, .J=4096, .I=1, .K=4096};


static const elem_t fc_8_w_alex1111[4096][1024+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_8_b_alex1111[1][1024] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_8_out_alex1111[1][1024+64] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_8_params_alex1111 = {.batch_size=1, .in_features=4096, .out_features=1024, .out_stride = (1024+64), .bias=1, .output_scale=10, .J=1024, .I=1, .K=4096};

#endif
//#endif

