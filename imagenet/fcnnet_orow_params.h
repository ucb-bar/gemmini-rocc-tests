#ifndef RESNET_MT_PARAMS_H
#define RESNET_MT_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_1_w_fcn1[147][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_1_b_fcn1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_1_in_fcn1[12544][147] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_fcn1[12544][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_fcn1_pooled[1][56][56][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_fcn1 = {.batch_size=1, .in_dim=224, .kernel_size=7, .in_channels=3, .out_channels=64, .out_stride=(64), .stride=2, .padding=3, .bias=1, .depthwise=0, .out_dim=112, .n_patches=12544, .patch_size=147, .pool_size=3, .pool_stride=2, .pool_padding=1, .out_dim_pooled=56, .output_scale=2, .I=12544, .J=64, .K=147, .res_scale=0, .dilation=1};


static const elem_t conv_2_w_fcn1[64][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_2_b_fcn1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_2_in_fcn1[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_fcn1[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .I=3136, .J=64, .K=64, .res_scale=0, .dilation=1};


static const elem_t conv_3_w_fcn1[576][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_3_b_fcn1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_3_in_fcn1[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_fcn1[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=8, .I=3136, .J=64, .K=576, .res_scale=0, .dilation=1};


static const elem_t conv_4_w_fcn1[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_4_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_4_in_fcn1[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_fcn1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=256, .K=64, .res_scale=0, .dilation=1};


static const elem_t conv_5_w_fcn1[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_5_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_5_in_fcn1[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_fcn1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .I=3136, .J=256, .K=64, .res_scale=0, .dilation=1};


static const elem_t conv_6_w_fcn1[256][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_6_b_fcn1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_6_in_fcn1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_6_out_fcn1[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_6_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .I=3136, .J=64, .K=256, .res_scale=0, .dilation=1};


static const elem_t conv_7_w_fcn1[576][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_7_b_fcn1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_7_in_fcn1[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_7_out_fcn1[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_7_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=8, .I=3136, .J=64, .K=576, .res_scale=0, .dilation=1};


static const elem_t conv_8_w_fcn1[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_8_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_8_in_fcn1[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_8_out_fcn1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_8_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .I=3136, .J=256, .K=64, .res_scale=0, .dilation=1};


static const elem_t conv_9_w_fcn1[256][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_9_b_fcn1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_9_in_fcn1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_9_out_fcn1[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_9_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=256, .res_scale=0, .dilation=1};


static const elem_t conv_10_w_fcn1[576][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_10_b_fcn1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_10_in_fcn1[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_10_out_fcn1[3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_10_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=8, .I=3136, .J=64, .K=576, .res_scale=0, .dilation=1};


static const elem_t conv_11_w_fcn1[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_11_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_11_in_fcn1[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_11_out_fcn1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_11_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .I=3136, .J=256, .K=64, .res_scale=0, .dilation=1};


static const elem_t conv_12_w_fcn1[256][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_12_b_fcn1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_12_in_fcn1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_12_out_fcn1[3136][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_12_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=128, .K=256, .res_scale=0, .dilation=1};


static const elem_t conv_13_w_fcn1[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_13_b_fcn1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_13_in_fcn1[784][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_13_out_fcn1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_13_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=128, .K=1152, .res_scale=0, .dilation=1};


static const elem_t conv_14_w_fcn1[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_14_b_fcn1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_14_in_fcn1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_14_out_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_14_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=128, .res_scale=0, .dilation=1};


static const elem_t conv_15_w_fcn1[256][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_15_b_fcn1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_15_in_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_15_out_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_15_params_fcn1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .out_channels=512, .out_stride=(512+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=256, .res_scale=0, .dilation=1};


static const elem_t conv_16_w_fcn1[512][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_16_b_fcn1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_16_in_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_16_out_fcn1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_16_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=128, .K=512, .res_scale=0, .dilation=1};


static const elem_t conv_17_w_fcn1[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_17_b_fcn1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_17_in_fcn1[784][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_17_out_fcn1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_17_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=9, .I=784, .J=128, .K=1152, .res_scale=0, .dilation=1};


static const elem_t conv_18_w_fcn1[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_18_b_fcn1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_18_in_fcn1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_18_out_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_18_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=128, .res_scale=0, .dilation=1};


static const elem_t conv_19_w_fcn1[512][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_19_b_fcn1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_19_in_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_19_out_fcn1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_19_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=128, .K=512, .res_scale=0, .dilation=1};


static const elem_t conv_20_w_fcn1[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_20_b_fcn1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_20_in_fcn1[784][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_20_out_fcn1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_20_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0, .dilation=1};


static const elem_t conv_21_w_fcn1[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_21_b_fcn1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_21_in_fcn1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_21_out_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_21_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=512, .K=128, .res_scale=0, .dilation=1};


static const elem_t conv_22_w_fcn1[512][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_22_b_fcn1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_22_in_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_22_out_fcn1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_22_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=128, .K=512, .res_scale=0, .dilation=1};


static const elem_t conv_23_w_fcn1[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_23_b_fcn1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_23_in_fcn1[784][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_23_out_fcn1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_23_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0, .dilation=1};


static const elem_t conv_24_w_fcn1[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_24_b_fcn1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_24_in_fcn1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_24_out_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_24_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=128, .res_scale=0, .dilation=1};


static const elem_t conv_25_w_fcn1[512][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_25_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_25_in_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_25_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_25_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=256, .K=512, .res_scale=0, .dilation=1};


static const elem_t conv_26_w_fcn1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_26_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_26_in_fcn1[784][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_26_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_26_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=256, .K=2304, .res_scale=0, .dilation=1};


static const elem_t conv_27_w_fcn1[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_27_b_fcn1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_27_in_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_27_out_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_27_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=256, .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=1024, .K=256, .res_scale=0, .dilation=1};


static const elem_t conv_28_w_fcn1[512][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_28_b_fcn1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_28_in_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_28_out_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_28_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=1024, .K=512, .res_scale=0, .dilation=1};


static const elem_t conv_29_w_fcn1[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_29_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_29_in_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_29_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_29_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=1024, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=256, .K=1024, .res_scale=0, .dilation=1};


static const elem_t conv_30_w_fcn1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_30_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_30_in_fcn1[784][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_30_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_30_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=2, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=256, .K=2304, .res_scale=0, .dilation=2};


static const elem_t conv_31_w_fcn1[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_31_b_fcn1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_31_in_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_31_out_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_31_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=256, .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=1024, .K=256, .res_scale=0, .dilation=1};


static const elem_t conv_32_w_fcn1[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_32_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_32_in_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_32_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_32_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=1024, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=256, .K=1024, .res_scale=0, .dilation=1};


static const elem_t conv_33_w_fcn1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_33_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_33_in_fcn1[784][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_33_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_33_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=2, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=256, .K=2304, .res_scale=0, .dilation=2};


static const elem_t conv_34_w_fcn1[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_34_b_fcn1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_34_in_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_34_out_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_34_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=256, .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=1024, .K=256, .res_scale=0, .dilation=1};


static const elem_t conv_35_w_fcn1[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_35_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_35_in_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_35_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_35_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=1024, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=256, .K=1024, .res_scale=0, .dilation=1};


static const elem_t conv_36_w_fcn1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_36_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_36_in_fcn1[784][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_36_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_36_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=2, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=256, .K=2304, .res_scale=0, .dilation=2};


static const elem_t conv_37_w_fcn1[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_37_b_fcn1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_37_in_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_37_out_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_37_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=256, .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=1024, .K=256, .res_scale=0, .dilation=1};


static const elem_t conv_38_w_fcn1[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_38_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_38_in_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_38_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_38_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=1024, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=256, .K=1024, .res_scale=0, .dilation=1};


static const elem_t conv_39_w_fcn1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_39_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_39_in_fcn1[784][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_39_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_39_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=2, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=256, .K=2304, .res_scale=0, .dilation=2};


static const elem_t conv_40_w_fcn1[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_40_b_fcn1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_40_in_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_40_out_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_40_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=256, .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=1024, .K=256, .res_scale=0, .dilation=1};


static const elem_t conv_41_w_fcn1[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_41_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_41_in_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_41_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_41_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=1024, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=256, .K=1024, .res_scale=0, .dilation=1};


static const elem_t conv_42_w_fcn1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_42_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_42_in_fcn1[784][2304] row_align(MAX_BLOCK_LEN);
static elem_t conv_42_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_42_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=256, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=2, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=256, .K=2304, .res_scale=0, .dilation=2};


static const elem_t conv_43_w_fcn1[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_43_b_fcn1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_43_in_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_43_out_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_43_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=256, .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=1024, .K=256, .res_scale=0, .dilation=1};


static const elem_t conv_44_w_fcn1[1024][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_44_b_fcn1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_44_in_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_44_out_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_44_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=1024, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=1024, .res_scale=0, .dilation=1};


static const elem_t conv_45_w_fcn1[4608][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_45_b_fcn1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_45_in_fcn1[784][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_45_out_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_45_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=512, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=2, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=512, .K=4608, .res_scale=0, .dilation=2};


static const elem_t conv_46_w_fcn1[512][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_46_b_fcn1[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_46_in_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_46_out_fcn1[784][(2048+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_46_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=2048, .K=512, .res_scale=0, .dilation=1};


static const elem_t conv_47_w_fcn1[1024][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_47_b_fcn1[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_47_in_fcn1[784][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_47_out_fcn1[784][(2048+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_47_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=1024, .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=2048, .K=1024, .res_scale=0, .dilation=1};


static const elem_t conv_48_w_fcn1[2048][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_48_b_fcn1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_48_in_fcn1[784][(2048+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_48_out_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_48_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=2048, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=2048, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=2048, .res_scale=0, .dilation=1};


static const elem_t conv_49_w_fcn1[4608][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_49_b_fcn1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_49_in_fcn1[784][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_49_out_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_49_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=512, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=4, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=4608, .res_scale=0, .dilation=4};


static const elem_t conv_50_w_fcn1[512][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_50_b_fcn1[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_50_in_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_50_out_fcn1[784][(2048+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_50_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=2048, .K=512, .res_scale=0, .dilation=1};


static const elem_t conv_51_w_fcn1[2048][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_51_b_fcn1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_51_in_fcn1[784][(2048+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_51_out_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_51_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=2048, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=2048, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=512, .K=2048, .res_scale=0, .dilation=1};


static const elem_t conv_52_w_fcn1[4608][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_52_b_fcn1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_52_in_fcn1[784][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_52_out_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_52_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=512, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=4, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=5, .I=784, .J=512, .K=4608, .res_scale=0, .dilation=4};


static const elem_t conv_53_w_fcn1[512][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_53_b_fcn1[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_53_in_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_53_out_fcn1[784][(2048+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_53_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=2048, .K=512, .res_scale=0, .dilation=1};


static const elem_t conv_54_w_fcn1[18432][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_54_b_fcn1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_54_in_fcn1[784][18432] row_align(MAX_BLOCK_LEN);
static elem_t conv_54_out_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_54_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=2048, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=18432, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=9, .I=784, .J=512, .K=18432, .res_scale=0, .dilation=1};


static const elem_t conv_55_w_fcn1[512][21] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_55_b_fcn1[21] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_55_in_fcn1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_55_out_fcn1[784][21] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_55_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .out_channels=21, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .out_stride=21, .output_scale=8, .I=784, .J=21, .K=512, .res_scale=0, .dilation=1};

static elem_t image_out_fcn1[1][224][224][21] row_align(1);

static const elem_t conv_56_w_fcn1[9216][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_56_b_fcn1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_56_in_fcn1[784][9216] row_align(MAX_BLOCK_LEN);
static elem_t conv_56_out_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_56_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=1024, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=9216, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=9, .I=784, .J=256, .K=9216, .res_scale=0, .dilation=1};


static const elem_t conv_57_w_fcn1[256][21] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_57_b_fcn1[21] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_57_in_fcn1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_57_out_fcn1[784][21] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_57_params_fcn1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=256, .out_channels=21, .out_stride=21, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=10, .I=784, .J=21, .K=256, .res_scale=0, .dilation=1};


#endif

