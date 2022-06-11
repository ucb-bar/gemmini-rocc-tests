#ifndef RESNET_MT_PARAMS_H
#define RESNET_MT_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_1_w_res1[147][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_1_b_res1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_1_in_res1[12544][147] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_res1[12544][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_res1_pooled[1][56][56][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_1_params_res1 = {.batch_size=1, .in_dim=224, .kernel_size=7, .in_channels=3, .in_stride = 3, .out_channels=64, .out_stride=(64), .stride=2, .padding=3, .bias=1, .depthwise=0, .out_dim=112, .n_patches=12544, .patch_size=147, .pool_size=3, .pool_stride=2, .pool_padding=1, .out_dim_pooled=56, .output_scale=8, .I=12544, .J=64, .K=147, .res_scale=0};


static const elem_t conv_2_w_res1[64][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_2_b_res1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_2_in_res1[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_res1[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_2_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=5, .I=3136, .J=64, .K=64, .res_scale=0};


static const elem_t conv_3_w_res1[576][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_3_b_res1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_3_in_res1[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_res1[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_3_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=9, .I=3136, .J=64, .K=576, .res_scale=0};


static const elem_t conv_4_w_res1[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_4_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_4_in_res1[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_res1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_4_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=256, .K=64, .res_scale=0};


static const elem_t conv_5_w_res1[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_5_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_5_in_res1[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_res1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_5_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=5, .I=3136, .J=256, .K=64, .res_scale=0};


static const elem_t conv_6_w_res1[256][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_6_b_res1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_6_in_res1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_6_out_res1[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_6_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=256, .res_scale=1};


static const elem_t conv_7_w_res1[576][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_7_b_res1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_7_in_res1[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_7_out_res1[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_7_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=576, .res_scale=1};


static const elem_t conv_8_w_res1[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_8_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_8_in_res1[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_8_out_res1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_8_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=256, .K=64, .res_scale=1};


static const elem_t conv_9_w_res1[256][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_9_b_res1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_9_in_res1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_9_out_res1[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_9_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=256, .res_scale=-1};


static const elem_t conv_10_w_res1[576][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_10_b_res1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_10_in_res1[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_10_out_res1[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_10_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=576, .res_scale=-1};


static const elem_t conv_11_w_res1[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_11_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_11_in_res1[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_11_out_res1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_11_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .I=3136, .J=256, .K=64, .res_scale=-1};


static const elem_t conv_12_w_res1[256][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_12_b_res1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_12_in_res1[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_12_out_res1[3136][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_12_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=128, .K=256, .res_scale=0};


static const elem_t conv_13_w_res1[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_13_b_res1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_13_in_res1[784][(1152+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_13_out_res1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_13_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=128, .in_stride=(128+64), .out_channels=128, .out_stride=(128+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_14_w_res1[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_14_b_res1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_14_in_res1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_14_out_res1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_14_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .in_stride=(128+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=512, .K=128, .res_scale=0};


static const elem_t conv_15_w_res1[256][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_15_b_res1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_15_in_res1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_15_out_res1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_15_params_res1 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=512, .out_stride=(512+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=512, .K=256, .res_scale=0};


static const elem_t conv_16_w_res1[512][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_16_b_res1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_16_in_res1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_16_out_res1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_16_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=128, .K=512, .res_scale=0};


static const elem_t conv_17_w_res1[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_17_b_res1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_17_in_res1[784][(1152+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_17_out_res1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_17_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .in_stride=(128+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_18_w_res1[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_18_b_res1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_18_in_res1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_18_out_res1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_18_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .in_stride=(128+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=128, .res_scale=0};


static const elem_t conv_19_w_res1[512][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_19_b_res1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_19_in_res1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_19_out_res1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_19_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=128, .K=512, .res_scale=0};


static const elem_t conv_20_w_res1[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_20_b_res1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_20_in_res1[784][(1152+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_20_out_res1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_20_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .in_stride=(128+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_21_w_res1[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_21_b_res1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_21_in_res1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_21_out_res1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_21_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .in_stride=(128+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=512, .K=128, .res_scale=0};


static const elem_t conv_22_w_res1[512][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_22_b_res1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_22_in_res1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_22_out_res1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_22_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=128, .K=512, .res_scale=0};


static const elem_t conv_23_w_res1[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_23_b_res1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_23_in_res1[784][(1152+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_23_out_res1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_23_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .in_stride=(128+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_24_w_res1[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_24_b_res1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_24_in_res1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_24_out_res1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_24_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .in_stride=(128+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=128, .res_scale=0};


static const elem_t conv_25_w_res1[512][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_25_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_25_in_res1[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_25_out_res1[784][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_25_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=256, .K=512, .res_scale=0};


static const elem_t conv_26_w_res1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_26_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_26_in_res1[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_26_out_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_26_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_27_w_res1[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_27_b_res1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_27_in_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_27_out_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_27_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=1024, .K=256, .res_scale=0};


static const elem_t conv_28_w_res1[512][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_28_b_res1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_28_in_res1[196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_28_out_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_28_params_res1 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=1024, .out_stride=(1024+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=6, .I=196, .J=1024, .K=512, .res_scale=0};


static const elem_t conv_29_w_res1[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_29_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_29_in_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_29_out_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_29_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=1024, .res_scale=1};


static const elem_t conv_30_w_res1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_30_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_30_in_res1[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_30_out_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_30_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=2304, .res_scale=1};


static const elem_t conv_31_w_res1[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_31_b_res1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_31_in_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_31_out_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_31_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=1024, .K=256, .res_scale=1};


static const elem_t conv_32_w_res1[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_32_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_32_in_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_32_out_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_32_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=6, .I=196, .J=256, .K=1024, .res_scale=0};


static const elem_t conv_33_w_res1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_33_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_33_in_res1[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_33_out_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_33_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_34_w_res1[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_34_b_res1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_34_in_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_34_out_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_34_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=9, .I=196, .J=1024, .K=256, .res_scale=0};


static const elem_t conv_35_w_res1[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_35_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_35_in_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_35_out_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_35_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=6, .I=196, .J=256, .K=1024, .res_scale=0};


static const elem_t conv_36_w_res1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_36_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_36_in_res1[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_36_out_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_36_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_37_w_res1[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_37_b_res1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_37_in_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_37_out_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_37_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=1024, .K=256, .res_scale=0};


static const elem_t conv_38_w_res1[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_38_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_38_in_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_38_out_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_38_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=1024, .res_scale=0};


static const elem_t conv_39_w_res1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_39_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_39_in_res1[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_39_out_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_39_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_40_w_res1[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_40_b_res1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_40_in_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_40_out_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_40_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=1024, .K=256, .res_scale=0};


static const elem_t conv_41_w_res1[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_41_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_41_in_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_41_out_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_41_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=1024, .res_scale=-1};


static const elem_t conv_42_w_res1[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_42_b_res1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_42_in_res1[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_42_out_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_42_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=-1};


static const elem_t conv_43_w_res1[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_43_b_res1[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_43_in_res1[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_43_out_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_43_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=6, .I=196, .J=1024, .K=256, .res_scale=-1};


static const elem_t conv_44_w_res1[1024][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_44_b_res1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_44_in_res1[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_44_out_res1[196][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_44_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=512, .K=1024, .res_scale=0};


static const elem_t conv_45_w_res1[4608][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_45_b_res1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_45_in_res1[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_45_out_res1[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_45_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=512, .in_stride=(512+64), .out_channels=512, .out_stride=(512+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=8, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_46_w_res1[512][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_46_b_res1[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_46_in_res1[49][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_46_out_res1[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_46_params_res1 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=5, .I=49, .J=2048, .K=512, .res_scale=0};


static const elem_t conv_47_w_res1[1024][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_47_b_res1[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_47_in_res1[49][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_47_out_res1[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_47_params_res1 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=2048, .out_stride=(2048+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=6, .I=49, .J=2048, .K=1024, .res_scale=0};


static const elem_t conv_48_w_res1[2048][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_48_b_res1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_48_in_res1[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_48_out_res1[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_48_params_res1 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=2048, .in_stride=(2048+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=2048, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=6, .I=49, .J=512, .K=2048, .res_scale=0};


static const elem_t conv_49_w_res1[4608][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_49_b_res1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_49_in_res1[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_49_out_res1[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_49_params_res1 = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=512, .in_stride=(512+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=8, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_50_w_res1[512][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_50_b_res1[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_50_in_res1[49][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_50_out_res1[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_50_params_res1 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=8, .I=49, .J=2048, .K=512, .res_scale=0};


static const elem_t conv_51_w_res1[2048][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_51_b_res1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_51_in_res1[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_51_out_res1[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_51_params_res1 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=2048, .in_stride=(2048+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=2048, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=512, .K=2048, .res_scale=0};


static const elem_t conv_52_w_res1[4608][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_52_b_res1[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_52_in_res1[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_52_out_res1[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_52_params_res1 = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=512, .in_stride=(512+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=8, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_53_w_res1[512][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_53_b_res1[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_53_in_res1[49][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_53_out_res1[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_53_params_res1 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=2048, .K=512, .res_scale=0};


static const elem_t fc_54_w_res1[2048][1024+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_54_b_res1[1][1024] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_54_out_res1[1][1024] row_align(MAX_BLOCK_LEN);
static struct FcParams fc_54_params_res1 = {.batch_size=1, .in_features=2048, .out_features=1024, .bias=1, .output_scale=9, .J=1024, .I=1, .K=2048, .out_stride = (1024+64)};

//#if NUM_CORE == 8
static const elem_t conv_1_w_res11[147][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_1_b_res11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_1_in_res11[12544][147] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_res11[12544][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_res11_pooled[1][56][56][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_1_params_res11 = {.batch_size=1, .in_dim=224, .kernel_size=7, .in_channels=3, .in_stride = 3, .out_channels=64, .out_stride=(64), .stride=2, .padding=3, .bias=1, .depthwise=0, .out_dim=112, .n_patches=12544, .patch_size=147, .pool_size=3, .pool_stride=2, .pool_padding=1, .out_dim_pooled=56, .output_scale=8, .I=12544, .J=64, .K=147, .res_scale=0};


static const elem_t conv_2_w_res11[64][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_2_b_res11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_2_in_res11[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_res11[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_2_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=5, .I=3136, .J=64, .K=64, .res_scale=0};


static const elem_t conv_3_w_res11[576][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_3_b_res11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_3_in_res11[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_res11[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_3_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=9, .I=3136, .J=64, .K=576, .res_scale=0};


static const elem_t conv_4_w_res11[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_4_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_4_in_res11[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_res11[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_4_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=256, .K=64, .res_scale=0};


static const elem_t conv_5_w_res11[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_5_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_5_in_res11[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_res11[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_5_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=5, .I=3136, .J=256, .K=64, .res_scale=0};


static const elem_t conv_6_w_res11[256][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_6_b_res11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_6_in_res11[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_6_out_res11[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_6_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=256, .res_scale=1};


static const elem_t conv_7_w_res11[576][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_7_b_res11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_7_in_res11[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_7_out_res11[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_7_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=576, .res_scale=1};


static const elem_t conv_8_w_res11[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_8_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_8_in_res11[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_8_out_res11[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_8_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=256, .K=64, .res_scale=1};


static const elem_t conv_9_w_res11[256][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_9_b_res11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_9_in_res11[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_9_out_res11[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_9_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=256, .res_scale=-1};


static const elem_t conv_10_w_res11[576][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_10_b_res11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_10_in_res11[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_10_out_res11[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_10_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=576, .res_scale=-1};


static const elem_t conv_11_w_res11[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_11_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_11_in_res11[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_11_out_res11[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_11_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .I=3136, .J=256, .K=64, .res_scale=-1};


static const elem_t conv_12_w_res11[256][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_12_b_res11[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_12_in_res11[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_12_out_res11[3136][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_12_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=128, .K=256, .res_scale=0};


static const elem_t conv_13_w_res11[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_13_b_res11[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_13_in_res11[784][(1152+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_13_out_res11[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_13_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=128, .in_stride=(128+64), .out_channels=128, .out_stride=(128+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_14_w_res11[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_14_b_res11[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_14_in_res11[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_14_out_res11[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_14_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .in_stride=(128+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=512, .K=128, .res_scale=0};


static const elem_t conv_15_w_res11[256][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_15_b_res11[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_15_in_res11[784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_15_out_res11[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_15_params_res11 = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=512, .out_stride=(512+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=512, .K=256, .res_scale=0};


static const elem_t conv_16_w_res11[512][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_16_b_res11[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_16_in_res11[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_16_out_res11[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_16_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=128, .K=512, .res_scale=0};


static const elem_t conv_17_w_res11[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_17_b_res11[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_17_in_res11[784][(1152+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_17_out_res11[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_17_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .in_stride=(128+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_18_w_res11[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_18_b_res11[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_18_in_res11[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_18_out_res11[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_18_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .in_stride=(128+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=128, .res_scale=0};


static const elem_t conv_19_w_res11[512][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_19_b_res11[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_19_in_res11[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_19_out_res11[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_19_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=128, .K=512, .res_scale=0};


static const elem_t conv_20_w_res11[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_20_b_res11[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_20_in_res11[784][(1152+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_20_out_res11[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_20_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .in_stride=(128+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_21_w_res11[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_21_b_res11[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_21_in_res11[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_21_out_res11[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_21_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .in_stride=(128+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=512, .K=128, .res_scale=0};


static const elem_t conv_22_w_res11[512][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_22_b_res11[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_22_in_res11[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_22_out_res11[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_22_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=128, .K=512, .res_scale=0};


static const elem_t conv_23_w_res11[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_23_b_res11[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_23_in_res11[784][(1152+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_23_out_res11[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_23_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .in_stride=(128+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_24_w_res11[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_24_b_res11[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_24_in_res11[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_24_out_res11[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_24_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .in_stride=(128+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=128, .res_scale=0};


static const elem_t conv_25_w_res11[512][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_25_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_25_in_res11[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_25_out_res11[784][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_25_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=256, .K=512, .res_scale=0};


static const elem_t conv_26_w_res11[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_26_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_26_in_res11[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_26_out_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_26_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_27_w_res11[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_27_b_res11[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_27_in_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_27_out_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_27_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=1024, .K=256, .res_scale=0};


static const elem_t conv_28_w_res11[512][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_28_b_res11[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_28_in_res11[196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_28_out_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_28_params_res11 = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=1024, .out_stride=(1024+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=6, .I=196, .J=1024, .K=512, .res_scale=0};


static const elem_t conv_29_w_res11[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_29_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_29_in_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_29_out_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_29_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=1024, .res_scale=1};


static const elem_t conv_30_w_res11[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_30_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_30_in_res11[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_30_out_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_30_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=2304, .res_scale=1};


static const elem_t conv_31_w_res11[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_31_b_res11[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_31_in_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_31_out_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_31_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=1024, .K=256, .res_scale=1};


static const elem_t conv_32_w_res11[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_32_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_32_in_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_32_out_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_32_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=6, .I=196, .J=256, .K=1024, .res_scale=0};


static const elem_t conv_33_w_res11[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_33_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_33_in_res11[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_33_out_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_33_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_34_w_res11[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_34_b_res11[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_34_in_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_34_out_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_34_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=9, .I=196, .J=1024, .K=256, .res_scale=0};


static const elem_t conv_35_w_res11[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_35_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_35_in_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_35_out_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_35_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=6, .I=196, .J=256, .K=1024, .res_scale=0};


static const elem_t conv_36_w_res11[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_36_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_36_in_res11[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_36_out_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_36_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_37_w_res11[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_37_b_res11[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_37_in_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_37_out_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_37_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=1024, .K=256, .res_scale=0};


static const elem_t conv_38_w_res11[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_38_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_38_in_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_38_out_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_38_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=1024, .res_scale=0};


static const elem_t conv_39_w_res11[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_39_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_39_in_res11[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_39_out_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_39_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_40_w_res11[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_40_b_res11[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_40_in_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_40_out_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_40_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=1024, .K=256, .res_scale=0};


static const elem_t conv_41_w_res11[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_41_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_41_in_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_41_out_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_41_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=1024, .res_scale=-1};


static const elem_t conv_42_w_res11[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_42_b_res11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_42_in_res11[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_42_out_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_42_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=-1};


static const elem_t conv_43_w_res11[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_43_b_res11[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_43_in_res11[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_43_out_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_43_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=6, .I=196, .J=1024, .K=256, .res_scale=-1};


static const elem_t conv_44_w_res11[1024][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_44_b_res11[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_44_in_res11[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_44_out_res11[196][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_44_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=512, .K=1024, .res_scale=0};


static const elem_t conv_45_w_res11[4608][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_45_b_res11[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_45_in_res11[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_45_out_res11[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_45_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=512, .in_stride=(512+64), .out_channels=512, .out_stride=(512+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=8, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_46_w_res11[512][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_46_b_res11[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_46_in_res11[49][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_46_out_res11[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_46_params_res11 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=5, .I=49, .J=2048, .K=512, .res_scale=0};


static const elem_t conv_47_w_res11[1024][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_47_b_res11[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_47_in_res11[49][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_47_out_res11[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_47_params_res11 = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=2048, .out_stride=(2048+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=6, .I=49, .J=2048, .K=1024, .res_scale=0};


static const elem_t conv_48_w_res11[2048][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_48_b_res11[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_48_in_res11[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_48_out_res11[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_48_params_res11 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=2048, .in_stride=(2048+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=2048, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=6, .I=49, .J=512, .K=2048, .res_scale=0};


static const elem_t conv_49_w_res11[4608][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_49_b_res11[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_49_in_res11[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_49_out_res11[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_49_params_res11 = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=512, .in_stride=(512+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=8, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_50_w_res11[512][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_50_b_res11[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_50_in_res11[49][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_50_out_res11[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_50_params_res11 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=8, .I=49, .J=2048, .K=512, .res_scale=0};


static const elem_t conv_51_w_res11[2048][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_51_b_res11[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_51_in_res11[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_51_out_res11[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_51_params_res11 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=2048, .in_stride=(2048+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=2048, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=512, .K=2048, .res_scale=0};


static const elem_t conv_52_w_res11[4608][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_52_b_res11[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_52_in_res11[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_52_out_res11[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_52_params_res11 = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=512, .in_stride=(512+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=8, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_53_w_res11[512][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_53_b_res11[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_53_in_res11[49][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_53_out_res11[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_53_params_res11 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=2048, .K=512, .res_scale=0};


static const elem_t fc_54_w_res11[2048][1024+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_54_b_res11[1][1024] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_54_out_res11[1][1024] row_align(MAX_BLOCK_LEN);
static struct FcParams fc_54_params_res11 = {.batch_size=1, .in_features=2048, .out_features=1024, .bias=1, .output_scale=9, .J=1024, .I=1, .K=2048, .out_stride = (1024+64)};

#endif
//#endif

