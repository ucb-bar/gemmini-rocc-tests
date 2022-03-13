#ifndef RESNET_MT_PARAMS_H
#define RESNET_MT_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_1_w[147][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_1_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_1_in[12544][147] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out[12544][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_pooled[1][56][56][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_1_params = {.batch_size=1, .in_dim=224, .kernel_size=7, .in_channels=3, .in_stride = 3, .out_channels=64, .out_stride=(64), .stride=2, .padding=3, .bias=1, .depthwise=0, .out_dim=112, .n_patches=12544, .patch_size=147, .pool_size=3, .pool_stride=2, .pool_padding=1, .out_dim_pooled=56, .output_scale=8, .I=12544, .J=64, .K=147, .res_scale=0};


static const elem_t conv_2_w[64][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_2_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_2_in[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_2_params = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=5, .I=3136, .J=64, .K=64, .res_scale=0};


static const elem_t conv_3_w[576][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_3_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_3_in[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_3_params = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=9, .I=3136, .J=64, .K=576, .res_scale=0};


static const elem_t conv_4_w[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_4_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_4_in[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_4_params = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=256, .K=64, .res_scale=0};


static const elem_t conv_5_w[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_5_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_5_in[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_5_params = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=5, .I=3136, .J=256, .K=64, .res_scale=0};


static const elem_t conv_6_w[256][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_6_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_6_in[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_6_out[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_6_params = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=256, .res_scale=1};


static const elem_t conv_7_w[576][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_7_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_7_in[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_7_out[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_7_params = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=576, .res_scale=1};


static const elem_t conv_8_w[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_8_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_8_in[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_8_out[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_8_params = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=256, .K=64, .res_scale=1};


static const elem_t conv_9_w[256][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_9_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_9_in[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_9_out[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_9_params = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=256, .res_scale=-1};


static const elem_t conv_10_w[576][64] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_10_b[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_10_in[3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_10_out[3136][64] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_10_params = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=64, .in_stride=(64), .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=64, .K=576, .res_scale=-1};


static const elem_t conv_11_w[64][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_11_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_11_in[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_11_out[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_11_params = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=64, .in_stride=(64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .I=3136, .J=256, .K=64, .res_scale=-1};


static const elem_t conv_12_w[256][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_12_b[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_12_in[3136][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_12_out[3136][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_12_params = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=3136, .J=128, .K=256, .res_scale=0};


static const elem_t conv_13_w[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_13_b[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_13_in[784][(1152+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_13_out[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_13_params = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=128, .in_stride=(128+64), .out_channels=128, .out_stride=(128+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_14_w[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_14_b[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_14_in[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_14_out[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_14_params = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .in_stride=(128+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=512, .K=128, .res_scale=0};


static const elem_t conv_15_w[256][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_15_b[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_15_in[784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_15_out[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_15_params = {.batch_size=1, .in_dim=56, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=512, .out_stride=(512+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=512, .K=256, .res_scale=0};


static const elem_t conv_16_w[512][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_16_b[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_16_in[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_16_out[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_16_params = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=128, .K=512, .res_scale=0};


static const elem_t conv_17_w[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_17_b[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_17_in[784][(1152+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_17_out[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_17_params = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .in_stride=(128+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_18_w[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_18_b[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_18_in[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_18_out[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_18_params = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .in_stride=(128+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=128, .res_scale=0};


static const elem_t conv_19_w[512][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_19_b[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_19_in[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_19_out[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_19_params = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=128, .K=512, .res_scale=0};


static const elem_t conv_20_w[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_20_b[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_20_in[784][(1152+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_20_out[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_20_params = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .in_stride=(128+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_21_w[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_21_b[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_21_in[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_21_out[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_21_params = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .in_stride=(128+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=512, .K=128, .res_scale=0};


static const elem_t conv_22_w[512][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_22_b[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_22_in[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_22_out[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_22_params = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=128, .K=512, .res_scale=0};


static const elem_t conv_23_w[1152][(128+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_23_b[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_23_in[784][(1152+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_23_out[784][(128+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_23_params = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=128, .in_stride=(128+64), .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=8, .I=784, .J=128, .K=1152, .res_scale=0};


static const elem_t conv_24_w[128][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_24_b[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_24_in[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_24_out[784][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_24_params = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=128, .in_stride=(128+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=128, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=784, .J=512, .K=128, .res_scale=0};


static const elem_t conv_25_w[512][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_25_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_25_in[784][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_25_out[784][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_25_params = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=6, .I=784, .J=256, .K=512, .res_scale=0};


static const elem_t conv_26_w[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_26_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_26_in[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_26_out[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_26_params = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_27_w[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_27_b[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_27_in[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_27_out[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_27_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=1024, .K=256, .res_scale=0};


static const elem_t conv_28_w[512][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_28_b[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_28_in[196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_28_out[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_28_params = {.batch_size=1, .in_dim=28, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=1024, .out_stride=(1024+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=6, .I=196, .J=1024, .K=512, .res_scale=0};


static const elem_t conv_29_w[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_29_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_29_in[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_29_out[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_29_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=1024, .res_scale=1};


static const elem_t conv_30_w[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_30_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_30_in[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_30_out[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_30_params = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=2304, .res_scale=1};


static const elem_t conv_31_w[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_31_b[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_31_in[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_31_out[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_31_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=1024, .K=256, .res_scale=1};


static const elem_t conv_32_w[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_32_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_32_in[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_32_out[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_32_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=6, .I=196, .J=256, .K=1024, .res_scale=0};


static const elem_t conv_33_w[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_33_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_33_in[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_33_out[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_33_params = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_34_w[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_34_b[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_34_in[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_34_out[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_34_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=9, .I=196, .J=1024, .K=256, .res_scale=0};


static const elem_t conv_35_w[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_35_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_35_in[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_35_out[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_35_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=6, .I=196, .J=256, .K=1024, .res_scale=0};


static const elem_t conv_36_w[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_36_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_36_in[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_36_out[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_36_params = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_37_w[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_37_b[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_37_in[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_37_out[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_37_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=1024, .K=256, .res_scale=0};


static const elem_t conv_38_w[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_38_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_38_in[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_38_out[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_38_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=1024, .res_scale=0};


static const elem_t conv_39_w[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_39_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_39_in[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_39_out[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_39_params = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=2304, .res_scale=0};


static const elem_t conv_40_w[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_40_b[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_40_in[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_40_out[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_40_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=1024, .K=256, .res_scale=0};


static const elem_t conv_41_w[1024][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_41_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_41_in[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_41_out[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_41_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=256, .K=1024, .res_scale=-1};


static const elem_t conv_42_w[2304][(256+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_42_b[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_42_in[196][(2304+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_42_out[196][(256+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_42_params = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=256, .in_stride=(256+64), .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=2304, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=8, .I=196, .J=256, .K=2304, .res_scale=-1};


static const elem_t conv_43_w[256][(1024+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_43_b[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_43_in[196][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_43_out[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_43_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=256, .in_stride=(256+64), .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=6, .I=196, .J=1024, .K=256, .res_scale=-1};


static const elem_t conv_44_w[1024][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_44_b[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_44_in[196][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_44_out[196][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_44_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=196, .J=512, .K=1024, .res_scale=0};


static const elem_t conv_45_w[4608][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_45_b[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_45_in[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_45_out[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_45_params = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=512, .in_stride=(512+64), .out_channels=512, .out_stride=(512+64), .stride=2, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=8, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_46_w[512][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_46_b[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_46_in[49][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_46_out[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_46_params = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=5, .I=49, .J=2048, .K=512, .res_scale=0};


static const elem_t conv_47_w[1024][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_47_b[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_47_in[49][(1024+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_47_out[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_47_params = {.batch_size=1, .in_dim=14, .kernel_size=1, .in_channels=1024, .in_stride=(1024+64), .out_channels=2048, .out_stride=(2048+64), .stride=2, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=1024, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=6, .I=49, .J=2048, .K=1024, .res_scale=0};


static const elem_t conv_48_w[2048][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_48_b[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_48_in[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_48_out[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_48_params = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=2048, .in_stride=(2048+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=2048, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=6, .I=49, .J=512, .K=2048, .res_scale=0};


static const elem_t conv_49_w[4608][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_49_b[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_49_in[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_49_out[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_49_params = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=512, .in_stride=(512+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=8, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_50_w[512][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_50_b[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_50_in[49][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_50_out[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_50_params = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=8, .I=49, .J=2048, .K=512, .res_scale=0};


static const elem_t conv_51_w[2048][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_51_b[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_51_in[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_51_out[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_51_params = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=2048, .in_stride=(2048+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=2048, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=512, .K=2048, .res_scale=0};


static const elem_t conv_52_w[4608][(512+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_52_b[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_52_in[49][4608] row_align(MAX_BLOCK_LEN);
static elem_t conv_52_out[49][(512+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_52_params = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=512, .in_stride=(512+64), .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=4608, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=8, .I=49, .J=512, .K=4608, .res_scale=0};


static const elem_t conv_53_w[512][(2048+64)] row_align(MAX_BLOCK_LEN);// = 
static const acc_t conv_53_b[(2048+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = { 
static elem_t conv_53_in[49][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_53_out[49][(2048+64)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_53_params = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=512, .in_stride=(512+64), .out_channels=2048, .out_stride=(2048+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=49, .J=2048, .K=512, .res_scale=0};


static const elem_t fc_54_w[2048][1024+64] row_align(MAX_BLOCK_LEN);// =
static const acc_t fc_54_b[1][1024] row_align_acc(MAX_BLOCK_LEN_ACC); 
static elem_t fc_54_out[1][1024] row_align(MAX_BLOCK_LEN);
static struct FcParams fc_54_params = {.batch_size=1, .in_features=2048, .out_features=1024, .bias=1, .output_scale=9, .J=1024, .I=1, .K=2048, .out_stride = (1024+64)};


#endif

