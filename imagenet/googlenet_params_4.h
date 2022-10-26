#ifndef GOOGLENET4_PARAMETERS_H
#define GOOGLENET4_PARAMETERS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

#define GOOGLE_BATCH 4

static const elem_t conv_1_w_google4[147][64] row_align(MAX_BLOCK_LEN); static const acc_t conv_1_b_google4[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_1_in_google4[GOOGLE_BATCH*12544][147] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_google4[GOOGLE_BATCH*12544][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_google4_pooled[GOOGLE_BATCH][56][56][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=224, .kernel_size=7, .in_channels=3, .out_channels=64, .out_stride=(64), .stride=2, .padding=3, .bias=1, .depthwise=0, .out_dim=112, .n_patches=12544, .patch_size=147, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=56, .output_scale=0.00390625, .I=GOOGLE_BATCH*12544, .J=64, .K=147, .res_scale=0};


static const elem_t conv_2_w_google4[64][64] row_align(MAX_BLOCK_LEN); static const acc_t conv_2_b_google4[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_2_in_google4[GOOGLE_BATCH*3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_google4[GOOGLE_BATCH*3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=56, .kernel_size=1, .in_channels=64, .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=64, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=0.03125, .I=GOOGLE_BATCH*3136, .J=64, .K=64, .res_scale=0};


static const elem_t conv_3_w_google4[576][192] row_align(MAX_BLOCK_LEN); static const acc_t conv_3_b_google4[192] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_3_in_google4[GOOGLE_BATCH*3136][576] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_google4[GOOGLE_BATCH*3136][192] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_google4_pooled[GOOGLE_BATCH][28][28][192] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=192, .out_stride=(192), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .n_patches=3136, .patch_size=576, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.015625, .I=GOOGLE_BATCH*3136, .J=192, .K=576, .res_scale=0};


static elem_t inception3a_out_google4[GOOGLE_BATCH*1][28][28][(256+64)] row_align(MAX_BLOCK_LEN);


static const elem_t conv_4_w_google4[192][64] row_align(MAX_BLOCK_LEN); static const acc_t conv_4_b_google4[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_4_in_google4[GOOGLE_BATCH*784][192] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=28, .kernel_size=1, .in_channels=192, .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=192, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.0078125, .I=GOOGLE_BATCH*784, .J=64, .K=192, .res_scale=0};


static const elem_t conv_5_w_google4[192][96] row_align(MAX_BLOCK_LEN); static const acc_t conv_5_b_google4[96] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_5_in_google4[GOOGLE_BATCH*784][192] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_google4[GOOGLE_BATCH*784][96] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=28, .kernel_size=1, .in_channels=192, .out_channels=96, .out_stride=(96), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=192, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.015625, .I=GOOGLE_BATCH*784, .J=96, .K=192, .res_scale=0};


static const elem_t conv_6_w_google4[864][(128+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_6_b_google4[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_6_in_google4[GOOGLE_BATCH*784][864] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_6_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=28, .kernel_size=3, .in_channels=96, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=864, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.0078125, .I=GOOGLE_BATCH*784, .J=128, .K=864, .res_scale=0};


static const elem_t conv_7_w_google4[192][16] row_align(MAX_BLOCK_LEN); static const acc_t conv_7_b_google4[16] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_7_in_google4[GOOGLE_BATCH*784][192] row_align(MAX_BLOCK_LEN);
static elem_t conv_7_out_google4[GOOGLE_BATCH*784][16] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_7_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=28, .kernel_size=1, .in_channels=192, .out_channels=16, .out_stride=(16), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=192, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.0078125, .I=GOOGLE_BATCH*784, .J=16, .K=192, .res_scale=0};


static const elem_t conv_8_w_google4[144][32] row_align(MAX_BLOCK_LEN); static const acc_t conv_8_b_google4[32] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_8_in_google4[GOOGLE_BATCH*784][144] row_align(MAX_BLOCK_LEN);
static elem_t conv_8_out_google4[GOOGLE_BATCH*784][32] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_8_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=28, .kernel_size=3, .in_channels=16, .out_channels=32, .out_stride=(32), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=144, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.015625, .I=GOOGLE_BATCH*784, .J=32, .K=144, .res_scale=0};


static elem_t pool_9_out_google4[GOOGLE_BATCH*1][28][28][192] row_align(MAX_BLOCK_LEN);
static const struct ConvParams pool_9_params_google4 = {.batch_size=GOOGLE_BATCH, .in_channels=192, .out_channels=192, .out_stride=(192), .out_dim=28, .pool_size=3, .pool_stride=1, .pool_padding=1, .out_dim_pooled=28};


static const elem_t conv_10_w_google4[192][32] row_align(MAX_BLOCK_LEN); static const acc_t conv_10_b_google4[32] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_10_in_google4[GOOGLE_BATCH*784][192] row_align(MAX_BLOCK_LEN);
static elem_t conv_10_out_google4[GOOGLE_BATCH*784][32] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_10_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=28, .kernel_size=1, .in_channels=192, .out_channels=32, .out_stride=(32), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=192, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.015625, .I=GOOGLE_BATCH*784, .J=32, .K=192, .res_scale=0};


static elem_t inception3b_out_google4[GOOGLE_BATCH*1][28][28][480] row_align(MAX_BLOCK_LEN);


static const elem_t conv_11_w_google4[256][(128+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_11_b_google4[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_11_in_google4[GOOGLE_BATCH*784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_11_out_google4[GOOGLE_BATCH*784][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_11_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=28, .kernel_size=1, .in_channels=256, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.015625, .I=GOOGLE_BATCH*784, .J=128, .K=256, .res_scale=0};


static const elem_t conv_12_w_google4[256][(128+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_12_b_google4[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_12_in_google4[GOOGLE_BATCH*784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_12_out_google4[GOOGLE_BATCH*784][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_12_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=28, .kernel_size=1, .in_channels=256, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.0078125, .I=GOOGLE_BATCH*784, .J=128, .K=256, .res_scale=0};


static const elem_t conv_13_w_google4[1152][192] row_align(MAX_BLOCK_LEN); static const acc_t conv_13_b_google4[192] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_13_in_google4[GOOGLE_BATCH*784][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_13_out_google4[GOOGLE_BATCH*784][192] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_13_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=192, .out_stride=(192), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.0078125, .I=GOOGLE_BATCH*784, .J=192, .K=1152, .res_scale=0};


static const elem_t conv_14_w_google4[256][32] row_align(MAX_BLOCK_LEN); static const acc_t conv_14_b_google4[32] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_14_in_google4[GOOGLE_BATCH*784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_14_out_google4[GOOGLE_BATCH*784][32] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_14_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=28, .kernel_size=1, .in_channels=256, .out_channels=32, .out_stride=(32), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.0078125, .I=GOOGLE_BATCH*784, .J=32, .K=256, .res_scale=0};


static const elem_t conv_15_w_google4[288][96] row_align(MAX_BLOCK_LEN); static const acc_t conv_15_b_google4[96] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_15_in_google4[GOOGLE_BATCH*784][288] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_15_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=28, .kernel_size=3, .in_channels=32, .out_channels=96, .out_stride=(96), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=288, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.0078125, .I=GOOGLE_BATCH*784, .J=96, .K=288, .res_scale=0};


static elem_t pool_16_out_google4[GOOGLE_BATCH*1][28][28][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams pool_16_params_google4 = {.batch_size=GOOGLE_BATCH, .in_channels=256, .out_channels=256, .out_stride=(256+64), .out_dim=28, .pool_size=3, .pool_stride=1, .pool_padding=1, .out_dim_pooled=28};


static const elem_t conv_17_w_google4[256][64] row_align(MAX_BLOCK_LEN); static const acc_t conv_17_b_google4[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_17_in_google4[GOOGLE_BATCH*784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_17_out_google4[GOOGLE_BATCH*784][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_17_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=28, .kernel_size=1, .in_channels=256, .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .n_patches=784, .patch_size=256, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=0.0078125, .I=GOOGLE_BATCH*784, .J=64, .K=256, .res_scale=0};

static elem_t pool_18_in_google4[GOOGLE_BATCH*1][28][28][480] row_align(MAX_BLOCK_LEN);
static elem_t pool_18_out_google4[GOOGLE_BATCH*1][14][14][480] row_align(MAX_BLOCK_LEN);
static const struct ConvParams pool_18_params_google4 = {.batch_size=GOOGLE_BATCH, .in_channels=480, .out_channels=480, .out_stride=(480), .out_dim=28, .pool_size=3, .pool_stride=2, .pool_padding=0, .out_dim_pooled=14};


static elem_t inception4a_out_google4[GOOGLE_BATCH*1][14][14][(512+64)] row_align(MAX_BLOCK_LEN);


static const elem_t conv_19_w_google4[480][192] row_align(MAX_BLOCK_LEN); static const acc_t conv_19_b_google4[192] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_19_in_google4[GOOGLE_BATCH*196][480] row_align(MAX_BLOCK_LEN);
static elem_t conv_19_out_google4[GOOGLE_BATCH*196][192] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_19_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=480, .out_channels=192, .out_stride=(192), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=480, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=192, .K=480, .res_scale=0};


static const elem_t conv_20_w_google4[480][96] row_align(MAX_BLOCK_LEN); static const acc_t conv_20_b_google4[96] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_20_in_google4[GOOGLE_BATCH*196][480] row_align(MAX_BLOCK_LEN);
static elem_t conv_20_out_google4[GOOGLE_BATCH*196][96] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_20_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=480, .out_channels=96, .out_stride=(96), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=480, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=96, .K=480, .res_scale=0};


static const elem_t conv_21_w_google4[864][208] row_align(MAX_BLOCK_LEN); static const acc_t conv_21_b_google4[208] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_21_in_google4[GOOGLE_BATCH*196][864] row_align(MAX_BLOCK_LEN);
static elem_t conv_21_out_google4[GOOGLE_BATCH*196][208] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_21_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=3, .in_channels=96, .out_channels=208, .out_stride=(208), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=864, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=208, .K=864, .res_scale=0};


static const elem_t conv_22_w_google4[480][16] row_align(MAX_BLOCK_LEN); static const acc_t conv_22_b_google4[16] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_22_in_google4[GOOGLE_BATCH*196][480] row_align(MAX_BLOCK_LEN);
static elem_t conv_22_out_google4[GOOGLE_BATCH*196][16] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_22_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=480, .out_channels=16, .out_stride=(16), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=480, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.00390625, .I=GOOGLE_BATCH*196, .J=16, .K=480, .res_scale=0};


static const elem_t conv_23_w_google4[144][48] row_align(MAX_BLOCK_LEN); static const acc_t conv_23_b_google4[48] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_23_in_google4[GOOGLE_BATCH*196][144] row_align(MAX_BLOCK_LEN);
static elem_t conv_23_out_google4[GOOGLE_BATCH*196][48] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_23_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=3, .in_channels=16, .out_channels=48, .out_stride=(48), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=144, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.015625, .I=GOOGLE_BATCH*196, .J=48, .K=144, .res_scale=0};


static elem_t pool_24_out_google4[GOOGLE_BATCH*1][14][14][480] row_align(MAX_BLOCK_LEN);
static const struct ConvParams pool_24_params_google4 = {.batch_size=GOOGLE_BATCH, .in_channels=480, .out_channels=480, .out_stride=(480), .out_dim=14, .pool_size=3, .pool_stride=1, .pool_padding=1, .out_dim_pooled=14};


static const elem_t conv_25_w_google4[480][64] row_align(MAX_BLOCK_LEN); static const acc_t conv_25_b_google4[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_25_in_google4[GOOGLE_BATCH*196][480] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_25_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=480, .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=480, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=64, .K=480, .res_scale=0};


static elem_t inception4b_out_google4[GOOGLE_BATCH*1][14][14][(512+64)] row_align(MAX_BLOCK_LEN);


static const elem_t conv_26_w_google4[512][160] row_align(MAX_BLOCK_LEN); static const acc_t conv_26_b_google4[160] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_26_in_google4[GOOGLE_BATCH*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_26_out_google4[GOOGLE_BATCH*196][160] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_26_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=160, .out_stride=(160), .out_stride=160, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=160, .K=512, .res_scale=0};


static const elem_t conv_27_w_google4[512][112] row_align(MAX_BLOCK_LEN); static const acc_t conv_27_b_google4[112] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_27_in_google4[GOOGLE_BATCH*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_27_out_google4[GOOGLE_BATCH*196][112] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_27_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=112, .out_stride=(112), .out_stride=112, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=112, .K=512, .res_scale=0};


static const elem_t conv_28_w_google4[1008][224] row_align(MAX_BLOCK_LEN); static const acc_t conv_28_b_google4[224] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_28_in_google4[GOOGLE_BATCH*196][1008] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_28_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=3, .in_channels=112, .out_channels=224, .out_stride=224, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1008, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.015625, .I=GOOGLE_BATCH*196, .J=224, .K=1008, .res_scale=0};


static const elem_t conv_29_w_google4[512][24] row_align(MAX_BLOCK_LEN); static const acc_t conv_29_b_google4[24] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_29_in_google4[GOOGLE_BATCH*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_29_out_google4[GOOGLE_BATCH*196][24] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_29_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=24, .out_stride=(24), .out_stride=24, .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.00390625, .I=GOOGLE_BATCH*196, .J=24, .K=512, .res_scale=0};


static const elem_t conv_30_w_google4[216][64] row_align(MAX_BLOCK_LEN); static const acc_t conv_30_b_google4[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_30_in_google4[GOOGLE_BATCH*196][216] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_30_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=3, .in_channels=24, .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=216, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=64, .K=216, .res_scale=0};


static elem_t pool_31_out_google4[GOOGLE_BATCH*1][14][14][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams pool_31_params_google4 = {.batch_size=GOOGLE_BATCH, .in_channels=512, .out_channels=512, .out_stride=(512+64), .out_dim=14, .pool_size=3, .pool_stride=1, .pool_padding=1, .out_dim_pooled=14};


static const elem_t conv_32_w_google4[512][64] row_align(MAX_BLOCK_LEN); static const acc_t conv_32_b_google4[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_32_in_google4[GOOGLE_BATCH*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_32_out_google4[GOOGLE_BATCH*196][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_32_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.00390625, .I=GOOGLE_BATCH*196, .J=64, .K=512, .res_scale=0};


static elem_t inception4c_out_google4[GOOGLE_BATCH*1][14][14][(512+64)] row_align(MAX_BLOCK_LEN);


static const elem_t conv_33_w_google4[512][(128+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_33_b_google4[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_33_in_google4[GOOGLE_BATCH*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_33_out_google4[GOOGLE_BATCH*196][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_33_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=128, .K=512, .res_scale=0};


static const elem_t conv_34_w_google4[512][(128+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_34_b_google4[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_34_in_google4[GOOGLE_BATCH*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_34_out_google4[GOOGLE_BATCH*196][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_34_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=128, .K=512, .res_scale=0};


static const elem_t conv_35_w_google4[1152][(256+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_35_b_google4[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_35_in_google4[GOOGLE_BATCH*196][1152] row_align(MAX_BLOCK_LEN);
static elem_t conv_35_out_google4[GOOGLE_BATCH*196][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_35_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=3, .in_channels=128, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1152, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.00390625, .I=GOOGLE_BATCH*196, .J=256, .K=1152, .res_scale=0};


static const elem_t conv_36_w_google4[512][24] row_align(MAX_BLOCK_LEN); static const acc_t conv_36_b_google4[24] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_36_in_google4[GOOGLE_BATCH*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_36_out_google4[GOOGLE_BATCH*196][24] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_36_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=24, .out_stride=(24), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.015625, .I=GOOGLE_BATCH*196, .J=24, .K=512, .res_scale=0};


static const elem_t conv_37_w_google4[216][64] row_align(MAX_BLOCK_LEN); static const acc_t conv_37_b_google4[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_37_in_google4[GOOGLE_BATCH*196][216] row_align(MAX_BLOCK_LEN);
static elem_t conv_37_out_google4[GOOGLE_BATCH*196][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_37_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=3, .in_channels=24, .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=216, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.015625, .I=GOOGLE_BATCH*196, .J=64, .K=216, .res_scale=0};


static elem_t pool_38_out_google4[GOOGLE_BATCH*1][14][14][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams pool_38_params_google4 = {.batch_size=GOOGLE_BATCH, .in_channels=512, .out_channels=512, .out_stride=(512+64), .out_dim=14, .pool_size=3, .pool_stride=1, .pool_padding=1, .out_dim_pooled=14};


static const elem_t conv_39_w_google4[512][64] row_align(MAX_BLOCK_LEN); static const acc_t conv_39_b_google4[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_39_in_google4[GOOGLE_BATCH*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_39_out_google4[GOOGLE_BATCH*196][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_39_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.00390625, .I=GOOGLE_BATCH*196, .J=64, .K=512, .res_scale=0};


static elem_t inception4d_out_google4[GOOGLE_BATCH*1][14][14][528] row_align(MAX_BLOCK_LEN);


static const elem_t conv_40_w_google4[512][112] row_align(MAX_BLOCK_LEN); static const acc_t conv_40_b_google4[112] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_40_in_google4[GOOGLE_BATCH*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_40_out_google4[GOOGLE_BATCH*196][112] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_40_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=112, .out_stride=(112), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=112, .K=512, .res_scale=0};


static const elem_t conv_41_w_google4[512][144] row_align(MAX_BLOCK_LEN); static const acc_t conv_41_b_google4[144] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_41_in_google4[GOOGLE_BATCH*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_41_out_google4[GOOGLE_BATCH*196][144] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_41_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=144, .out_stride=(144), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=144, .K=512, .res_scale=0};


static const elem_t conv_42_w_google4[1296][288] row_align(MAX_BLOCK_LEN); static const acc_t conv_42_b_google4[288] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_42_in_google4[GOOGLE_BATCH*196][1296] row_align(MAX_BLOCK_LEN);
static elem_t conv_42_out_google4[GOOGLE_BATCH*196][288] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_42_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=3, .in_channels=144, .out_channels=288, .out_stride=(288), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1296, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.00390625, .I=GOOGLE_BATCH*196, .J=288, .K=1296, .res_scale=0};


static const elem_t conv_43_w_google4[512][32] row_align(MAX_BLOCK_LEN); static const acc_t conv_43_b_google4[32] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_43_in_google4[GOOGLE_BATCH*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_43_out_google4[GOOGLE_BATCH*196][32] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_43_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=32, .out_stride=(32), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=32, .K=512, .res_scale=0};


static const elem_t conv_44_w_google4[288][64] row_align(MAX_BLOCK_LEN); static const acc_t conv_44_b_google4[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_44_in_google4[GOOGLE_BATCH*196][288] row_align(MAX_BLOCK_LEN);
static elem_t conv_44_out_google4[GOOGLE_BATCH*196][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_44_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=3, .in_channels=32, .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=288, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=64, .K=288, .res_scale=0};


static elem_t pool_45_out_google4[GOOGLE_BATCH*1][14][14][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams pool_45_params_google4 = {.batch_size=GOOGLE_BATCH, .in_channels=512, .out_channels=512, .out_stride=(512+64), .out_dim=14, .pool_size=3, .pool_stride=1, .pool_padding=1, .out_dim_pooled=14};


static const elem_t conv_46_w_google4[512][64] row_align(MAX_BLOCK_LEN); static const acc_t conv_46_b_google4[64] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_46_in_google4[GOOGLE_BATCH*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_46_out_google4[GOOGLE_BATCH*196][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_46_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=512, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.00390625, .I=GOOGLE_BATCH*196, .J=64, .K=512, .res_scale=0};


static elem_t inception4e_out_google4[GOOGLE_BATCH*1][14][14][832] row_align(MAX_BLOCK_LEN);


static const elem_t conv_47_w_google4[528][(256+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_47_b_google4[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_47_in_google4[GOOGLE_BATCH*196][528] row_align(MAX_BLOCK_LEN);
static elem_t conv_47_out_google4[GOOGLE_BATCH*196][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_47_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=528, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=528, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=256, .K=528, .res_scale=0};


static const elem_t conv_48_w_google4[528][160] row_align(MAX_BLOCK_LEN); static const acc_t conv_48_b_google4[160] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_48_in_google4[GOOGLE_BATCH*196][528] row_align(MAX_BLOCK_LEN);
static elem_t conv_48_out_google4[GOOGLE_BATCH*196][160] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_48_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=528, .out_channels=160, .out_stride=(160), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=528, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.00390625, .I=GOOGLE_BATCH*196, .J=160, .K=528, .res_scale=0};


static const elem_t conv_49_w_google4[1440][320] row_align(MAX_BLOCK_LEN); static const acc_t conv_49_b_google4[320] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_49_in_google4[GOOGLE_BATCH*196][1440] row_align(MAX_BLOCK_LEN);
static elem_t conv_49_out_google4[GOOGLE_BATCH*196][320] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_49_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=3, .in_channels=160, .out_channels=320, .out_stride=(320), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=1440, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.0078125, .I=GOOGLE_BATCH*196, .J=320, .K=1440, .res_scale=0};


static const elem_t conv_50_w_google4[528][32] row_align(MAX_BLOCK_LEN); static const acc_t conv_50_b_google4[32] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_50_in_google4[GOOGLE_BATCH*196][528] row_align(MAX_BLOCK_LEN);
static elem_t conv_50_out_google4[GOOGLE_BATCH*196][32] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_50_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=528, .out_channels=32, .out_stride=(32), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=528, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.015625, .I=GOOGLE_BATCH*196, .J=32, .K=528, .res_scale=0};


static const elem_t conv_51_w_google4[288][(128+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_51_b_google4[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_51_in_google4[GOOGLE_BATCH*196][288] row_align(MAX_BLOCK_LEN);
static elem_t conv_51_out_google4[GOOGLE_BATCH*196][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_51_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=3, .in_channels=32, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=288, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.00390625, .I=GOOGLE_BATCH*196, .J=128, .K=288, .res_scale=0};


static elem_t pool_52_out_google4[GOOGLE_BATCH*1][14][14][528] row_align(MAX_BLOCK_LEN);
static const struct ConvParams pool_52_params_google4 = {.batch_size=GOOGLE_BATCH, .in_channels=528, .out_channels=528, .out_stride=(528), .out_dim=14, .pool_size=3, .pool_stride=1, .pool_padding=1, .out_dim_pooled=14};


static const elem_t conv_53_w_google4[528][(128+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_53_b_google4[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_53_in_google4[GOOGLE_BATCH*196][528] row_align(MAX_BLOCK_LEN);
static elem_t conv_53_out_google4[GOOGLE_BATCH*196][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_53_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=14, .kernel_size=1, .in_channels=528, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .n_patches=196, .patch_size=528, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=0.015625, .I=GOOGLE_BATCH*196, .J=128, .K=528, .res_scale=0};


static elem_t pool_54_in_google4[GOOGLE_BATCH*1][14][14][832] row_align(MAX_BLOCK_LEN);
static elem_t pool_54_out_google4[GOOGLE_BATCH*1][7][7][832] row_align(MAX_BLOCK_LEN);
static const struct ConvParams pool_54_params_google4 = {.batch_size=GOOGLE_BATCH, .in_channels=832, .out_channels=832, .out_stride=(832), .out_dim=14, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=7};


static elem_t inception5a_out_google4[GOOGLE_BATCH*1][7][7][832] row_align(MAX_BLOCK_LEN);

static const elem_t conv_55_w_google4[832][(256+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_55_b_google4[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_55_in_google4[GOOGLE_BATCH*49][832] row_align(MAX_BLOCK_LEN);
static elem_t conv_55_out_google4[GOOGLE_BATCH*49][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_55_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=7, .kernel_size=1, .in_channels=832, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=832, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=0.0078125, .I=GOOGLE_BATCH*49, .J=256, .K=832, .res_scale=0};


static const elem_t conv_56_w_google4[832][160] row_align(MAX_BLOCK_LEN); static const acc_t conv_56_b_google4[160] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_56_in_google4[GOOGLE_BATCH*49][832] row_align(MAX_BLOCK_LEN);
static elem_t conv_56_out_google4[GOOGLE_BATCH*49][160] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_56_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=7, .kernel_size=1, .in_channels=832, .out_channels=160, .out_stride=(160), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=832, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=0.0078125, .I=GOOGLE_BATCH*49, .J=160, .K=832, .res_scale=0};


static const elem_t conv_57_w_google4[1440][320] row_align(MAX_BLOCK_LEN); static const acc_t conv_57_b_google4[320] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_57_in_google4[GOOGLE_BATCH*49][1440] row_align(MAX_BLOCK_LEN);
static elem_t conv_57_out_google4[GOOGLE_BATCH*49][320] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_57_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=7, .kernel_size=3, .in_channels=160, .out_channels=320, .out_stride=(320), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=1440, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=0.0078125, .I=GOOGLE_BATCH*49, .J=320, .K=1440, .res_scale=0};


static const elem_t conv_58_w_google4[832][32] row_align(MAX_BLOCK_LEN); static const acc_t conv_58_b_google4[32] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_58_in_google4[GOOGLE_BATCH*49][832] row_align(MAX_BLOCK_LEN);
static elem_t conv_58_out_google4[GOOGLE_BATCH*49][32] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_58_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=7, .kernel_size=1, .in_channels=832, .out_channels=32, .out_stride=(32), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=832, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=0.0078125, .I=GOOGLE_BATCH*49, .J=32, .K=832, .res_scale=0};


static const elem_t conv_59_w_google4[288][(128+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_59_b_google4[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_59_in_google4[GOOGLE_BATCH*49][288] row_align(MAX_BLOCK_LEN);
static elem_t conv_59_out_google4[GOOGLE_BATCH*49][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_59_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=7, .kernel_size=3, .in_channels=32, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=288, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=0.00390625, .I=GOOGLE_BATCH*49, .J=128, .K=288, .res_scale=0};


static elem_t pool_60_out_google4[GOOGLE_BATCH*1][7][7][832] row_align(MAX_BLOCK_LEN);
static const struct ConvParams pool_60_params_google4 = {.batch_size=GOOGLE_BATCH, .in_channels=832, .out_channels=832, .out_stride=(832), .out_dim=7, .pool_size=3, .pool_stride=1, .pool_padding=1, .out_dim_pooled=7};


static const elem_t conv_61_w_google4[832][(128+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_61_b_google4[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_61_in_google4[GOOGLE_BATCH*49][832] row_align(MAX_BLOCK_LEN);
static elem_t conv_61_out_google4[GOOGLE_BATCH*49][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_61_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=7, .kernel_size=1, .in_channels=832, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=832, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=0.00390625, .I=GOOGLE_BATCH*49, .J=128, .K=832, .res_scale=0};


static elem_t inception5b_out_google4[GOOGLE_BATCH*1][7][7][(1024+64)] row_align(MAX_BLOCK_LEN);


static const elem_t conv_62_w_google4[832][(384+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_62_b_google4[(384+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_62_in_google4[GOOGLE_BATCH*49][832] row_align(MAX_BLOCK_LEN);
static elem_t conv_62_out_google4[GOOGLE_BATCH*49][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_62_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=7, .kernel_size=1, .in_channels=832, .out_channels=384, .out_stride=(384+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=832, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=0.03125, .I=GOOGLE_BATCH*49, .J=384, .K=832, .res_scale=0};


static const elem_t conv_63_w_google4[832][192] row_align(MAX_BLOCK_LEN); static const acc_t conv_63_b_google4[192] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_63_in_google4[GOOGLE_BATCH*49][832] row_align(MAX_BLOCK_LEN);
static elem_t conv_63_out_google4[GOOGLE_BATCH*49][192] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_63_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=7, .kernel_size=1, .in_channels=832, .out_channels=192, .out_stride=(192), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=832, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=0.015625, .I=GOOGLE_BATCH*49, .J=192, .K=832, .res_scale=0};


static const elem_t conv_64_w_google4[1728][(384+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_64_b_google4[(384+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_64_in_google4[GOOGLE_BATCH*49][1728] row_align(MAX_BLOCK_LEN);
static elem_t conv_64_out_google4[GOOGLE_BATCH*49][(384+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_64_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=7, .kernel_size=3, .in_channels=192, .out_channels=384, .out_stride=(384+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=1728, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=0.015625, .I=GOOGLE_BATCH*49, .J=384, .K=1728, .res_scale=0};


static const elem_t conv_65_w_google4[832][48] row_align(MAX_BLOCK_LEN); static const acc_t conv_65_b_google4[48] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_65_in_google4[GOOGLE_BATCH*49][832] row_align(MAX_BLOCK_LEN);
static elem_t conv_65_out_google4[GOOGLE_BATCH*49][48] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_65_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=7, .kernel_size=1, .in_channels=832, .out_channels=48, .out_stride=(48), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=832, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=0.00390625, .I=GOOGLE_BATCH*49, .J=48, .K=832, .res_scale=0};


static const elem_t conv_66_w_google4[432][(128+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_66_b_google4[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_66_in_google4[GOOGLE_BATCH*49][432] row_align(MAX_BLOCK_LEN);
static elem_t conv_66_out_google4[GOOGLE_BATCH*49][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_66_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=7, .kernel_size=3, .in_channels=48, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=432, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=0.015625, .I=GOOGLE_BATCH*49, .J=128, .K=432, .res_scale=0};


static elem_t pool_67_out_google4[GOOGLE_BATCH*1][7][7][832] row_align(MAX_BLOCK_LEN);
static elem_t pool_67_out_google42[1][7][7][832] row_align(MAX_BLOCK_LEN);
static const struct ConvParams pool_67_params_google4 = {.batch_size=GOOGLE_BATCH, .in_channels=832, .out_channels=832, .out_stride=(832), .out_dim=7, .pool_size=3, .pool_stride=1, .pool_padding=1, .out_dim_pooled=7};


static const elem_t conv_68_w_google4[832][(128+64)] row_align(MAX_BLOCK_LEN); static const acc_t conv_68_b_google4[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);static elem_t conv_68_in_google4[GOOGLE_BATCH*49][832] row_align(MAX_BLOCK_LEN);
static elem_t conv_68_out_google4[GOOGLE_BATCH*49][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_68_params_google4 = {.batch_size=GOOGLE_BATCH, .in_dim=7, .kernel_size=1, .in_channels=832, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .n_patches=49, .patch_size=832, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=0.015625, .I=GOOGLE_BATCH*49, .J=128, .K=832, .res_scale=0};


static const elem_t fc_69_w_google4[1000][(1024+64)] row_align(MAX_BLOCK_LEN);
static const acc_t fc_69_b_google4[1000][1] row_align_acc(MAX_BLOCK_LEN_ACC) = {{-733.0},{-812.0},{-31.0},{108.0},{2302.0},{737.0},{420.0},{-1350.0},{-1333.0},{-326.0},{-312.0},{-418.0},{-1100.0},{96.0},{-11.0},{155.0},{197.0},{-679.0},{1333.0},{-816.0},{1258.0},{2530.0},{333.0},{1092.0},{99.0},{1030.0},{1213.0},{793.0},{172.0},{-1234.0},{-678.0},{148.0},{707.0},{-901.0},{648.0},{-415.0},{-1261.0},{-716.0},{-70.0},{-988.0},{34.0},{1070.0},{1378.0},{-1810.0},{928.0},{453.0},{938.0},{-949.0},{669.0},{508.0},{445.0},{-2388.0},{1875.0},{695.0},{1650.0},{737.0},{1416.0},{-148.0},{934.0},{1418.0},{2089.0},{-237.0},{275.0},{-640.0},{-140.0},{1400.0},{646.0},{767.0},{733.0},{75.0},{-166.0},{-169.0},{-187.0},{407.0},{276.0},{769.0},{-438.0},{801.0},{1185.0},{872.0},{-134.0},{1486.0},{836.0},{-1087.0},{-883.0},{-294.0},{-90.0},{-2317.0},{-732.0},{698.0},{-586.0},{282.0},{1865.0},{-308.0},{-820.0},{488.0},{-269.0},{-1170.0},{-270.0},{-291.0},{-1071.0},{-142.0},{506.0},{977.0},{1073.0},{-1607.0},{929.0},{246.0},{-587.0},{-122.0},{-469.0},{1797.0},{-556.0},{-1249.0},{-200.0},{-9.0},{-111.0},{-1677.0},{-1180.0},{435.0},{532.0},{-1286.0},{-2100.0},{-1454.0},{112.0},{-650.0},{-437.0},{817.0},{739.0},{937.0},{-875.0},{669.0},{592.0},{742.0},{357.0},{-342.0},{30.0},{41.0},{1830.0},{-432.0},{-572.0},{-23.0},{-389.0},{818.0},{-1102.0},{-1330.0},{690.0},{506.0},{-370.0},{1148.0},{-1177.0},{-1022.0},{-607.0},{684.0},{227.0},{-311.0},{-226.0},{-938.0},{-857.0},{-418.0},{-1048.0},{-56.0},{1176.0},{-37.0},{-506.0},{-543.0},{26.0},{-1388.0},{-25.0},{-778.0},{353.0},{-475.0},{138.0},{-141.0},{101.0},{182.0},{-792.0},{325.0},{286.0},{-0.0},{-169.0},{-1018.0},{132.0},{-965.0},{-48.0},{581.0},{-880.0},{-698.0},{342.0},{-197.0},{-94.0},{188.0},{1000.0},{-869.0},{-152.0},{-39.0},{89.0},{-296.0},{-958.0},{355.0},{395.0},{3.0},{191.0},{485.0},{922.0},{106.0},{-839.0},{33.0},{-197.0},{1112.0},{-935.0},{422.0},{-58.0},{-352.0},{44.0},{319.0},{-407.0},{738.0},{62.0},{-373.0},{-255.0},{-1109.0},{-45.0},{715.0},{-295.0},{651.0},{692.0},{370.0},{-643.0},{1029.0},{-164.0},{-169.0},{980.0},{580.0},{542.0},{-92.0},{-283.0},{-461.0},{403.0},{1303.0},{37.0},{427.0},{-411.0},{-1199.0},{73.0},{264.0},{-983.0},{410.0},{-271.0},{-279.0},{100.0},{-1328.0},{665.0},{-1309.0},{-626.0},{-275.0},{623.0},{142.0},{51.0},{-551.0},{-41.0},{23.0},{-172.0},{285.0},{-324.0},{-357.0},{5.0},{-620.0},{-1563.0},{675.0},{1306.0},{451.0},{753.0},{846.0},{1100.0},{203.0},{1355.0},{810.0},{376.0},{711.0},{177.0},{45.0},{-435.0},{-24.0},{-330.0},{31.0},{-574.0},{883.0},{750.0},{771.0},{149.0},{-23.0},{-487.0},{-108.0},{275.0},{1180.0},{-693.0},{1185.0},{352.0},{367.0},{164.0},{274.0},{57.0},{119.0},{326.0},{1836.0},{264.0},{830.0},{-22.0},{-1079.0},{897.0},{-77.0},{832.0},{1601.0},{814.0},{-114.0},{-19.0},{1258.0},{1794.0},{-256.0},{573.0},{-1612.0},{-843.0},{-541.0},{-898.0},{-996.0},{-497.0},{-590.0},{-556.0},{1160.0},{449.0},{696.0},{179.0},{-1127.0},{-51.0},{-493.0},{1961.0},{-282.0},{-1814.0},{-477.0},{-489.0},{-646.0},{353.0},{505.0},{-285.0},{-878.0},{501.0},{-272.0},{-413.0},{-162.0},{140.0},{477.0},{-199.0},{-229.0},{-687.0},{-1281.0},{-232.0},{-797.0},{603.0},{-559.0},{-372.0},{588.0},{1107.0},{600.0},{386.0},{-547.0},{489.0},{135.0},{171.0},{-40.0},{-475.0},{-456.0},{284.0},{867.0},{-419.0},{-693.0},{-350.0},{-641.0},{219.0},{1001.0},{-226.0},{212.0},{-527.0},{-567.0},{-98.0},{-507.0},{-639.0},{-738.0},{-552.0},{-1055.0},{-157.0},{35.0},{-1625.0},{-1660.0},{-1105.0},{-543.0},{-1698.0},{-588.0},{-675.0},{832.0},{247.0},{-1050.0},{499.0},{1012.0},{178.0},{626.0},{-857.0},{433.0},{-1332.0},{-536.0},{-354.0},{-95.0},{-337.0},{-247.0},{327.0},{-1061.0},{131.0},{-584.0},{2105.0},{1956.0},{-609.0},{830.0},{-736.0},{18.0},{1158.0},{385.0},{-1476.0},{-860.0},{-735.0},{-679.0},{-612.0},{-923.0},{369.0},{-466.0},{538.0},{-50.0},{600.0},{87.0},{-528.0},{-899.0},{741.0},{151.0},{-533.0},{291.0},{416.0},{262.0},{1243.0},{-1038.0},{-1028.0},{840.0},{-865.0},{194.0},{-965.0},{536.0},{294.0},{492.0},{-426.0},{946.0},{-326.0},{-111.0},{1699.0},{-1380.0},{-217.0},{178.0},{445.0},{1155.0},{-961.0},{-1508.0},{418.0},{-1217.0},{612.0},{-959.0},{-82.0},{1167.0},{-627.0},{-1247.0},{-2995.0},{-1612.0},{475.0},{167.0},{1502.0},{-680.0},{100.0},{73.0},{1157.0},{296.0},{-156.0},{1446.0},{-12.0},{396.0},{-1483.0},{-955.0},{-306.0},{773.0},{-824.0},{317.0},{-1216.0},{-547.0},{238.0},{-59.0},{471.0},{757.0},{-982.0},{-372.0},{527.0},{-925.0},{-147.0},{-309.0},{530.0},{-1014.0},{-117.0},{685.0},{29.0},{-243.0},{-1378.0},{260.0},{-1005.0},{-58.0},{-479.0},{903.0},{-660.0},{-765.0},{359.0},{1035.0},{-1609.0},{-1377.0},{899.0},{536.0},{-1003.0},{1342.0},{1464.0},{-1029.0},{-250.0},{1604.0},{-83.0},{-720.0},{204.0},{-1070.0},{138.0},{1098.0},{-466.0},{-670.0},{956.0},{27.0},{-239.0},{-425.0},{-1015.0},{-621.0},{163.0},{1237.0},{-947.0},{456.0},{-294.0},{1137.0},{-186.0},{-152.0},{-1126.0},{1144.0},{-123.0},{87.0},{-1411.0},{-679.0},{-1892.0},{576.0},{-1666.0},{-283.0},{-1161.0},{1325.0},{-632.0},{-695.0},{-615.0},{-766.0},{50.0},{-54.0},{-107.0},{15.0},{-1353.0},{-1714.0},{-74.0},{-82.0},{406.0},{-868.0},{-88.0},{138.0},{-488.0},{-571.0},{-1356.0},{818.0},{-1607.0},{-734.0},{-179.0},{-94.0},{-263.0},{589.0},{-1593.0},{-1250.0},{78.0},{114.0},{-292.0},{-194.0},{-138.0},{449.0},{-185.0},{-233.0},{-1037.0},{913.0},{-336.0},{493.0},{901.0},{-474.0},{1030.0},{-1352.0},{-871.0},{-881.0},{-978.0},{900.0},{-312.0},{2078.0},{680.0},{-101.0},{356.0},{199.0},{1259.0},{1943.0},{495.0},{-197.0},{1307.0},{434.0},{-678.0},{634.0},{176.0},{-438.0},{1432.0},{139.0},{-543.0},{-701.0},{-15.0},{-965.0},{478.0},{-654.0},{1238.0},{718.0},{-321.0},{-1123.0},{2002.0},{46.0},{1326.0},{-824.0},{1453.0},{2572.0},{1087.0},{53.0},{527.0},{542.0},{-98.0},{530.0},{-28.0},{-55.0},{-484.0},{1083.0},{873.0},{-975.0},{346.0},{-200.0},{573.0},{-136.0},{352.0},{265.0},{-692.0},{-287.0},{-1464.0},{1182.0},{-70.0},{543.0},{-874.0},{-631.0},{-1402.0},{1075.0},{217.0},{-292.0},{690.0},{1548.0},{971.0},{918.0},{-93.0},{64.0},{-90.0},{-168.0},{-1322.0},{-247.0},{-1251.0},{-692.0},{-213.0},{-499.0},{-1063.0},{-1031.0},{-517.0},{579.0},{277.0},{-1050.0},{647.0},{922.0},{19.0},{509.0},{-946.0},{-43.0},{-645.0},{-1161.0},{-218.0},{-256.0},{-965.0},{-143.0},{726.0},{834.0},{488.0},{-1741.0},{715.0},{-401.0},{1357.0},{38.0},{291.0},{-1005.0},{584.0},{-1047.0},{-312.0},{-359.0},{-467.0},{-1251.0},{-555.0},{-766.0},{144.0},{814.0},{-327.0},{1626.0},{244.0},{220.0},{394.0},{82.0},{-799.0},{-930.0},{-932.0},{785.0},{275.0},{1045.0},{-140.0},{423.0},{1461.0},{-262.0},{-839.0},{-98.0},{145.0},{341.0},{687.0},{-323.0},{579.0},{-761.0},{-1453.0},{-1018.0},{-1256.0},{129.0},{1143.0},{1001.0},{-183.0},{-22.0},{371.0},{-326.0},{-2527.0},{763.0},{1212.0},{870.0},{-1298.0},{-376.0},{-792.0},{-4.0},{-305.0},{-426.0},{-222.0},{688.0},{-1148.0},{-664.0},{476.0},{-431.0},{308.0},{407.0},{847.0},{-83.0},{-1801.0},{-1242.0},{-457.0},{975.0},{-3090.0},{-907.0},{121.0},{152.0},{824.0},{152.0},{-1060.0},{527.0},{-117.0},{2002.0},{-1226.0},{-673.0},{357.0},{-635.0},{-381.0},{447.0},{-944.0},{-953.0},{-753.0},{350.0},{-820.0},{-1545.0},{-790.0},{-100.0},{605.0},{1084.0},{-1401.0},{312.0},{475.0},{-896.0},{-1754.0},{-477.0},{-594.0},{1154.0},{600.0},{1721.0},{-329.0},{-1212.0},{975.0},{-1305.0},{552.0},{-493.0},{-1554.0},{-117.0},{2486.0},{-2071.0},{-10.0},{804.0},{732.0},{504.0},{-688.0},{812.0},{-770.0},{-451.0},{1248.0},{972.0},{-316.0},{-752.0},{313.0},{-850.0},{-2459.0},{-413.0},{263.0},{-2.0},{-50.0},{-723.0},{-1777.0},{-1888.0},{520.0},{-1367.0},{508.0},{381.0},{-1060.0},{-191.0},{140.0},{-867.0},{-1005.0},{-472.0},{590.0},{980.0},{-248.0},{35.0},{508.0},{-203.0},{-451.0},{394.0},{-529.0},{-376.0},{-1311.0},{11.0},{584.0},{573.0},{376.0},{-489.0},{326.0},{2849.0},{-1369.0},{-1674.0},{694.0},{-696.0},{-1659.0},{-1961.0},{918.0},{-12.0},{1321.0},{1708.0},{243.0},{-96.0},{135.0},{-174.0},{1025.0},{924.0},{1346.0},{847.0},{1490.0},{1410.0},{1839.0},{148.0},{2598.0},{202.0},{541.0},{1362.0},{279.0},{933.0},{-261.0},{-445.0},{-123.0},{-1943.0},{-1118.0},{930.0},{-48.0},{-1369.0},{607.0},{-520.0},{-602.0},{-310.0},{76.0},{-641.0},{-1216.0},{709.0},{683.0},{-855.0},{-502.0},{-179.0},{-375.0},{-361.0},{59.0},{-385.0},{-212.0},{-1667.0},{-107.0},{-1026.0},{-759.0},{-1507.0},{423.0},{528.0},{1057.0},{-234.0},{330.0},{415.0},{922.0},{734.0},{140.0},{-1972.0},{-635.0},{-801.0},{-839.0},{794.0},{768.0},{-428.0},{-1021.0},{1470.0},{-523.0},{1208.0},{-57.0},{-872.0},{668.0},{547.0},{386.0},{-419.0},{1676.0},{-938.0},{1851.0},{780.0},{2226.0},{2098.0},{2503.0},{3966.0},{1120.0},{2115.0},{1886.0},{813.0},{-154.0},{94.0},{1371.0},{996.0},{41.0},{-458.0},{-715.0},{1294.0},{372.0},{210.0},{-555.0},{-1149.0},{-687.0},{-815.0},{171.0},{-980.0},{-629.0},{1120.0}};
static elem_t fc_69_out_google4[GOOGLE_BATCH*1000][1] row_align(MAX_BLOCK_LEN);
static const struct FcParams fc_69_params_google4 = {.batch_size=GOOGLE_BATCH, .in_features=1024, .out_features=1000, .out_stride=1, .bias=1, .output_scale=7, .I=GOOGLE_BATCH*1000, .J=1, .K=1024};

#undef GOOGLE_BATCH

#endif

