#ifndef YOLOLITE_PARAMS_H
#define YOLOLITE_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_1_w_yololite1[27][(16+48)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_1_b_yololite1[16] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_1_out_yololite1[50176][(16+48)] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_yololite1_pooled[1][112][112][(16+48)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_yololite1 = {.batch_size=1, .in_dim=224, .kernel_size=3, .in_channels=3, .out_channels=16, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=224, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=112, .output_scale=10, .res_scale=0};


static const elem_t conv_2_w_yololite1[144][(32+32)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_2_b_yololite1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_2_out_yololite1[12544][(32+32)] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_yololite1_pooled[1][56][56][(32+32)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_yololite1 = {.batch_size=1, .in_dim=112, .kernel_size=3, .in_channels=16, .out_channels=32, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=112, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .res_scale=0};


static const elem_t conv_3_w_yololite1[288][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_3_b_yololite1[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_3_out_yololite1[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_yololite1_pooled[1][28][28][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_yololite1 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=32, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .res_scale=0};


static const elem_t conv_4_w_yololite1[576][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_4_b_yololite1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_4_out_yololite1[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_yololite1_pooled[1][14][14][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_yololite1 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=64, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .res_scale=0};


static const elem_t conv_5_w_yololite1[1152][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_5_b_yololite1[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_5_out_yololite1[196][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_yololite1_pooled[1][7][7][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_yololite1 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .res_scale=0};

static const elem_t conv_6_w_yololite1[1152][(256+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_6_b_yololite1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_6_out_yololite1[49][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_6_params_yololite1 = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=128, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .res_scale=0};

static const elem_t conv_7_w_yololite1[256][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_7_b_yololite1[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_7_out_yololite1[49][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_7_params_yololite1 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=256, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .I=49, .J=128, .K=256, .res_scale=0};


//#if NUM_CORE == 8
static const elem_t conv_1_w_yololite11[27][(16+48)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_1_b_yololite11[16] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_1_out_yololite11[50176][(16+48)] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_yololite11_pooled[1][112][112][(16+48)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_yololite11 = {.batch_size=1, .in_dim=224, .kernel_size=3, .in_channels=3, .out_channels=16, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=224, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=112, .output_scale=10, .res_scale=0};


static const elem_t conv_2_w_yololite11[144][(32+32)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_2_b_yololite11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_2_out_yololite11[12544][(32+32)] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_yololite11_pooled[1][56][56][(32+32)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_yololite11 = {.batch_size=1, .in_dim=112, .kernel_size=3, .in_channels=16, .out_channels=32, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=112, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .res_scale=0};


static const elem_t conv_3_w_yololite11[288][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_3_b_yololite11[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_3_out_yololite11[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_yololite11_pooled[1][28][28][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_yololite11 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=32, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .res_scale=0};


static const elem_t conv_4_w_yololite11[576][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_4_b_yololite11[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_4_out_yololite11[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_yololite11_pooled[1][14][14][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_yololite11 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=64, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .res_scale=0};


static const elem_t conv_5_w_yololite11[1152][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_5_b_yololite11[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_5_out_yololite11[196][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_yololite11_pooled[1][7][7][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_yololite11 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .res_scale=0};

static const elem_t conv_6_w_yololite11[1152][(256+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_6_b_yololite11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_6_out_yololite11[49][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_6_params_yololite11 = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=128, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .res_scale=0};

static const elem_t conv_7_w_yololite11[256][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_7_b_yololite11[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_7_out_yololite11[49][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_7_params_yololite11 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=256, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .I=49, .J=128, .K=256, .res_scale=0};



//#if NUM_CORE == 8
static const elem_t conv_1_w_yololite111[27][(16+48)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_1_b_yololite111[16] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_1_out_yololite111[50176][(16+48)] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_yololite111_pooled[1][112][112][(16+48)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_yololite111 = {.batch_size=1, .in_dim=224, .kernel_size=3, .in_channels=3, .out_channels=16, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=224, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=112, .output_scale=10, .res_scale=0};


static const elem_t conv_2_w_yololite111[144][(32+32)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_2_b_yololite111[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_2_out_yololite111[12544][(32+32)] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_yololite111_pooled[1][56][56][(32+32)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_yololite111 = {.batch_size=1, .in_dim=112, .kernel_size=3, .in_channels=16, .out_channels=32, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=112, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .res_scale=0};


static const elem_t conv_3_w_yololite111[288][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_3_b_yololite111[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_3_out_yololite111[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_yololite111_pooled[1][28][28][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_yololite111 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=32, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .res_scale=0};


static const elem_t conv_4_w_yololite111[576][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_4_b_yololite111[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_4_out_yololite111[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_yololite111_pooled[1][14][14][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_yololite111 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=64, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .res_scale=0};


static const elem_t conv_5_w_yololite111[1152][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_5_b_yololite111[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_5_out_yololite111[196][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_yololite111_pooled[1][7][7][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_yololite111 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .res_scale=0};

static const elem_t conv_6_w_yololite111[1152][(256+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_6_b_yololite111[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_6_out_yololite111[49][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_6_params_yololite111 = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=128, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .res_scale=0};

static const elem_t conv_7_w_yololite111[256][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_7_b_yololite111[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_7_out_yololite111[49][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_7_params_yololite111 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=256, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .I=49, .J=128, .K=256, .res_scale=0};


//#if NUM_CORE == 8
static const elem_t conv_1_w_yololite1111[27][(16+48)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_1_b_yololite1111[16] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_1_out_yololite1111[50176][(16+48)] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_yololite1111_pooled[1][112][112][(16+48)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_yololite1111 = {.batch_size=1, .in_dim=224, .kernel_size=3, .in_channels=3, .out_channels=16, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=224, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=112, .output_scale=10, .res_scale=0};


static const elem_t conv_2_w_yololite1111[144][(32+32)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_2_b_yololite1111[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_2_out_yololite1111[12544][(32+32)] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_yololite1111_pooled[1][56][56][(32+32)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_yololite1111 = {.batch_size=1, .in_dim=112, .kernel_size=3, .in_channels=16, .out_channels=32, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=112, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .res_scale=0};


static const elem_t conv_3_w_yololite1111[288][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_3_b_yololite1111[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_3_out_yololite1111[3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_yololite1111_pooled[1][28][28][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_yololite1111 = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=32, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .res_scale=0};


static const elem_t conv_4_w_yololite1111[576][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_4_b_yololite1111[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_4_out_yololite1111[784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_yololite1111_pooled[1][14][14][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_yololite1111 = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=64, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .res_scale=0};


static const elem_t conv_5_w_yololite1111[1152][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_5_b_yololite1111[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_5_out_yololite1111[196][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_yololite1111_pooled[1][7][7][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_yololite1111 = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .res_scale=0};

static const elem_t conv_6_w_yololite1111[1152][(256+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_6_b_yololite1111[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_6_out_yololite1111[49][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_6_params_yololite1111 = {.batch_size=1, .in_dim=7, .kernel_size=3, .in_channels=128, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .res_scale=0};

static const elem_t conv_7_w_yololite1111[256][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_7_b_yololite1111[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_7_out_yololite1111[49][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_7_params_yololite1111 = {.batch_size=1, .in_dim=7, .kernel_size=1, .in_channels=256, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .I=49, .J=128, .K=256, .res_scale=0};



#endif
//#endif

