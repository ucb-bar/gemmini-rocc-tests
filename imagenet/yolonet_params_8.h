#ifndef YOLOV2NET8_MT_PARAMS_H
#define YOLOV2NET8_MT_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

#define BATCH_SIZE 8


static const elem_t conv_1_w_yolo8[363][(32+32)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_1_b_yolo8[32] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_1_out_yolo8[BATCH_SIZE*50176][(32+32)] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_yolo8_pooled[BATCH_SIZE][112][112][(32+32)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=224, .kernel_size=3, .in_channels=3, .out_channels=32, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=224, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=112, .output_scale=10, .res_scale=0};


static const elem_t conv_2_w_yolo8[288][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_2_b_yolo8[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_2_out_yolo8[BATCH_SIZE*12544][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_yolo8_pooled[BATCH_SIZE][56][56][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=112, .kernel_size=3, .in_channels=32, .out_channels=64, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=112, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .res_scale=0};


static const elem_t conv_3_w_yolo8[576][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_3_b_yolo8[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_3_out_yolo8[BATCH_SIZE*3136][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .res_scale=0};


static const elem_t conv_4_w_yolo8[128][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_4_b_yolo8[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_4_out_yolo8[BATCH_SIZE*3136][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=56, .kernel_size=1, .in_channels=128, .out_channels=64, .out_stride=(64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=56, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=7, .I=BATCH_SIZE*3136, .J=64, .K=128, .res_scale=0};


static const elem_t conv_5_w_yolo8[576][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_5_b_yolo8[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_5_out_yolo8[BATCH_SIZE*3136][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_yolo8_pooled[BATCH_SIZE][28][28][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=56, .kernel_size=3, .in_channels=64, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=28, .output_scale=9, .res_scale=0};


static const elem_t conv_6_w_yolo8[1152][(256+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_6_b_yolo8[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_6_out_yolo8[BATCH_SIZE*784][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_6_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .res_scale=0};


static const elem_t conv_7_w_yolo8[256][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_7_b_yolo8[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_7_out_yolo8[BATCH_SIZE*784][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_7_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=28, .kernel_size=1, .in_channels=256, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=28, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .I=BATCH_SIZE*784, .J=128, .K=256, .res_scale=0};


static const elem_t conv_8_w_yolo8[1152][(256+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_8_b_yolo8[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_8_out_yolo8[BATCH_SIZE*784][(256+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_8_out_yolo8_pooled[BATCH_SIZE][14][14][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_8_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=28, .kernel_size=3, .in_channels=128, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=14, .output_scale=9, .res_scale=0};


static const elem_t conv_9_w_yolo8[2304][(512+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_9_b_yolo8[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_9_out_yolo8[BATCH_SIZE*196][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_9_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=14, .kernel_size=3, .in_channels=256, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .res_scale=0};


static const elem_t conv_10_w_yolo8[512][(256+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_10_b_yolo8[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_10_out_yolo8[BATCH_SIZE*196][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_10_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=BATCH_SIZE*196, .J=256, .K=512, .res_scale=0};


static const elem_t conv_11_w_yolo8[2304][(512+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_11_b_yolo8[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_11_out_yolo8[BATCH_SIZE*196][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_11_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=14, .kernel_size=3, .in_channels=256, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .res_scale=0};


static const elem_t conv_12_w_yolo8[512][(256+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_12_b_yolo8[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_12_out_yolo8[BATCH_SIZE*196][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_12_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=14, .kernel_size=1, .in_channels=512, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=14, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .I=BATCH_SIZE*196, .J=256, .K=512, .res_scale=0};

static const elem_t conv_13_w_yolo8[2304][(512+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_13_b_yolo8[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_13_out_yolo8[BATCH_SIZE*196][(512+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_13_out_yolo8_pooled[BATCH_SIZE][7][7][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_13_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=14, .kernel_size=3, .in_channels=256, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .res_scale=0};


static const elem_t conv_14_w_yolo8[4608][(1024+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_14_b_yolo8[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_14_out_yolo8[BATCH_SIZE*49][(1024+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_14_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=7, .kernel_size=3, .in_channels=512, .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .res_scale=0};


static const elem_t conv_15_w_yolo8[1024][(512+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_15_b_yolo8[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_15_out_yolo8[BATCH_SIZE*49][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_15_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=7, .kernel_size=1, .in_channels=1024, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=BATCH_SIZE*49, .J=512, .K=1024, .res_scale=0};


static const elem_t conv_16_w_yolo8[4608][(1024+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_16_b_yolo8[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_16_out_yolo8[BATCH_SIZE*49][(1024+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_16_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=7, .kernel_size=3, .in_channels=512, .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .res_scale=0};


static const elem_t conv_17_w_yolo8[1024][(512+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_17_b_yolo8[(512+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_17_out_yolo8[BATCH_SIZE*49][(512+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_17_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=7, .kernel_size=1, .in_channels=1024, .out_channels=512, .out_stride=(512+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=BATCH_SIZE*49, .J=512, .K=1024, .res_scale=0};


static const elem_t conv_18_w_yolo8[4608][(1024+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_18_b_yolo8[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_18_out_yolo8[BATCH_SIZE*49][(1024+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_18_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=7, .kernel_size=3, .in_channels=512, .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .res_scale=0};


static const elem_t conv_19_w_yolo8[1024][(1024+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_19_b_yolo8[(1024+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_19_out_yolo8[BATCH_SIZE*49][(1024+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_19_params_yolo8 = {.batch_size=BATCH_SIZE, .in_dim=7, .kernel_size=1, .in_channels=1024, .out_channels=1024, .out_stride=(1024+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=7, .I=BATCH_SIZE*49, .J=1024, .K=1024, .res_scale=0};

#undef BATCH_SIZE
#endif

