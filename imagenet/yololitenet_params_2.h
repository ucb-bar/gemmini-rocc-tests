#ifndef YOLOLITE2_PARAMS_H
#define YOLOLITE2_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

#define BATCH_SIZE 2


static const elem_t conv_1_w_yololite2[27][(16+48)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_1_b_yololite2[16] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_1_out_yololite2[BATCH_SIZE*50176][(16+48)] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_yololite2_pooled[BATCH_SIZE][112][112][(16+48)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_1_params_yololite2 = {.batch_size=BATCH_SIZE, .in_dim=224, .kernel_size=3, .in_channels=3, .out_channels=16, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=224, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=112, .output_scale=10, .res_scale=0};


static const elem_t conv_2_w_yololite2[144][(32+32)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_2_b_yololite2[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_2_out_yololite2[BATCH_SIZE*12544][(32+32)] row_align(MAX_BLOCK_LEN);
static elem_t conv_2_out_yololite2_pooled[BATCH_SIZE][56][56][(32+32)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_2_params_yololite2 = {.batch_size=BATCH_SIZE, .in_dim=112, .kernel_size=3, .in_channels=16, .out_channels=32, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=112, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=56, .output_scale=6, .res_scale=0};


static const elem_t conv_3_w_yololite2[288][64] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_3_b_yololite2[64] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_3_out_yololite2[BATCH_SIZE*3136][64] row_align(MAX_BLOCK_LEN);
static elem_t conv_3_out_yololite2_pooled[BATCH_SIZE][28][28][64] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_3_params_yololite2 = {.batch_size=BATCH_SIZE, .in_dim=56, .kernel_size=3, .in_channels=32, .out_channels=64, .out_stride=64, .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=56, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=28, .output_scale=7, .res_scale=0};


static const elem_t conv_4_w_yololite2[576][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_4_b_yololite2[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_4_out_yololite2[BATCH_SIZE*784][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_4_out_yololite2_pooled[BATCH_SIZE][14][14][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_4_params_yololite2 = {.batch_size=BATCH_SIZE, .in_dim=28, .kernel_size=3, .in_channels=64, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=28, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=14, .output_scale=7, .res_scale=0};


static const elem_t conv_5_w_yololite2[1152][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_5_b_yololite2[(128+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_5_out_yololite2[BATCH_SIZE*196][(128+64)] row_align(MAX_BLOCK_LEN);
static elem_t conv_5_out_yololite2_pooled[BATCH_SIZE][7][7][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_5_params_yololite2 = {.batch_size=BATCH_SIZE, .in_dim=14, .kernel_size=3, .in_channels=128, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=14, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .res_scale=0};

static const elem_t conv_6_w_yololite2[1152][(256+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_6_b_yololite2[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_6_out_yololite2[BATCH_SIZE*49][(256+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_6_params_yololite2 = {.batch_size=BATCH_SIZE, .in_dim=7, .kernel_size=3, .in_channels=128, .out_channels=256, .out_stride=(256+64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .res_scale=0};

static const elem_t conv_7_w_yololite2[256][(128+64)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_7_b_yololite2[(256+64)] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_7_out_yololite2[BATCH_SIZE*49][(128+64)] row_align(MAX_BLOCK_LEN);
static const struct ConvParams conv_7_params_yololite2 = {.batch_size=BATCH_SIZE, .in_dim=7, .kernel_size=1, .in_channels=256, .out_channels=128, .out_stride=(128+64), .stride=1, .padding=0, .bias=1, .depthwise=0, .out_dim=7, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=7, .output_scale=9, .I=BATCH_SIZE*49, .J=128, .K=256, .res_scale=0};

#undef BATCH_SIZE
#endif

