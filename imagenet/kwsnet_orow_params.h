#ifndef RES26NET_MT_PARAMS_H
#define RES26NET_MT_PARAMS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

static const elem_t conv_1_w_kws[27][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_1_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_1_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static elem_t conv_1_out_kws_pooled[1][48][48][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_1_params_kws = {.batch_size=1, .in_dim=96, .kernel_size=3, .in_channels=3, .in_stride = 3, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=96, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_2_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_2_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_2_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_2_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_3_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_3_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_3_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_3_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_4_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_4_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_4_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_4_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_5_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_5_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_5_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_5_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};

static const elem_t conv_6_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_6_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_6_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_6_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_7_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_7_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_7_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_7_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_8_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_8_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_8_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_8_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_9_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_9_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_9_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_9_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_10_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_10_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_10_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_10_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_11_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_11_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_11_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_11_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};

static const elem_t conv_12_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_12_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_12_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_12_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_13_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_13_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_13_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_13_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_14_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_14_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_14_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_14_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_15_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_15_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_15_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_15_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_16_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_16_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_16_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_16_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};

static const elem_t conv_17_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_17_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_17_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_17_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_18_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_18_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_18_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_18_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_19_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_19_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_19_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_19_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_20_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_20_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_20_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_20_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_21_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_21_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_21_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_21_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_22_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_22_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_22_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_22_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};

static const elem_t conv_23_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_23_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_23_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_23_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_24_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_24_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_24_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_24_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};


static const elem_t conv_25_w_kws[405][(45+19)] row_align(MAX_BLOCK_LEN);// =
static const acc_t conv_25_b_kws[45] row_align_acc(MAX_BLOCK_LEN_ACC);// = {
static elem_t conv_25_out_kws[9216][(45+19)] row_align(MAX_BLOCK_LEN);
static struct ConvParams conv_25_params_kws = {.batch_size=1, .in_dim=48, .kernel_size=3, .in_channels=45, .in_stride = 64, .out_channels=45, .out_stride=(64), .stride=1, .padding=1, .bias=1, .depthwise=0, .out_dim=48, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=48, .output_scale=8, .res_scale=0};



#endif

