#ifndef C95AAE97_875E_4B5B_AEA4_1F2773C54BC2
#define C95AAE97_875E_4B5B_AEA4_1F2773C54BC2

#include <include/gemmini_params.h>
#include <include/gemmini_nn.h>

static const elem_t conv_1_w[64][3][3][3] = {0};
static  elem_t conv_1_in[224][224][3] = {0};
static  elem_t conv_1_out[224][224][64] = {0};
static  elem_t conv_1_b[224][224][64] = {0};
static const struct ConvParamsSimple conv_1_params = {.batch_size=1, .in_dim=224, .kernel_size=3, .in_channels=3, .out_channels=64, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=224, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=224, .output_scale=1.0};

static const elem_t conv_2_w[64][3][3][64] = {0};
static  elem_t conv_2_in[224][224][64] = {0};
static  elem_t conv_2_out[112][112][64] = {0};
static  elem_t conv_2_b[112][112][64] = {0};
static const struct ConvParamsSimple conv_2_params = {.batch_size=1, .in_dim=224, .kernel_size=3, .in_channels=64, .out_channels=64, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=224, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=112, .output_scale=1.0};

static const elem_t conv_3_w[128][3][3][64] = {0};
static  elem_t conv_3_in[112][112][64] = {0};
static  elem_t conv_3_out[112][112][128] = {0};
static  elem_t conv_3_b[112][112][128] = {0};
static const struct ConvParamsSimple conv_3_params = {.batch_size=1, .in_dim=112, .kernel_size=3, .in_channels=64, .out_channels=128, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=112, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=112, .output_scale=1.0};

static const elem_t conv_4_w[128][3][3][128] = {0};
static  elem_t conv_4_in[112][112][128] = {0};
static  elem_t conv_4_out[56][56][128] = {0};
static  elem_t conv_4_b[56][56][128] = {0};
static const struct ConvParamsSimple conv_4_params = {.batch_size=1, .in_dim=112, .kernel_size=3, .in_channels=128, .out_channels=128, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=112, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=56, .output_scale=1.0};

static const elem_t conv_5_w[256][3][3][128] = {0};
static  elem_t conv_5_in[56][56][128] = {0};
static  elem_t conv_5_out[56][56][256] = {0};
static  elem_t conv_5_b[56][56][256] = {0};
static const struct ConvParamsSimple conv_5_params = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=128, .out_channels=256, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=56, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=1.0};

static const elem_t conv_6_w[256][3][3][256] = {0};
static  elem_t conv_6_in[56][56][256] = {0};
static  elem_t conv_6_out[56][56][256] = {0};
static  elem_t conv_6_b[56][56][256] = {0};
static const struct ConvParamsSimple conv_6_params = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=256, .out_channels=256, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=56, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=56, .output_scale=1.0};

static const elem_t conv_7_w[256][3][3][256] = {0};
static  elem_t conv_7_in[56][56][256] = {0};
static  elem_t conv_7_out[28][28][256] = {0};
static  elem_t conv_7_b[28][28][256] = {0};
static const struct ConvParamsSimple conv_7_params = {.batch_size=1, .in_dim=56, .kernel_size=3, .in_channels=256, .out_channels=256, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=56, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=28, .output_scale=1.0};

static const elem_t conv_8_w[512][3][3][256] = {0};
static  elem_t conv_8_in[28][28][256] = {0};
static  elem_t conv_8_out[28][28][512] = {0};
static  elem_t conv_8_b[28][28][512] = {0};
static const struct ConvParamsSimple conv_8_params = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=256, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=28, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=1.0};

static const elem_t conv_9_w[512][3][3][512] = {0};
static  elem_t conv_9_in[28][28][512] = {0};
static  elem_t conv_9_out[28][28][512] = {0};
static  elem_t conv_9_b[28][28][512] = {0};
static const struct ConvParamsSimple conv_9_params = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=512, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=28, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=1.0};

static const elem_t conv_10_w[512][3][3][512] = {0};
static  elem_t conv_10_in[28][28][512] = {0};
static  elem_t conv_10_out[14][14][512] = {0};
static  elem_t conv_10_b[14][14][512] = {0};
static const struct ConvParamsSimple conv_10_params = {.batch_size=1, .in_dim=28, .kernel_size=3, .in_channels=512, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=28, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_11_w[512][3][3][512] = {0};
static  elem_t conv_11_in[14][14][512] = {0};
static  elem_t conv_11_out[14][14][512] = {0};
static  elem_t conv_11_b[14][14][512] = {0};
static const struct ConvParamsSimple conv_11_params = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=512, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=14, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_12_w[512][3][3][512] = {0};
static  elem_t conv_12_in[14][14][512] = {0};
static  elem_t conv_12_out[14][14][512] = {0};
static  elem_t conv_12_b[14][14][512] = {0};
static const struct ConvParamsSimple conv_12_params = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=512, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=14, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_13_w[512][3][3][512] = {0};
static  elem_t conv_13_in[14][14][512] = {0};
static  elem_t conv_13_out[7][7][512] = {0};
static  elem_t conv_13_b[7][7][512] = {0};
static const struct ConvParamsSimple conv_13_params = {.batch_size=1, .in_dim=14, .kernel_size=3, .in_channels=512, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=14, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=7, .output_scale=1.0};

static const acc_t fc_14_b[4096][1] = {0};
static const elem_t fc_14_w[25088][4096] = {0};
static elem_t fc_14_out[4096][1] row_align(1);
static const struct FcParams fc_14_params = {.batch_size=1, .in_features=25088, .out_features=4096, .bias=1, .output_scale=(1.0 / (1 << 10)), .I=4, .J=4096, .K=25088};















#endif /* C95AAE97_875E_4B5B_AEA4_1F2773C54BC2 */
