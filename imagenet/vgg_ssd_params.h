#ifndef F2F15EE8_05C8_43F3_96E4_FAF8C462ABFB
#define F2F15EE8_05C8_43F3_96E4_FAF8C462ABFB

#include <include/gemmini_params.h>
#include <include/gemmini_nn.h>

static const elem_t conv_1_w[64][3][3][3] = {0};
static  elem_t conv_1_in[300][300][3] = {0};
static  elem_t conv_1_out[300][300][64] = {0};
static  elem_t conv_1_b[300][300][64] = {0};//1_1
static const struct ConvParamsSimple conv_1_params = {.batch_size=1, .in_dim=300, .kernel_size=3, .in_channels=3, .out_channels=64, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=300, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=224, .output_scale=1.0};

static const elem_t conv_2_w[64][3][3][64] = {0};
static  elem_t conv_2_out[150][150][64] = {0};
static  elem_t conv_2_b[150][150][64] = {0};//1_2
static const struct ConvParamsSimple conv_2_params = {.batch_size=1, .in_dim=300, .kernel_size=3, .in_channels=64, .out_channels=64, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=300, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=150, .output_scale=1.0};

static const elem_t conv_3_w[128][3][3][64] = {0};
static  elem_t conv_3_out[150][150][128] = {0};
static  elem_t conv_3_b[150][150][128] = {0};//2_1
static const struct ConvParamsSimple conv_3_params = {.batch_size=1, .in_dim=150, .kernel_size=3, .in_channels=64, .out_channels=128, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=150, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=112, .output_scale=1.0};

static const elem_t conv_4_w[128][3][3][128] = {0};
static  elem_t conv_4_out[75][75][128] = {0};
static  elem_t conv_4_b[75][75][128] = {0};//2_2
static const struct ConvParamsSimple conv_4_params = {.batch_size=1, .in_dim=150, .kernel_size=3, .in_channels=128, .out_channels=128, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=150, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=75, .output_scale=1.0};

static const elem_t conv_5_w[256][3][3][128] = {0};
static  elem_t conv_5_out[75][75][256] = {0};
static  elem_t conv_5_b[75][75][256] = {0};//3_1
static const struct ConvParamsSimple conv_5_params = {.batch_size=1, .in_dim=75, .kernel_size=3, .in_channels=128, .out_channels=256, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=75, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=75, .output_scale=1.0};

static const elem_t conv_6_w[256][3][3][256] = {0};
static  elem_t conv_6_out[75][75][256] = {0};
static  elem_t conv_6_b[75][75][256] = {0};//3_2
static const struct ConvParamsSimple conv_6_params = {.batch_size=1, .in_dim=75, .kernel_size=3, .in_channels=256, .out_channels=256, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=75, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=75, .output_scale=1.0};

static const elem_t conv_7_w[256][3][3][256] = {0};
static  elem_t conv_7_out[38][38][256] = {0};
static  elem_t conv_7_b[38][38][256] = {0};//3_3
static const struct ConvParamsSimple conv_7_params = {.batch_size=1, .in_dim=75, .kernel_size=3, .in_channels=256, .out_channels=256, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=75, .pool_size=2, .pool_stride=2, .pool_padding=1, .out_dim_pooled=38, .output_scale=1.0};

static const elem_t conv_8_w[512][3][3][256] = {0};
static  elem_t conv_8_out[38][38][512] = {0};
static  elem_t conv_8_b[38][38][512] = {0};//4_1
static const struct ConvParamsSimple conv_8_params = {.batch_size=1, .in_dim=38, .kernel_size=3, .in_channels=256, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=38, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=1.0};

static const elem_t conv_9_w[512][3][3][512] = {0};
static  elem_t conv_9_out[38][38][512] = {0};
static  elem_t conv_9_b[38][38][512] = {0};//4_2
static const struct ConvParamsSimple conv_9_params = {.batch_size=1, .in_dim=38, .kernel_size=3, .in_channels=512, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=38, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=28, .output_scale=1.0};

static const elem_t conv_10_w[512][3][3][512] = {0};
static  elem_t conv_10_out[19][19][512] = {0};
static  elem_t conv_10_b[19][19][512] = {0};//4_3
static const struct ConvParamsSimple conv_10_params = {.batch_size=1, .in_dim=38, .kernel_size=3, .in_channels=512, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=38, .pool_size=2, .pool_stride=2, .pool_padding=0, .out_dim_pooled=19, .output_scale=1.0};

static const elem_t conv_11_w[512][3][3][512] = {0};
static  elem_t conv_11_out[19][19][512] = {0};
static  elem_t conv_11_b[19][19][512] = {0};//5_1
static const struct ConvParamsSimple conv_11_params = {.batch_size=1, .in_dim=19, .kernel_size=3, .in_channels=512, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=19, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=19, .output_scale=1.0};

static const elem_t conv_12_w[512][3][3][512] = {0};
static  elem_t conv_12_out[19][19][512] = {0};
static  elem_t conv_12_b[19][19][512] = {0};//5_2
static const struct ConvParamsSimple conv_12_params = {.batch_size=1, .in_dim=19, .kernel_size=3, .in_channels=512, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=19, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=19, .output_scale=1.0};

static const elem_t conv_13_w[512][3][3][512] = {0};
static  elem_t conv_13_out[19][19][512] = {0};
static  elem_t conv_13_b[19][19][512] = {0};//5_3
static const struct ConvParamsSimple conv_13_params = {.batch_size=1, .in_dim=19, .kernel_size=3, .in_channels=512, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=19, .pool_size=3, .pool_stride=1, .pool_padding=1, .out_dim_pooled=19, .output_scale=1.0};


static const elem_t conv_aux_6_1_w[1024][1][1][512] = {0};
static  elem_t conv_aux_6_1_out[19][19][1024] = {0};
static  elem_t conv_aux_6_1_b[19][19][1024] = {0};//6_1
static const struct ConvParamsSimple conv_aux_6_1_params = {.batch_size=1, .in_dim=19, .kernel_size=1, .in_channels=512, .out_channels=1024, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=19, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=19, .output_scale=1.0};


static const elem_t conv_aux_7_1_w[1024][3][3][1024] = {0};
static  elem_t conv_aux_7_1_out[19][19][1024] = {0};
static  elem_t conv_aux_7_1_b[19][19][1024] = {0};//7_1
static const struct ConvParamsSimple conv_aux_7_1_params = {.batch_size=1, .in_dim=19, .kernel_size=3, .in_channels=1024, .out_channels=1024, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=19, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=19, .output_scale=1.0};



// EXTRA AUXILLIARY CONV FUNCTIONS

static const elem_t conv_aux_8_1_w[256][1][1][1024] = {0};
static  elem_t conv_aux_8_1_out[19][19][256] = {0};
static  elem_t conv_aux_8_1_b[19][19][512] = {0};//
static const struct ConvParamsSimple conv_aux_8_1_params = {.batch_size=1, .in_dim=19, .kernel_size=1, .in_channels=1024, .out_channels=256, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=19, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=19, .output_scale=1.0};

static const elem_t conv_aux_8_2_w[512][3][3][256] = {0};
static  elem_t conv_aux_8_2_out[10][10][512] = {0};
static  elem_t conv_aux_8_2_b[10][10][512] = {0};//
static const struct ConvParamsSimple conv_aux_8_2_params = {.batch_size=1, .in_dim=19, .kernel_size=3, .in_channels=256, .out_channels=512, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=10, .output_scale=1.0};


static const elem_t conv_aux_9_1_w[128][1][1][512] = {0};
static  elem_t conv_aux_9_1_out[10][10][128] = {0};
static  elem_t conv_aux_9_1_b[10][10][128] = {0};//
static const struct ConvParamsSimple conv_aux_9_1_params = {.batch_size=1, .in_dim=10, .kernel_size=1, .in_channels=512, .out_channels=128, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=10, .output_scale=1.0};

static const elem_t conv_aux_9_2_w[256][3][3][128] = {0};
static  elem_t conv_aux_9_2_out[5][5][256] = {0};
static  elem_t conv_aux_9_2_b[5][5][256] = {0};//
static const struct ConvParamsSimple conv_aux_9_2_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=128, .out_channels=256, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=5, .output_scale=1.0};

static const elem_t conv_aux_10_1_w[128][1][1][256] = {0};
static  elem_t conv_aux_10_1_out[5][5][128] = {0};
static  elem_t conv_aux_10_1_b[5][5][128] = {0};//
static const struct ConvParamsSimple conv_aux_10_1_params = {.batch_size=1, .in_dim=5, .kernel_size=1, .in_channels=256, .out_channels=128, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=5, .output_scale=1.0};

static const elem_t conv_aux_10_2_w[256][3][3][128] = {0};
static  elem_t conv_aux_10_2_out[3][3][256] = {0};
static  elem_t conv_aux_10_2_b[3][3][256] = {0};//
static const struct ConvParamsSimple conv_aux_10_2_params = {.batch_size=1, .in_dim=5, .kernel_size=3, .in_channels=128, .out_channels=256, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=3, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=3, .output_scale=1.0};


static const elem_t conv_aux_11_1_w[128][1][1][256] = {0};
static  elem_t conv_aux_11_1_out[3][3][512] = {0};
static  elem_t conv_aux_11_1_b[3][3][512] = {0};
static const struct ConvParamsSimple conv_aux_11_1_params = {.batch_size=1, .in_dim=3, .kernel_size=1, .in_channels=256, .out_channels=128, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=3, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=3, .output_scale=1.0};

static const elem_t conv_aux_11_2_w[256][3][3][128] = {0};
static  elem_t conv_aux_11_2_out[1][1][512] = {0};
static  elem_t conv_aux_11_2_b[1][1][512] = {0};
static const struct ConvParamsSimple conv_aux_11_2_params = {.batch_size=1, .in_dim=3, .kernel_size=3, .in_channels=128, .out_channels=256, .stride=1, .padding=1, .bias=0, .depthwise=0, .out_dim=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=1, .output_scale=1.0};

/*
*  LOCALIZATION PREDICTION CONVOLUTIONS
*/

static const elem_t conv_loc_pred_1_w[16][3][3][512] = {0};
static  elem_t conv_loc_pred_1_out[38][38][16] = {0};
static  elem_t conv_loc_pred_1_b[38][38][16] = {0};
static const struct ConvParamsSimple conv_loc_pred_1_params = {.batch_size=1, .in_dim=38, .kernel_size=3, .in_channels=512, .out_channels=16, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=38, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_loc_pred_2_w[24][3][3][1024] = {0};
static  elem_t conv_loc_pred_2_out[19][19][24] = {0};
static  elem_t conv_loc_pred_2_b[19][19][24] = {0};
static const struct ConvParamsSimple conv_loc_pred_2_params = {.batch_size=1, .in_dim=19, .kernel_size=3, .in_channels=1024, .out_channels=24, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=19, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_loc_pred_3_w[24][3][3][512] = {0};
static  elem_t conv_loc_pred_3_out[10][10][24] = {0};
static  elem_t conv_loc_pred_3_b[10][10][24] = {0};
static const struct ConvParamsSimple conv_loc_pred_3_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=512, .out_channels=24, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_loc_pred_4_w[24][3][3][256] = {0};
static  elem_t conv_loc_pred_4_out[5][5][24] = {0};
static  elem_t conv_loc_pred_4_b[5][5][24] = {0};
static const struct ConvParamsSimple conv_loc_pred_4_params = {.batch_size=1, .in_dim=5, .kernel_size=3, .in_channels=256, .out_channels=24, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_loc_pred_5_w[16][3][3][256] = {0};
static  elem_t conv_loc_pred_5_out[3][3][16] = {0};
static  elem_t conv_loc_pred_5_b[3][3][16] = {0};
static const struct ConvParamsSimple conv_loc_pred_5_params = {.batch_size=1, .in_dim=3, .kernel_size=3, .in_channels=256, .out_channels=16, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=3, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_loc_pred_6_w[16][3][3][256] = {0};
static  elem_t conv_loc_pred_6_out[1][1][16] = {0};
static  elem_t conv_loc_pred_6_b[1][1][16] = {0};
static const struct ConvParamsSimple conv_loc_pred_6_params = {.batch_size=1, .in_dim=1, .kernel_size=3, .in_channels=256, .out_channels=16, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

/*
*  CLASS PREDICTION CONVOLUTIONS
*/
static const elem_t conv_class_pred_1_w[64][3][3][512] = {0};
static  elem_t conv_class_pred_1_out[38][38][16] = {0};
static  elem_t conv_class_pred_1_b[38][38][16] = {0};
static const struct ConvParamsSimple conv_class_pred_1_params = {.batch_size=1, .in_dim=38, .kernel_size=3, .in_channels=512, .out_channels=64, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=38, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_class_pred_2_w[96][3][3][1024] = {0};
static  elem_t conv_class_pred_2_out[19][19][16] = {0};
static  elem_t conv_class_pred_2_b[19][19][16] = {0};
static const struct ConvParamsSimple conv_class_pred_2_params = {.batch_size=1, .in_dim=19, .kernel_size=3, .in_channels=1024, .out_channels=96, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=19, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_class_pred_3_w[96][3][3][512] = {0};
static  elem_t conv_class_pred_3_out[10][10][16] = {0};
static  elem_t conv_class_pred_3_b[10][10][16] = {0};
static const struct ConvParamsSimple conv_class_pred_3_params = {.batch_size=1, .in_dim=10, .kernel_size=3, .in_channels=512, .out_channels=96, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=10, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_class_pred_4_w[96][3][3][256] = {0};
static  elem_t conv_class_pred_4_out[5][5][16] = {0};
static  elem_t conv_class_pred_4_b[5][5][16] = {0};
static const struct ConvParamsSimple conv_class_pred_4_params = {.batch_size=1, .in_dim=5, .kernel_size=3, .in_channels=256, .out_channels=96, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=5, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_class_pred_5_w[64][3][3][256] = {0};
static  elem_t conv_class_pred_5_out[3][3][16] = {0};
static  elem_t conv_class_pred_5_b[3][3][16] = {0};
static const struct ConvParamsSimple conv_class_pred_5_params = {.batch_size=1, .in_dim=3, .kernel_size=3, .in_channels=256, .out_channels=64, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=3, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};

static const elem_t conv_class_pred_6_w[64][3][3][256] = {0};
static  elem_t conv_class_pred_6_out[1][1][16] = {0};
static  elem_t conv_class_pred_6_b[1][1][16] = {0};
static const struct ConvParamsSimple conv_class_pred_6_params = {.batch_size=1, .in_dim=1, .kernel_size=3, .in_channels=256, .out_channels=64, .stride=1, .padding=0, .bias=0, .depthwise=0, .out_dim=1, .pool_size=1, .pool_stride=1, .pool_padding=0, .out_dim_pooled=14, .output_scale=1.0};


#endif /* F2F15EE8_05C8_43F3_96E4_FAF8C462ABFB */
