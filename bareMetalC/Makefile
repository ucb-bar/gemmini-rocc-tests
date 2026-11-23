include $(abs_top_srcdir)/Makefrag

tests = \
	mvin_mvout \
	mvin_mvout_zeros \
	mvin_mvout_stride \
	mvin_mvout_block_stride \
	mvin_mvout_acc \
	mvin_mvout_acc_zero_stride \
	mvin_mvout_acc_stride \
	mvin_mvout_acc_full \
	mvin_mvout_acc_full_stride \
	mvin_mvout_spad \
	matmul_os \
	matmul_ws \
	matmul \
	matmul_spad \
	raw_hazard \
	aligned \
	padded \
	mvin_scale \
	conv \
	conv_stride \
	conv_rect \
	conv_rect_pool \
	conv_with_pool \
	conv_with_rot180 \
	conv_with_kernel_dilation \
	conv_with_input_dilation \
	conv_with_input_dilation_and_rot180 \
	conv_with_input_dilation_and_neg_padding \
	conv_trans_output_1203 \
	conv_trans_weight_1203 \
	conv_trans_weight_0132 \
	conv_trans_input_3120 \
	conv_trans_input_3120_with_kernel_dilation \
	conv_first_layer \
	conv_dw \
	conv_perf \
	conv_dw_perf \
	tiled_matmul_os \
	tiled_matmul_ws \
	tiled_matmul_ws_At \
	tiled_matmul_ws_Bt \
	tiled_matmul_ws_full_C \
	tiled_matmul_ws_low_D \
	tiled_matmul_ws_igelu \
	tiled_matmul_ws_layernorm \
	tiled_matmul_ws_softmax \
	tiled_matmul_ws_perf \
	tiled_matmul_cpu \
	tiled_matmul_option \
	transpose \
	matrix_add \
	resadd \
	resadd_stride \
	global_average \
	gemmini_counter \
	template \

# Plain BF16 Gemmini (gemmini_params_bf16_dim*_vpu.h): BF16 elements, no MX scaling
HAS_BF16_ELEM := $(shell grep -q 'define ELEM_T_EXP_BITS 8' $(abs_top_srcdir)/include/gemmini_params.h 2>/dev/null && grep -q 'define ELEM_T_SIG_BITS 8' $(abs_top_srcdir)/include/gemmini_params.h && echo yes)
ifeq ($(HAS_BF16_ELEM),yes)
tests += tiled_matmul_bf16 mvout_spad_vpu_bf16
endif

include $(abs_top_srcdir)/common.mk
