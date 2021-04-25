// See LICENSE for license details.

#ifndef GEMMINI_MT_H
#define GEMMINI_MT_H

#include "include/threadpool.h"
#include "include/gemmini.h"

typedef struct args_tiled_conv_auto_t {
  int batch_size; int in_dim; int in_channels;
  int out_channels; int out_dim;
  int stride; int dilation; int padding; int kernel_dim;
  bool wrot180;

  int out_channels_stride; int weight_out_channels_stride;

  const elem_t * input;
  const elem_t * weights;
  const acc_t * bias;
  elem_t * output;

  int act; acc_scale_t scale; size_t relu6_shift;
  int pool_size; int pool_stride; int pool_padding; bool pool_ceil_dim;

  enum tiled_matmul_type_t tiled_conv_type;
} args_tiled_conv_auto_t;

static void worker_tiled_conv_auto(void * args_ptr) {
  args_tiled_conv_auto_t * args = args_ptr;

  tiled_conv_A_stride_auto(
    args->batch_size, args->in_dim, args->in_channels,
    args->out_channels, args->out_dim,
    args->stride, args->dilation, args->padding, args->kernel_dim,
    args->wrot180,

    args->out_channels_stride, args->weight_out_channels_stride,

    args->input,
    args->weights,
    args->bias,
    args->output,

    args->act, args->scale, args->relu6_shift,
    args->pool_size, args->pool_stride, args->pool_padding, args->pool_ceil_dim,

    args->tiled_conv_type
  );
}

// Batch parallel convolution

static void tiled_conv_batch_parallel_auto(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int dilation, int padding, int kernel_dim,
        bool wrot180,

        int out_channels_stride, int weight_out_channels_stride,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,

        enum tiled_matmul_type_t tiled_conv_type) {

  args_tiled_conv_auto_t args[THREADS];

  const bool no_pool = pool_stride == 0;
  if (no_pool) {
      pool_size = 1;
      pool_stride = 1;
      pool_padding = 0;
  }

  int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;
  if (pool_ceil_dim)
    pool_out_dim += (out_dim + 2*pool_padding - pool_size) % pool_stride != 0;

  const int batches_per_thread = batch_size / THREADS + (batch_size % THREADS != 0);

  for (int t = 0; t < THREADS; t++) {
    const int batch = t * batches_per_thread;

    const int batches = batch + batches_per_thread <= batch_size ?
      batches_per_thread :
      batch_size - batch;

    args[t].batch_size = batches;
    args[t].in_dim = in_dim;
    args[t].in_channels = in_channels;
    args[t].out_channels = out_channels;
    args[t].out_dim = out_dim;
    args[t].stride = stride;
    args[t].dilation = dilation;
    args[t].padding = padding;
    args[t].kernel_dim = kernel_dim;
    args[t].wrot180 = wrot180;

    args[t].out_channels_stride = out_channels_stride;
    args[t].weight_out_channels_stride = weight_out_channels_stride;

    args[t].input = (elem_t*)input + batch*in_dim*in_dim*in_channels;
    args[t].weights = weights;
    args[t].bias = bias;
    args[t].output = (elem_t*)output + batch*pool_out_dim*pool_out_dim*out_channels_stride;

    args[t].act = act;
    args[t].scale = scale;
    args[t].relu6_shift = relu6_shift;
    args[t].pool_size = pool_size;
    args[t].pool_stride = pool_stride;
    args[t].pool_padding = pool_padding;
    args[t].pool_ceil_dim = pool_ceil_dim;

    args[t].tiled_conv_type = tiled_conv_type;

    SET_TASK(t, worker_tiled_conv_auto, &args[t]);
  }
  RUN_TASKS();
}

// Out-channel parallel convolution

static void tiled_conv_outchannel_parallel_auto(
        int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int dilation, int padding, int kernel_dim,
        bool wrot180,

        int out_channels_stride, int weight_out_channels_stride,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale, size_t relu6_shift,
        int pool_size, int pool_stride, int pool_padding, bool pool_ceil_dim,

        enum tiled_matmul_type_t tiled_conv_type) {

  args_tiled_conv_auto_t args[THREADS];

  const int outchannel_per_thread = out_channels / THREADS + (out_channels % THREADS != 0);

  for (int t = 0; t < THREADS; t++) {
    const int outchannel = t * outchannel_per_thread;

    const int outchannels = outchannel + outchannel_per_thread <= out_channels ?
      outchannel_per_thread :
      out_channels - outchannel;

    args[t].batch_size = batch_size;
    args[t].in_dim = in_dim;
    args[t].in_channels = in_channels;
    args[t].out_channels = outchannels;
    args[t].out_dim = out_dim;
    args[t].stride = stride;
    args[t].dilation = dilation;
    args[t].padding = padding;
    args[t].kernel_dim = kernel_dim;
    args[t].wrot180 = wrot180;

    args[t].out_channels_stride = out_channels_stride;
    args[t].weight_out_channels_stride = weight_out_channels_stride;

    args[t].input = input;
    args[t].weights = (elem_t*)weights + outchannel;
    args[t].bias = bias;
    args[t].output = (elem_t*)output + outchannel;

    args[t].act = act;
    args[t].scale = scale;
    args[t].relu6_shift = relu6_shift;
    args[t].pool_size = pool_size;
    args[t].pool_stride = pool_stride;
    args[t].pool_padding = pool_padding;
    args[t].pool_ceil_dim = pool_ceil_dim;

    args[t].tiled_conv_type = tiled_conv_type;

    SET_TASK(t, worker_tiled_conv_auto, &args[t]);
  }
  RUN_TASKS();
}

#endif

