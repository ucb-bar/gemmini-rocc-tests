// See LICENSE for license details.

#ifndef GEMMINI_MT_H
#define GEMMINI_MT_H

#include "include/threadpool.h"
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

//struct for nn_auto_extended split
typedef struct args_matmul_auto_t{
          size_t dim_I; size_t dim_J; size_t dim_K;
          size_t stride_C;
          const elem_t * A; const elem_t * B;
          const void * D; elem_t* C;
          int act; acc_scale_t scale; size_t relu6_shift; bool repeating_bias;
          enum tiled_matmul_type_t tiled_matmul_type;
          bool check; char * layer_name;
} args_matmul_auto_t;

typedef struct args_matmul2_auto_t{
          size_t dim_I; size_t dim_J; size_t dim_K;
          size_t stride_A; size_t stride_B; size_t stride_D; size_t stride_C;
          const elem_t * A; const elem_t * B;
          const void * D; elem_t* C;
          int act; acc_scale_t scale; size_t relu6_shift; bool repeating_bias;
          enum tiled_matmul_type_t tiled_matmul_type;
          bool check; char * layer_name;
} args_matmul2_auto_t;

typedef struct args_tiled_conv_auto_t {
  int batch_size; int in_dim; int in_channels;
  int out_channels; int out_dim;
  int stride; int dilation; int padding; int kernel_dim;
  bool wrot180;

  int out_channels_stride; int weight_out_channels_stride; int porow_max; int porow_start;

  const elem_t * input;
  const elem_t * weights;
  const acc_t * bias;
  elem_t * output;

  int act; acc_scale_t scale; size_t relu6_shift;
  int pool_size; int pool_stride; int pool_padding; bool pool_ceil_dim;

  enum tiled_matmul_type_t tiled_conv_type;
  int thread;
} args_tiled_conv_auto_t;

static void worker_matmul_extended_auto(void * args_ptr) {
  if (args_ptr == NULL)
    return;

  args_matmul_auto_t * args = args_ptr;

  tiled_matmul_nn_auto_extended(args->dim_I, args->dim_J, args->dim_K,
          args->stride_C,
          args->A, args->B,
          args->D, args->C,
          args->act, args->scale, args->relu6_shift, args->repeating_bias,
          args->tiled_matmul_type,
          args->check, args->layer_name);
}

static void worker_matmul2_extended_auto(void * args_ptr) {
  if (args_ptr == NULL)
    return;

  args_matmul2_auto_t * args = args_ptr;

  tiled_matmul_nn_auto_extended2(args->dim_I, args->dim_J, args->dim_K,
          args->stride_A, args->stride_B, args->stride_D, args->stride_C,
          args->A, args->B,
          args->D, args->C,
          args->act, args->scale, args->relu6_shift, args->repeating_bias,
          args->tiled_matmul_type,
          args->check, args->layer_name);
}

// static uint64_t time_taken[4];

static void worker_tiled_conv_auto(void * args_ptr) {
  if (args_ptr == NULL) {
    // printf("Skipping\n");
    return;
  }

  uint64_t start = read_cycles();

  args_tiled_conv_auto_t * args = args_ptr;

  tiled_conv_A_stride_auto(
    args->batch_size, args->in_dim, args->in_channels,
    args->out_channels, args->out_dim,
    args->stride, args->dilation, args->padding, args->kernel_dim,
    args->wrot180,

    args->out_channels_stride, args->weight_out_channels_stride, args->porow_max, args->porow_start,

    args->input,
    args->weights,
    args->bias,
    args->output,

    args->act, args->scale, args->relu6_shift,
    args->pool_size, args->pool_stride, args->pool_padding, args->pool_ceil_dim,

    args->tiled_conv_type
  );

  uint64_t end = read_cycles();

  // printf("trt %d: %llu\n", args->thread, end-start);
  // time_taken[args->thread] = end-start;
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
    args[t].porow_max = -1;

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

static void matmul_extended_auto_norun(size_t dim_I, size_t dim_J, size_t dim_K,
        size_t stride_C,
        const elem_t * A, const elem_t * B,
        const void * D, elem_t* C,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name, int thread, args_matmul_auto_t args){

    //define args

    args.dim_I = dim_I;
    args.dim_J = dim_J;
    args.dim_K = dim_K;

    args.stride_C = stride_C;

    args.A = A;
    args.B = B;

    args.D = D;
    args.C = C;

    args.act = act;
    args.scale = scale;
    args.relu6_shift = relu6_shift;
    args.repeating_bias = repeating_bias;

    args.tiled_matmul_type = tiled_matmul_type;

    args.check = check;
    args.layer_name = layer_name;

    SET_TASK(thread, worker_matmul_extended_auto, &args);
}


// Out-row parallel convolution
static void tiled_conv_outrow_parallel_auto(
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

  const int outrow_per_thread = pool_out_dim / THREADS + (pool_out_dim % THREADS != 0);

  for (int t = 0; t < THREADS; t++) {
    const int outrow = t * outrow_per_thread;
    const bool skip = outrow >= pool_out_dim;

    const int outrows = outrow + outrow_per_thread <= pool_out_dim ?
      outrow_per_thread :
      pool_out_dim - outrow;

    args[t].batch_size = batch_size;
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
    args[t].porow_max = outrows;
    args[t].porow_start = outrow;

    args[t].input = input;
    args[t].weights = (elem_t*)weights;
    args[t].bias = bias == NULL ? NULL : (bias);
    args[t].output = (elem_t*)output;

    args[t].act = act;
    args[t].scale = scale;
    args[t].relu6_shift = relu6_shift;
    args[t].pool_size = pool_size;
    args[t].pool_stride = pool_stride;
    args[t].pool_padding = pool_padding;
    args[t].pool_ceil_dim = pool_ceil_dim;

    args[t].tiled_conv_type = tiled_conv_type;

    args[t].thread = t;

    SET_TASK(t, worker_tiled_conv_auto, skip ? NULL : &args[t]);
  }
  RUN_TASKS();

  // for (int i = 0; i < 4; i++) {
  //   printf("time_taken[%d] = %llu\n", i, time_taken[i]);
  // }
}


// Out-channel parallel convolution
static void tiled_conv_outchannel_norun(
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

        enum tiled_matmul_type_t tiled_conv_type, int t, args_tiled_conv_auto_t args) {

    args.batch_size = batch_size;
    args.in_dim = in_dim;
    args.in_channels = in_channels;
    args.out_channels = out_channels;
    args.out_dim = out_dim;
    args.stride = stride;
    args.dilation = dilation;
    args.padding = padding;
    args.kernel_dim = kernel_dim;
    args.wrot180 = wrot180;

    args.out_channels_stride = out_channels_stride;
    args.weight_out_channels_stride = weight_out_channels_stride;
    args.porow_max = -1;
    args.porow_start = 0;

    args.input = input;
    args.weights = (elem_t*)weights;
    args.bias = bias;
    args.output = (elem_t*)output;

    args.act = act;
    args.scale = scale;
    args.relu6_shift = relu6_shift;
    args.pool_size = pool_size;
    args.pool_stride = pool_stride;
    args.pool_padding = pool_padding;
    args.pool_ceil_dim = pool_ceil_dim;

    args.tiled_conv_type = tiled_conv_type;

    SET_TASK(t, worker_tiled_conv_auto, &args);
}

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
  // printf("outchannel_per_thread: %d\n", outchannel_per_thread);
  // printf("outchannels: %d\n", out_channels);

  for (int t = 0; t < THREADS; t++) {
    const int outchannel = t * outchannel_per_thread;
    const bool skip = outchannel >= out_channels;

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
    args[t].porow_max = -1;
    args[t].porow_start = 0;

    args[t].input = input;
    args[t].weights = (elem_t*)weights + outchannel;
    args[t].bias = bias == NULL ? NULL : (bias + outchannel);
    args[t].output = (elem_t*)output + outchannel;

    args[t].act = act;
    args[t].scale = scale;
    args[t].relu6_shift = relu6_shift;
    args[t].pool_size = pool_size;
    args[t].pool_stride = pool_stride;
    args[t].pool_padding = pool_padding;
    args[t].pool_ceil_dim = pool_ceil_dim;

    args[t].tiled_conv_type = tiled_conv_type;

    args[t].thread = t;

    SET_TASK(t, worker_tiled_conv_auto, skip ? NULL : &args[t]);
  }
  RUN_TASKS();

  // for (int i = 0; i < 4; i++) {
  //   printf("time_taken[%d] = %llu\n", i, time_taken[i]);
  // }
}

static void tiled_conv_outchannel_and_outrow_parallel_auto(
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

  const int outrow_per_thread = pool_out_dim / 2 + (pool_out_dim % 2 != 0);
  const int outchannel_per_thread = out_channels / 2 + (out_channels % 2 != 0);

  for (int t = 0; t < THREADS; t++) {
    const int outrow = (t/2) * outrow_per_thread;
    const int outchannel = (t%2) * outchannel_per_thread;
    const bool skip = outchannel >= out_channels;

    const int outrows = outrow + outrow_per_thread <= pool_out_dim ?
      outrow_per_thread :
      pool_out_dim - outrow;

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
    args[t].porow_max = outrows;
    args[t].porow_start = outrow;

    args[t].input = input;
    args[t].weights = (elem_t*)weights + outchannel;
    args[t].bias = bias == NULL ? NULL : (bias + outchannel);
    args[t].output = (elem_t*)output + outchannel;

    args[t].act = act;
    args[t].scale = scale;
    args[t].relu6_shift = relu6_shift;
    args[t].pool_size = pool_size;
    args[t].pool_stride = pool_stride;
    args[t].pool_padding = pool_padding;
    args[t].pool_ceil_dim = pool_ceil_dim;

    args[t].tiled_conv_type = tiled_conv_type;

    args[t].thread = t;

    SET_TASK(t, worker_tiled_conv_auto, skip ? NULL : &args[t]);
  }
  RUN_TASKS();

  // for (int i = 0; i < 4; i++) {
  //   printf("time_taken[%d] = %llu\n", i, time_taken[i]);
  // }
}

// I-parallel matmul
static void tiled_matmul_nn_I_parallel_auto_extended(size_t dim_I, size_t dim_J, size_t dim_K,
        size_t stride_C,
        const elem_t * A, const elem_t * B,
        const void * D, elem_t * C,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
  args_matmul2_auto_t args[THREADS];

  const int dim_I_per_thread = dim_I / THREADS + (dim_I % THREADS != 0);

  for (int t = 0; t < THREADS; t++) {
    const int I = t * dim_I_per_thread;
    const bool skip = I >= dim_I;

    const int I_dim = I + dim_I_per_thread <= dim_I ?
      dim_I_per_thread :
      dim_I - I;

    args[t].dim_I = I_dim;
    args[t].dim_J = dim_J;
    args[t].dim_K = dim_K;

    args[t].stride_A = dim_K;
    args[t].stride_B = dim_J;
    args[t].stride_D = dim_J;
    args[t].stride_C = stride_C;

    args[t].A = A + I * dim_K;
    args[t].B = B;
    args[t].D = (D == NULL || repeating_bias) ? D : ((acc_t*)D + I * dim_J);
    args[t].C = C + I * dim_J;

    args[t].act = act;
    args[t].scale = scale;
    args[t].relu6_shift = relu6_shift;
    args[t].repeating_bias = repeating_bias;

    args[t].tiled_matmul_type = tiled_matmul_type;

    args[t].check = check;
    args[t].layer_name = layer_name;

    SET_TASK(t, worker_matmul2_extended_auto, skip ? NULL : &args[t]);
  }

  RUN_TASKS();
}

// J-parallel matmul
static void tiled_matmul_nn_J_parallel_auto_extended(size_t dim_I, size_t dim_J, size_t dim_K,
        size_t stride_C,
        const elem_t * A, const elem_t * B,
        const void * D, elem_t * C,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
  args_matmul2_auto_t args[THREADS];

  const int dim_J_per_thread = dim_J / THREADS + (dim_J % THREADS != 0);

  for (int t = 0; t < THREADS; t++) {
    const int J = t * dim_J_per_thread;
    const bool skip = J >= dim_J;

    const int J_dim = J + dim_J_per_thread <= dim_J ?
      dim_J_per_thread :
      dim_J - J;

    args[t].dim_I = dim_I;
    args[t].dim_J = J_dim;
    args[t].dim_K = dim_K;

    args[t].stride_A = dim_K;
    args[t].stride_B = dim_J;
    args[t].stride_D = dim_J;
    args[t].stride_C = stride_C;

    args[t].A = A;
    args[t].B = B + J;
    args[t].D = (D == NULL || repeating_bias) ? D : ((acc_t*)D + J);
    args[t].C = C + J;

    args[t].act = act;
    args[t].scale = scale;
    args[t].relu6_shift = relu6_shift;
    args[t].repeating_bias = repeating_bias;

    args[t].tiled_matmul_type = tiled_matmul_type;

    args[t].check = check;
    args[t].layer_name = layer_name;

    SET_TASK(t, worker_matmul2_extended_auto, skip ? NULL : &args[t]);
  }

  RUN_TASKS();
}

// IJ-parallel matmul
static void tiled_matmul_nn_IJ_parallel_auto_extended(size_t dim_I, size_t dim_J, size_t dim_K,
        size_t stride_C,
        const elem_t * A, const elem_t * B,
        const void * D, elem_t * C,
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
  args_matmul2_auto_t args[THREADS];

  const int dim_I_per_thread = dim_I / 2 + (dim_I % 2 != 0);
  const int dim_J_per_thread = dim_J / 2 + (dim_J % 2 != 0);

  for (int t = 0; t < THREADS; t++) {
    const int I = (t / 2) * dim_I_per_thread;
    const int J = (t % 2) * dim_J_per_thread;
    const bool skip = I >= dim_I || J >= dim_J;

    const int I_dim = I + dim_I_per_thread <= dim_I ?
      dim_I_per_thread :
      dim_I - I;

    const int J_dim = J + dim_J_per_thread <= dim_J ?
      dim_J_per_thread :
      dim_J - J;

    args[t].dim_I = I_dim;
    args[t].dim_J = J_dim;
    args[t].dim_K = dim_K;

    args[t].stride_A = dim_K;
    args[t].stride_B = dim_J;
    args[t].stride_D = dim_J;
    args[t].stride_C = stride_C;

    args[t].A = A + I * dim_K;
    args[t].B = B + J;
    args[t].D = (D == NULL || repeating_bias) ? D : ((acc_t*)D + I * dim_J + J);
    args[t].C = C + I * dim_J + J;

    args[t].act = act;
    args[t].scale = scale;
    args[t].relu6_shift = relu6_shift;
    args[t].repeating_bias = repeating_bias;

    args[t].tiled_matmul_type = tiled_matmul_type;

    args[t].check = check;
    args[t].layer_name = layer_name;

    SET_TASK(t, worker_matmul2_extended_auto, skip ? NULL : &args[t]);
  }

  RUN_TASKS();
}

#endif

