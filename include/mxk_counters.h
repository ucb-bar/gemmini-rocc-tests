// Gemmini perf counters for the mxk kernels (-DMK_CTR): the timed loop runs MXK_CTR_ITERS times; iteration
// ITERS-2 counts the mesh busy split (PERF_HW, as mx_bench_matmul pass 1), iteration ITERS-1 the load path
// (PERF_LD_HW / PERF_LITTLE, as pass 2). Earlier iterations only warm the caches. Without -DMK_CTR: no-ops.
#ifndef MXK_COUNTERS_H
#define MXK_COUNTERS_H
#include <stdint.h>
#include <stdio.h>

#ifdef MK_CTR
#define MXK_CTR_ITERS 3
static const int mxk_ev[8] = { EXE_ACTIVE_CYCLE, MAIN_EX_CYCLES, MAIN_LD_CYCLES, MAIN_ST_CYCLES,
                               MAIN_LD_EX_CYCLES, MAIN_ST_EX_CYCLES, MAIN_LD_ST_CYCLES, MAIN_LD_ST_EX_CYCLES };
static const char *mxk_evn[8] = { "exe_active", "ex", "ld", "st", "ld_ex", "st_ex", "ld_st", "ld_st_ex" };
// slots 5 and 7 are external counters: cleared only by counter_reset, read live after the fence
static const int mxk_lev[8] = { LOAD_ACTIVE_CYCLE, LOAD_DMA_WAIT_CYCLE, RDMA_ACTIVE_CYCLE, RDMA_XACT_FULL_CYCLES,
                                RDMA_TL_WAIT_CYCLES, RDMA_TOTAL_LATENCY, LOAD_CMD_TRACKER_FULL_CYCLES, RDMA_BYTES_REC };
static const char *mxk_levn[8] = { "ld_active", "ld_dma_wait", "rdma_active", "xact_full", "tl_wait",
                                   "inflight_sum", "ld_cmd_full", "rdma_bytes" };

static void mxk_ctr_begin(int it) {
  const int *e = it == MXK_CTR_ITERS - 2 ? mxk_ev : it == MXK_CTR_ITERS - 1 ? mxk_lev : 0;
  if (!e) return;
  counter_reset();
  counter_snapshot_reset();
  for (int i = 0; i < 8; i++) counter_configure(i, e[i]);
}

static void mxk_ctr_end(int it, uint64_t cyc) {   // after the iteration's final fence
  uint32_t c[8];
  if (it == MXK_CTR_ITERS - 2) {
    counter_snapshot_take();
    for (int i = 0; i < 8; i++) c[i] = counter_read(i);
    counter_snapshot_reset();
    uint64_t busy = 0;
    printf("PERF_HW cycles=%lu", (unsigned long) cyc);
    for (int i = 0; i < 8; i++) { printf(" %s=%u", mxk_evn[i], c[i]); if (i) busy += c[i]; }
    printf(" idle=%ld\n", (long) cyc - (long) busy);
  } else if (it == MXK_CTR_ITERS - 1) {
    counter_snapshot_take();
    for (int i = 0; i < 8; i++) c[i] = counter_read(i);
    counter_snapshot_reset();
    c[5] = counter_read(5);
    c[7] = counter_read(7);
    printf("PERF_LD_HW cycles=%lu", (unsigned long) cyc);
    for (int i = 0; i < 8; i++) printf(" %s=%u", mxk_levn[i], c[i]);
    printf("\n");
    const uint64_t reqs = c[7] / 64;
    printf("PERF_LITTLE avg_inflight=%lu.%lu lat_per_64B=%lu xact_full=%lu%% tl_wait=%lu%% ld_cmd_full=%lu%% "
           "dma_Bpc=%lu.%02lu\n",
           (unsigned long) ((uint64_t) c[5] / cyc), (unsigned long) ((uint64_t) c[5] * 10 / cyc % 10),
           (unsigned long) (reqs ? c[5] / reqs : 0), (unsigned long) ((uint64_t) c[3] * 100 / cyc),
           (unsigned long) ((uint64_t) c[4] * 100 / cyc), (unsigned long) ((uint64_t) c[6] * 100 / cyc),
           (unsigned long) (c[7] / cyc), (unsigned long) ((uint64_t) c[7] * 100 / cyc % 100));
  }
}
#else
static inline void mxk_ctr_begin(int it) { (void) it; }
static inline void mxk_ctr_end(int it, uint64_t cyc) { (void) it; (void) cyc; }
#endif
#endif
