// Phase cycle timing (rdcycle) + Gemmini HW perf counters for the MX perf tests.
#ifndef MX_PERF_H
#define MX_PERF_H

#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

// RTL only (Spike returns dummy counts); counter ops hang if num_counter=0, -DMX_PERF_NO_HW_COUNTERS opts out.
#if !defined(GEMMINI_MMIO) && !defined(MX_PERF_NO_HW_COUNTERS)
#define MX_PERF_HW_COUNTERS
#endif

#define MX_PERF_NCTR 8

typedef struct {
  uint64_t t0, t_sc, t_lda, t_ld, t_ex, t_st;   // t_sc/t_lda optional: load-phase split
  uint32_t ctr[MX_PERF_NCTR];
  uint32_t ld_a[MX_PERF_NCTR], ld_b[MX_PERF_NCTR];   // load-phase counters, cumulative at t_lda / t_ld
  int ld_ctr;
} mx_perf_t;

static const int mx_perf_events[MX_PERF_NCTR] = {
  EXE_ACTIVE_CYCLE, LOOP_MATMUL_ACTIVE_CYCLES, MAIN_EX_CYCLES, MAIN_ST_EX_CYCLES,
  SCRATCHPAD_A_WAIT_CYCLE, EXE_OVERLAP_HAZ_CYCLE, MAIN_ST_CYCLES, RESERVATION_STATION_FULL_CYCLES,
};
static const char *mx_perf_names[MX_PERF_NCTR] = {
  "exe_active", "loop_active", "ex_only", "st_ex", "spad_a_wait", "overlap_haz", "st_only", "rs_full",
};

static const int mx_perf_ld_events[MX_PERF_NCTR] = {
  LOAD_ACTIVE_CYCLE, LOAD_DMA_WAIT_CYCLE, LOAD_SCRATCHPAD_WAIT_CYCLE, RDMA_ACTIVE_CYCLE,
  RDMA_XACT_FULL_CYCLES, RDMA_TOTAL_LATENCY, RDMA_TL_WAIT_CYCLES, RESERVATION_STATION_FULL_CYCLES,
};
#define MX_PERF_LD_INFLIGHT 5   // slot of RDMA_TOTAL_LATENCY (external) in mx_perf_ld_events
static const char *mx_perf_ld_names[MX_PERF_NCTR] = {
  "ld_active", "ld_dma_wait", "ld_dma_bp", "rdma_active", "xact_full", "inflight_sum", "rdma_tl_wait", "rs_full",
};

// Macro so the fence is the including test's gemmini_fence (radiance tests redefine it).
#define MX_PERF_MARK(t) do { gemmini_fence(); (t) = read_cycles(); } while (0)

static void mx_perf_ctr_start(mx_perf_t *p) {
#ifdef MX_PERF_HW_COUNTERS
  counter_snapshot_reset();
  for (int i = 0; i < MX_PERF_NCTR; i++)
    counter_configure(i, mx_perf_events[i]);
#endif
}

// Call after MX_PERF_MARK (counter ops bypass the RS, so they need the fence).
static void mx_perf_ctr_stop(mx_perf_t *p) {
#ifdef MX_PERF_HW_COUNTERS
  counter_snapshot_take();
  for (int i = 0; i < MX_PERF_NCTR; i++)
    p->ctr[i] = counter_read(i);
  counter_snapshot_reset();
#endif
}

// Quad-throughput ideal (FP4 / LUT 4-bit operands: 4 MACs per PE per cycle).
#define MX_PERF_IDEAL_QUAD(M, N, K, dim) ((uint64_t)(M) * (N) * (K) / (4 * (uint64_t)(dim) * (dim)))

static void mx_perf_report_ideal(const char *name, int M, int N, int K, int dim, uint64_t ideal, const mx_perf_t *p) {
  uint64_t ld = p->t_ld - p->t0, ex = p->t_ex - p->t_ld, st = p->t_st - p->t_ex;
  uint64_t util = ex ? ideal * 1000 / ex : 0;
  printf("PERF %s M=%d N=%d K=%d DIM=%d load=%lu compute=%lu mvout=%lu total=%lu ideal=%lu util=%lu.%lu%%\n",
         name, M, N, K, dim, (unsigned long)ld, (unsigned long)ex, (unsigned long)st,
         (unsigned long)(p->t_st - p->t0), (unsigned long)ideal, (unsigned long)(util / 10), (unsigned long)(util % 10));
#ifdef MX_PERF_HW_COUNTERS
  printf("PERF_HW %s", name);
  for (int i = 0; i < MX_PERF_NCTR; i++)
    printf(" %s=%u", mx_perf_names[i], p->ctr[i]);
  // other = compute cycles with neither execute nor store busy (front-end / fence / idle)
  uint64_t busy = (uint64_t)p->ctr[2] + p->ctr[3] + p->ctr[6];
  printf(" other=%ld\n", (long)(ex - busy));
#endif
}

static void mx_perf_report(const char *name, int M, int N, int K, int dim, const mx_perf_t *p) {
  mx_perf_report_ideal(name, M, N, K, dim, (uint64_t)M * N * K / ((uint64_t)dim * dim), p);
}

// Load-phase counters: start after the scale load (MARK t_sc), snapshot at t_lda and t_ld.
static void mx_perf_ld_ctr_start(mx_perf_t *p) {
#ifdef MX_PERF_HW_COUNTERS
  counter_reset();   // external counters (inflight_sum) only clear on a global reset
  counter_snapshot_reset();
  for (int i = 0; i < MX_PERF_NCTR; i++)
    counter_configure(i, mx_perf_ld_events[i]);
  p->ld_ctr = 1;
#endif
}

static void mx_perf_ld_ctr_read(uint32_t *dst) {
#ifdef MX_PERF_HW_COUNTERS
  counter_snapshot_take();
  for (int i = 0; i < MX_PERF_NCTR; i++)
    dst[i] = counter_read(i);
  counter_snapshot_reset();
  // CounterFile snapshot picks internal-vs-external by the READ index (io.addr), so external slots
  // snapshot as 0: read the external one (inflight_sum) live — the DMA is idle after the fence.
  dst[MX_PERF_LD_INFLIGHT] = counter_read(MX_PERF_LD_INFLIGHT);
#endif
}

// Load-phase split: scale loader vs mvin A vs mvin B, with bytes/cycle (x100) for each.
static void mx_perf_report_load(const char *name, int scale_bytes, int a_bytes, int b_bytes, const mx_perf_t *p) {
  uint64_t sc = p->t_sc - p->t0, la = p->t_lda - p->t_sc, lb = p->t_ld - p->t_lda;
  uint64_t r_sc = sc ? (uint64_t)scale_bytes * 100 / sc : 0;
  uint64_t r_a = la ? (uint64_t)a_bytes * 100 / la : 0;
  uint64_t r_b = lb ? (uint64_t)b_bytes * 100 / lb : 0;
  printf("PERF_LD %s scales=%lu(%dB,%lu.%02luB/c) mvin_a=%lu(%dB,%lu.%02luB/c) mvin_b=%lu(%dB,%lu.%02luB/c)\n",
         name, (unsigned long)sc, scale_bytes, (unsigned long)(r_sc / 100), (unsigned long)(r_sc % 100),
         (unsigned long)la, a_bytes, (unsigned long)(r_a / 100), (unsigned long)(r_a % 100),
         (unsigned long)lb, b_bytes, (unsigned long)(r_b / 100), (unsigned long)(r_b % 100));
  if (p->ld_ctr) {
    printf("PERF_LD_HW %s A:", name);
    for (int i = 0; i < MX_PERF_NCTR; i++)
      printf(" %s=%u", mx_perf_ld_names[i], p->ld_a[i]);
    printf("\nPERF_LD_HW %s B:", name);
    for (int i = 0; i < MX_PERF_NCTR; i++)
      printf(" %s=%u", mx_perf_ld_names[i], p->ld_b[i] - p->ld_a[i]);
    printf("\n");
    // Little's law: inflight_sum = sum over cycles of occupied xact slots (reserved -> last beat).
    // avg in flight = sum / phase cycles; avg latency per 64B request = sum / (bytes / 64).
    uint64_t sa = p->ld_a[MX_PERF_LD_INFLIGHT], sb = p->ld_b[MX_PERF_LD_INFLIGHT] - p->ld_a[MX_PERF_LD_INFLIGHT];
    printf("PERF_LD_LITTLE %s A: avg_inflight=%lu.%lu lat64=%lu  B: avg_inflight=%lu.%lu lat64=%lu\n", name,
           (unsigned long)(la ? sa * 10 / la / 10 : 0), (unsigned long)(la ? sa * 10 / la % 10 : 0),
           (unsigned long)(a_bytes ? sa * 64 / a_bytes : 0),
           (unsigned long)(lb ? sb * 10 / lb / 10 : 0), (unsigned long)(lb ? sb * 10 / lb % 10 : 0),
           (unsigned long)(b_bytes ? sb * 64 / b_bytes : 0));
  }
}

#endif
