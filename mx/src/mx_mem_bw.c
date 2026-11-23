// Memory characterization, no compute: DMA mvin (cold DRAM vs clean L2 hit, 64B vs 16B rows), the MX scale
// loader, and mvout. Standalone only. Source data = the 128x128 test's A_in/B_in (.rodata, never touched by
// the CPU, so the only L2 client is Gemmini: warm phases are clean hits with no L1 probes).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_128x128.h"
#include "include/mx_perf.h"

#ifdef SPIKE_SIM
int main() { printf("skipped: standalone-only test\n"); return 0; }
#else

#define DIM 16
#define N 128                       // A_in/B_in are N x N bytes
#define TILES (N / DIM)

typedef struct {
  const char *name;
  uint64_t cyc;
  uint32_t c[MX_PERF_NCTR];
  int bytes, nreq;
} phase_t;

static void phase_begin(void) {
  gemmini_fence();
  counter_reset();                  // clears the external inflight_sum
  counter_snapshot_reset();
  for (int i = 0; i < MX_PERF_NCTR; i++)
    counter_configure(i, mx_perf_ld_events[i]);
}

static void phase_end(phase_t *p, uint64_t t0) {
  gemmini_fence();
  p->cyc = read_cycles() - t0;
  counter_snapshot_take();
  for (int i = 0; i < MX_PERF_NCTR; i++)
    p->c[i] = counter_read(i);
  counter_snapshot_reset();
  p->c[MX_PERF_LD_INFLIGHT] = counter_read(MX_PERF_LD_INFLIGHT);   // external: snapshot reads it as 0
}

static void report(const phase_t *p) {
  uint64_t bpc = p->cyc ? (uint64_t)p->bytes * 100 / p->cyc : 0;
  uint64_t s = p->c[MX_PERF_LD_INFLIGHT];
  uint64_t avg = p->cyc ? s * 10 / p->cyc : 0;
  printf("MEMBW %s bytes=%d cyc=%lu B/c=%lu.%02lu reqs=%d avg_inflight=%lu.%lu lat_req=%lu |",
         p->name, p->bytes, (unsigned long)p->cyc, (unsigned long)(bpc / 100), (unsigned long)(bpc % 100),
         p->nreq, (unsigned long)(avg / 10), (unsigned long)(avg % 10),
         (unsigned long)(p->nreq ? s / p->nreq : 0));
  for (int i = 0; i < MX_PERF_NCTR; i++)
    if (i != MX_PERF_LD_INFLIGHT) printf(" %s=%u", mx_perf_ld_names[i], p->c[i]);
  printf("\n");
}

// mvin an N x N byte matrix as DIM x DIM tiles, `t` k-tiles per mvin (row = DIM*t bytes, block stride DIM).
static void mvin_matrix(phase_t *p, const char *name, const uint8_t *src, uint32_t sp_base, int t) {
  p->name = name;
  p->bytes = N * N;
  p->nreq = N * (TILES / t);        // one DMA row request per matrix row per mvin
  phase_begin();
  uint64_t t0 = read_cycles();
  gemmini_config_ld(N);
  for (int i = 0; i < TILES; i++)
    for (int k = 0; k < TILES; k += t)
      gemmini_extended_mvin((void *)(src + i * DIM * N + k * DIM), sp_base + (i * TILES + k) * DIM, DIM * t, DIM);
  phase_end(p, t0);
}

static void scale_load(phase_t *p, const char *name) {
  p->name = name;
  p->bytes = sizeof(A_scales_row);
  p->nreq = sizeof(A_scales_row) / 8;   // loader issues one 8B Get at a time
  phase_begin();
  uint64_t t0 = read_cycles();
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  phase_end(p, t0);
}

static uint8_t out_buf[N * N] __attribute__((aligned(64)));   // .bss: CPU-zeroed at boot, so mvout lines may be L1-owned

int main() {
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

  const uint32_t sp_a = 0, sp_b = TILES * TILES * DIM;
  phase_t ph[8];
  int n = 0;

  mvin_matrix(&ph[n++], "A_cold_64B", (const uint8_t *)A_in, sp_a, 4);
  mvin_matrix(&ph[n++], "A_warm_64B", (const uint8_t *)A_in, sp_a, 4);
  mvin_matrix(&ph[n++], "B_cold_16B", (const uint8_t *)B_in, sp_b, 1);
  mvin_matrix(&ph[n++], "B_warm_16B", (const uint8_t *)B_in, sp_b, 1);
  scale_load(&ph[n++], "scale_cold");
  scale_load(&ph[n++], "scale_warm");

  // mvout: 16KB spad -> DRAM, one tile per mvout (16B rows), flat. Write path: load counters mostly idle.
  {
    phase_t *p = &ph[n++];
    p->name = "mvout_16B";
    p->bytes = N * N;
    p->nreq = N * N / DIM;
    phase_begin();
    uint64_t t0 = read_cycles();
    gemmini_config_st(DIM * sizeof(uint8_t));
    for (int r = 0; r < N * N / DIM; r += DIM)
      gemmini_extended_mvout(out_buf + r * DIM, sp_a + r, DIM, DIM);
    phase_end(p, t0);
  }

  for (int i = 0; i < n; i++)
    report(&ph[i]);

  // sanity: the warm A mvin landed in the spad and the mvout copied it back (first tile row)
  int errors = 0;
  for (int c = 0; c < DIM; c++)
    if (out_buf[c] != A_in[0][c]) errors++;
  printf("mx_mem_bw %s (%d mismatches in spot check)\n", errors ? "FAILED" : "PASSED", errors);
  return errors != 0;
}
#endif
