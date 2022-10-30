//#define CACHE_SIZE (1e6/DIM)
#define DRAM_MAX_UTIL 150
//#define DRAM_BW 0.5
#ifndef NUM_CORE
#define NUM_CORE 8 // for 8 cores
#endif
//#define NUM_CORE 8

#ifndef NUM_GROUP
#define NUM_GROUP 2
#endif
#ifndef NUM_CORE
#define NUM_CORE 8
#endif

#ifndef WORKLOAD_CORE
#define WORKLOAD_CORE 2
#endif

#ifndef SUB_CORE
#define SUB_CORE 4 // 8 -> 4 + 4
#endif

#define SUB_GROUP 2 // 4 -> 2 + 2
#define NUM_SUB_GROUP 4 // total 4 sub groups

// we use 8 cores
#if NUM_CORE == 8
#define DRAM_BW 1
#define CACHE_SIZE (2e6/DIM)
#else
#define DRAM_BW 0.5
#define CACHE_SIZE (1e6/DIM)
#endif

#define MAX_WORKLOAD 600
//#define NUM_WORKLOAD 8//(8*3) // 1, 2, 4 batches

#ifndef total_workloads
#define total_workloads 200
#define QUEUE_DEPTH 10
#endif


// dram_bw -1: disable bandwidth modulation (window, target load to 0)
// dram_bw 0: monitor gemmini_dram_util and priority score 
// dram_bw 0-100: use dram_bw given to compute window, target load 
static int gemmini_dram_util[NUM_SUB_GROUP] = {0};
//static int gemmini_dram_util[NUM_GROUP][SUB_GROUP] = {0}; // only the cid == 0 updates it
static int gemmini_score[NUM_SUB_GROUP] = {0}; // priority score scaled to 100 (for bw division when it gets over the limit)
static uint64_t total_queue_togo[MAX_WORKLOAD] = {0};
static uint64_t total_queue_conv[MAX_WORKLOAD] = {0};
static int gemmini_queue_id[NUM_CORE] = {0};
static int64_t gemmini_start_time[NUM_CORE] = {0}; // inner_start - temp_cycles

static int total_queue_type[MAX_WORKLOAD] = {-1};
static uint64_t total_queue_dispatch[MAX_WORKLOAD] = {0}; // dispatched time (in order)
static uint64_t total_queue_finish[SUB_CORE][MAX_WORKLOAD] = {0};
static int total_queue_status[MAX_WORKLOAD] = {-1}; // -1: not assigned, 0: in assigned queue, >= 1: part
static int total_queue_priority[MAX_WORKLOAD] = {-1}; // 0 - 11
static uint64_t total_queue_target[MAX_WORKLOAD] = {0};
static uint64_t total_queue_runtime_thread[SUB_CORE][MAX_WORKLOAD] = {0}; // for checking purpose (end to end runtime)
static uint64_t total_queue_runtime_total[SUB_CORE][MAX_WORKLOAD] = {0}; // for checking purpose (end to end runtime)

#define MAX_ITER (int)(total_workloads / QUEUE_DEPTH)
static int gemmini_workload_assigned[NUM_GROUP][SUB_GROUP][MAX_ITER][QUEUE_DEPTH] = {-1};
static uint64_t gemmini_runtime[NUM_CORE] = {0}; // to track real runtime without thread create overhead

static int gemmini_workload_grouped[NUM_GROUP][SUB_GROUP][MAX_ITER][QUEUE_DEPTH] = {-1};
//static bool gemmini_done[NUM_GROUP][SUB_GROUP] = {0};
static bool gemmini_done[NUM_SUB_GROUP] = {0};
static bool gemmini_terminate[NUM_SUB_GROUP] = {0};
static bool gemmini_terminate_receive[NUM_SUB_GROUP] = {0};
static uint64_t global_time = {0};

// dram_bw -1: disable bandwidth modulation (window, target load to 0)
// dram_bw 0: monitor gemmini_bw and priority score 
// dram_bw 0-100: use dram_bw given to compute window, target load 
//static int gemmini_bw[NUM_GROUP] = {0}; // only the cid == 0 updates it
//static int gemmini_score[NUM_GROUP] = {0}; // priority score scaled to 100 (for bw division when it gets over the limit)


#define MAX(X, Y) (X > Y ? X : Y)
#define MIN(X, Y) (X < Y ? X : Y)

static uint64_t read_cycles() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;

    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbff8);
    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbffc);
    // return *mtime;
}
