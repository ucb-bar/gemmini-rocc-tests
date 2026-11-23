// -DGEMMINI_MMIO: issue Gemmini commands through Radiance's MMIO registers instead of RoCC.
// Register map: cyclotron libgemmini/gemmini_radiance.h. Scales, LUTs and requant still go through Gemmini's own DMAs,
// so only the command transport, the fence and perf counters change. Included by gemmini.h.
#ifndef GEMMINI_MMIO_H
#define GEMMINI_MMIO_H

#include <stdint.h>

#ifndef GEMMINI_MMIO_BASE
#define GEMMINI_MMIO_BASE 0x84000UL   // the cluster's Gemmini block (GPU-local)
#endif
#define GEMMINI_MMIO_INST  (GEMMINI_MMIO_BASE + 0x00)   // W: RoCC instruction word, issues {funct, RS1, RS2}
#define GEMMINI_MMIO_READY (GEMMINI_MMIO_BASE + 0x08)   // R: 1 when an INST store would be taken now
#define GEMMINI_MMIO_RS1   (GEMMINI_MMIO_BASE + 0x10)
#define GEMMINI_MMIO_RS2   (GEMMINI_MMIO_BASE + 0x18)
#define GEMMINI_MMIO_BUSY  (GEMMINI_MMIO_BASE + 0x20)   // R: commands, loops, DMAs or loaders still active

#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_MMIO_RS1) = (uint64_t)(rs1); \
    *((volatile uint64_t *) GEMMINI_MMIO_RS2) = (uint64_t)(rs2); \
    while (!*((volatile uint32_t *) GEMMINI_MMIO_READY)) ; \
    *((volatile uint32_t *) GEMMINI_MMIO_INST) = 0x7B | (3 << 12) | (1 << 15) | (2 << 20) | ((uint32_t)(funct) << 25); \
  }

#undef gemmini_fence
#define gemmini_fence() { while (*((volatile uint32_t *) GEMMINI_MMIO_BUSY)) ; }

// No response register in the map: perf counter ops are no-ops and read 0.
#undef gemmini_counter_access
#define gemmini_counter_access(rd, config_reg) { (rd) = 0; (void) (config_reg); }

#endif
