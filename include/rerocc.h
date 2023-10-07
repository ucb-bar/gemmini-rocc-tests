#ifndef REROCC_H
#define REROCC_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "rocc.h"

#define read_csr(reg) ({ unsigned long __tmp; \
  asm volatile ("csrr %0, " #reg : "=r"(__tmp)); \
  __tmp; })

#define write_csr(reg, val) ({ \
  asm volatile ("csrw " #reg ", %0" :: "rK"(val)); })

#define swap_csr(reg, val) ({ unsigned long __tmp; \
  asm volatile ("csrrw %0, " #reg ", %1" : "=r"(__tmp) : "rK"(val)); \
  __tmp; })

#define set_csr(reg, bit) ({ unsigned long __tmp; \
  asm volatile ("csrrs %0, " #reg ", %1" : "=r"(__tmp) : "rK"(bit)); \
  __tmp; })

#define clear_csr(reg, bit) ({ unsigned long __tmp; \
  asm volatile ("csrrc %0, " #reg ", %1" : "=r"(__tmp) : "rK"(bit)); \
  __tmp; })

#define RR_CFG_ACQ_MASK 0x100
#define RR_CFG_MGR_MASK 0x0ff

#define CSR_RROPC0 0x800
#define CSR_RROPC1 0x801
#define CSR_RROPC2 0x802
#define CSR_RROPC3 0x803
#define CSR_RRBAR 0x804

#define CSR_RRCFG0 0x810
#define CSR_RRCFG1 0x811
#define CSR_RRCFG2 0x812
#define CSR_RRCFG3 0x813
#define CSR_RRCFG4 0x814
#define CSR_RRCFG5 0x815
#define CSR_RRCFG6 0x816
#define CSR_RRCFG7 0x817
#define CSR_RRCFG8 0x818
#define CSR_RRCFG9 0x819
#define CSR_RRCFG10 0x81a
#define CSR_RRCFG11 0x81b
#define CSR_RRCFG12 0x81c
#define CSR_RRCFG13 0x81d
#define CSR_RRCFG14 0x81e
#define CSR_RRCFG15 0x81f

#define LIST_OF_RR_CSRS \
  F(CSR_RRBAR) \
  F(CSR_RROPC0) \
  F(CSR_RROPC1) \
  F(CSR_RROPC2) \
  F(CSR_RROPC3) \
  F(CSR_RRCFG0) \
  F(CSR_RRCFG1) \
  F(CSR_RRCFG2) \
  F(CSR_RRCFG3) \
  F(CSR_RRCFG4) \
  F(CSR_RRCFG5) \
  F(CSR_RRCFG6) \
  F(CSR_RRCFG7) \
  F(CSR_RRCFG8) \
  F(CSR_RRCFG9) \
  F(CSR_RRCFG10) \
  F(CSR_RRCFG11) \
  F(CSR_RRCFG12) \
  F(CSR_RRCFG13) \
  F(CSR_RRCFG14) \
  F(CSR_RRCFG15)

static inline uint64_t swap_rr_csr(uint64_t cfgId, uint64_t wdata) {
  uint64_t ret = 0;
  switch (cfgId) {
#define F(c) case c: { ret = swap_csr(c, wdata); break; }
    LIST_OF_RR_CSRS
#undef F
  default: { printf("swap_rr_csr illegal csr %lx\n", cfgId); abort(); }
  }
  return ret;
}

static inline uint64_t read_rr_csr(uint64_t cfgId) {
  uint64_t ret = 0;
  switch (cfgId) {
#define F(c) case c: { ret = read_csr(c); break; }
    LIST_OF_RR_CSRS
#undef F
  default: { printf("read_rr_csr illegal csr %lx\n", cfgId); abort(); }
  }
  return ret;
}

static inline void write_rr_csr(uint64_t cfgId, uint64_t wdata) {
  swap_rr_csr(cfgId, wdata);
}

static bool rr_acquire_single(uint32_t cfgId, uint64_t accelId) {
  uint32_t csrid = CSR_RRCFG0 + cfgId;
  uint64_t w = RR_CFG_ACQ_MASK | (accelId & RR_CFG_MGR_MASK);
  write_rr_csr(csrid, w);
  return (read_rr_csr(csrid) & RR_CFG_ACQ_MASK) != 0;
}

static void rr_release(uint32_t cfgId) {
  uint32_t csrid = CSR_RRCFG0 + cfgId;
  uint64_t w = 0;
  write_rr_csr(csrid, w);
}

static bool rr_acquire_multi(uint32_t cfgId, uint64_t* accelIds, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (rr_acquire_single(cfgId, accelIds[i])) return true;
  return false;
}

static void rr_set_opc(uint8_t opc, uint32_t cfgId) {
  write_rr_csr(CSR_RROPC0 + opc, cfgId);
}

static void rr_fence(uint32_t cfgId) {
  write_rr_csr(CSR_RRBAR, cfgId);
  asm volatile("fence");
}

#endif
