#=============================================================================
# variables
#=============================================================================
XLEN ?= 64

# Support Linux gcc from riscv-gnu-toolchain and from system packages
# riscv64-unknown-linux-gnu-gcc is built from riscv-gnu-toolchain, 
# comes with Firesim's tools. riscv64-linux-gnu-gcc comes from a system package
CC_LINUX_PRESENT	:= $(shell command -v \
												riscv$(XLEN)-unknown-linux-gnu-gcc 2> /dev/null)
CC_LINUX					:= $(if $(CC_LINUX_PRESENT),\
												riscv$(XLEN)-unknown-linux-gnu-gcc,\
												riscv$(XLEN)-linux-gnu-gcc)
CC_BAREMETAL			:= riscv$(XLEN)-unknown-elf-gcc
CC_PK				 			:= riscv$(XLEN)-unknown-elf-gcc

ENV_P := $(abs_top_srcdir)/riscv-tests/env/p
ENV_V := $(abs_top_srcdir)/riscv-tests/env/v

# must define list of tests to run
include $(TARGET_MAKEFILE)

tests_baremetal := $(tests:=-baremetal)
tests_pk				:= $(tests:=-pk)
tests_linux			:= $(tests:=-linux)

#=============================================================================
# high-level targets
#=============================================================================
.PHONY: all baremetal pk linux clean default

default: all

all: baremetal pk linux

baremetal: $(tests_baremetal)

pk: $(tests_pk)

linux: $(tests_linux)

clean:
	rm -rf $(tests_baremetal) $(tests_pk) $(tests_linux)

#=============================================================================
# gcc/g++ configuration flags 
#=============================================================================
BENCH_COMMON    := $(abs_top_srcdir)/riscv-tests/benchmarks/common
GEMMINI_HEADERS := $(wildcard $(abs_top_srcdir)/include/*.h)

ID_STRING ?=
LIBS ?=
LDFLAGS ?=

#----------------------------------------------------------------------------
CFLAGS_ALL := \
  $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-ffast-math \
	-march=rv64gc \
	-lm \
	-lgcc \
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-DID_STRING=$(ID_STRING) \

CFLAGS_BAREMETAL := \
	$(CFLAGS_ALL) \
	-DGEMMINI_BAREMETAL=1 \
	-O2 \
	-I$(BENCH_COMMON) \
	-fno-common \
	-fno-builtin-printf \
	-nostdlib \
	-nostartfiles \
	-T $(BENCH_COMMON)/test.ld \
	-static \

CFLAGS_PK := \
	$(CFLAGS_ALL) \
	-DGEMMINI_PK=1 \
	-O3 \
	-static \

CFLAGS_LINUX := \
	$(CFLAGS_ALL) \
	-DGEMMINI_LINUX=1 \
	-O3 \

#----------------------------------------------------------------------------
vpath %.c $(src_dir)

%-baremetal: %.c $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LDFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

%-pk: %.c $(GEMMINI_HEADERS)
	$(CC_PK) $(CFLAGS_PK) $< $(LDFLAGS) -o $@ $(LIBS)

%-linux: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS_LINUX) $< $(LDFLAGS) -o $@ $(LIBS)

