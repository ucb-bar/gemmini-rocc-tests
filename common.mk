# Build flags and rules shared by bareMetalC/ and mx/. Each sets $(tests) first, then includes this.

tests_baremetal = $(tests:=-baremetal)

ifeq ($(findstring spike,$(RUNNER)),spike)
CFLAGS := $(CFLAGS) -DSPIKE_SIM
# Currently don't support conv or conv-with-pool on spike
runs_baremetal = $(addsuffix .run,$(filter-out conv-baremetal conv_with_pool-baremetal,$(tests_baremetal)))
else
# Don't run very long benchmarks for RTL sim
runs_baremetal = $(addsuffix .run,$(filter-out tiled_matmul_cpu-baremetal tiled_matmul_option-baremetal,$(tests_baremetal)))
endif
# Nornaml configs dont have ext spad
runs_baremetal := $(filter-out matmul_spad-baremetal.run mvin_mvout_spad-baremetal.run,$(runs_baremetal))

ifdef BAREMETAL_ONLY
	tests_linux =
	tests_pk =
else
	tests_linux = $(tests:=-linux)
	tests_pk = $(tests:=-pk)
endif

BENCH_COMMON = $(abs_top_srcdir)/riscv-tests/benchmarks/common
# The matmul_*.h data headers are generated (gen_matmul_llama.py), so they must be prerequisites
# too. Without them a regenerated golden leaves the ELF untouched and the test silently re-runs the
# OLD data -- which still PASSES, because the stale header is self-consistent.
GEMMINI_HEADERS = $(abs_top_srcdir)/include/gemmini.h $(abs_top_srcdir)/include/gemmini_params.h $(abs_top_srcdir)/include/gemmini_testutils.h $(wildcard $(abs_top_srcdir)/include/matmul_*.h)



CFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-fno-tree-loop-distribute-patterns \
	-march=rv64gc -Wa,-march=rv64gc \
	-lm \
	-lgcc \
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(BENCH_COMMON) \
	-DID_STRING=$(ID_STRING) \
	-DPRINT_TILE=0 \

# Extra flags appended without clobbering CFLAGS (e.g. -DGEMMINI_MMIO).
CFLAGS := $(CFLAGS) $(EXTRA_CFLAGS)

CFLAGS_PK := \
	$(CFLAGS) \
	-static \
	-DBAREMETAL=1 \

# Appended AFTER the sources on the link line, where a library has to be to resolve anything.
LIBS := -lm

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-nostdlib \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1 \

all: $(tests_baremetal) $(tests_linux) $(tests_pk)

vpath %.c $(src_dir)

%-baremetal: %.c $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

%-linux: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS) $< $(LFLAGS) -o $@

%-pk: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS_PK) $< $(LFLAGS) -o $@

run-baremetal: $(runs_baremetal)

%-baremetal.run: %-baremetal
	$(RUNNER)$(abs_top_srcdir)/build/$(notdir $(src_dir))/$^

junk += $(tests_baremetal) $(tests_linux) $(tests_pk)

