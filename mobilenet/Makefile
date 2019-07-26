include $(abs_top_srcdir)/Makefrag

ifdef BAREMETAL_ONLY
	tests =
else
	tests = \
		test \
		mobilenet32
endif

BENCH_COMMON = $(RISCVTOOLS)/riscv-tests/benchmarks/common

CFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-march=rv64gc -Wa,-march=rv64gcxhwacha \
	-lm \
	-lgcc \
	-I$(RISCVTOOLS) \
	-I$(RISCVTOOLS)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(BENCH_COMMON) \
	-DID_STRING=$(ID_STRING) \

	#-nostdlib \
	#-nostartfiles \
	#-static \
	#-T $(BENCH_COMMON)/test.ld \

all: $(tests)

vpath %.c $(src_dir)

%: %.c
	$(CC_LINUX) $(CFLAGS) $< $(LFLAGS) -o $@
	# $(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

junk += $(tests)
