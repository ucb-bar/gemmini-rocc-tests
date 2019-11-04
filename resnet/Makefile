include $(abs_top_srcdir)/Makefrag

ifdef BAREMETAL_ONLY
	tests =
else
	tests = \
		resnet50-16 \
		resnet50-32 \
		resnet152-16 \
		resnet152-32
endif

BENCH_COMMON = $(abs_top_srcdir)/riscv-tests/benchmarks/common

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
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(BENCH_COMMON) \
	-DID_STRING=$(ID_STRING) \

all: $(tests)

vpath %.c $(src_dir)

%: %.c
	$(CC_LINUX) $(CFLAGS) $< $(LFLAGS) -o $@

junk += $(tests)
