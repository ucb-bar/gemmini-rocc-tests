include $(abs_top_srcdir)/Makefrag

tests = \
	mvin_mvout \
	mvin_mvout_stride \
	mvin_mvout_acc \
	mvin_mvout_acc_stride \
	matmul_os \
	matmul_ws \
	matmul \
	tiled_matmul_os \
	raw_hazard

BENCH_COMMON = $(RISCVTOOLS)/riscv-tests/benchmarks/common

CFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-static \
	-std=gnu99 \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-march=rv64gc -Wa,-march=rv64gcxhwacha \
	-nostdlib \
	-nostartfiles \
	-lm \
	-lgcc \
	-T $(BENCH_COMMON)/test.ld \
	-I$(RISCVTOOLS) \
	-I$(RISCVTOOLS)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(BENCH_COMMON) \
	-DID_STRING=$(ID_STRING) \

all: $(tests)

vpath %.c $(src_dir)

%: %.c
	$(CC) $(CFLAGS) $< $(LFLAGS) -o $@ $(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

junk += $(tests)
