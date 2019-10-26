include $(abs_top_srcdir)/Makefrag

tests = \
	hello

tests_baremetal = $(tests:=-baremetal)
ifdef BAREMETAL_ONLY
	tests_linux =
else
	tests_linux = $(tests:=-linux)
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

CPPFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu++98 \
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

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-nostdlib \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1 \

CPPFLAGS_BAREMETAL := \
	$(CPPFLAGS) \
	-nostdlib \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1 \

all: $(tests_baremetal) $(tests_linux)

vpath %.cpp $(src_dir)

syscalls.o crt.o: 
	$(CC_BAREMETAL) -c $(CFLAGS_BAREMETAL) $(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S)

%-baremetal: %.cpp syscalls.o crt.o
	$(CPP_BAREMETAL) -c $(CPPFLAGS_BAREMETAL) $< $(LFLAGS) -o $@.o
	$(CPP_BAREMETAL) $(CPPFLAGS_BAREMETAL) $@.o syscalls.o crt.o $(LFLAGS) -o $@

%-linux: %.cpp
	$(CPP_LINUX) $(CPPFLAGS) $< $(LFLAGS) -o $@

junk += $(tests_baremetal) $(tests_linux) $(addsuffix .o, $(tests_baremetal)) syscalls.o crt.o

