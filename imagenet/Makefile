include $(abs_top_srcdir)/Makefrag

tests = \
	priority_mp8_dynamic_h \
	priority_mp8_static_h \
	priority_mp8_dynamica_h \
	priority_mp8_statica_h \
	priority_mp8_dynamicb_h \
	priority_mp8_staticb_h \
	priority_mp8_dynamic_l \
	priority_mp8_static_l \
	priority_mp8_dynamica_l \
	priority_mp8_statica_l \
	priority_mp8_dynamicb_l \
	priority_mp8_staticb_l \
	priority_mp8_dynamic \
	priority_mp8_static \
	priority_mp8_dynamica \
	priority_mp8_statica \
	priority_mp8_dynamicb \
	priority_mp8_staticb \
	motivation_4 \
	motivation_3 \
	motivation_2 \
	motivation_1 \
	#yolonet_batch1_8 \
	#kwsnet_batch1_8 \
	#resnet_batch1_8 \
	#alexnet_batch1_8 \
	#googlenet_batch1_8 \
	#yololitenet_batch1_8 \
	#squeezenet_batch1_8 \
	#yolonet_batch1_4 \
	#kwsnet_batch1_4 \
	#resnet_batch1_4 \
	#alexnet_batch1_4 \
	#googlenet_batch1_4 \
	#yololitenet_batch1_4 \
	#squeezenet_batch1_4 \
	#kwsnet_batch1_2 \
	#yolonet_batch1_2 \
	#yololitenet_batch1_2 \
	#resnet_batch1_2 \
	#alexnet_batch1_2 \
	#googlenet_batch1_2 \
	#squeezenet_batch1_2 \
	#kwsnet_batch1_1 \
	#yolonet_batch1_1 \
	#yololitenet_batch1_1 \
	#resnet_batch1_1 \
	#alexnet_batch1_1 \
	#googlenet_batch1_1 \
	#squeezenet_batch1_1 \

tests_baremetal = $(tests:=-baremetal)
runs_baremetal = $(addsuffix .run,$(tests_baremetal))
ifdef BAREMETAL_ONLY
	tests_linux =
else
	tests_linux = $(tests:=-linux)
endif

BENCH_COMMON = $(abs_top_srcdir)/riscv-tests/benchmarks/common
GEMMINI_HEADERS = $(abs_top_srcdir)/include/gemmini.h $(abs_top_srcdir)/include/gemmini_params.h $(abs_top_srcdir)/include/gemmini_nn.h $(abs_top_srcdir)/include/gemmini_testutils.h

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
	-lpthread \
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(BENCH_COMMON) \
	-DID_STRING=$(ID_STRING) \
	-Wno-incompatible-pointer-types

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-nostdlib \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1 \

all: $(tests_linux)

vpath %.c $(src_dir)

%-baremetal: %.c $(src_dir)/images.h $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

%-linux: %.c $(src_dir)/images.h $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS) $< $(LFLAGS) -o $@

run-baremetal: $(runs_baremetal)

%-baremetal.run: %-baremetal
	$(RUNNER)$^

junk += $(tests_baremetal) $(tests_linux)

