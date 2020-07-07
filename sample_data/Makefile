include $(abs_top_srcdir)/Makefrag

tests = \
	nips \
	email-Eu-core \
	kos \
	socfb-Amherst41\
	GL6 \
	G17 \
	G23 \
	collegemsg \
	socfb-American75 \
	socfb-Brown11 \
	CL-trial3 \
	socfb-cal65 \
	mooc_actions \
	musae_DE_edges \
	musae_ES_edges \
	musae_RU_edges \
	soc-sign-bitcoin-otc \
	musae_ENGB_edges \
	bio-CE \
	web-EPA \
	ia-escorts \
	tech-pgp \
	ca-Erdos \
	poli3 \
	scc_rt_lolgop \
	scc_rt_gmanews \


tests_baremetal = $(tests:=-baremetal)
ifdef BAREMETAL_ONLY
	tests_linux =
else
	tests_linux = $(tests:=-linux)
endif

BENCH_COMMON = $(abs_top_srcdir)/riscv-tests/benchmarks/common
GEMMINI_HEADERS = $(abs_top_srcdir)/include/gemmini.h $(abs_top_srcdir)/include/gemmini_params.h $(abs_top_srcdir)/include/gemmini_testutils.h

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

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-nostdlib \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1 \

all: $(tests_baremetal) $(tests_linux)

vpath %.c $(src_dir)
vpath %.h $(src_dir)

%-baremetal: %.c %.h $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

%-linux: %.c %.h $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS) $< $(LFLAGS) -o $@

junk += $(tests_baremetal) $(tests_linux)

