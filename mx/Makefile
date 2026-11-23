# MX tests: one-off sources in src/, template tests from tests.txt (see ./mxtests.py).
include $(abs_top_srcdir)/Makefrag

mx_tests = \
	matmul_tiled_fp4_64x64 \
	matmul_tiled_fp6_128x128 \
	matmul_tiled_fp6_128x128x512 \
	matmul_tiled_fp6_128x128x512_requant \
	matmul_tiled_fp6_64x64_chain \
	matmul_tiled_fp6_128x128_chain \
	matmul_tiled_fp6_128x128_chunk_timing \
	matmul_tiled_fp6_128x128_chunk4_timing \
	matmul_tiled_fp6_128x128_lut_blocked \
	matmul_tiled_fp6_128x128_lut_dramloop \
	matmul_tiled_fp4_128x128 \
	matmul_tiled_fp4_128x128x512 \
	matmul_tiled_fp8_64x64 \
	matmul_tiled_asym_e4m3s_e4m3_dramloop \
	matmul_tiled_fp8_64x64_DRAMMvout \
	matmul_tiled_fp8_128x128 \
	matmul_tiled_fp8_128x128_chunked \
	matmul_tiled_fp8_128x128_chunked4 \
	matmul_tiled_fp8_128x128_dramloop \
	matmul_tiled_fp8_64x64_dramloop_dim4 \
	matmul_tiled_fp8_64x64_dramloop_dim8 \
	matmul_tiled_fp8_64x64_dramloop_dim16 \
	matmul_tiled_fp8_64x64_dramloop_dim32 \
	matmul_tiled_fp8_64x96x64_dramloop_dim16 \
	matmul_tiled_fp8_64x96x64_dramloop_dim32 \
	matmul_tiled_fp8_64x96x64_dim32 \
	matmul_tiled_fp8_64x96x64_dim8 \
	matmul_tiled_fp8_64x96x64_dim4 \
	matmul_tiled_fp8_64x96x64_dramloop_dim8 \
	matmul_tiled_fp8_64x96x64_dramloop_dim4 \
	matmul_tiled_fp8_64x96x64_requant_dim32 \
	matmul_tiled_fp8_128x128_dramloop_nc \
	matmul_tiled_fp8_128x128_dramloop_nc4 \
	matmul_tiled_fp8_128x128_chunked_2d \
	matmul_tiled_fp8_128x128_dramloop_nc_2d \
	matmul_tiled_fp8_128x128_dramloop_nc_wait \
	matmul_tiled_fp8_128x128_dramloop_ls \
	matmul_tiled_fp8_128x128_dramloop_ls4 \
	matmul_tiled_fp8_128x128_dramloop_kt \
	matmul_tiled_fp8_128x128_dramloop_relu \
	mx_mem_bw \
	matmul_tiled_fp8_128x128x256 \
	matmul_tiled_fp8_128x128x256_DRAMMvout \
	matmul_tiled_fp8_64x64_requant \
	matmul_tiled_fp4_128x128x64_requant_dim32 \
	matmul_tiled_asym_fp4_fp6_64x64_dim32 \
	matmul_tiled_asym_fp6_fp4_64x64_dim32 \
	matmul_tiled_asym_fp4_fp6_128x128x256_dim32 \
	matmul_tiled_asym_fp4_fp6_64x64_dim8 \
	matmul_tiled_asym_fp6_fp4_64x64_dim8 \
	matmul_tiled_fp8_64x64_chain \
	matmul_tiled_fp8_128x128_chain \
	matmul_tiled_fp8_128x128_requant \
	matmul_tiled_fp8_64x96x64 \
	matmul_tiled_fp8_96x32x32 \
	matmul_tiled_fp8_96x96x64 \
	matmul_tiled_fp8_32x32x32 \
	matmul_tiled_fp4_64x64_requant \
	matmul_tiled_fp4_128x128_requant \
	matmul_tiled_fp4_128x128x512_requant \
	matmul_tiled_asym_fp4_fp6_64x64 \
	matmul_tiled_asym_fp6_fp4_64x64 \

.DEFAULT_GOAL := all
mx_tests.mk: $(src_dir)/tests.txt $(src_dir)/mxtests.py
	python3 $(src_dir)/mxtests.py mk > $@
include mx_tests.mk
mx_tests += $(mx_default_tests)

# MX tests need a header generated with use_mx_scaling=true (#define HAS_MX_SCALING).
HAS_MX_SCALING := $(shell grep -q 'define HAS_MX_SCALING' $(abs_top_srcdir)/include/gemmini_params.h 2>/dev/null && echo yes)
ifeq ($(HAS_MX_SCALING),yes)
tests = $(mx_tests)
endif

include $(abs_top_srcdir)/common.mk

CFLAGS_BAREMETAL += -I$(src_dir)
CFLAGS += -I$(src_dir)
vpath %.c $(src_dir)/src

MX_TEMPLATE_DEPS = $(src_dir)/include/mx_test_common.h $(wildcard $(src_dir)/templates/*.c)

# Per-test extras: its params header (params=) and its generated data set (data lines).
mx_extra_deps = $(if $(MX_PARAMS_$(1)),params/$(MX_PARAMS_$(1))/include/gemmini_params.h) \
                $(if $(MX_BLOB_$(1)),data/$(MX_BLOB_$(1)).h data/$(MX_BLOB_$(1))_blob.o)
mx_extra_flags = $(if $(MX_PARAMS_$(1)),-iquote params/$(MX_PARAMS_$(1))) $(if $(MX_BLOB_$(1)),-Idata)

.SECONDEXPANSION:
$(addsuffix -baremetal,$(mx_default_tests) $(mx_mxk_tests)): %-baremetal: $(GEMMINI_HEADERS) $(MX_TEMPLATE_DEPS) \
    $$(src_dir)/$$(MX_SRC_$$*) $$(call mx_extra_deps,$$*)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $(call mx_extra_flags,$*) $(MX_DEFS_$*) $(src_dir)/$(MX_SRC_$*) $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(if $(MX_BLOB_$*),data/$(MX_BLOB_$*)_blob.o) $(LIBS)

# Kernel tests (lutk, mxk_*): not in `all`; build with `make mxk`.
.PHONY: mxk
mxk: $(mx_mxk_tests:=-baremetal)

params/%/include/gemmini_params.h: $(abs_top_srcdir)/include/gemmini_params_%.h
	mkdir -p $(dir $@) && cp $< $@

OBJCOPY ?= riscv$(XLEN)-unknown-elf-objcopy
data/%.bin data/%.h: $(wildcard $(src_dir)/golden/*.py)
	mkdir -p data && cd $(src_dir)/golden && MX_DATA_DIR=$(CURDIR)/data python3 $(MX_GEN_$*)

.PRECIOUS: data/%.bin data/%.h
data/%_blob.o: data/%.bin
	cd data && $(OBJCOPY) -I binary -O elf64-littleriscv -B riscv --set-section-alignment .data=64 $*.bin $*_blob.o
