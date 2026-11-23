# MX tests: all from tests.txt (see ./mxtests.py).
include $(abs_top_srcdir)/Makefrag

.DEFAULT_GOAL := all
# Re-made when a config's precision or a committed data header changes (decides which headers to regenerate).
mx_tests.mk: $(src_dir)/tests.txt $(src_dir)/mxtests.py $(wildcard $(src_dir)/src/*.c) \
             $(wildcard $(abs_top_srcdir)/include/gemmini_params_*.h $(abs_top_srcdir)/include/matmul_*.h)
	python3 $(src_dir)/mxtests.py mk > $@
include mx_tests.mk
mx_tests = $(mx_default_tests)

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

# Generated data (mx/golden) lives in $(MXDATA), shared by all build dirs: data headers at
# <config>/include/<name>.h, kernel data sets at <name>.bin/.h.
MXDATA = $(src_dir)/data

# Per-test extras: its params header (params=), its data headers (found as "include/<name>.h" via -iquote) and
# its data set (data lines).
mx_extra_deps = $(if $(MX_PARAMS_$(1)),params/$(MX_PARAMS_$(1))/include/gemmini_params.h) \
                $(addprefix $(MXDATA)/,$(addsuffix .h,$(MX_HDRS_$(1)))) \
                $(if $(MX_BLOB_$(1)),$(MXDATA)/$(MX_BLOB_$(1)).h data/$(MX_BLOB_$(1))_blob.o)
mx_extra_flags = $(if $(MX_PARAMS_$(1)),-iquote params/$(MX_PARAMS_$(1))) \
                 $(foreach h,$(MX_HDRS_$(1)),-iquote $(MXDATA)/$(firstword $(subst /, ,$(h)))) \
                 $(if $(MX_BLOB_$(1)),-I$(MXDATA))

.SECONDEXPANSION:
# mx_tests.mk: a test re-links when it switches between a committed and a regenerated data header.
$(addsuffix -baremetal,$(mx_default_tests) $(mx_mxk_tests)): %-baremetal: $(GEMMINI_HEADERS) $(MX_TEMPLATE_DEPS) \
    mx_tests.mk $$(src_dir)/$$(MX_SRC_$$*) $$(call mx_extra_deps,$$*)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $(call mx_extra_flags,$*) $(MX_DEFS_$*) $(src_dir)/$(MX_SRC_$*) $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(if $(MX_BLOB_$*),data/$(MX_BLOB_$*)_blob.o) $(LIBS)

# Kernel tests (lutk, mxk_*): not in `all`; build with `make mxk`.
.PHONY: mxk
mxk: $(mx_mxk_tests:=-baremetal)

params/%/include/gemmini_params.h: $(abs_top_srcdir)/include/gemmini_params_%.h
	mkdir -p $(dir $@) && cp $< $@

# A data header whose committed precision differs from its config's is regenerated (mx/golden, --params).
cfg_h = $(foreach p,$(1),$(abs_top_srcdir)/include/gemmini_params_$(p).h)
$(addprefix $(MXDATA)/,$(addsuffix .h,$(mx_hdrs))): $(MXDATA)/%.h: $$(call cfg_h,$$(MX_HPREC_$$*))
	cd $(src_dir) && python3 mxtests.py data $(notdir $*) --params $(MX_HPREC_$*)

# Kernel data sets: generated once (one torch thread each), again only when their config header changes.
GOLDEN_ENV = OMP_NUM_THREADS=1 MX_DATA_DIR=$$T
OBJCOPY ?= riscv$(XLEN)-unknown-elf-objcopy
$(MXDATA)/%.bin $(MXDATA)/%.h: $$(call cfg_h,$$(MX_BPREC_$$*))
	T=$(MXDATA)/$*.$$$$.d; rm -rf $$T && mkdir -p $$T && cd $(src_dir)/golden && \
	  env $(GOLDEN_ENV) python3 $(MX_GEN_$*) $(if $(MX_BPREC_$*),--params $(call cfg_h,$(MX_BPREC_$*))) > $$T/log 2>&1 \
	    || { tail -20 $$T/log; exit 1; }; \
	  mv $$T/$*.bin $$T/$*.h $(MXDATA)/ && rm -rf $$T

.PRECIOUS: $(MXDATA)/%.bin $(MXDATA)/%.h
data/%_blob.o: $(MXDATA)/%.bin
	mkdir -p data && cd $(MXDATA) && $(OBJCOPY) -I binary -O elf64-littleriscv -B riscv --set-section-alignment .data=64 $*.bin $(CURDIR)/data/$*_blob.o
