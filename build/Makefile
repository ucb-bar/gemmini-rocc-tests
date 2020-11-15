prefix		:= /usr/local
abs_top_srcdir	:= /scratch/sakim/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/..
XLEN            := 64
RISCVTOOLS      := @RISCVTOOLS@
ROCC = examples

.PHONY: all bareMetalC clean imagenet mlps
all: bareMetalC imagenet mlps

vars = \
	abs_top_srcdir=$(abs_top_srcdir) \
	XLEN=$(XLEN) \
	PREFIX=$(ROCC)-$@ \
	src_dir=$(abs_top_srcdir)/$@ \
	RISCVTOOLS=$(RISCVTOOLS)

bareMetalC:
	mkdir -p $@
	$(MAKE) -C $@ -f $(abs_top_srcdir)/$@/Makefile $(vars)

imagenet:
	mkdir -p $@
	$(MAKE) -C $@ -f $(abs_top_srcdir)/$@/Makefile $(vars)

mlps:
	mkdir -p $@
	$(MAKE) -C $@ -f $(abs_top_srcdir)/$@/Makefile $(vars)

clean:
	$(MAKE) -C bareMetalC -f $(abs_top_srcdir)/bareMetalC/Makefile abs_top_srcdir=$(abs_top_srcdir) PREFIX=$(ROCC)-bareMetalC clean
	$(MAKE) -C imagenet -f $(abs_top_srcdir)/imagenet/Makefile abs_top_srcdir=$(abs_top_srcdir) PREFIX=$(ROCC)-imagenet clean
	$(MAKE) -C mlps -f $(abs_top_srcdir)/mlps/Makefile abs_top_srcdir=$(abs_top_srcdir) PREFIX=$(ROCC)-mlps clean

