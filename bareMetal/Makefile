include $(abs_top_srcdir)/Makefrag

tests = \
	accumulator \
	systolic_toggle_mode

tests_p = $(addprefix $(PREFIX)-p-, $(tests))
tests_v = $(addprefix $(PREFIX)-v-, $(tests))

CFLAGS := $(CFLAGS) \
	-march=rv64gc -Wa,-march=rv64gcxhwacha \
	-mcmodel=medany \
	-static \
	-fvisibility=hidden \
	-nostdlib \
	-nostartfiles \
	-I$(abs_top_srcdir) \
	-I$(RISCVTOOLS) \
	-DID_STRING=$(ID_STRING) \

all: $(tests_p) $(tests_v)

vpath %.S $(src_dir)

$(PREFIX)-p-%: %.S
	$(CC_BAREMETAL) $(CFLAGS) -I$(ENV_P) -T$(ENV_P)/link.ld $< $(LFLAGS) -o $@ $(LIBS)

$(PREFIX)-v-%: %.S
	$(CC_BAREMETAL) $(CFLAGS) -I$(ENV_V) -T$(ENV_V)/link.ld $< $(LFLAGS) -o $@ $(LIBS)

junk += $(tests_p) $(tests_v)
