include $(abs_top_srcdir)/Makefrag

tests = \
	accumulator \
	gemmini \
	mvin_mvout

CFLAGS := $(CFLAGS) \
	-Wall \
	-static \
	-I$(abs_top_srcdir) \
	-lm \
	-O2 \

tests_pk = $(addprefix $(PREFIX)-, $(tests))

vpath %.c $(src_dir)

all: $(tests_pk)

$(PREFIX)-%: %.c
	$(CC_BAREMETAL) $(CFLAGS) $< -o $@ $(LFLAGS)

junk += $(tests_pk)
