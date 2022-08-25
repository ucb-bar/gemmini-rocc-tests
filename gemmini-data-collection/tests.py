from collections import namedtuple

GemminiTest = namedtuple("GemminiTest", "keywords replacement template_file new_file")

# MODIFY LIST BELOW TO SPECIFY TESTS TO RUN
# arguments: array of keywords, array of values for keywords, name of C file in the templates folder, name of output C file to be placed in bareMetalC

tests = [
    GemminiTest(["DIM_I", "DIM_J", "DIM_K"], ["128", "256", "64"], "matmul_template", "tiled_matmul_ws_perf-128_256_64"),
    GemminiTest(["DIM_I", "DIM_J", "DIM_K"], ["64", "32", "128"], "matmul_template", "tiled_matmul_ws_perf-64_32_128"),

    GemminiTest(["IN_DIM", "IN_CHANNELS", "OUT_CHANNELS", "KERNEL_DIM", "STRIDE", "PADDING"], ["224", "3", "64", "7", "2", "3"], "conv_template", "conv-perf_224-3-64-7-2-3"),
]

