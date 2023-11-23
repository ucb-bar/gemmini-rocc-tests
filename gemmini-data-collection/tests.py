from collections import namedtuple
import pickle

GemminiTest = namedtuple("GemminiTest", "keywords replacement template_file new_file")

# MODIFY LIST BELOW TO SPECIFY TESTS TO RUN
# arguments: array of keywords, array of values for keywords, name of C file in the templates folder, name of output C file to be placed in bareMetalC

with open("layers/layers.pickle", "rb") as p:
    layers = pickle.load(p)

tests = []

for layer in layers:
    layer_name = layer["prob_name"] 
    del layer["prob_name"]
    if "I" in layer: # matmul
        matmul_layer_keys = ["I", "K", "J", "TILE_OCOLS", "TILE_KCHS", "TILE_OCHS", "SPATIAL_TILE_KCHS", "SPATIAL_TILE_OCHS", "PERM_STR"]
        matmul_template_keys = ["DIM_I", "DIM_K", "DIM_J", "TILE_OCOLS", "TILE_KCHS", "TILE_OCHS", "SPATIAL_TILE_KCHS", "SPATIAL_TILE_OCHS", "PERM_STR"]
        tests.append(GemminiTest(matmul_template_keys, [layer[k] for k in matmul_layer_keys], "matmul_template_map", layer_name))
    else: # conv
        vals = []
        for val in layer.values():
            vals.append(str(val))
        tests.append(GemminiTest(list(layer.keys()), vals, "conv_template_map", layer_name)) 

"""
tests = [
    #GemminiTest(["DIM_I", "DIM_J", "DIM_K"], ["6272", "64", "64"], "matmul_template", "tiled_matmul_ws_perf-128_256_64"),
    #GemminiTest(["DIM_I", "DIM_J", "DIM_K"], ["64", "32", "128"], "matmul_template", "tiled_matmul_ws_perf-64_32_128"),
    #GemminiTest(["DIM_I", "DIM_J", "DIM_K", "TILE_DIM_I", "TILE_DIM_J", "TILE_DIM_K"], ["64", "32", "128", "1", "1", "2"], "matmul_template_map", "tiled_matmul_ws_perf-64_32_128"),
    GemminiTest(["IN_DIM", "IN_CHANNELS", "OUT_CHANNELS", "KERNEL_DIM", "STRIDE", "PADDING", "TILE_BATCHES", "TILE_OCOLS", "TILE_OROWS", "TILE_OCHS", "TILE_KCOLS", "TILE_KROWS", "TILE_KCHS"], ["224", "3", "64", "7", "2", "3", "2", "4", "4", "4", "4", "4", "3"], "conv_template_map", "conv-perf_224-3-64-7-2-3"),
    #GemminiTest(["IN_DIM", "IN_CHANNELS", "OUT_CHANNELS", "KERNEL_DIM", "STRIDE", "PADDING"], ["224", "3", "64", "7", "2", "3"], "conv_template", "conv-perf_224-3-64-7-2-3"),
]
"""
