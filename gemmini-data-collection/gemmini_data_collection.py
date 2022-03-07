#TODO: GENERALIZE PATHS

import sys

def main(keywords, replacement, template_file, new_file):
    """
    Search TEMPLATE_FILE for KEYWORDS and replace respective keywords with REPLACEMENT. Write changes to NEW_FILE.
    Update Makefile with new filename for target.
    """
    assert (len(keywords) == len(replacement)), "Number of keywords needs to be the same as number of replacement words"

    with open('templates/'+template_file+'.c', 'r') as file:
        filedata = file.read()

    for i in range(len(keywords)):
        filedata = filedata.replace('%'+keywords[i]+'%', replacement[i])

    with open('../bareMetalC/'+new_file+'.c', 'w') as file:
        file.write(filedata)

    print("Created " + new_file + " from " + template_file)

    with open('../bareMetalC/Makefile', 'r') as file:
        filedata = file.read()

    filedata = filedata.replace("tests = \\", "tests = \\\n\t"+new_file+"\\")

    with open('../bareMetalC/Makefile', 'w') as file:
        filedata = file.write(filedata)

    print("Updated Makefile")

    with open('../../../data-collection.sh', 'r') as file:
        filedata = file.read()

    filedata = filedata + "./scripts/run-vcs.sh " + new_file + " > data-collection-output/" + new_file + ".txt &\n"

    with open('../../../data-collection.sh', 'w') as file:
        filedata = file.write(filedata)

    print("Updated data-collection.sh script")

    with open('./clean.sh', 'r') as file:
        filedata = file.read()

    filedata = filedata + "rm ../bareMetalC/"+new_file+".c\n"

    with open('./clean.sh', 'w') as file:
        filedata = file.write(filedata)

    print("Updated clean.sh script")

if __name__ == "__main__":
    #USE PANDAS AND CSV
    with open('../../../data-collection.sh', 'w') as file:
        file.write("#!/bin/bash\n\nmkdir -p data-collection-output\n")

    with open('./clean.sh', 'w') as file:
        file.write("#!/bin/bash\n\nrm -rf ../../../data-collection-output\nrm ../../../data-collection.sh\ncp og_baremetal_Makefile ../bareMetalC/Makefile\n")
        
    main(["DIM_I", "DIM_J", "DIM_K"], ["128", "128", "128"], "tiled_matmul_ws_perf_template", "tiled_matmul_ws_perf-128_128_128")
    main(["DIM_I", "DIM_J", "DIM_K"], ["512", "32", "512"], "tiled_matmul_ws_perf_template", "tiled_matmul_ws_perf-512_32_512")
    main(["DIM_I", "DIM_J", "DIM_K"], ["512", "512", "512"], "tiled_matmul_ws_perf_template", "tiled_matmul_ws_perf-512_512_512")
    main(["DIM_I", "DIM_J", "DIM_K"], ["1024", "1024", "1024"], "tiled_matmul_ws_perf_template", "tiled_matmul_ws_perf-1024_1024_1024")

    main(["DIM_I", "DIM_J", "DIM_K"], ["128", "512", "512"], "tiled_matmul_ws_perf_template", "tiled_matmul_ws_perf-128_512_512")
    main(["DIM_I", "DIM_J", "DIM_K"], ["128", "2048", "512"], "tiled_matmul_ws_perf_template", "tiled_matmul_ws_perf-128_2048_512")
    main(["DIM_I", "DIM_J", "DIM_K"], ["128", "512", "2048"], "tiled_matmul_ws_perf_template", "tiled_matmul_ws_perf-128_512_2048")

    main(["IN_DIM", "IN_CHANNELS", "OUT_CHANNELS", "KERNEL_DIM", "PADDING", "STRIDE"], ["224", "3", "64", "7", "2", "3"], "conv-perf_template", "conv-perf_224-3-64-7-2-3")
    main(["IN_DIM", "IN_CHANNELS", "OUT_CHANNELS", "KERNEL_DIM", "PADDING", "STRIDE"], ["56", "64", "64", "1", "1", "1"], "conv-perf_template", "conv-perf_56-64-64-1-1-1")
    main(["IN_DIM", "IN_CHANNELS", "OUT_CHANNELS", "KERNEL_DIM", "PADDING", "STRIDE"], ["14", "256", "256", "3", "1", "1"], "conv-perf_template", "conv-perf_14-256-256-3-1-1")
    main(["IN_DIM", "IN_CHANNELS", "OUT_CHANNELS", "KERNEL_DIM", "PADDING", "STRIDE"], ["7", "512", "512", "3", "1", "1"], "conv-perf_template", "conv-perf_7-512-512-3-1-1")
