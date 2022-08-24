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

    with open('../../../data-collection-vcs.sh', 'r') as file:
        filedata = file.read()

    filedata = filedata + "./scripts/run-vcs.sh " + new_file + " > data-collection-output/" + new_file + "-vcs.txt &\n"

    with open('../../../data-collection-vcs.sh', 'w') as file:
        filedata = file.write(filedata)

    print("Updated data-collection-vcs.sh script")

    with open('../../../data-collection-midas.sh', 'r') as file:
        filedata = file.read()

    filedata = filedata + "./scripts/run-midas.sh $1" + new_file + " > data-collection-output/" + new_file + "-midas.txt &\n"

    with open('../../../data-collection-midas.sh', 'w') as file:
        filedata = file.write(filedata)

    print("Updated data-collection-midas.sh script")

    with open('../../../data-collection-spike.sh', 'r') as file:
        filedata = file.read()

    filedata = filedata + "./scripts/run-spike.sh " + new_file + " > data-collection-output/" + new_file + "-spike.txt &\n"

    with open('../../../data-collection-spike.sh', 'w') as file:
        filedata = file.write(filedata)

    print("Updated data-collection-spike.sh script")

    with open('clean.sh', 'a') as file:
        file.write('rm ../bareMetalC/' + new_file + '.c\n')

    print("Updated clean.sh script")


if __name__ == "__main__":
    with open('../../../data-collection-vcs.sh', 'w') as file:
        file.write("#!/bin/bash\n\nmkdir -p data-collection-output\n")

    with open('../../../data-collection-verilator.sh', 'w') as file:
        file.write("#!/bin/bash\n\nmkdir -p data-collection-output\n")

    with open('../../../data-collection-midas.sh', 'w') as file:
        file.write("#!/bin/bash\n\nmkdir -p data-collection-output\n")

    with open('../../../data-collection-spike.sh', 'w') as file:
        file.write("#!/bin/bash\n\nmkdir -p data-collection-output\n")

    with open('clean.sh', 'w') as file:
        file.write('#!/bin/bash\n\nrm -rf ../../../data-collection-output\nrm ../../../data-collection-vcs.sh\nrm ../../../data-collection-verilator.sh\nrm ../../../data-collection-midas.sh\nrm ../../../data-collection-spike.sh\ncp og_baremetal_Makefile ../bareMetalC/Makefile\ncd ..\n./build.sh clean\ncd gemmini-data-collection\n')

    # call to main: array of keywords, array of values for keywords, name of C file in the templates folder, name of output C file to be placed in bareMetalC
    main(["DIM_I", "DIM_J", "DIM_K"], ["128", "128", "128"], "tiled_matmul_ws_perf_template", "tiled_matmul_ws_perf-128_128_128")
    main(["DIM_I", "DIM_J", "DIM_K"], ["512", "32", "512"], "tiled_matmul_ws_perf_template", "tiled_matmul_ws_perf-512_32_512")
    main(["DIM_I", "DIM_J", "DIM_K"], ["512", "512", "512"], "tiled_matmul_ws_perf_template", "tiled_matmul_ws_perf-512_512_512")
    main(["DIM_I", "DIM_J", "DIM_K"], ["1024", "1024", "1024"], "tiled_matmul_ws_perf_template", "tiled_matmul_ws_perf-1024_1024_1024")

    main(["IN_DIM", "IN_CHANNELS", "OUT_CHANNELS", "KERNEL_DIM", "STRIDE", "PADDING"], ["224", "3", "64", "7", "2", "3"], "conv-perf_template", "conv-perf_224-3-64-7-2-3")
    main(["IN_DIM", "IN_CHANNELS", "OUT_CHANNELS", "KERNEL_DIM", "STRIDE", "PADDING"], ["56", "64", "64", "1", "1", "0"], "conv-perf_template", "conv-perf_56-64-64-1-1-1")
    main(["IN_DIM", "IN_CHANNELS", "OUT_CHANNELS", "KERNEL_DIM", "STRIDE", "PADDING"], ["14", "256", "256", "3", "1", "1"], "conv-perf_template", "conv-perf_14-256-256-3-1-1")
    main(["IN_DIM", "IN_CHANNELS", "OUT_CHANNELS", "KERNEL_DIM", "STRIDE", "PADDING"], ["7", "512", "512", "3", "1", "1"], "conv-perf_template", "conv-perf_7-512-512-3-1-1")
