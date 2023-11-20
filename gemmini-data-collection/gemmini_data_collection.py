import sys
import random
import argparse

import tests

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

    with open('../../../data-collection-verilator.sh', 'r') as file:
        filedata = file.read()

    filedata = filedata + "./scripts/run-verilator.sh " + new_file + " > data-collection-output/" + new_file + "-verilator.txt &\n"

    with open('../../../data-collection-verilator.sh', 'w') as file:
        filedata = file.write(filedata)

    print("Updated data-collection-verilator.sh script")

    with open('../../../data-collection-midas.sh', 'r') as file:
        filedata = file.read()

    filedata = filedata + "./scripts/run-midas.sh $1 " + new_file + " > data-collection-output/" + new_file + "-midas.txt &\n"

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

def one_file(tests, pe_dim, spad_size, acc_size):
    """
    Search TEMPLATE_FILE for KEYWORDS and replace respective keywords with REPLACEMENT. Write changes to NEW_FILE.
    Update Makefile with new filename for target.
    """
    # Make conv file
    with open('templates/conv_tilings.c', 'r') as file:
        filedata = file.read()

    conv_keywords = []
    num_convs = 0
    for test_i, test in enumerate(reversed(tests)):
        keywords, replacement, template_file, _ = test
        assert (len(keywords) == len(replacement)), "Number of keywords needs to be the same as number of replacement words"

        if template_file == "conv_template_map":
            conv_keywords = keywords
            for i in range(len(keywords)):
                filedata = filedata.replace('%'+keywords[i]+'%', '%'+keywords[i]+'%' + "," + str(replacement[i]))
            num_convs += 1
    if test_i == len(tests)-1:
        for i in range(len(conv_keywords)):
            filedata = filedata.replace('%'+conv_keywords[i]+'%,', "")
    filedata = filedata.replace("%NUM_LAYERS%", str(num_convs))
    filedata = filedata.replace("%PE_DIM%", str(pe_dim))
    filedata = filedata.replace("%SPAD_SIZE%", str(spad_size))
    filedata = filedata.replace("%ACC_SIZE%", str(acc_size))
    new_conv_file = "conv_tilings"
    with open('../bareMetalC/'+new_conv_file+'.c', 'w') as file:
        file.write(filedata)
    print("Created " + new_conv_file + " from conv_tilings.c")

    # Make matmul file
    with open('templates/matmul_tilings.c', 'r') as file:
        filedata = file.read()

    matmul_keywords = []
    num_matmuls = 0
    for test_i, test in enumerate(reversed(tests)):
        keywords, replacement, template_file, _ = test
        assert (len(keywords) == len(replacement)), "Number of keywords needs to be the same as number of replacement words"

        if template_file == "matmul_template_map":
            matmul_keywords = keywords
            for i in range(len(keywords)):
                filedata = filedata.replace('%'+keywords[i]+'%', '%'+keywords[i]+'%' + "," + str(replacement[i]))
            num_matmuls += 1
    if test_i == len(tests)-1:
        for i in range(len(matmul_keywords)):
            filedata = filedata.replace('%'+matmul_keywords[i]+'%,', "")

    filedata = filedata.replace("%NUM_LAYERS%", str(num_matmuls))
    filedata = filedata.replace("%PE_DIM%", str(pe_dim))
    filedata = filedata.replace("%SPAD_SIZE%", str(spad_size))
    filedata = filedata.replace("%ACC_SIZE%", str(acc_size))
    new_matmul_file = "matmul_tilings"
    with open('../bareMetalC/'+new_matmul_file+'.c', 'w') as file:
        file.write(filedata)
    print("Created " + new_matmul_file + " from matmul_tilings.c")

    # Update Makefile
    with open('../bareMetalC/Makefile', 'r') as file:
        filelines = file.readlines()

    # filedata = filedata.replace("tests = \\", "tests = \\\n\t"+new_conv_file+" \\")
    # filedata = filedata.replace("tests = \\", "tests = \\\n\t"+new_matmul_file+" \\")
    for l in range(len(filelines)):
        if "tests = " in filelines[l]:
            tests_str = "tests = "
            if num_convs > 0:
                tests_str += " " + new_conv_file + " "
            if num_matmuls > 0:
                tests_str += " " + new_matmul_file + " "
            filelines[l] = tests_str + "\n"
            break

    with open('../bareMetalC/Makefile', 'w') as file:
        filedata = file.writelines(filelines)

    print("Updated Makefile")

def per_pe_dim(tests):
    """
    Search TEMPLATE_FILE for KEYWORDS and replace respective keywords with REPLACEMENT. Write changes to NEW_FILE.
    Update Makefile with new filename for target.
    """
    random.seed(0)
    pe_dims = [2, 4, 8, 16, 32, 64, 128]
    test_groups = {pe_dim: {"conv_template_map": [], "matmul_template_map": []} for pe_dim in pe_dims}
    for test in tests:
        keywords, replacement, template_file, _ = test
        replace_dict = dict(zip(keywords, replacement))
        for pe_dim in pe_dims:
            if int(replace_dict["SPATIAL_TILE_KCHS"]) <= pe_dim and int(replace_dict["SPATIAL_TILE_OCHS"]) <= pe_dim:
                if (int(replace_dict["SPATIAL_TILE_KCHS"]) <= pe_dim // 2) and (int(replace_dict["TILE_KCHS"]) > int(replace_dict["SPATIAL_TILE_KCHS"])):
                    break
                elif (int(replace_dict["SPATIAL_TILE_OCHS"]) <= pe_dim // 2) and (int(replace_dict["TILE_OCHS"]) > int(replace_dict["SPATIAL_TILE_OCHS"])):
                    break
                else:
                    test_groups[pe_dim][template_file].append(test)
                    break
    # print(test_groups)
    print([len(test_groups[p]["conv_template_map"]) for p in test_groups])
    print([len(test_groups[p]["matmul_template_map"]) for p in test_groups])

    counts = {
        "conv_template_map": 500,
        "matmul_template_map": 200,
    }
    for pe_dim in test_groups:
        for t in test_groups[pe_dim]:
            if len(test_groups[pe_dim][t]) > counts[t]:
                test_groups[pe_dim][t] = random.choices(test_groups[pe_dim][t], k=counts[t])
    print([len(test_groups[p]["conv_template_map"]) for p in test_groups])
    print([len(test_groups[p]["matmul_template_map"]) for p in test_groups])

    files_created = []
    for pe_dim in test_groups:
        for t in test_groups[pe_dim]:
            layer_type = "conv" if "conv" in t else "matmul"
            # Make conv file
            with open(f'templates/{layer_type}_tilings.c', 'r') as file:
                filedata = file.read()

            for test_i, test in enumerate(reversed(test_groups[pe_dim][t])):
                keywords, replacement, template_file, _ = test
                assert (len(keywords) == len(replacement)), "Number of keywords needs to be the same as number of replacement words"
                for i in range(len(keywords)):
                    filedata = filedata.replace('%'+keywords[i]+'%', '%'+keywords[i]+'%' + "," + str(replacement[i]))

            for i in range(len(keywords)):
                filedata = filedata.replace('%'+keywords[i]+'%,', "")
            filedata = filedata.replace("%NUM_LAYERS%", str(len(test_groups[pe_dim][t])))
            filedata = filedata.replace("%PE_DIM%", str(pe_dim))
            new_file = f"{layer_type}_tilings_{pe_dim}pe"
            with open('../bareMetalC/'+new_file+'.c', 'w') as file:
                file.write(filedata)
            files_created.append(new_file)
            print("Created " + new_file)

    # with open('../bareMetalC/Makefile', 'r') as file:
    #     filedata = file.read()

    # for new_file in files_created:
    #     if new_file not in filedata:
    #         filedata = filedata.replace("tests = \\", "tests = \\\n\t"+new_file+"\\")

    # with open('../bareMetalC/Makefile', 'w') as file:
    #     filedata = file.write(filedata)

    # print("Updated Makefile")

def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')

    parser.add_argument('--pe_dim',
                        type=int,
                        help='Gemmini systolic array size',
                        required=True,
                        )
    parser.add_argument('--spad_size',
                        type=int,
                        help='Gemmini scratchpad size',
                        required=True,
                        )
    parser.add_argument('--acc_size',
                        type=int,
                        help='Gemmini accumulator size',
                        required=True,
                        )
    return parser

if __name__ == "__main__":
    # for fname in 'vcs', 'verilator', 'midas', 'spike':
    #     with open('../../../data-collection-' + fname + '.sh', 'w') as file:
    #         file.write("#!/bin/bash\n\nmkdir -p data-collection-output\n")

    with open('clean.sh', 'w') as file:
        file.write('#!/bin/bash\n\nrm -rf ../../../data-collection-output\nrm ../../../data-collection-vcs.sh\nrm ../../../data-collection-verilator.sh\nrm ../../../data-collection-midas.sh\nrm ../../../data-collection-spike.sh\ncp og_baremetal_Makefile ../bareMetalC/Makefile\ncd ..\n./build.sh clean\ncd gemmini-data-collection\n')

    # for test in tests.tests:
    #     main(*test)
    parser = construct_argparser()
    args = parser.parse_args()
    one_file(tests.tests, args.pe_dim, args.spad_size, args.acc_size)
    # per_pe_dim(tests.tests)

    # for fname in 'vcs', 'verilator', 'midas', 'spike':
    #     with open('../../../data-collection-' + fname + '.sh', 'a') as file:
    #         file.write("wait\n")

