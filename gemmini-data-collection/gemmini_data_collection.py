import sys
import tests
import argparse

def one_file(tests, pe_dim):
    """
    Search TEMPLATE_FILE for KEYWORDS and replace respective keywords with REPLACEMENT. Write changes to NEW_FILE.
    Update Makefile with new filename for target.
    """
    binaries = []
    for test_i, test in enumerate(tests):
        keywords, replacement, template_file, _ = test
        assert (len(keywords) == len(replacement)), "Number of keywords needs to be the same as number of replacement words"

        if template_file == "conv_template_map":
            with open('templates/conv_tilings.c', 'r') as file:
                filedata = file.read()
            new_file = "conv_tilings_" + str(test_i)
        elif template_file == "matmul_template_map":
            # Make matmul file
            with open('templates/matmul_tilings.c', 'r') as file:
                filedata = file.read()
            new_file = "matmul_tilings_" + str(test_i)

        binaries.append(new_file)
        filedata = filedata.replace("%NUM_LAYERS%", "1")
        filedata = filedata.replace("%PE_DIM%", str(pe_dim))
        for i in range(len(keywords)):
            filedata = filedata.replace('%'+keywords[i]+'%', str(replacement[i]))
        with open('../bareMetalC/'+new_file+'.c', 'w') as file:
            file.write(filedata)
        print("Created " + new_file)

    # Update Makefile
    with open('../bareMetalC/Makefile', 'r') as file:
        filelines = file.readlines()

    # filedata = filedata.replace("tests = \\", "tests = \\\n\t"+new_conv_file+" \\")
    # filedata = filedata.replace("tests = \\", "tests = \\\n\t"+new_matmul_file+" \\")
    for l in range(len(filelines)):
        if "tests = " in filelines[l]:
            tests_str = "tests = "
            for binary in binaries:
                tests_str += " " + binary + " "
            filelines[l] = tests_str + "\n"
            break

    with open('../bareMetalC/Makefile', 'w') as file:
        filedata = file.writelines(filelines)

    print("Updated Makefile")

def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')

    parser.add_argument('--pe_dim',
                        type=int,
                        help='Gemmini systolic array size',
                        required=True,
                        )
    return parser

if __name__ == "__main__":
    with open('clean.sh', 'w') as file:
        file.write('#!/bin/bash\n\nrm -rf ../../../data-collection-output\nrm ../../../data-collection-vcs.sh\nrm ../../../data-collection-verilator.sh\nrm ../../../data-collection-midas.sh\nrm ../../../data-collection-spike.sh\ncp og_baremetal_Makefile ../bareMetalC/Makefile\ncd ..\n./build.sh clean\ncd gemmini-data-collection\n')

    parser = construct_argparser()
    args = parser.parse_args()
    one_file(tests.tests, args.pe_dim)
