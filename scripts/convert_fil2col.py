#!/usr/bin/python

import numpy as np
import sys

# take a input kernel in and return it in a im2col'ed format
# input must be formed like a C array starting at the curly brace:
# example:
#
#  a 2x2x3x2 (fW, fW, inC, outC) kernel
#  outC1:
#  ch1:   ch2:   ch3:
#  1 2    5 6     9 10
#  3 4    7 8    11 12
#
#  outC2:
#  ch1:   ch2:   ch3:
#  13 14  17 18  21 22
#  15 16  19 20  23 24
#
#  input file must have (but all on one line):
#  {
#      {
#      {{  1,  13},{  5,  17},{  9,  21}},
#      {{  2,  14},{  6,  18},{ 10,  22}}
#      },
#      {
#      {{  3,  15},{  7,  19},{ 11,  23}},
#      {{  4,  16},{  8,  20},{ 12,  24}}
#      }
#  }

if len(sys.argv) != 5:
    sys.exit("[ERROR] Incorrect # of inputs (filterWidth, filterInChannels, filterOutChannels, pad)")

filterWidth = int(sys.argv[1])
filterInChannels = int(sys.argv[2])
filterOutChannels = int(sys.argv[3])
if sys.argv[4] == "True":
    padOutput = True
else:
    padOutput = False

##### FIRST GET THE OG KERNELS

# contains the 2D old im2col'ed array in this format:
# L1: X
# L2: Y
# L2: {{,,,,},{,,,,},...}
in_file = "cifar_quant_params_old_stripped.h"

with open(in_file, 'r') as inFile:
    # get the right sizes
    X = int(inFile.readline())
    Y = int(inFile.readline())

    #print(X,Y)

    oldMatrix = np.zeros((X, Y))

    matStrRows = inFile.readline().split("},{") # split by rows
    # cleanup
    for i in range(0, len(matStrRows)):
        matStrRows[i] = matStrRows[i].replace('{', '').replace('}', '')

    for i in range(0, len(matStrRows)):
        assert(len(matStrRows) == X)
        matStrRowWithCol = matStrRows[i].split(',')
        for j in range(0, len(matStrRowWithCol)):
            assert(len(matStrRowWithCol) == Y)
            oldMatrix[i][j] = int(matStrRowWithCol[j])

    ## convert the old im2col to the new
    kernels = np.zeros((filterWidth, filterWidth, filterInChannels, filterOutChannels))

    for j in range(0, filterOutChannels):
        for i in range(0, filterWidth*filterWidth*filterInChannels):

            inC = i / (filterWidth * filterWidth)
            outC = j
            sub = i % (filterWidth * filterWidth)
            row = sub / filterWidth
            col = sub % filterWidth

            #print(i, j, row, col, inChannel, outChannel)

            kernels[row][col][inC][outC] = oldMatrix[i][j]

    #print(kernels)

##### PUT KERNEL INTO NEW .H File
#np.set_printoptions(threshold=np.nan) # print all elems
#print('static const elem_t kernel[{}][{}][{}][{}] row_align(1) = '.format(filterWidth, filterWidth, filterInChannels, filterOutChannels)),
#kernelStr = np.array2string(kernels, separator=',').replace('[','{').replace(']','}')
#for i in range(0, len(kernelStr)):
#    print(kernelStr[i]),
#print(";"),
#sys.exit("Done")

#for i in range(0, filterWidth):
#    for j in range(0, filterWidth):
#        for k in range(0, filterInChannels):
#            print('{'),
#            for l in range(0, filterOutChannels):
#                print('{}'.format(int(kernels[i][j][k][l]))),
#                if l != filterOutChannels - 1:
#                    print(','),
#            print('}'),
#
#            if not (i == filterWidth - 1 and j == filterWidth - 1 and k == filterInChannels - 1):
#                print(','),
#
#print('};')

##### THEN CONVERT THE KERNEL TO THE BOTH IM2COL FORMA

padRow = 192
padCol = 64
extraColPad = padCol - filterOutChannels
extraRowPad = padRow - filterWidth*filterWidth*filterInChannels

if padOutput:
    dimX = padRow
    dimY = padCol
else:
    dimX = filterWidth*filterWidth*filterInChannels
    dimY = filterOutChannels

print('static const elem_t NAME[{}][{}] row_align(1) = '.format(dimX, dimY)),

print('{'),

for i in range(0, filterWidth):
    for j in range(0, filterWidth):
        for k in range(0, filterInChannels):
            print('{'),

            for l in range(0, filterOutChannels):
                print('{}'.format(int(kernels[i][j][k][l]))),
                if (l != filterOutChannels - 1) or padOutput:
                    print(','),

            if padOutput:
                for l in range(0, extraColPad):
                    print('0'),
                    if l != extraColPad - 1:
                        print(','),

            print('}'),

            if (not (i == filterWidth - 1 and j == filterWidth - 1 and k == filterInChannels - 1)) or padOutput:
                print(','),

if padOutput:
    for i in range(0, extraRowPad):
        print('{' + ('0, ' * (padCol - 1)) + '0}'),
        if i != extraRowPad - 1:
            print(','),

print('};')

print('static const elem_t NAME[{}][{}] row_align(1) = '.format(dimX, dimY)),

print('{'),

for k in range(0, filterInChannels):
    for i in range(0, filterWidth):
        for j in range(0, filterWidth):
            print('{'),

            for l in range(0, filterOutChannels):
                print('{}'.format(int(kernels[i][j][k][l]))),
                if (l != filterOutChannels - 1) or padOutput:
                    print(','),

            if padOutput:
                for idx in range(0, extraColPad):
                    print('0'),
                    if idx != extraColPad - 1:
                        print(','),

            print('}'),

            if (not (i == filterWidth - 1 and j == filterWidth - 1 and k == filterInChannels - 1)) or padOutput:
                print(','),

if padOutput:
    for i in range(0, extraRowPad):
        print('{' + ('0, ' * (padCol - 1)) + '0}'),
        if i != extraRowPad - 1:
            print(','),

print('};')
