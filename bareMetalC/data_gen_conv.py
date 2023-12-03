import random

NO_BIAS = True
IN_DIM = 7
STRIDE = 1
IN_CHANNELS = 88
OUT_CHANNELS = (64*2)
KERNEL_DIM = 3
PADDING = 1
BATCH_SIZE = 1

OUT_DIM = int((IN_DIM + 2*PADDING - KERNEL_DIM) / STRIDE) + 1
PATCH_SIZE = KERNEL_DIM * KERNEL_DIM * IN_CHANNELS
#int(random.random()*5)-2
IA = [[[int(random.random()*5)-2 for ich in range(IN_CHANNELS)] for icol in range(IN_DIM)] for irow in range(IN_DIM)]
W = [[[[int(random.random()*5)-2 for ich in range(IN_CHANNELS)] for krow in range(KERNEL_DIM)] for kcol in range(KERNEL_DIM)] for och in range(OUT_CHANNELS)]
D = [int(random.random()*5)-2 for och in range(OUT_CHANNELS)]

OA = [[[int(random.random()*5)-2 for ich in range(OUT_CHANNELS)] for icol in range(OUT_DIM)] for irow in range(OUT_DIM)]


for orow in range(OUT_DIM):
    for ocol in range(OUT_DIM):
        for och in range(OUT_CHANNELS):
            result = D[och] if NO_BIAS is False else 0
            for krow in range(KERNEL_DIM):
                for kcol in range(KERNEL_DIM):
                    for kch in range(IN_CHANNELS):
                        irow = orow * STRIDE + krow - PADDING
                        icol = ocol * STRIDE + kcol - PADDING
                        pixel = 0
                        if irow < 0 or irow >= IN_DIM or icol < 0 or icol >= IN_DIM:
                            pixel = 0
                        else:
                            pixel = IA[irow][icol][kch]
                        result += W[och][krow][kcol][kch] * pixel
            if result > 127:
                result = 127
            elif result < -128:
                result = -128
            OA[orow][ocol][och] = result


W_F = [[0 for och in range(OUT_CHANNELS)] for patch in range(PATCH_SIZE)]

for och in range(OUT_CHANNELS):
    for krow in range(KERNEL_DIM):
        for kcol in range(KERNEL_DIM):
            for ich in range(IN_CHANNELS):
                wmatrow = krow * KERNEL_DIM * IN_CHANNELS + kcol * IN_CHANNELS + ich
                W_F[wmatrow][och] = W[och][krow][kcol][ich]


w_array = "{"
for i in range(PATCH_SIZE):
    w_array += "{"
    for j in range(OUT_CHANNELS):
        w_array += str(W_F[i][j])
        if j != OUT_CHANNELS - 1:
            w_array += ","
    w_array += "}"
    if i != PATCH_SIZE - 1:
        w_array += ","
w_array += "};"

D_array = "{"
for i in range(OUT_CHANNELS):
    D_array += str(D[i])
    if i != OUT_CHANNELS - 1:
        D_array += ","
D_array += "};"

input_array = "{"
input_array += "{"
for i in range(IN_DIM):
    input_array += "{"
    for j in range(IN_DIM):
        input_array += "{"
        for k in range(IN_CHANNELS):
            input_array += str(IA[i][j][k])
            if k != IN_CHANNELS - 1:
                input_array += ","
        input_array += "}"
        if j != IN_DIM - 1:
            input_array += ","
    input_array += "}"
    if i != IN_DIM - 1:
        input_array += ","
input_array += "}"
input_array += "};"

result_array = "{"
result_array += "{"
for i in range(OUT_DIM):
    result_array += "{"
    for j in range(OUT_DIM):
        result_array += "{"
        for k in range(OUT_CHANNELS):
            result_array += str(OA[i][j][k])
            if k != OUT_CHANNELS - 1:
                result_array += ","
        result_array += "}"
        if j != OUT_DIM - 1:
            result_array += ","
    result_array += "}"
    if i != OUT_DIM - 1:
        result_array += ","
result_array += "}"
result_array += "};"

print("#define BATCH_SIZE ", BATCH_SIZE)
print("#define IN_CHANNELS ", IN_CHANNELS)
print("#define OUT_CHANNELS ", OUT_CHANNELS)
print("#define IN_DIM ", IN_DIM)
print("#define OUT_DIM ", OUT_DIM)
print("#define KERNEL_DIM ", KERNEL_DIM)
print("#define PATCH_SIZE ", PATCH_SIZE)
print("#define PADDING ", PADDING)
print("#define STRIDE ", STRIDE)

print("static elem_t weights_mat[PATCH_SIZE][OUT_CHANNELS] row_align(MAX_BLOCK_LEN) = ", w_array)

print("static acc_t bias[OUT_CHANNELS] row_align(MAX_BLOCK_LEN_ACC) = ", D_array)

print("static elem_t input[BATCH_SIZE][IN_DIM][IN_DIM][IN_CHANNELS] row_align(MAX_BLOCK_LEN) = ", input_array)


print("static elem_t output[BATCH_SIZE][OUT_DIM][OUT_DIM][OUT_CHANNELS] row_align(MAX_BLOCK_LEN) = ", result_array)
