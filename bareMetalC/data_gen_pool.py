import random

NO_BIAS = False
IN_DIM = 14
STRIDE = 2
IN_CHANNELS = 128
OUT_CHANNELS = 128
KERNEL_DIM = 3
PADDING = 1
BATCH_SIZE = 1

POOL_SIZE = 3
POOL_STRIDE = 1
POOL_PADDING = 1

OUT_DIM = int((IN_DIM + 2*PADDING - KERNEL_DIM) / STRIDE) + 1
PATCH_SIZE = KERNEL_DIM * KERNEL_DIM * IN_CHANNELS
POOL_OUT_DIM = int((OUT_DIM + 2 * POOL_PADDING - POOL_SIZE) / POOL_STRIDE + 1)
#int(random.random()*5)-2
IA = [[[int(random.random()*5)-2 for ich in range(IN_CHANNELS)] for icol in range(IN_DIM)] for irow in range(IN_DIM)]
W = [[[[int(random.random()*5)-2 for ich in range(IN_CHANNELS)] for krow in range(KERNEL_DIM)] for kcol in range(KERNEL_DIM)] for och in range(OUT_CHANNELS)]
D = [int(random.random()*5)-2 for och in range(OUT_CHANNELS)]

OA = [[[0 for ich in range(OUT_CHANNELS)] for icol in range(OUT_DIM)] for irow in range(OUT_DIM)]
POA = [[[-127 for ich in range(OUT_CHANNELS)] for icol in range(POOL_OUT_DIM)] for irow in range(POOL_OUT_DIM)]


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

for orow in range(POOL_OUT_DIM):
    for ocol in range(POOL_OUT_DIM):
        for ch in range(OUT_CHANNELS):
            for wrow in range(POOL_SIZE):
                for wcol in range(POOL_SIZE):
                    irow = orow * POOL_STRIDE + wrow - POOL_PADDING
                    icol = ocol * POOL_STRIDE + wcol - POOL_PADDING
                    pixel = 0
                    if not(irow < 0 or irow >= OUT_DIM or icol < 0 or icol >= OUT_DIM):
                        pixel = OA[irow][icol][ch]

                    if pixel > POA[orow][ocol][ch]:
                        POA[orow][ocol][ch] = pixel


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

pooled_result_array = "{"
pooled_result_array += "{"
for i in range(POOL_OUT_DIM):
    pooled_result_array += "{"
    for j in range(POOL_OUT_DIM):
        pooled_result_array += "{"
        for k in range(OUT_CHANNELS):
            pooled_result_array += str(POA[i][j][k])
            if k != OUT_CHANNELS - 1:
                pooled_result_array += ","
        pooled_result_array += "}"
        if j != POOL_OUT_DIM - 1:
            pooled_result_array += ","
    pooled_result_array += "}"
    if i != POOL_OUT_DIM - 1:
        pooled_result_array += ","
pooled_result_array += "}"
pooled_result_array += "};"

print("#define BATCH_SIZE ", BATCH_SIZE)
print("#define IN_CHANNELS ", IN_CHANNELS)
print("#define OUT_CHANNELS ", OUT_CHANNELS)
print("#define IN_DIM ", IN_DIM)
print("#define OUT_DIM ", OUT_DIM)
print("#define POOL_OUT_DIM ", POOL_OUT_DIM)
print("#define KERNEL_DIM ", KERNEL_DIM)
print("#define PATCH_SIZE ", PATCH_SIZE)
print("#define PADDING ", PADDING)
print("#define STRIDE ", STRIDE)
print("#define POOL_SIZE ", POOL_SIZE)
print("#define POOL_PADDING ", POOL_PADDING)
print("#define POOL_STRIDE ", POOL_STRIDE)

print("static elem_t weights_mat[PATCH_SIZE][OUT_CHANNELS] row_align(MAX_BLOCK_LEN) = ", w_array)

print("static acc_t bias[OUT_CHANNELS] row_align(MAX_BLOCK_LEN_ACC) = ", D_array)

print("static elem_t input[BATCH_SIZE][IN_DIM][IN_DIM][IN_CHANNELS] row_align(MAX_BLOCK_LEN) = ", input_array)


print("static elem_t output[BATCH_SIZE][OUT_DIM][OUT_DIM][OUT_CHANNELS] row_align(MAX_BLOCK_LEN) = ", result_array)
print("static elem_t pool_output[BATCH_SIZE][POOL_OUT_DIM][POOL_OUT_DIM][OUT_CHANNELS] row_align(MAX_BLOCK_LEN) = ", pooled_result_array)
