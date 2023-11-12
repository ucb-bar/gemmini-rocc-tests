import random

NO_BIAS = False
MAT_DIM_I = 96
MAT_DIM_J = 192
MAT_DIM_K = 128

A = [[int(random.random()*5)-2 for x in range(MAT_DIM_K)] for y in range(MAT_DIM_I)]
B = [[int(random.random()*5)-2 for x in range(MAT_DIM_J)] for y in range(MAT_DIM_K)]
D = [[int(random.random()*5) for x in range(MAT_DIM_J)] for y in range(MAT_DIM_I)]
if NO_BIAS:
    D = [[0 for x in range(MAT_DIM_J)] for y in range(MAT_DIM_I)]

result = [[0 for x in range(MAT_DIM_J)] for y in range(MAT_DIM_I)]


for i in range(MAT_DIM_I):
    for j in range(MAT_DIM_J):
        result[i][j] = D[i][j]
        for k in range(MAT_DIM_K):
            result[i][j] += A[i][k]*B[k][j]


A_array = "{"
for i in range(MAT_DIM_I):
    A_array += "{"
    for j in range((MAT_DIM_K)):
        A_array += str(A[i][j])
        if j != (MAT_DIM_K) - 1:
            A_array += ","
    A_array += "}"
    if i != (MAT_DIM_I) - 1:
        A_array += ","
A_array += "};"

B_array = "{"
for i in range((MAT_DIM_K)):
    B_array += "{"
    for j in range((MAT_DIM_J)):
        B_array += str(B[i][j])
        if j != (MAT_DIM_J) - 1:
            B_array += ","
    B_array += "}"
    if i != (MAT_DIM_K) - 1:
        B_array += ","
B_array += "};"


D_array = "{"
for i in range((MAT_DIM_I)):
    D_array += "{"
    for j in range((MAT_DIM_J)):
        D_array += str(D[i][j])
        if j != (MAT_DIM_J) - 1:
            D_array += ","
    D_array += "}"
    if i != (MAT_DIM_I) - 1:
        D_array += ","
D_array += "};"


result_array = "{"
for i in range((MAT_DIM_I)):
    result_array += "{"
    for j in range((MAT_DIM_J)):
        result_array += str(result[i][j])
        if j != (MAT_DIM_J) - 1:
            result_array += ","
    result_array += "}"
    if i != (MAT_DIM_I) - 1:
        result_array += ","
result_array += "};"

print("#define MAT_DIM_I ", MAT_DIM_I)
print("#define MAT_DIM_J ", MAT_DIM_J)
print("#define MAT_DIM_K ", MAT_DIM_K)

print("static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN) = ", A_array)

print("static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = ", B_array)

print("static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN_ACC) = ", D_array)

print("static elem_t gold[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = ", result_array)

