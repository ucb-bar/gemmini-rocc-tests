import random

NO_BIAS = True
MAT_DIM_I = 56
MAT_DIM_J = (64*8)
MAT_DIM_K = 156
repeating_bias = 1 #True

A = [[int(random.random()*5)-2 for x in range(MAT_DIM_K)] for y in range(MAT_DIM_I)]
B = [[int(random.random()*5)-2 for x in range(MAT_DIM_J)] for y in range(MAT_DIM_K)]
D = [[int(random.random()*3) for x in range(MAT_DIM_J)] for y in range(MAT_DIM_I)]
if repeating_bias == 1:
    for y in range(MAT_DIM_I):
        D[y] = D[0]

#A = [[x%2 for x in range(MAT_DIM_K)] for y in range(MAT_DIM_I)]
#B = [[1 for x in range(MAT_DIM_J)] for y in range(MAT_DIM_K)]

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
print("#define REPEATING_BIAS ", repeating_bias)

print("static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN) = ", A_array)

print("static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = ", B_array)

print("static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN_ACC) = ", D_array)

print("static elem_t gold[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = ", result_array)

