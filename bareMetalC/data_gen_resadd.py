import random

MAT_DIM_I = 220
MAT_DIM_J = 178

A_SCALE = 1
B_SCALE = 1
C_SCALE = 1
RELU = True

A = [[int(random.random()*5)-2 for x in range(MAT_DIM_J)] for y in range(MAT_DIM_I)]
B = [[int(random.random()*5)-2 for x in range(MAT_DIM_J)] for y in range(MAT_DIM_I)]

result = [[0 for x in range(MAT_DIM_J)] for y in range(MAT_DIM_I)]


for i in range(MAT_DIM_I):
    for j in range(MAT_DIM_J):
        result[i][j] = A_SCALE*A[i][j]+B_SCALE*B[i][j]
        if result[i][j] < 0 and RELU:
            result[i][j] = 0


A_array = "{"
for i in range(MAT_DIM_I):
    A_array += "{"
    for j in range((MAT_DIM_J)):
        A_array += str(A[i][j])
        if j != (MAT_DIM_J) - 1:
            A_array += ","
    A_array += "}"
    if i != (MAT_DIM_I) - 1:
        A_array += ","
A_array += "};"

B_array = "{"
for i in range((MAT_DIM_I)):
    B_array += "{"
    for j in range((MAT_DIM_J)):
        B_array += str(B[i][j])
        if j != (MAT_DIM_J) - 1:
            B_array += ","
    B_array += "}"
    if i != (MAT_DIM_I) - 1:
        B_array += ","
B_array += "};"


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
print("#define A_SCALE ", A_SCALE)
print("#define B_SCALE ", B_SCALE)
print("#define C_SCALE ", C_SCALE)

print("static elem_t A[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = ", A_array)

print("static elem_t B[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = ", B_array)

print("static elem_t gold[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = ", result_array)

