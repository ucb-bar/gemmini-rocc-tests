import numpy as np
import random
from numpy.linalg import matrix_rank
from sklearn import datasets
from numpy.linalg import inv

def print_arr(array_type, array_name, array_sz, pyarr):
    print("{} {}[{}][{}] = " .format(array_type, array_name, array_sz, array_sz))
    print("{", end='', flush=True)
    for r in range(len(pyarr)):
        row = pyarr[r]
        print("{", end='', flush=True)
        print(", ".join(map(str, row)), end='', flush=True)
        print("}", end='', flush=True)
        if (r != len(pyarr) - 1):
            print(",")
    print("};")
    print("\n")


def print_arr2(array_type, array_name, array_sz_r, array_sz_c, pyarr):
    print("{} {}[{}][{}] = " .format(array_type, array_name, array_sz_r, array_sz_c))
    print("{", end='', flush=True)
    for r in range(len(pyarr)):
        row = pyarr[r]
        print("{", end='', flush=True)
        print(", ".join(map(str, row)), end='', flush=True)
        print("}", end='', flush=True)
        if (r != len(pyarr) - 1):
            print(",")
    print("};")
    print("\n")

def print_vec(array_type, array_name, array_sz, pyarr):
    print("{} {}[{}] = " .format(array_type, array_name, array_sz))
    print("{", end='', flush=True)
    row = pyarr
    print(", ".join(map(str, row)), end='', flush=True)
    print("};")
    print("\n")

dim = 4
A_dim = 512
D_dim = 128
total_dim = A_dim + D_dim
stride = total_dim#+dim
A_small_dim = 3 # 3x3 at most
# A   B
# C   D
# S = D-C*A^(-1)*B
#generate random spd matrix
D = np.random.randint(1, 5, size=(D_dim,D_dim))
D = np.tril(D)
while matrix_rank(D) < D_dim:
    D = np.random.randint(1, 5, size=(D_dim,D_dim))
    D = np.tril(D)

for i in range(D_dim):
    for j in range(i+1, D_dim):
        D[i, j] = D[j, i]

C = np.zeros((D_dim, A_dim))
A = np.zeros((A_dim, A_dim))
M = np.zeros((total_dim, total_dim))

while matrix_rank(M) < total_dim:
    for i in range(total_dim):
        for j in range(i): # lower dimension
            if i < A_dim:
                small_index = (int)(i / A_small_dim)
                if j >= (small_index * A_small_dim) and j < (small_index + 1) * A_small_dim:
                    M[i, j] = random.randrange(1, 5)
                    A[i, j] = M[i, j]
            else:
                if j < A_dim:
                    M[i, j] = random.randrange(1, 4)
                    C[i-A_dim, j] = M[i, j]
                else:
                    M[i, j] = D[i - A_dim, j - A_dim]

    for i in range(total_dim):
        for j in range(i+1, total_dim):
            M[i, j] = M[j, i]

    for i in range(A_dim):
        for j in range(i+1, A_dim):
            A[i, j] = A[j, i]
    #print(matrix_rank(M), matrix_rank(A))

M = np.float32(M)

# D-CA^-1B = D-CA^-1C^T

M = np.pad(M, ((0,stride-total_dim),(0,stride-total_dim)), 'constant')
#x = x.reshape(dim)
#b = b.reshape(dim)
A_inv = inv(A)
A_inv = np.float32(A_inv)
CA_inv = np.matmul(C, A_inv)
CA_inv = np.float32(CA_inv)
S = np.subtract(D, np.matmul(CA_inv, np.transpose(C)))
S = np.float32(S)

print("#define MAT_DIM_S {}".format(stride))
print("#define MAT_DIM {}".format(total_dim))
print("#define A_DIM {}".format(A_dim))
print("#define D_DIM {}".format(D_dim))
print_arr('elem_t', 'M', 'MAT_DIM_S', M)
print_arr('elem_t', 'D', 'D_DIM', D)
print_arr('elem_t', 'A', 'A_DIM', A)
print_arr('elem_t', 'A_inv_gold', 'A_DIM', A_inv)
print_arr2('elem_t', 'C', 'D_DIM', 'A_DIM', C)
print_arr2('elem_t', 'CAinv', 'D_DIM', 'A_DIM', CA_inv)
print_arr('elem_t', 'S', 'D_DIM', S)
