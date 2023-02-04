import numpy as np
from numpy.linalg import matrix_rank
from sklearn import datasets

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

def print_vec(array_type, array_name, array_sz, pyarr):
    print("{} {}[{}] = " .format(array_type, array_name, array_sz))
    print("{", end='', flush=True)
    row = pyarr
    print(", ".join(map(str, row)), end='', flush=True)
    print("};")
    print("\n")

dim = 128
stride = dim #+ 4
#generate random spd matrix
A = np.random.randint(1, 5, size=(dim,dim))
A = np.tril(A) # lower triangular matrix
while matrix_rank(A) < dim:
    A = np.random.randint(1, 5, size=(dim,dim))
    A = np.tril(A) # lower triangular matrix
    #print(matrix_rank(A))

x = np.random.randint(-2, 3, size=(dim, 1))
b = np.matmul(A,x)

A = np.float32(A)
b = np.float32(b)
x = np.float32(x)

A = np.pad(A, ((0,stride-dim),(0,stride-dim)), 'constant')
x = x.reshape(dim)
b = b.reshape(dim)
print("#define MAT_DIM_S {}".format(stride))
print("#define MAT_DIM {}".format(dim))
print_arr('elem_t', 'A', 'MAT_DIM_S', A)
print_vec('elem_t', 'dx_gold', 'MAT_DIM', x)
print_vec('elem_t', 'b_vec', 'MAT_DIM', b)
