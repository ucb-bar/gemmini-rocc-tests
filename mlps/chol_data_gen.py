import numpy as np
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
block_dim = 16
num_block = int(dim/block_dim)
stride = dim + 4
#generate random spd matrix
A = np.random.randint(-2, 3, size=(dim,dim))
A = np.dot(A, A.transpose())
A = np.float32(A)
#A = A.astype(int)
L = np.linalg.cholesky(A)
L = np.float32(L)
#L = L.astype(int)
#invL = np.linalg.inv(L)
#invL = np.float32(invL)

x = np.random.randint(-2, 3, size=(dim, 1))
x = np.float32(x)
b = np.matmul(L,x)

b = np.float32(b)
A = np.pad(A, ((0,stride-dim),(0,stride-dim)), 'constant')
L = np.pad(L, ((0,stride-dim),(0,stride-dim)), 'constant')
x = x.reshape(dim)
b = b.reshape(dim)

print("#define MAT_DIM_S {}".format(stride))
print("#define MAT_DIM {}".format(dim))
print("#define NUM_BLOCK {}".format(num_block))
print("#define BLOCK_DIM {}".format(block_dim))
print_arr('elem_t', 'in_A', 'MAT_DIM_S', A)
print_arr('elem_t', 'gold_L', 'MAT_DIM_S', L)
#print_arr('elem_t', 'gold_invL', 'MAT_DIM', invL)
print_vec('elem_t', 'dx_gold', 'MAT_DIM', x)
print_vec('elem_t', 'b_vec', 'MAT_DIM', b)
