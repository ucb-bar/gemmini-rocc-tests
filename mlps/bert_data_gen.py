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
input_len = 128
hidden_dim = 768
expansion_dim = 3072
num_heads = 12
stride = dim
#generate random spd matrix
input = np.random.randint(1, 5, size=(input_len, hidden_dim))
Wq = np.random.randint(-3, 3, size=(hidden_dim, hidden_dim))
Wk = np.random.randint(-3, 3, size=(hidden_dim, hidden_dim))
Wv = np.random.randint(-3, 3, size=(hidden_dim, hidden_dim))

Wq = np.float32(Wq)
Wv = np.float32(Wv)
Wk = np.float32(Wk)
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
