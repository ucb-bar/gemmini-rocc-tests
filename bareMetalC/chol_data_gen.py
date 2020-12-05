import numpy as np
from sklearn import datasets

def print_arr(array_type, array_name, array_sz, pyarr):
    print("{} {}[{}][{}] = " .format(array_type, array_name, array_sz, array_sz))
    print("{", end='', flush=True)
    for row in pyarr:
        print("{", end='', flush=True)
        print(", ".join(map(str, row)), end='', flush=True)
        print("}", end='', flush=True)
        if (row != pyarr[-1]).any():
            print(",")
    print("};")
    print("\n")

dim = 512
num_block = 8
block_dim = (int)(dim/num_block)
#generate random spd matrix
A = np.random.randint(-3, 4, size=(dim,dim))
A = np.dot(A, A.transpose())
#A = datasets.make_sparse_spd_matrix(dim, alpha=0.2, smallest_coef=0.5, largest_coef=0.9)
A = np.float32(A)
L = np.linalg.cholesky(A)
L = np.float32(L)
invL = np.linalg.inv(L)
invL = np.float32(invL)

print("#define MAT_DIM {}".format(dim))
print("#define num_block {}".format(num_block))
print("#define block_dim {}".format(block_dim))
print_arr('elem_t', 'in_A', 'MAT_DIM', A)
print_arr('elem_t', 'gold_L', 'MAT_DIM', L)
print_arr('elem_t', 'gold_invL', 'MAT_DIM', invL)
