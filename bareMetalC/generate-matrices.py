import sys
import random

if len(sys.argv) != 5:
    print("usage: {} I J K NO_BIAS".format(sys.argv[0]))
    sys.exit(1)

if sys.argv[4] != "1" and sys.argv[4] != "0":
    print('NO_BIAS must be "1" or "0"')
    sys.exit(1)

I = int(sys.argv[1])
J = int(sys.argv[2])
K = int(sys.argv[3])
NO_BIAS = sys.argv[4] == "1"

def rand_print(X, Y):
    for x in range(X):
        print("\t{", end="")
        for y in range(Y):
            end = ", " if y != Y-1 else ""
            print(random.randint(-64, 64), end=end)

        end = ",\n" if x != X-1 else "\n"
        print("}", end=end)

print("elem_t full_A[{}][{}] = {{".format(I, K))
rand_print(I, K)
print("};")

print("elem_t full_B[{}][{}] = {{".format(K, J))
rand_print(K, J)
print("};")

if NO_BIAS:
    print("ACC_T full_D[{}][{}];".format(I, J))
else:
    print("ACC_T full_D[{}][{}] = {{".format(I, J))
    rand_print(I, J)
    print("};")

