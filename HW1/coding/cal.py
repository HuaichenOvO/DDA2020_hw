import numpy as np

A = np.array([
    [1,2,3],
    [0,0,1],
    [1,0,0]
])

B  = np.array([
    [1,1,1],
    [2,2,2],
    [3,3,3]
])

mat_A = np.matrix(A)
mat_B = np.matrix(B)

if __name__ == '__main__':
    print(A*B)
