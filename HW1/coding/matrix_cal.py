import numpy as np

A_mat = np.matrix([
    [1,2,3],
    [3,4,5],
    [1,2,2]])

B_mat = np.matrix([
    [1,2,3],
    [1,2,3],
    [1,2,3]])

C_mat = np.matrix([
    [1,1,0],
    [1,0,1],
    [0,1,1]])        

if __name__ == '__main__':

    print((A_mat*B_mat))
    print()
    print((B_mat*C_mat))
