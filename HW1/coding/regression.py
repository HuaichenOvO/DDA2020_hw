import numpy as np
from numpy import linalg as la


"""
Input type
:X type: numpy.ndarray
:y type: numpy.ndarray

Return type
:w type: numpy.ndarray
:XT type: numpy.ndarray
:InvXTX type: numpy.ndarray

"""
# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_119010176(X, y):
    # your code goes here
    mat_X = np.matrix(X)
    mat_y = np.matrix(y)
    XTX = mat_X.T * mat_X

    InvXTX = np.matrix([])
    XT = np.matrix([])
    w = np.matrix([])
    
    # none ridge case
    if (la.det(XTX)>0):

        InvXTX = la.inv(XTX)
        XT = mat_X.T
        w = InvXTX*XT*mat_y

    # ridge case
    else:
        mat_lambda = np.matrix([
            [0,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1]
        ])

        InvXTX = la.inv(XTX + mat_lambda)
        XT = mat_X.T
        w = InvXTX*XT*mat_y

    # return in this order
    return w.getA(), XT.getA(), InvXTX.getA()


# if __name__ == '__main__':
#     X = np.array([
#         [1,2],
#         [4,3],
#         [5,6],
#         [3,8],
#         [9,10]
#         ])

#     y = np.array([
#         [-1],
#         [0],
#         [1],
#         [0],
#         [0]
#     ])

#     print(A1_119010176(X,y))