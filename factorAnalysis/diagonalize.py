import numpy as np

def diagonalize(mat):
    '''Makes all the off-diagonal elements equal to
    zero for any given matrix.'''

    # Get the diagonal elements
    diag_ele = np.diag(mat)

    # Create a diagonal identity matirx
    e =  np.eye(mat.shape[0], mat.shape[1])

    # Diagonalize the matrix
    diag = diag_ele * e

    return diag
