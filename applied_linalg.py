"""applied_linalg.py
    Matrix manipulation operators
"""
from numpy import *
from numpy.linalg.linalg import isComplexType
import warnings

"""
    Matrix manipulation function
"""
# Kronecker delta function
def krondelta(i, j):
    if i == j:
        return 1
    else:
        return 0

# Vector to Matrix
def smat(vec): # unittest: OK
    nsize = int((-1 + sqrt(1+8*len(vec))) / 2)
    mat = empty((nsize,nsize))
    pivot = 0
    for i in range(nsize):
        for j in range(i,nsize):
            if i == j:
                if not isclose(vec[pivot].imag, 0):
                    print(f'{vec[pivot]}')
                    raise ValueError('The value is complex.')
                mat[i, i] = vec[pivot]
            else:
                mat[i, j] = mat[j, i] = vec[pivot] / sqrt(2)
            pivot += 1
    return mat


# Matrix sum
def matsum(mat):
    num = len(mat)
    sumval = zeros(mat[0].shape)
    for i in range(num):
        sumval += mat[i]
    return sumval


# Determine (i, j) element from vectorized matrix
# e.g. svec(mat)=[1,2*sqrt(2),3], ijvec(svec(mat), 2) -> (0,1) (i.e., row_elem=1, col_elem=2)
def ijvec(vec, i):
    vecsize = len(vec)
    matsize = int((-1 + sqrt(1+8*vecsize)) / 2)
    pivot = 0
    row = 0
    col = 0
    if i > vecsize - 1:
        raise ValueError(f'Invalid input on second argument \'i\'' +
                         'ijvec->vecsize is {vecsize} but index i is {i}')
    while pivot < i:
        pivot += 1
        if pivot == (col + 1) * matsize - col*(col+1)/2:
            col += 1
            row = col
        else:
            row += 1
    return row, col


# Matrix to Vector
def svec(mat): # unittest: OK
    nsize = len(mat)
    vec = empty(int(nsize*(nsize+1)/2))
    pivot = 0
    for i in range(nsize):
        for j in range(i,nsize):
            if i == j:
                vec[pivot] = mat[j, i]
            else:
                vec[pivot] = sqrt(2) * mat[j, i]
            pivot += 1
    return vec

# Is the matrix M is positive definite?
def is_pos_def(M):
    return all(linalg.eigvals(M) > 0)

# Matrix inner product of two square matrices A, B
def matIP(matA, matB): # unittest: OK
    try:
        return trace(matA@matB)
    except ValueError as e:
        print(e)
        print(matA)
        print(matB)

# Matrix norm (Frobenius norm)
def mat_norm(mat): # unittest: OK
    return sqrt(matIP(mat, mat))

# Matrix l1-norm
def mat_1norm(mat):
    sizen = len(mat)
    norm_sum = 0
    for i in range(sizen):
        for j in range(sizen):
            norm_sum += abs(mat[i, j])
    return norm_sum

# Matrix to the power 1/2
def mat_1half(A):
    eig_A_diag_sqrt = diag(sqrt(linalg.eig(A)[0]))
    P = linalg.eig(A)[1]
    invP = linalg.inv(P)
    return P@eig_A_diag_sqrt@invP

# Jordan matrix product (symmetrize product)
def JProd_mat(matA, matB):
    return (matA @ matB + matB @ matA) / 2 

# Matrix to the power -1/2
def mat_minus1half(A):
    eig_A_diag_frac_sqrt = diag(1/sqrt(linalg.eigvals(A)))
    P = linalg.eig(A)[1]
    invP = linalg.inv(P)
    return P@eig_A_diag_frac_sqrt@invP

"""
    Jordan algebra computation functions
"""

# R matrix
def R(dim):
    RR = - eye(dim)
    RR[0, 0] = 1
    return RR

# Arrow matrix
def Arw(vec):
    nsize = len(vec)
    return block([[vec[0], vec[1:nsize]],
                  [array([vec[1:nsize]]).T, vec[0]*eye(nsize-1)]])

# Jordan product
def JProd(vec1, vec2):
    vecsize = len(vec1)
    return block([vec1@vec2,vec1[0]*vec2[1:vecsize]+vec2[0]*vec1[1:vecsize]])

# Unit vector whose only 1st element is 1
def unitvec(dim):
    vec = zeros(dim)
    vec[0] = 1
    return vec

# Determinant of vector
def detvec(vec):
    return vec[0]**2 - linalg.norm(vec[1:len(vec)])**2

# Vector inverse
''' Will be removed for computation time '''
''' Recommend to use vec_power(vec, power) '''
'''
def invvec(vec):
    R = -eye(len(vec))
    R[0,0] = 1
    return R@vec / detvec(vec)
'''

# Vector to the power 1/2, -1/2 or -1
def vec_power(vec, power):
    if is_int_cone(vec) == False:
        print(vec)
        raise ValueError('Input vector is not a interior point of cone.')
    nsize = len(vec)
    # Eigenvalues for the vector
    eigval1 = vec[0] + linalg.norm(vec[1:nsize])
    eigval2 = vec[0] - linalg.norm(vec[1:nsize])
    # Jordan frame of the vector
    c1 = 1/2*block([1, vec[1:nsize]/linalg.norm(vec[1:nsize])])
    c2 = 1/2*block([1, -vec[1:nsize]/linalg.norm(vec[1:nsize])])
    return (eigval1**power)*c1 + (eigval2**power)*c2

# Is the given vector is a interior point of second-order cone?
def is_int_cone(vec):
    if len(vec) == 0 or vec[0] > linalg.norm(vec[1:len(vec)]):
        return True
    else:
        return False
