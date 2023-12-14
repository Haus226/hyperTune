import numpy as np
from scipy.linalg import cholesky as cho, cho_solve

def backwardSubstituition(U:np.array, b:np.array) -> np.array:
    '''
    For upper triangular matrix
    '''
    N = U.shape[0]
    b_ = np.copy(b)
    b_[N - 1] /= U[N - 1, N - 1]
    for idx in range(N - 2, -1, -1):
        b_[idx] = (b_[idx] - (U[idx, idx + 1:N] @ b_[idx + 1:N])) / U[idx, idx]
    return b_

def forwardSubstituition(L:np.array, b:np.array) -> np.array:
    '''
    For lower triangular matrix
    '''
    # if L.ndim > 1:
    N = L.shape[0]
    b_ = np.copy(b)
    b_[0] /= L[0, 0]
    for idx in range(1, N):
        b_[idx] = (b_[idx] - (L[idx, :idx] @ b_[:idx])) / L[idx, idx]
    return b_



def cholesky(A:np.array) -> np.array:
    A_ = np.copy(A)
    N = A_.shape[0]
    for idx in range(N):
        A_[idx:N, idx] -= (A_[idx:N, :idx] @ A_[idx, :idx].T)
        # if np.abs(A_[idx, idx]) < 1e-10: 
        #     A_[idx:N, idx] = 0
        # else:
        A_[idx:N, idx] /= -np.sqrt(-A_[idx, idx]) if A_[idx, idx] < 0 else np.sqrt(A_[idx, idx])

    return np.tril(A_)

def cholesky_solve(L:np.array, b:np.array) -> np.array:
    return backwardSubstituition(L.T, forwardSubstituition(L, b))
