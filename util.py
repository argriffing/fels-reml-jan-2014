"""
"""

import numpy as np

from algopy import ones_like, log, det


def assert_square(A):
    #assert_equal(len(A.shape), 2)
    #assert_equal(A.shape[0], A.shape[1])
    pass

def assert_symmetric(A):
    assert_square(A)
    #assert_allclose(A, A.T)

def centering(n):
    return np.identity(n) - np.ones((n, n)) / n

def centering_like(A):
    assert_square(A)
    n = A.shape[0]
    return np.identity(n) - np.ones_like(A) / n

def doubly_centered(A):
    assert_square(A)
    S = A.copy()
    S -= np.mean(A, axis=0)[np.newaxis, :]
    S -= np.mean(A, axis=1)[:, np.newaxis]
    S += np.mean(A)
    return S

def augmented(A):
    """
    @param A: doubly centered symmetric nxn matrix of rank n-1
    @return: full rank symmetric matrix whose rows sum to 1
    """
    assert_square(A)
    n = A.shape[0]
    #assert_allclose(dot(ones(n), A), zeros(n), atol=1e-12)
    #assert_allclose(dot(A, ones(n)), zeros(n), atol=1e-12)
    return A + ones_like(A) / n

def restored(A):
    """
    @param A: full rank symmetric matrix whose rows sum to 1
    @return: a doubly centered but otherwise full rank matrix
    """
    assert_symmetric(A)
    n = A.shape[0]
    return A - ones_like(A) / n

def log_pdet(A):
    """
    @param A: doubly centered symmetric nxn matrix of rank n-1
    @return: log of pseudo determinant
    """
    return log(det(augmented(A)))

