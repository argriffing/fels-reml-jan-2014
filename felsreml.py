"""
"""
from __future__ import print_function, division, absolute_import


import numpy as np
from numpy.testing import assert_allclose, assert_equal

import scipy
import scipy.linalg

import algopy
from algopy import (log, det, trace, dot, inv, reciprocal, sqrt,
        ones, ones_like, zeros, zeros_like)

LOG2PI = np.log(2 * np.pi)


def assert_square(A):
    assert_equal(len(A.shape), 2)
    assert_equal(A.shape[0], A.shape[1])

def assert_symmetric(A):
    assert_square(A)
    assert_allclose(A, A.T)

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
    assert_allclose(dot(ones(n), A), zeros(n), atol=1e-12)
    assert_allclose(dot(A, ones(n)), zeros(n), atol=1e-12)
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

def ugly_log_pdet(A):
    v = scipy.linalg.eigvals(A)
    v_pos = [x for x in v if abs(x) > 1e-8]
    return log(v_pos).sum()

def log_likelihood(A, x):
    """
    @param A: doubly centered symmetric nxn matrix of rank n-1
    @param x: a vector of data
    """
    #NOTE: this formula is wrong on wikipedia
    assert_symmetric(A)
    n = A.shape[0]
    A_pinv = restored(inv(augmented(A)))
    a = (n-1) * LOG2PI + log_pdet(A)
    b = dot(x, dot(A_pinv, x))
    return -0.5 * (a + b)

def log_likelihoods(A, xs):
    """
    @param A: doubly centered symmetric nxn matrix of rank n-1
    @param xs: vectors of data
    """
    #NOTE: this formula is wrong on wikipedia
    assert_symmetric(A)
    n = A.shape[0]
    A_pinv = restored(inv(augmented(A)))
    a = (n-1) * LOG2PI + log_pdet(A)
    bs = np.array([dot(x, dot(A_pinv, x)) for x in xs])
    return -0.5 * (a + bs)

def differential_entropy(A):
    """
    @param A: doubly centered symmetric nxn matrix of rank n-1
    """
    #NOTE: this formula is wrong on wikipedia
    assert_symmetric(A)
    n = A.shape[0]
    return 0.5 * ((n-1) * (1 + LOG2PI) + log_pdet(A))

def kl_divergence(A, B):
    """
    @param A: doubly centered symmetric nxn matrix of rank n-1
    @param B: doubly centered symmetric nxn matrix of rank n-1
    """
    assert_symmetric(A)
    assert_symmetric(B)
    n = A.shape[0]
    B_pinv = restored(inv(augmented(B)))
    stein_loss = (
            trace(dot(B_pinv, A)) -
            (log_pdet(B_pinv) + log_pdet(A)) - (n-1))
    return 0.5 * stein_loss


def main():

    # define the dimensionality
    n = 4
    H = centering(n)

    # sample a random symmetric matrix
    A_raw = np.random.rand(n, n)
    A = np.dot(A_raw, A_raw.T)
    assert_allclose(H, centering_like(A))

    # check the matrix centering code
    HAH_slow = np.dot(H, np.dot(A, H))
    HAH_fast = doubly_centered(A)
    assert_allclose(HAH_slow, HAH_fast)

    # check the pseudoinversion
    HAH = HAH_slow
    HAH_pinv_direct = scipy.linalg.pinvh(HAH)
    HAH_pinv_clever = restored(inv(augmented(HAH)))
    assert_allclose(HAH_pinv_direct, HAH_pinv_clever)

    # check the logarithm of the pseudo determinant
    logpdet_direct = np.log(scipy.linalg.eigvalsh(HAH)[1:]).sum()
    logpdet_clever = log_pdet(HAH)
    assert_allclose(logpdet_direct, logpdet_clever)

    # check the log likelihood
    print('average log likelihoods:')
    for nsamples in (100, 1000, 10000, 100000):
        X = np.random.multivariate_normal(np.zeros(n), HAH, size=nsamples)
        ll = log_likelihoods(HAH, X).mean()
        print(ll)
    print()

    # check differential entropy
    print('differential entropy:')
    print(differential_entropy(HAH))
    print()

    # make another covariance matrix
    B_raw = A_raw + 0.5 * np.random.rand(n, n)
    B = np.dot(B_raw, B_raw.T)
    HBH = doubly_centered(B)

    # check another expected log likelihood
    print('average log likelihoods:')
    for nsamples in (100, 1000, 10000, 100000):
        X = np.random.multivariate_normal(np.zeros(n), HAH, size=nsamples)
        ll = log_likelihoods(HBH, X).mean()
        print(ll)
    print()

    # check cross entropy
    print('cross entropy from A to B:')
    print(differential_entropy(HAH) + kl_divergence(HAH, HBH))
    print()


if __name__ == '__main__':
    main()

