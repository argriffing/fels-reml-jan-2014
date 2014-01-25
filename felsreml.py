"""
"""
from __future__ import print_function, division, absolute_import

from functools import partial

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import scipy
import scipy.linalg
import scipy.optimize
import scipy.stats

import algopy
from algopy import (exp, log, det, trace, dot, inv, reciprocal, sqrt,
        ones, ones_like, zeros, zeros_like, diag)

LOG2PI = np.log(2 * np.pi)


def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))


def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))


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

def cross_entropy(A, B):
    # return differential_entropy(A) + kl_divergence(A, B)
    # Note that trace(dot(A, B)) == sum(A * B)
    assert_symmetric(A)
    assert_symmetric(B)
    n = A.shape[0]
    B_pinv = restored(inv(augmented(B)))
    #return 0.5 * ((n-1) * LOG2PI + trace(dot(B_pinv, A)) + log_pdet(B))
    return 0.5 * ((n-1) * LOG2PI + (B_pinv * A).sum() + log_pdet(B))


def centered_tree_covariance(B, nleaves, v):
    """
    @param B: rows of this unweighted incidence matrix are edges
    @param nleaves: number of leaves
    @param v: vector of edge variances
    """
    #TODO: track the block multiplication through the schur complement
    W = diag(reciprocal(v))
    L = dot(B.T, dot(W, B))
    #print('full laplacian matrix:')
    #print(L)
    #print()
    nvertices = v.shape[0]
    ninternal = nvertices - nleaves
    Laa = L[:nleaves, :nleaves]
    Lab = L[:nleaves, nleaves:]
    Lba = L[nleaves:, :nleaves]
    Lbb = L[nleaves:, nleaves:]
    L_schur = Laa - dot(Lab, dot(inv(Lbb), Lba))
    L_schur_pinv = restored(inv(augmented(L_schur)))
    #print('schur laplacian matrix:')
    #print(L_schur)
    #print()
    #print('pinv of schur laplacian matrix:')
    #print(L_schur_pinv)
    #print()
    return L_schur_pinv


def cross_entropy_trees(B, nleaves, va, vb):
    """
    Internal vertices are expected to follow leaves.
    Rows of B correspond to edges, and columns of B correspond to vertices.
    The test variances are last.
    @param B: rows of this unweighted incidence matrix are edges
    @param nleaves: number of leaves
    @param va: vector of reference edge variances
    @param vb: vector of test edge variances
    """

    # Get the number of edges, vertices, and internal vertices,
    # and check the shapes of the input arrays.
    assert_equal(len(va.shape), 1)
    assert_equal(len(vb.shape), 1)
    assert_equal(len(B.shape), 2)
    nedges = B.shape[0]
    nvertices = B.shape[1]
    assert_equal(nvertices-1, nedges)
    ninternal = nvertices - nleaves
    assert_equal(va.shape[0], nedges)
    assert_equal(vb.shape[0], nedges)

    # Get the centered covariance matrices and compute the cross entropy.
    A = centered_tree_covariance(B, nleaves, va)
    B = centered_tree_covariance(B, nleaves, vb)
    return cross_entropy(A, B)


def demo_covariances():

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
    print(cross_entropy(HAH, HBH))
    print()

def prune_cherry(v1, v2, x1, x2):
    """
    @param v1: terminal branch length of the first leaf
    @param v2: terminal branch length of the second leaf
    @param x1: observed data at the first leaf
    @param x2: observed data at the second leaf
    @return: ll, delta, x12
    """
    v12 = v1 + v2
    sigma = sqrt(v12)
    delta = v1*v2 / v12
    x12 = (v2*x1 + v1*x2) / v12
    ll = scipy.stats.norm.logpdf(x2 - x1, loc=0, scale=sigma)
    return ll, delta, x12

def custom_pruning(v, x):
    """
    Do Felsenstein REML pruning using a hardcoded tree.
    Branch lengths are variances.
    Return the log likelihood.
    @param v: branch lengths
    @param x: data vector
    @return: ll
    """
    nleaves = x.shape[0]
    ll01, delta01, x01 = prune_cherry(v[0], v[1], x[0], x[1])
    ll23, delta23, x23 = prune_cherry(v[2], v[3], x[2], x[3])
    v45 = v[4] + delta01 + delta23
    sigma45 = sqrt(v45)
    ll45 = scipy.stats.norm.logpdf(x23 - x01, loc=0, scale=sigma45)
    pruning_adjustment = 0.5 * log(nleaves)
    return pruning_adjustment + ll01 + ll23 + ll45

def custom_centered_cov(v):
    """
    Hardcoded covariance matrix relating leaves.
    Constructed using an arbitrary root and then doubly centered.
    """
    C = np.array([
        [v[0], 0, 0, 0],
        [0, v[1], 0, 0],
        [0, 0, v[2]+v[4], v[4]],
        [0, 0, v[4], v[3]+v[4]],
        ], dtype=float)
    return doubly_centered(C)


def demo_trees():

    # six vertices
    # five edges
    nvertices = 6
    nleaves = 4
    nedges = 5
    log_va = np.random.randn(nedges)
    va = np.exp(log_va)
    vb = np.exp(log_va + 0.5 * np.random.randn(nedges))

    # The B matrix defines the tree shape.
    # Each row of B is an edge.
    # The first four columns correspond to leaf vertices.
    B = np.array([
        [1, 0, 0, 0, -1, 0],
        [0, 1, 0, 0, -1, 0],
        [0, 0, 1, 0, 0, -1],
        [0, 0, 0, 1, 0, -1],
        [0, 0, 0, 0, 1, -1],
        ], dtype=float)

    # Compute the centered coveriance matrices
    # corresponding to the reference and test branch lengths.
    LA = centered_tree_covariance(B, nleaves, va)
    LB = centered_tree_covariance(B, nleaves, vb)

    # Hardcoded created centered tree covariance.
    ccova = custom_centered_cov(va)
    ccovb = custom_centered_cov(vb)
    assert_allclose(LA, ccova)
    assert_allclose(LB, ccovb)

    print('incidence matrix:')
    print(B)
    print()
    print('reference branch lengths:')
    print(va)
    print()
    print('centered covariance matrix for the reference branch lengths:')
    print(LA)
    print()
    print('test branch lengths:')
    print(vb)
    print()
    print('cross entropy:')
    print(cross_entropy_trees(B, nleaves, va, vb))
    print()

    # Sample a bunch of data vectors from the tree
    # using the reference branch lengths,
    # and directly applying the univariate conditional normal distribution
    # of difference across branches associated with Brownian motion.
    # Center each data vector.
    # Use a an arbitrary root.
    print('sampling a bunch of data from distribution A...')
    vsqrt = np.sqrt(va)
    xs = []
    nsamples = 1000
    for i in range(nsamples):
        y = np.zeros(nvertices)
        y[4] = 0
        y[5] = np.random.normal(y[4], vsqrt[4])
        y[0] = np.random.normal(y[4], vsqrt[0])
        y[1] = np.random.normal(y[4], vsqrt[1])
        y[2] = np.random.normal(y[5], vsqrt[2])
        y[3] = np.random.normal(y[5], vsqrt[3])
        x = y[:nleaves]
        x -= x.mean()
        xs.append(x)
    X = np.array(xs)
    print()
    print('sample data covariance matrix:')
    print(dot(X.T, X)/nsamples)
    print()

    # check the log likelihood using matrix algebra
    print('average log likelihoods using matrix algebra,')
    print('computed using parameters B for data sampled from parameters A:')
    ll_average_matrix = log_likelihoods(LB, xs).mean()
    print(ll_average_matrix)
    print()

    # check the log likelihood using felsenstein pruning
    print('average log likelihoods using felsenstein pruning,')
    print('computed using parameters B for data sampled from parameters A:')
    ll_average_pruning = np.array([custom_pruning(vb, x) for x in xs]).mean()
    print(ll_average_pruning)
    print()

    d = ll_average_pruning - ll_average_matrix
    print('difference of log likelihoods:')
    print(d)
    print()
    print('exp of difference of log likelihoods:')
    print(exp(d))
    print()

    f = partial(cross_entropy_trees, B, nleaves, va)
    g = partial(eval_grad, f)
    h = partial(eval_hess, f)
    G = g(vb)
    H = h(vb)
    print('gradient of cross entropy:')
    print(G)
    print()
    print('hessian of cross entropy:')
    print(H)
    print()
    print('eigenvalues of hessian of cross entropy:')
    print(scipy.linalg.eigvalsh(H))
    print()
    
    print('minimizing cross entropy...')
    result = scipy.optimize.minimize(f, vb, jac=g, hess=h, method='trust-ncg')
    xopt = result.x
    F = f(xopt)
    G = g(xopt)
    H = h(xopt)
    print()
    print('branch lengths at minimum:')
    print(xopt)
    print()
    print('minimum cross entropy:')
    print(F)
    print()
    print('gradient at minimum:')
    print(G)
    print()
    print('hessian at minimum:')
    print(H)
    print()
    print('eigenvalues of hessian at minimum:')
    print(scipy.linalg.eigvalsh(H))
    print()


def demo_small_tree():
    nvertices = 3
    nleaves = 2
    nedges = 2
    v = np.exp(np.random.randn(2))
    v1, v2 = v.tolist()

    # define the shape of the tree
    B = np.array([
        [1, 0, -1],
        [0, 1, -1],
        ], dtype=float)

    # construct the centered covariance matrix using matrix algebra
    L = centered_tree_covariance(B, nleaves, v)

    # construct the centered covariance matrix using direct methods
    C = np.array([
        [v1, 0],
        [0, v2],
        ], dtype=float)
    C = doubly_centered(C)
    assert_allclose(L, C)

    # sample centered data
    vsqrt = np.sqrt(v)
    xs = []
    nsamples = 1000
    for i in range(nsamples):
        x = np.zeros(nleaves)
        x[0] = np.random.normal(0, vsqrt[0])
        x[1] = np.random.normal(0, vsqrt[1])
        x -= x.mean()
        xs.append(x)

    # check the log likelihood using matrix algebra
    print('average log likelihoods using matrix algebra')
    ll_average_matrix = log_likelihoods(L, xs).mean()
    print(ll_average_matrix)
    print()

    # check the log likelihood using felsenstein pruning
    print('average log likelihoods using felsenstein pruning')
    lls = []
    for x in xs:
        ll = scipy.stats.norm.logpdf(x[1] - x[0], loc=0, scale=sqrt(v1 + v2))
        pruning_adjustment = 0.5 * log(nleaves)
        lls.append(pruning_adjustment + ll)
    ll_average_pruning = np.mean(lls)
    print(ll_average_pruning)
    print()

    d = ll_average_pruning - ll_average_matrix
    print('difference of log likelihoods:')
    print(d)
    print()
    print('exp of difference of log likelihoods:')
    print(exp(d))
    print()


def demo_medium_tree():
    nvertices = 4
    nleaves = 3
    nedges = 3
    v = np.exp(np.random.randn(nedges))

    # define the shape of the tree
    B = np.array([
        [1, 0, 0, -1],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        ], dtype=float)

    # construct the centered covariance matrix using matrix algebra
    L = centered_tree_covariance(B, nleaves, v)

    # construct the centered covariance matrix using direct methods
    C = np.array([
        [v[0], 0, 0],
        [0, v[1], 0],
        [0, 0, v[2]],
        ], dtype=float)
    C = doubly_centered(C)
    assert_allclose(L, C)

    # sample centered data
    vsqrt = np.sqrt(v)
    xs = []
    nsamples = 1000
    for i in range(nsamples):
        x = np.zeros(nleaves)
        x[0] = np.random.normal(0, vsqrt[0])
        x[1] = np.random.normal(0, vsqrt[1])
        x[2] = np.random.normal(0, vsqrt[2])
        x -= x.mean()
        xs.append(x)

    # check the log likelihood using matrix algebra
    print('average log likelihoods using matrix algebra')
    ll_average_matrix = log_likelihoods(L, xs).mean()
    print(ll_average_matrix)
    print()

    # check the log likelihood using felsenstein pruning
    print('average log likelihoods using felsenstein pruning')
    lls = []
    for x in xs:
        ll01, d01, x01 = prune_cherry(v[0], v[1], x[0], x[1])
        ll = scipy.stats.norm.logpdf(x[2] - x01, loc=0, scale=sqrt(v[2] + d01))
        pruning_adjustment = 0.5 * log(nleaves)
        lls.append(pruning_adjustment + ll + ll01)
    ll_average_pruning = np.mean(lls)
    print(ll_average_pruning)
    print()

    d = ll_average_pruning - ll_average_matrix
    print('difference of log likelihoods:')
    print(d)
    print()
    print('exp of difference of log likelihoods:')
    print(exp(d))
    print()


def check_multivariate_normal_log_likelihood():

    # sample a full rank covariance matrix and some data
    n = 5
    x = np.random.randn(n)
    X = np.random.randn(n, n)
    A = dot(X.T, X)

    # get the log likelihood directly
    a = n * LOG2PI + np.linalg.slogdet(A)[1]
    b = dot(x, dot(inv(A), x))
    ll_direct = -0.5 * (a + b)

    # get the log likelihood using scipy multivariate distributions
    ll_scipy = scipy.stats.multivariate_normal.logpdf(
            x, mean=np.zeros(n), cov=A)

    print('log likelihood computed directly:')
    print(ll_direct)
    print()

    print('log likelihood computed using scipy:')
    print(ll_scipy)
    print()


def main():
    demo_trees()
    #demo_small_tree()
    #demo_medium_tree()
    #check_multivariate_normal_log_likelihood()


if __name__ == '__main__':
    main()

