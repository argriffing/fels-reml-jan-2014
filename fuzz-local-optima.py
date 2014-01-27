"""
Look for local optima that are not global optima.

"""
from __future__ import print_function, division, absolute_import

from functools import partial
import itertools
import random

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import scipy
import scipy.linalg
import scipy.optimize
import scipy.stats

import algopy
from algopy import (exp, log, det, trace, dot, inv, reciprocal, sqrt,
        ones, ones_like, zeros, zeros_like, diag)

from util import (assert_square, assert_symmetric, centering, centering_like,
        doubly_centered, augmented, restored, log_pdet)

from sampletrees import sample_tree

LOG2PI = np.log(2 * np.pi)


def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))

def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))

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
    try:
        L_schur = Laa - dot(Lab, dot(inv(Lbb), Lba))
    except ValueError:
        print(L)
        raise
    L_schur_pinv = restored(inv(augmented(L_schur)))
    #print('schur laplacian matrix:')
    #print(L_schur)
    #print()
    #print('pinv of schur laplacian matrix:')
    #print(L_schur_pinv)
    #print()
    return L_schur_pinv

def cross_entropy_trees_log_lengths(B, nleaves, log_va, log_vb):
    return cross_entropy_trees(B, nleaves, exp(log_va), exp(log_vb))

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



def main():
    nsamples = 0
    largest_error = None
    for nsamples in itertools.count():

        # Pick a random number of leaves in the unrooted bifurcating tree.
        # The number of leaves determines the total number of nodes.
        nleaves = random.randrange(3, 10)
        ninternal = nleaves - 2
        nvertices = nleaves + ninternal
        nedges = nvertices - 1

        # Sample the shape of the tree.
        B = sample_tree(nleaves)

        # Sample branch log lengths for the reference covariance matrix.
        log_va = np.random.randn(nedges)
        va = exp(log_va)

        # Sample branch log lengths for the initial guess.
        log_vb = np.random.randn(nedges)

        # Define the cross entropy function and gradient and hessian.
        f = partial(cross_entropy_trees_log_lengths, B, nleaves, log_va)
        g = partial(eval_grad, f)
        h = partial(eval_hess, f)

        # Use a trust region conjugate gradient search for a local optimum.
        result = scipy.optimize.minimize(
                f, log_vb, jac=g, hess=h, method='trust-ncg')
        log_xopt = result.x
        xopt = exp(log_xopt)

        # Compute the branch length error.
        error = np.linalg.norm(xopt - va)

        # If the error is large then report some error.
        if largest_error is None or error > largest_error:
            F = f(log_xopt)
            G = g(log_xopt)
            H = h(log_xopt)
            largest_error = error
            print('new largest error:', largest_error)
            print('iteration:', nsamples + 1)
            print('number of leaves:', nleaves)
            print('incidence matrix:')
            print(B)
            print('true branch lengths:')
            print(va)
            print('locally optimal branch lengths:')
            print(xopt)
            print()


if __name__ == '__main__':
    main()

