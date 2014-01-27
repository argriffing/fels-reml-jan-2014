"""
Look for instances of non-convexity.

"""
from __future__ import print_function, division, absolute_import

from functools import partial
import itertools
import random

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import scipy.optimize

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

# TODO copypasted
def cross_entropy(A, B):
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
    Lbb_inv = inv(Lbb)
    try:
        Lbb_inv_Lba = dot(Lbb_inv, Lba)
    except ValueError:
        print(L)
        print('Laa:', Laa, Laa.shape, Laa.dtype)
        print('Lab:', Lab, Lab.shape, Lab.dtype)
        print('Lba:', Lba, Lba.shape, Lba.dtype)
        print('Lbb_inv:', Lbb_inv, Lbb_inv.shape, Lbb_inv.dtype)
        raise
    L_schur = Laa - dot(Lab, Lbb_inv_Lba)
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
        #nleaves = 2
        nleaves = random.randrange(3, 10)
        ninternal = nleaves - 2
        nvertices = nleaves + ninternal
        nedges = nvertices - 1

        # Sample the shape of the tree.
        B = sample_tree(nleaves)

        # Sample the reference branch lengths.
        vref = np.exp(np.random.randn(nedges))

        # Sample two vectors of branch lengths.
        va = np.exp(np.random.randn(nedges))
        vb = np.exp(np.random.randn(nedges))

        # Sample a convex combination parameter.
        t = np.random.rand()

        # Sample the point between the two points.
        vc = t * vb + (1 - t) * va

        # Compute the cross entropy at the three points.
        fa = cross_entropy_trees(B, nleaves, vref, va)
        fb = cross_entropy_trees(B, nleaves, vref, vb)
        fc = cross_entropy_trees(B, nleaves, vref, vc)

        # Check the quasi-convexity condition.
        quasiconvexity_fail = False
        if fc > max(fa, fb):
            quasiconvexity_fail = True
            print('the quasi-convexity condition is violated')

        # Check the convexity condition.
        convexity_fail = False
        #if fc > t * fb + (1 - t) * fa:
            #convexity_fail = True
            #print('the convexity condition is violated')

        # Report some stuff if a convexity condition is violated.
        if convexity_fail or quasiconvexity_fail:
            print('iteration:', nsamples + 1)
            print('number of leaves:', nleaves)
            print('incidence matrix:')
            print(B)
            print('reference branch lengths:')
            print(vref)
            print('mixing parameter:')
            print(t)
            print('branch lengths va, vb, vc:')
            print(va)
            print(vb)
            print(vc)
            print('function values fa, fb, fc:')
            print(fa)
            print(fb)
            print(fc)
            print()


if __name__ == '__main__':
    main()

