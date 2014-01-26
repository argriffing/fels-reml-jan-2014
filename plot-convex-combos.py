
import numpy as np

from numpy.testing import assert_equal

import matplotlib
#matplotlib.use('gtkagg')
#matplotlib.use('qtagg')
matplotlib.use('agg')

import matplotlib.pyplot as plt

from algopy import diag, dot, inv, reciprocal

from util import (assert_square, assert_symmetric, centering, centering_like,
        doubly_centered, augmented, restored, log_pdet)

LOG2PI = np.log(2 * np.pi)


#XXX copypasted
def cross_entropy(A, B):
    # return differential_entropy(A) + kl_divergence(A, B)
    # Note that trace(dot(A, B)) == sum(A * B)
    assert_symmetric(A)
    assert_symmetric(B)
    n = A.shape[0]
    B_pinv = restored(inv(augmented(B)))
    #return 0.5 * ((n-1) * LOG2PI + trace(dot(B_pinv, A)) + log_pdet(B))
    return 0.5 * ((n-1) * LOG2PI + (B_pinv * A).sum() + log_pdet(B))


#XXX copypasted
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


#XXX copypasted
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

    directions = []
    for i in range(nedges):
        d = np.zeros(nedges)
        d[i] = 1
        directions.append(d)
    d = np.ones(nedges)
    directions.append(d)
    for i, d in enumerate(directions):

        x = np.linspace(0, 1, 101)
        y = []
        for t in x:
            combo = vb + t * d * (va - vb)
            #combo = vb.copy()
            #combo[i] = t * va[i] + (1 - t) * vb[i]
            ci = cross_entropy_trees(B, nleaves, va, combo)
            y.append(ci)

        plt.plot(x, y)
        filename = 'foo-%s.png' % i
        plt.savefig(filename)

    # Compute the centered coveriance matrices
    # corresponding to the reference and test branch lengths.
    #LA = centered_tree_covariance(B, nleaves, va)
    #LB = centered_tree_covariance(B, nleaves, vb)


if __name__ == '__main__':
    main()
