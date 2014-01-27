"""
Sample tree shapes without edge weights.

"""
from __future__ import print_function, division, absolute_import

import random

import numpy as np
from numpy.testing import assert_equal


def sample_tree(nleaves):
    """
    Return an incidence matrix, where each row represents an edge.
    """
    if nleaves < 2:
        raise ValueError
    ninternal = nleaves - 2
    nvertices = ninternal + nleaves
    nedges = nvertices - 1
    first_edge = (0, 1)
    arr = [first_edge]
    next_leaf = 2
    next_internal = nleaves
    for i in range(nleaves-2):

        # pick an edge uniformly at random
        idx = random.randrange(len(arr))

        # swap the picked with the last edge in the list
        arr[-1], arr[idx] = arr[idx], arr[-1]

        # pop the last edge off of the list, getting the vertex indices
        va, vb = arr.pop()

        # define the two new vertices to add
        vleaf = next_leaf
        vinternal = next_internal
        next_leaf += 1
        next_internal += 1

        # add the new internal vertex to the middle of the deleted edge
        arr.append((va, vinternal))
        arr.append((vinternal, vb))

        # connect the new internal vertex to the new leaf vertex
        arr.append((vinternal, vleaf))

    # check that we used the right number of leaves and internal nodes
    assert_equal(next_leaf, nleaves)
    assert_equal(next_internal, nvertices)

    # construct the incidence matrix
    B = np.zeros((nedges, nvertices))
    for i, (va, vb) in enumerate(arr):
        B[i, va] = 1
        B[i, vb] = -1
    
    # return the incidence matrix
    return B


def main():
    print(sample_tree(5))


if __name__ == '__main__':
    main()

