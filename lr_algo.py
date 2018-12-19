#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""LR algo.


"""

import numpy as np
import scipy
from scipy import linalg
from copy import copy


class LR(object):
    def __init__(self, k, A):
        # max num steps
        self.ksteps = k
        self.A = A
        self.k = 0

    def lr_decomp(self, A):
        """Implements LU from scipy.

        """
        # multiply L by P
        P, L, U = scipy.linalg.lu(A, permute_l=False)
        return P, L, U

    def get_P(self, A):
        """Get Permutation matrix for row permutation of A.

        Steps: get col max of A.
        If that is not on diag, swap rows so that max is on diag.

        :param A:
        :return:
        """
        n = A.shape[1]
        P = np.eye(n, dtype=np.float32)

        # make it so that max will move to the diag via multip by P
        for j in range(n-1):
            row_max = j + np.argmax(np.abs(A[j:, j]), axis=0)

            # under the elem of diag dominant elems but then
            # reset index from j on
            # print(row_max)
            # interchange entire row j with row_max
            # print(Pnew)
            tmp = P[j, :].copy()
            P[j, :] = P[row_max, :]
            P[row_max, :] = tmp

        return P

    def lr_decomp1(self, A):
        """

        :param A: square matrix
        :return: P, L, U
        """

        A = np.asarray(A, dtype=np.float32)
        P = self.get_P(A)
        # do decomposition on PA
        a = np.dot(P, A)

        n = a.shape[1]
        # assume symmetric and square matrix
        u = np.zeros((n, n), dtype=np.float32)
        d = np.zeros((n, n), dtype=np.float32)

        # d is unit lower triangular
        for j in range(n):
            d[j, j] = 1.0
            for i in range(j+1):
                ps = np.sum(np.array([d[i, k] * u[k, j] for k in range(i)]))
                u[i, j] = a[i, j] - ps

            for i in range(j, n):
                ps1 = np.sum(np.array([d[i, k] * u[k, j] for k in range(j)]))
                d[i, j] = (a[i, j] - ps1) / u[j, j]

        # equivalence to L = Pd
        L = np.dot(P, d)

        return L, u

    def __call__(self):

        anext = np.array(self.A)
        aprev = np.empty_like(self.A)

        while not np.allclose(aprev, anext) and self.k < self.ksteps:
            aprev = anext
            L, U = self.lr_decomp1(aprev)
            anext = np.dot(U, L)
            self.k += 1

        return anext

    def __repr__(self):
        return ('It took {} steps'.format(self.k))


def main():
     # a = np.arange(9).reshape(3, 3)
     # print(a)

     a = np.array([[1, 2, 4], [2, 1, 3], [3, 2, 4]])

     # a = np.array([[7, 3, -1, 2],
     #  [3, 8, 1, -4],
     #  [-1, 1, 4, -1],
     #  [2, -4, -1, 6]])

     lr = LR(100, a)
     L, U = lr.lr_decomp1(a)
     # print(L)
     # P = lr.get_P(a)
     # print(P)


     # P1, L1, U1 = lr.lr_decomp(a)
     # print(L1)
     # print(P1)
     #
     # # and algo for eignevals
     # # that will be on the diagonal
     anext = lr()
     #
     print(anext)
     print(np.linalg.eigvals(a))

     # works


if __name__ == '__main__':
    main()
