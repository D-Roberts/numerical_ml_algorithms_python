#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Problem: Principal components.
"""

import numpy as np


np.random.seed(1234)


class LR(object):
    def __init__(self, p=4, n=50, k=2):
        self.p = p
        self.n = n
        self.k = k # num comps

    def dgp(self, p=4, n=50):
        """

        :param size: square cov matrix size
        :return: X, cov_mat
        """

        # pos def cov matrix
        cov_mat = np.random.normal(size=p*p).reshape(p, p)
        cov_mat = np.dot(cov_mat, cov_mat.T)

        # mean vec
        miu = np.random.normal(size=p)

        # generate X; no intercept
        X = np.random.multivariate_normal(miu, cov_mat, size=n)
        return X

    def centerm(self, A):

        if A is None:
            raise ValueError("Matrix should be not none.")

        colmeans = np.mean(A, axis=0)

        return A - colmeans

    def get_cov_mat_eig(self, A):
        """

        :param A: a symmetric square matrix
        :return: eigenvals and right eigenvects
        """
        lams, V = np.linalg.eig(A)
        return lams, V

    def project(self):
        """topkv multip x

        :param x: a sample from original p dim space nXp
        :return: y, sample projected onto kdim pca space nXk
        """
        # self.topk_v is projection matrix of size p X k
        y = np.dot(self.X, self.topk_v)

        return y

    def __call__(self):
        """

        :return: top k principal components
        """
        self.X = self.dgp(self.p, self.n)
        self.Xc = self.centerm(self.X)
        self.covmat = np.cov(self.Xc.T)
        self.lam, self.Q = np.linalg.eig(self.covmat)

        # check first vec and val (first eigenvalue and vector)
        assert np.allclose(self.lam[0] * self.Q[:,0], np.dot(self.covmat, self.Q[:,0]))

        # top k eigenvals and coresp vectors, in desc order
        topk_lam_ind = np.array(np.argsort(self.lam)[-self.k:][::-1])
        self.topk_lam = self.lam[topk_lam_ind]
        self.topk_v = self.Q[:,topk_lam_ind]

        return self.topk_lam, self.topk_v

    def __repr__(self):
        return 'Print topk principal components and project onto subspace'



def main():

    # Example serves as adhoc unittest

    lr = LR(4, 50)
    X = lr.dgp()
    Xc = lr.centerm(X)
    print(np.allclose(np.mean(Xc, axis=0), [0.0, 0.0, 0.0, 0.0]))

    # get eigendecomp of covmat
    covmat = np.cov(X.T)
    print(covmat.shape)
    lams, v = np.linalg.eig(covmat)
    print(lams)
    print(v)
    vinv = np.linalg.inv(v)
    d = np.diag(lams)

    # check decomposition a = QLamQinv
    print(np.allclose(np.dot(np.dot(v, d), vinv), covmat))

    # call and test the object
    lr()
    print(lr)
    projected_scores = lr.project()
    print(projected_scores.shape)



if __name__ == '__main__':
    main()
