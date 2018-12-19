#!/usr/bin/env python3
"""

LSA example.
"""

import numpy as np
from numpy import linalg
import sklearn
from sklearn import metrics

def main():
	X = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0],[1, 1, 0]]
	Xt = np.transpose(X)
	rm = np.dot(X, Xt)
	print(rm)
	dm = np.dot(Xt, X)
	# eigenval and right eigenv
	e, u = linalg.eig(rm)
	print(e)
	print(u)
	print(np.diag(e))
	ut = np.transpose(u)
	test_rm = np.dot(np.dot(u,np.diag(e)),ut)
	U = np.transpose(ut)
	e1, Vt = linalg.eig(dm)
	print(e1)
	print(np.dot(U,np.diag(e)))
	
	# pick top k=2 eignvals and eigenvectors
	lam2 = e1[:2]
	print(lam2)
	# first doc right vect
	Vk = Vt[:2]
	print(Vk)
	dmat = np.diag(lam2)
	dmatinv = np.linalg.inv(dmat)
	# pretend new query is a vector from initial mat
	t = X[0]
	print(t)
	# project onto the term embedding space
	that = np.dot(dmatinv,np.dot(Vk,t))
	print(that)
	cs = sklearn.metrics.pairwise.cosine_similarity(that.reshape(1,-1), that.reshape(1,-1))
	print(cs)

if __name__ == '__main__':
	main()
