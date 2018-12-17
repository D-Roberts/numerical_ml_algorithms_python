#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""LR algo.


"""

import numpy as np 
import scipy
from scipy import linalg


class LR(object):
	def __init__(self, k, A):
		# max num steps
		self.ksteps = k
		self.A = A
		self.k = 0

	def lr_decomp(self, A):
		"""Implements LU

		"""
		# multiply L by P
		L, U = linalg.lu(A, permute_l=True)
		return L, U

	def get_P(self, A):

		n = A.shape[1]
		P = np.eye(n, dtype=np.float32)

		# where the column max are in A
		print(A)
		max_loc = np.argmax(np.abs(A), axis=0)
		print(max_loc)

		# make it so that max will move to the diag via multip by P
		for j in range(n):
			row = max(range(j, n), key=lambda i: abs(A[i, j]))
			print(row)
			if j != row:
				tmp = P[j]
				P[j] = P[row]
				P[row] = tmp
		print(P)

		# TODO: fix this and put some checks in place (value error)

		return P

	def lr_decomp1(self, A):
		
		A = np.asarray(A, dtype=np.float32)

		P = self.get_P(A)
		a = np.dot(P, A)
		print(a)

		n = a.shape[1]
		# assume symmetric and square matrix
		u = np.zeros_like(a, dtype=np.float32)
		d = np.zeros_like(a, dtype=np.float32)

		# d is unit lower triangular
		u[0,0] = a[0,0]

		for j in range(n):
			d[j, j] = 1.0
			for i in range(1, j):
				ps = np.sum(np.array([d[i, k] * u[k, j] for k in range(i)]))
				u[i, j] = a[i, j] - ps
				print(u[i, j])

			for i in range(j, n):
				ps1 = np.sum(np.array([d[i, k] * u[k, j] for k in range(j)]))
				
				if u[j, j] == 0:
					raise ValueError("U should not be 0")
				d[i, j] = 1.0 * (a[i, j] - ps1)/u[j, j]
				
				print(d[i, j])

		# not correct because of the absence of permut matrix P

		return P, d, u


	def __call__(self):
		anext = np.array(self.A)
		aprev = np.empty_like(self.A)
		k = 0

		while not np.allclose(aprev, anext) and self.k < self.ksteps:
			aprev= anext
			L, U = self.lr_decomp(aprev)
			anext = np.dot(U, L)
			self.k += 1

		return aprev, anext

	def __repr__(self):
		return('It took {} steps'.format(self.k))


def main():
	a = np.arange(9).reshape(3,3)
	a = (a + a.T)/2
	lr = LR(100, a)
	L, U = lr.lr_decomp(a)

	print(L)
	print(U)
	print(np.dot(L, U))

	preva, finala  = lr()

	print(np.allclose(preva, finala))

	print('preva', preva)
	print('finala', finala)

	print(np.linalg.eig(a)[0])
	print(lr)


	print("self decomp:\n")
	print(lr.lr_decomp1(a))

if __name__ == '__main__':
	main()