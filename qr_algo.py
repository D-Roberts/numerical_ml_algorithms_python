#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""QR algo.

QR algo to find eigenvalues. Preceded by LR.

Employs QR decomposition. Gram-Schmidt algo implemented.
https://en.wikipedia.org/wiki/QR_decomposition

"""

import numpy as np 


def inner_prod(v, w):
	"""Assume real vectors.

	"""
	return np.dot(np.transpose(v), w)

def project(u, a):

	return (inner_prod(u, a)/inner_prod(u, u)) * u


class QR(object):
	def __init__(self, A):
		self.A = A

	def qr_decomp(self, A):

		if A is None:
			raise ValueError("A must be not None")

		# Q, R = np.linalg.qr(A)
		# Q is orthogonal and R is upper triangular

		m = A.shape[0]
		n = A.shape[1]
		
		uj = []
		uj.append(A[:, 0].reshape(m, 1))

		ej = []
		ej.append(uj[0]/np.linalg.norm(uj[0]))
		
		for k in range(1, n):
			partial_sum = np.zeros(A.shape[1]).reshape(m, 1)
			# print("partial sum", partial_sum.shape)
			for i in range(k):
				partial_sum += project(uj[i], A[:,k].reshape(m, 1))
			uj.append(A[:,k].reshape(m, 1) - partial_sum)
			ej.append(uj[k]/np.linalg.norm(uj[k]))

		Q = ej
		print('Q is', Q)

		R = np.zeros((n, n))
		for i in range(n):
			for j in range(i, n):
				R[i,j] = inner_prod(ej[j], A[:, i])

		return np.array(Q).reshape(3, 3), R

	def __call__(self):
		# Starting vals come from qr decomp
		Q0, R0 = np.linalg.qr(self.A)
		Qprev, Rprev = Q0, R0

		print("Check if Q0 contains eigenvectors", np.dot(self.A, Q0[:,0]))
		print("print product first eigenval and A", -3.80353437 * Q0[:,0])

		ksteps = 1000 

		for k in range(ksteps):
			A_next = np.dot(Rprev, Qprev)
			Qprev, Rprev = np.linalg.qr(A_next)

		return A_next


def main():

	a = np.arange(9).reshape((3,3))
	# ttransform to symmetrical
	a = (a + a.T)/2
	print(a)

	qr_obj = QR(a)
	Q, R = qr_obj.qr_decomp(a)

	# to check qr decomp - nope, sth not right
	print(np.allclose(a, np.dot(Q, R))) 

	# Check eigenvals of qr algo
	print(qr_obj())
	print(np.linalg.eig(a)[0]) # still the last not quite the same
	
	

if __name__ == '__main__':
	main()
