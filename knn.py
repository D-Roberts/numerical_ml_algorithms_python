#!/usr/bin/env python3
# -- * -- utf-8 -- * --

"""3-nn

Very simple 2 class implementation.

Majority vote by dist.

function interpolation f(x) = y.
"""


import numpy as np

np.random.seed(1234)

def get_dist(x1, x2):
	"""Euclid dist bet 2 vecs

	"""
	return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN(object):
	def __init__(self, X, y, k=3):
		self.X_train = X
		self.y_train = y
		self.k = k

	def predict(self, x):
		"""Make pred/interpolate at new point.

		"""
		M = self.X_train.shape[0]
		x = np.asarray(x)

		dist = []

		for i in range(M):
			d = get_dist(self.X_train[i, :], x)
			dist.append(d)
			# order is equal to indeces i

		# sort and get lowest k indices
		
		topk_ind = np.argsort(dist)[:self.k]
		vote = np.bincount(self.y_train[topk_ind]).argmax()
		print(np.bincount(self.y_train[topk_ind]))

		return vote


def main():

	# Examples serving as unittest
	
	X = np.random.normal(0, 1, size = 40).reshape((20, 2))
	y = np.random.randint(2, size = 20)
	# print(X)
	# print(y)
	
	test = KNN(X, y, 3)
	y_hat = test.predict([1, 2])
	print(y_hat)


if __name__ == '__main__':
	main()
