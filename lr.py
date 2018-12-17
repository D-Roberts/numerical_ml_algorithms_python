#!/usr/bin/env python3
"""

Linear Regression class matrix formulation with least squares analytical formula for coefs.
"""

import numpy as np 
from numpy import linalg


class LR(object):
	def __init__(self, X=None, y=None):
		self.X = X
		self.y = y

	def __call__(self):
		# perhaps the fit method should be in call dunder
		pass

	def __repr__(self):
		return 'Linear regression with {} features and 1 output'.format(len(self.X[0]))

	def fit(self):
		X = np.array(self.X)
		y = np.array(self.y).reshape(len(self.y), 1)

		# add intercept
		intercept = np.ones(X.shape, dtype=np.float32).reshape((X.shape[0], 1))
		X_train = np.concatenate((intercept, X), axis = 1)

		xx_inv = linalg.inv(np.dot(np.transpose(X_train), X_train))
		print(xx_inv.shape)

		coefs = np.dot(np.dot(xx_inv, np.transpose(X_train)), y)
		return coefs

	def predict(self, coefs, x_new):
		X_predict = np.array(x_new)
		intercept = np.ones(X_predict.shape).reshape((X_predict.shape[0], 1))
		X_predict = np.concatenate((intercept, X_predict), axis = 1)

		return np.dot(X_predict, coefs)

		

def dgp(p):
	pass

def main():

	# inputs_x = [[2], [7], [12]]
	inputs_x = None

	# test assert 
	assert inputs_x is not None

	model = LR(X=inputs_x, y=[1, 2, 3])
	# print(model)

	if inputs_x is not None:
		coefs = model.fit()
		print(coefs)
	else:
		raise RuntimeError('X input must be provided.')

	x_new = [[5]]
	print(model.predict(coefs, x_new))




if __name__ == '__main__':
	main()

