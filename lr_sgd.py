#!/usr/bin/env python3
"""

Linear regression algorithm trained with SGD.

"""

import numpy as np 

def gradient(beta, X, y):
	Xt = np.transpose(X)
	
	return (1.0/X.shape[0]) * (2 * np.dot((np.dot(np.transpose(X), X)), beta)- 2 * np.dot(Xt, y))

def loss(preds, y):
	err = preds - y

	return (1.0/preds.shape[0]) * np.dot(np.transpose(err), err)


def add_intercept(X):

		intercept = np.ones(X.shape[0], dtype=np.float32).reshape((X.shape[0], 1))

		return np.concatenate((intercept, X), axis=1)


class LR_SGD(object):
	def __init__(self, X, y, epochs, lr, tol=1e-5):
		self.X = X
		self.y = y
		self.epochs = epochs
		# batch is all data
		self.lr = lr
		self.tol = tol


	def train(self):
		X_train = add_intercept(self.X)
		
		# initialize betas
		beta = np.array([0] * X_train.shape[1])

		for i in range(1, X_train.shape[1]):
			beta[i] = 0.5

		# forward pass
		epoch = 1

		while True:
			grad = gradient(beta, X_train, self.y)
			beta_prev = beta
			beta = beta - self.lr * grad 
			epoch += 1

			if epoch % 5:

				print('beta shape', beta.shape)
				print('X shape', X_train.shape)

				preds = self.predict(beta, X_train)
				
				# loss on train set
				print('Loss at epoch {} is {}'.format(epoch, loss(preds, self.y)))

			if all(beta - beta_prev < self.tol) or epoch >= self.epochs:
				break

		return beta

	def predict(self, beta, X_pred):

		# assumes intercept added

		return np.dot(X_pred, beta)


def main():
	X = np.random.normal(0, 1, 100)
	y = 0.5 * X + 3
	# univariate here
	X = X.reshape((X.shape[0], 1)) 


	lr_obj = LR_SGD(X, y, 10, 0.1)
	beta = lr_obj.train()
	print(beta) #[2.60151767 0.54183088]


if __name__ == '__main__':
	main()