import numpy as np 


class SoftM(object):
	def __init__(self, p=5, n=100, batch_size=4, lr=0.1, epochs=1):
		self.batch_size = 4
		self.X = np.random.randn(100, 5)
		self.y = (np.random.randn(100, 3) > 0).astype(int)
		self.epochs=epochs

	def train(self):
		# initialize weight and bias
		w0 = np.random.randn(5, 3)
		b0 = np.random.randn(3, 1)
		#. concat
		theta0 = np.concatenate([w0.T, b0], axis=1)
		print(theta0)

		# concatenate intercept to features X
		# print(self.X.shape)
		intercept = np.ones(self.X.shape[0]).reshape(self.X.shape[0],1)
		# print(intercept.shape)
		X = np.concatenate([intercept, self.X], axis=1)

		# get it

		def get_it(X, start, batch_size):
			# unshuffled for now
			while True:
				yield X[start:start+batch_size, :]

		def affine(Xi, theta):
			'''afine transformation only for one feature vector'''
			return np.dot(X, theta)

		# for now get predictions
		def softmax(X):
			expn = np.exp(X, axis=1)
			# normalize
			return expn/np.sum(expn, axis=1), np.sum(expn, axis=1)

		def grads(y, Xi, theta_prev):
			'''directly get gradient'''
			# k dimmensional k=num classes
			part = softmax(Xi)[1]
			# should have eps at least
			return np.dot(y, Xi)*(1-1/part)

		theta = theta0
		for i in self.epochs:
			g = get_it(X, i, self.batch_size)
			for j in range(self.batch_size):
				yhat = softmax(affine(X[j,:], theta))
				grad = grads(y[j, :], X[j,:], theta)
				theta = theta - self.lr * grad 

		self.theta = theta

		return self.theta

				

	def predict(self):
		# get self test
		return np.argmax(softmax(self.theta, affine(theta, X_test), axis=1))

	def __call__(self):
		print(self.y)

	def __repr__(self):
		print("softMax classification implementation with k=3 classes and 5 features")


def main():
	sm = SoftM()

	# generate data and fit model and use to predict
	sm()
	sm.train()

if __name__ == "__main__":
	main()
