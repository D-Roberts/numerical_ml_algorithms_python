"""
pi

"""


import numpy as np


class Pi(object):
	def __init__(self, N=10E6, seed=1234):
		self.N = N
		self.seed = seed

	def __call__(self):
		np.random.seed(self.seed)
		# std dev is O(1/sqrt(N)) here
		A = 0
		i = 0

		while i <= self.N:
			u1 = np.random.uniform(-1, 1)
			u2 = np.random.uniform(-1, 1)
			if (u1**2 + u2**2) <= 1:
				A += 1
			i += 1
		pi_es = 4.0 * A / self.N
		print(pi_es)

		return pi_es

def std_err(M=1000):
	
	pies = []

	for i in range(M):
		pi = Pi(seed=1234+i)
		pi_es = pi()
		pies.append(pi_es)
	
	return np.std(np.array(pies))

def main():
	pi_es = Pi()
	print(pi_es()) # 3.1409108
	print(std_err())

if __name__ == '__main__':
	main()