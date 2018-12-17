"""

Generate samples of normal correlated samples
with correlation rho.

"""

import numpy as np 


class Uncorrelated(object):
	def __init__(self, sample_size):
		self.sample_size = sample_size

	def __call__(self):
		return np.random.normal(0,1,self.sample_size), np.random.normal(0,1,self.sample_size)


class Correlated(Uncorrelated):
	def __init__(self, rho, sample_size):
		self.rho = rho
		self.sample_size = sample_size


	def __call__(self):

		z1, z2 = np.random.normal(0,1,self.sample_size), np.random.normal(0,1,self.sample_size)
	
		return z1, self.rho * z1 - np.sqrt(1- self.rho ** 2) * z2
		


def main():
	unc = Uncorrelated(5)
	z1, z2 = unc()
	print(z1)

	cor = Correlated(0.5, 50)
	y1, y2 = cor()
	print(y1)
	print(y2)
	c = np.corrcoef(y1, y2)
	assert c[0][1] - 0.5 < 0.1

if __name__ == "__main__":
	main()