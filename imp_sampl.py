#/usr/bin/env python3
# -*- coding: utf-8 -*-

'''MC evaluation of integral using importance sampling.

The exponential distribution should have likely had a different rate parameter found
through grid search to minimize variance or plotting. The rate param lambda could be a 
simult hyperparam.
'''

import numpy as np 

class IntegralIS(object):
	def __init__(self, M=1000, nit=100):
		self.M = M
		self.nit = nit

	def f_func(self, x):
		return (np.exp(-x)/(1+(x-1)**2))

	def g_cdf(self, x):
		return -np.exp(x)

	def importance_sampling(self):
		r = np.random.uniform(size=self.M)
		# calculate weights
		inverse_cdf_g = np.exp(r)
		w = self.f_func(inverse_cdf_g) / r
		return w

	def MC_simulation(self):
		'''Generate M uniform rand var
		evaluate the integral in nit runs and take expectation
		'''
		MC_runs = []
		for i in range(self.nit):
			w = self.importance_sampling()
			avg = np.mean(w)
			MC_runs.append(avg)
		MC_mean = np.mean(np.array(MC_runs))
		MC_std = np.std(np.array(MC_runs))
		print("{:2f}, {:2f}".format(MC_mean, MC_std))

	def __call__(self):
		'''Evaluate integral'''
		self.MC_simulation()


def main():
	H = IntegralIS(1000, 1000)
	H()

if __name__ == '__main__':
	main()