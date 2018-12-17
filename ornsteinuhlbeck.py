#!/usr/bin/env python3

"""

Numerical algorithm implemented in the OpenAI codebase.
Replicate. OlhsteinUhlenback.
https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab.
https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
"""

import numpy as np 

class ActiveNoise(object):
	def __init__(self, mu, sig, T):
		self.mu = mu
		self.sig = sig
		self.T = T

	def __call__(self):
		return np.random.normal(self.mu, self.sig, self.T+1)

	def reset(self):
		pass


class OU(ActiveNoise):
	def __init__(self, mu, sig, deltat, T, theta, x0=None):
		self.x0 = x0
		self.mu = mu
		self.sig = sig
		self.deltat = deltat
		self.T = T 
		self.theta = theta 

	def get_ou_path(self):

		x = np.zeros(self.T+1)
		x[0] = self.x0

		z = np.random.normal(self.mu, self.sig, size=self.T+1)

		for t in range(self.T):
			x[t+1] = x[t] + self.theta*(self.mu - x[t]) * self.deltat + self.sig * z[t]

		return x

	def __call__(self):
		
		self.get_ou_path()

	def reset(self):
		"""Theri implem

		Generate one value at a time for one n at a time.
		mu must be an array since shape is taken.
		process gets reset to starting value if provided or given.
		"""
		pass

	def __repr__(self):

		return 'OrnsteinUhlenbeck (mu={} and sigma={})'.format(self.mu, self.sig)

def main():
	
	test = OU(1, 0, 1, 3, 1)
	print(test)
	test()

if __name__ == "__main__":
	main()


