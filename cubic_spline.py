#!/usr/bin/python3

"""

Implement cubic splines numerical algorithm in Python3 with the goal of interpolation.
"""

import time
import numpy as np 


class CubicSpline(object):

	def __init__(self, k, x, y):
		"""x, y arrays of size k+1
		"""
		self.k = k
		self.x = x
		self.y = y

	def get_coefs(self):

		# guard against division by 0
		eps = 10e-8
		a = [yi for yi in self.y]
		b = [0] * self.k
		d = [0] * self.k
		miu = [0] * self.k

		h = [self.x[i+1] - self.x[i] for i in range(self.k)] 
		alpha = [(3/(h[i]+eps))*(a[i+1] - a[i])-(3/(h[i-1]+eps))*(a[i] - a[i-1]) for i in range(1,self.k)]

		c = [0] * (self.k + 1)
		l = [0] * (self.k + 1)
		z = [0] * (self.k + 1)
		

		for i in range(1, self.k-1):
			l[i] = 2 * (self.x[i+1] - self.x[i-1]) - h[i-1]*miu[i-1]
			miu[i] = float(h[i]) / l[i]
			z[i] = (alpha[i] - h[i-1]*z[i-1])/float(l[i])

		l[self.k] = 1

		for j in range(self.k-1,-1,-1):
			c[j] = z[j] - miu[j]*c[j+1]
			b[j] = (a[j+1] - a[j])/h[j] - (h[j]*(c[j+1] + 2 * c[j])/3)
			d[j] = (c[j+1] - c[j])/(3*h[j])

		return [(a[i],b[i],c[i],d[i],self.x[i]) for i in range(self.k)]
	
	def __call__(self):
		print(self.get_coefs())

	def __repr__(self):
		return " ".join([str(xi) for xi in self.x])

	
def main():
	
	test = CubicSpline(4, [1,2,3,4,5], [1,2,3,4,5])
	print(test)
	test()

	# TODO: add an actual test similar to what they have on mxnet


if __name__ == '__main__':
	main()