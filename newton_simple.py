
"""

Inputs coefficients as polynomial and get zero by Newton.

"""

import numpy as np
from numpy import polynomial 


class Newton(object):
	def __init__(self, polyn_coefs, theta0, tol, steps):
		self.theta0 = theta0
		self.tol = tol
		self.steps = steps
		self.polyn_coefs = np.array(polyn_coefs)

	def __call__(self):
		thetas =[]
		theta_prev = self.theta0
		theta_cur = float('inf')

		while len(thetas) < self.steps and theta_cur - theta_prev > self.tol:

			# derivat coefficients
			grad = np.polynomial.polynomial.polyder(self.polyn_coefs)
			step = np.polynomial.polynomial.polyval(theta_prev, self.polyn_coefs)/\
					np.polynomial.polynomial.polyval(theta_prev, grad)
			theta_cur = theta_prev - step 
			thetas.append(theta_cur)

		return thetas 

	def __repr__(self):
		return 'Should print the polynomial.'


def main():

	n = Newton((1, -1, 1), 0, 10e-3, 100)
	thetas = n()
	print('Last iteration theta is theta={}'.format(thetas[-1]))

if __name__ == '__main__':
	main()

