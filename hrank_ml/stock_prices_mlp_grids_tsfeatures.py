#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Predict stock prices.

HackerRank problem.

The ML formulation is to predict the 1 step ahead return sign for
buy and sell decisions.

The portfolio weights decision may be a separate model.
"""

import os
import random
import math

import numpy as np 
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.pipeline import Pipeline



DATA_DIR = '.'



class StockP(object):
	def __init__(self, filename, batch_size, 
				epochs, hidden_size, lr, l2,
				n_estimators, 
				mlp=True,
				tune=False):
		self.filename = filename
		self.batch_size = batch_size
		self.h = hidden_size
		# what model to fit
		self.mlp = mlp
		# learning rate
		self.lr = lr 
		# l2 regularization
		self.l2 = l2
		# epochs
		self.epochs = epochs
		self.n_estimators = n_estimators
		self.tune = tune

	def _load_data(self):
		"""Build a dict with prices.

		Last number in the list is the current price.
		In real life preprocessing would be necessary for data issues.
		"""
		d = {}
		with open(os.path.join(DATA_DIR, self.filename), 'r') as f:
			for line in f.readlines():
				line = line.strip().split()
				d[line[0]] = [float(x) for x in line[1:]]
				# print(len(line[1:]))
		# 9 stocks with 505 days of history
		#print(d)
		return d

	def _get_returns(self, d):
		returns = {}
		for item in d:
			# turn to percentages
			temp = [0] + [100*(math.log(d[item][i]) - math.log(d[item][i-1])) \
			for i in range(1, len(d[item]))]
			returns[item] = temp
		# print('returns', returns)
		# padd with a 0 to maintain size
		
		return returns

	def parse_data(self, prices, n_steps=5):
		"""Get X_train, y_train, X_test, y_test.

		Use 5 day history for each 1 ahead.
		Include prediction for all stocks in the same model
		in a multitask fashion.

		Intend to use the simple sklearn MLPClassifier to predict the sign
		of the 1 step ahead return, which has validation step.

		Standardize.

		Get label code and splits here.
		"""
		returns = self._get_returns(prices)

		stocks = returns.keys()
		d = []
		for stock in returns:
			d.append(returns[stock])
		d = np.array(d).T
		# print('raw return data with 9 cols', d.shape)

		# number of days in dataset
		# columns are the different stocks
		# 9 stocks together
		T = len(d[:, 0])
		X = np.zeros((9*(T-n_steps), n_steps))
		y = np.zeros(9*(T-n_steps))

		# bundle up the stocks in the same dataset
		# for joint prediction
		for j in range(len(stocks)):
			for i in range(T-n_steps):
				X[j*(T-n_steps)+i:j*(T-n_steps)+i+n_steps, :] = d[i:i+n_steps, j]
				# label is sign; 1 is plus.
				y[j*(T-n_steps)+i:j*(T-n_steps)+i+n_steps] = int(d[i+n_steps, j] > 0)

		# split
		X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

		# Standardize with all train data; stdscaler does it per feature
		m = np.mean(X_train)
		# print(m)
		s = np.std(X_train)
		X_train = (X_train - m) / s
		# print(X_train)
		X_test = (X_test - m) / s
		return X_train, X_test, y_train, y_test

	def build_model(self):
		if self.mlp:
			self.net = MLPClassifier(hidden_layer_sizes=(self.h, self.h),
							batch_size=self.batch_size,
							random_state=42,
							early_stopping=True,
							validation_fraction=0.2,
							learning_rate_init=self.lr,
							alpha=self.l2,
							max_iter=self.epochs) 
		else:
			self.net = RandomForestClassifier(n_estimators=self.n_estimators)

	def train(self, X_train, y_train):
		"""Train model using.

		Find best params via grid search.
		Simplify to 2 hidden layers.
		"""
		self.net.fit(X_train, y_train)

	def predict(self, X_test):
		"""Return preds.

		"""
		preds = self.net.predict(X_test)
		return preds

	def evaluation(self, preds, y_test):
		"""Accuracy, precision, recall."""
		acc = accuracy_score(y_test, preds)
		prec, rec, f1, supp = precision_recall_fscore_support(y_test, preds)
		print('Accuracy, precision, recall and support on test: ', (acc, prec, rec, supp))
		print(235/(235+215))

	def _grid_s(self, X_train, y_train, cv=2):
		"""Basic hyperpar tune."""
		pipe = Pipeline(steps=[('m', self.net)])
		if self.mlp:
			param_grid = {'m__batch_size':[32],
						  'm__learning_rate_init':[0.01, 0.001],
						  'm__alpha':[0.01, 0.001]}	
		else:	
			param_grid = {'m__n_estimators':[100, 200, 2000]}

		search = GridSearchCV(pipe, param_grid, cv=cv)
		search.fit(X_train, y_train)
			# print("Best parameter (CV score=%0.3f):" % search.best_score_)
		print(search.best_params_)

	def __call__(self):
		"""Call train and predict and print output.

		Use the call dunder as a form of model module manager.
		"""
		d = self._load_data()
		X_train, X_test, y_train, y_test = self.parse_data(d)
		self.build_model()
	
		# a basic grid search
		# self._grid_s(X_train, y_train)

		# train with best params and predict
		self.train(X_train, y_train)
		preds = self.predict(X_test)
		# # print(preds)
		self.evaluation(preds, y_test)
		# a 20% lift above support with MLP; mLP need some architect tune
		# rf does better

def main():
	sp = StockP('bigger_data_set.txt',
				batch_size=32,
				lr=0.001,
				l2=0.01,
				hidden_size=100,
				epochs=1000, 
				n_estimators=2000,
				mlp=True)
	sp()

if __name__ == '__main__':
	main()