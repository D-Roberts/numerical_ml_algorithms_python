#!/bin/python3

import math
import os
import random
import re
import sys
import numpy as np
from numpy import polyfit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def preprocess():
    data = np.genfromtxt('trainingdata.txt', dtype=str, delimiter=',')
    features = []
    y = []

    for i in range(data.shape[0]):
        features.append(float(data[i][0]))
        y.append(float(data[i][1]))


    # Fit a piecewise model with output 8 if X is above a threshold a linear
    # model otherwise
    # needed to plot the data to see this
    ntrain = 100
    X_train, X_test = np.array(features[:ntrain]).reshape(-1,1), np.array(features[ntrain:]).reshape(-1,1)
    y_train, y_test = np.array(y[:ntrain]), np.array(y[ntrain:])

    # model1 train
    # the max val of X to include in the linear model piece
    cutoff = max(X_train[np.where(y_train<8)])
    # print(cutoff)
    X_train, y_train = X_train[np.where(y_train<8)], y_train[np.where(y_train<8)]
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return cutoff, model
    

if __name__ == '__main__':
    timeCharged = float(input())
    cutoff, model = preprocess()
    if timeCharged > cutoff:
        print(8.00)
    else:
        print(np.round(model.predict(np.array(timeCharged).reshape(1, -1))[0], 2))
    
