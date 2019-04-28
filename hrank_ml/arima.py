#!/usr/bin/env python3

"""BUILD and ARIMA model.

Since one series and short.
Rather than build a feature based ML model as typically do.
Or build horizon features for one-step ahead prediction
and recurrently feed one step at a time for recurrent 
prediction.

500 days to predict next 30.
"""

import os 

import numpy as np 
from statsmodels.tsa.arima_model import ARIMA 

def load_data():
    # Parse directly
    N = int(input())
    X = []
    for i in range(500):
        X.append(int(input()))
    return np.array(X)

def main():
    X = load_data()

    path = '.'
    files = os.listdir(path)
    # print(files)

    # build the model directly since the fitting process test for 
    # AR lag MA lag and need for differentiation for unit root
    # even this one times out with too many lags which gives the good model
    m = ARIMA(X, order=(5,0,3))
    model = m.fit(disp=0)
    preds = model.forecast(steps=30)

    # predict 30 steps
    for i in range(len(preds[0])):
        print(int(preds[0][i]))

if __name__ == '__main__':
    main()
