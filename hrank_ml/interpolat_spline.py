# Enter your code here. Read input from STDIN. Print output to STDOUT

"""Interpolation of missing values.

This simple choice because:
- univariate
- few time points
- time points can be consec
- a more classical ML model not worth the trouble.
- assume future points can be used since they exist in the historical data
and this is backfil. Some people may assume otw.
- features built from time and date not likely to help too much.

"""

import numpy as np 
from scipy.interpolate import UnivariateSpline

def load_data():
    N = int(input())
    X = np.zeros((N, 2))
    # print(X)
    for i in range(N):
        line = input().split()
        # index
        X[i,0] = i
        # value with missing
        if line[2][0] == 'M':
            X[i,1] = np.nan
        else:
            X[i,1] = float(line[2])
    return X


def main():
    X = load_data()
    # print(X)
    N = X.shape[0]

    # interpolating spline
    # set 0 weights for the nans
    w = np.isnan(X[:,1])

    # set the nans with zeros
    X[w, 1] = 0.
    spl = UnivariateSpline(X[:,0], X[:,1], w=~w)
    spl.set_smoothing_factor(0.5)

    # print what interpolated in place of missings
    spls = spl(X[:,0])
    for i in range(N):
        if w[i]:
            print(np.round(spls[i],2))

if __name__ == '__main__':
    main()