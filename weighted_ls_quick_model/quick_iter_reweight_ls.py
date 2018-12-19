"""

Reiterated weighted least squares.
"""


import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import math

import scipy

import statsmodels as sm
from statsmodels import tsa
from statsmodels.tsa import stattools
from statsmodels import regression
from statsmodels.regression import linear_model

from data_utils import load_data
from data_utils import get_diagnostic_plots
from data_utils import get_data_label

from models import fit_wls_model


home_dir = os.getenv("HOME")

# TODO: create check if exist remove and create these folders in code
project_dir = os.path.join(home_dir, 'voleon')
data_dir = os.path.join(project_dir, 'data')
figure_dir = os.path.join(project_dir, 'figures')


def main():

    #1. Data processing section

    # Load data
    list_of_datasets = ['data_'+'1'+'_'+str(i)+'.csv' for i in range(1, 5)]

    # List of datasets
    nj = [0]
    dj = []
    Xj = []
    yj = []

    for j in range(4):

        d = load_data(data_dir, list_of_datasets[j])
        dj.append(d)
        print(len(d))
        nj.append(len(d))
        X, y = get_data_label(d)
        Xj.append(X)
        yj.append(y)


    nj = np.cumsum(nj)

    #2. First minimum baseline model simple OLS linear regression, ignoring additional information

    ols_params = []
    ols_aic = []
    ols_rsq = []
    ols_se = []

    for j in range(4):

        w = np.ones(Xj[j].shape[0])
        params, aic, rsq, bse, residj = fit_wls_model(Xj[j], yj[j], w)

        # Save baseline model stats
        ols_params.append(params)
        ols_aic.append(aic)
        ols_rsq.append(rsq)
        ols_se.append(bse)

        # Diagnostic plots
        get_diagnostic_plots(Xj[j], yj[j], residj, j, figure_dir)



    #3. Iterative reweighted least squares


    w = np.ones(nj[4])

    nit = 15


    for i in range(nit):

        irwls_params = []
        irwls_aic = []
        irwls_rsq = []
        irwls_se = []

        resids = []

        for j in range(4):

            wj = w[nj[j]:nj[j+1]]

            params, aic, rsq, bse, residj = fit_wls_model(Xj[j], yj[j], w=1. / (wj ** 2))

            irwls_params.append(params)
            irwls_aic.append(aic)
            irwls_rsq.append(rsq)
            irwls_se.append(bse)

            # concat residj with resid from the other 4 models
            resids.extend(list(residj))


        sq_resid = np.asarray(resids) ** 2
        sq_resid_ols = linear_model.OLS(sq_resid, X)
        sq_resid_ols_res = sq_resid_ols.fit()
        print("current error params", sq_resid_ols_res.params)

        vars = sq_resid_ols.predict(sq_resid_ols_res.params, X)
        w = 1 / vars

    print(irwls_aic)
    print(ols_aic)
    print(ols_params)
    print(irwls_params)
    print(irwls_se)
    print(ols_se)



if __name__ == "__main__":
    main()


