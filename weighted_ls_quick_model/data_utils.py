"""

data utils
"""
import os

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import statsmodels as sm


def load_data(data_dir, file_name):

    y = np.genfromtxt(os.path.join(data_dir, file_name),
                      skip_header=1,
                      delimiter=',',
                      dtype='f4')

    return y


def get_data_label(d):
    x = np.array([sample[0] for sample in d])
    X = sm.tools.tools.add_constant(x)
    y = np.array([sample[1] for sample in d])

    return X, y


def get_diagnostic_plots(Xj, yj, residj, j, figure_dir):

    # TODO: add labels; titles and put in 4 plot to include in pdf

    # Scatterplots
    plt.scatter([sample[1] for sample in Xj], yj)
    plt.savefig(os.path.join(figure_dir, 'scatterxy' + str(j + 1) + '.png'))
    plt.close()

    # Residual plots
    plt.plot(residj)
    plt.savefig(os.path.join(figure_dir, 'res_ols' + str(j+1) + '.png'))
    plt.close()

    # And residuals histograms
    plt.hist(residj, bins=21)
    plt.savefig(os.path.join(figure_dir, 'hist_r' + str(j+1) + '.png'))
    plt.close()

    # Autocorrelation
    plt.acorr(residj)
    plt.savefig(os.path.join(figure_dir, 'acf_r' + str(j+1) + '.png'))
    plt.close()
