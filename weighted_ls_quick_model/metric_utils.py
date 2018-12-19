"""
Metrics used in evaluation quality of estimation.
"""

import numpy as np
import math

def mae(actual, predicted):

    return np.sum(np.abs(actual - predicted))/len(actual)

def rmse(actual, predicted):

    return math.sqrt(np.sum((actual-predicted)**2)/len(actual))

