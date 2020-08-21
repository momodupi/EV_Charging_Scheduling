import sys


import numpy as np
import pandas as pd

from numpy.linalg import matrix_rank, norm

from scipy.optimize import linprog
from scipy.optimize import minimize
from scipy.linalg import null_space


from scipy.stats import gamma

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

import json
import pickle
import time


    
def phi_quadratic(x, Q, b):
    return  x.T.dot(Q.dot(x))+b.T.dot(x)


def bootstrapping(x, z):
    p_0 = np.zeros(shape=((len(x)+1)*len(x),1))
    # print(p_0)
    K = len(x[:,0])

    def phi_dist(p):
        _p = p.reshape( (len(x)+1, len(x)) )
        Q = _p[:-1,:]
        b = _p[-1,:]
        print(Q, b)
        # return norm( (phi_quadratic(Q, b) - z), 2)
        # return (phi_quadratic(x, Q, b) - z)**2
        sum_dist = 0
        for k in range(K):
            sum_dist = (phi_quadratic(x[:,i], Q, b) - z)**2
        return sum_dist
    
    result = minimize(phi_dist, p_0, method='CG')
    print(result)
    return result.x

x = np.array( [1,2,3,4,5] ).T
z = np.exp( x[0] )
print(x,z)
res = bootstrapping(x,z)
res = res.reshape( (len(x)+1, len(x)) )
Q = res[:-1,:]
b = res[-1,:]
print(x.T.dot(Q.dot(x))+b.T.dot(x), z)


# def bagging(X, n):
#     x_size = len(X[:,0])
    

