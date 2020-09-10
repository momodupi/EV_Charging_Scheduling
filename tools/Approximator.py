import sys


import numpy as np
import pandas as pd

from numpy.linalg import matrix_rank, norm

from scipy.optimize import linprog
from scipy.optimize import minimize
from scipy.linalg import null_space

from sklearn import svm


import json
import pickle
import time
import copy 

import logging

class Approximator(object):
    def __init__(self, method='quadratic'):
        self.method = method
        
        self.kernel = {
            'quadratic': self.phi_quadratic,
        }

        self.theta = {
            'quadratic': self.theta_quadratic,
        }

        self.parameter = {}

    def theta_quadratic(self, p, x_size):
        _p = p.reshape( (x_size+1, x_size) )
        Q = _p[:-1,:]
        b = _p[-1,:]
        self.parameter = {
            'Q': Q,
            'b': b
        }

    def phi_quadratic(self, x):
        _x = x[0]
        Q = self.parameter['Q']
        b = self.parameter['b']
        return _x.T.dot(Q.dot(_x))+b.T.dot(_x)

    def bootstrapping(self, x, z):
        p_0 = np.zeros(shape=((len(x[:,0])+1)*len(x[:,0]),1))
        # print(p_0)
        K = len(x[0,:])
        x_size = len(x[:,0])

        def phi_dist_boosts(p):
            sum_dist = 0
            for k in range(K):
                self.theta[self.method](p, x_size)
                phi = self.kernel[self.method]( [x[:,k]] )
                
                sum_dist += (phi - z[k])**2
            
            return sum_dist/K
        
        result = minimize(phi_dist_boosts, p_0, method='CG')
        # print(result)
        # return result.x
        self.theta[self.method](result.x, x_size)

        # return self.theta_quadratic(result.x, x_size)
        return self.kernel[self.method]

    def bagging(self, x, z, sets):
        xk = {}
        zk = {}
        for k in range(sets):
            xk[k] = []
            zk[k] = []

        K = len(x[0,:])
        for k in range(K):
            i = np.random.choice(range(sets))
            xk[i].append(x[:,k])
            zk[i].append(z[k])

        self.cur = []
        for i in range(sets):
            _x = np.array(xk[i]).T
            _z = np.array(zk[i])
            # print(_x,_z)
            _cur = copy.deepcopy( self.bootstrapping(_x, _z) )
            self.cur.append(_cur)

        self.alpha = np.ones(sets)
        
        def phi_bagging(x):
            phi_hat = np.zeros(sets)
            for i in range(sets):
                phi_hat[i] = self.cur[i]( x )
            return self.alpha.dot( phi_hat )

        alpha_0 = np.ones(sets)

        def phi_dist_bagging(alpha):
            dist_sum = 0
            self.alpha = alpha
            for k in range(K):
                dist_sum += ( phi_bagging( [x[:,k]] ) - z[k])**2
                # print(dist_sum)
            return dist_sum

        result = minimize(phi_dist_bagging, alpha_0, method='CG')
        self.alpha = result.x
        return phi_bagging

    
    def sklearn_svm(self, x, z, setting):
        kernel = setting['kernel']
        degree = setting['degree']
        gamma = setting['gamma']
        tol = setting['tol']
        regr = svm.SVR(
            kernel=kernel, 
            degree=degree, 
            gamma=gamma, 
            tol=tol
            )
        regr.fit(x.T, z)

        return regr.predict

    def check(self, cur, x, z):
        err = 0
        for i in range(len(z)):
            err += ( cur([x[:,i]]) - z[i])**2
        logging.info(f'{err}')



def demo(X):

    x = np.array(
        [
            np.random.uniform(0,10,X),
            np.random.uniform(0,20,X),
        ]
    )
    z = np.zeros(len(x[0,:]))

    for i,_z in enumerate(z):
        z[i] = ( np.sum(x[:,i]) )**2
        # z[i] = np.sum(x[:,i])

    ap = Approximator()
    cur = ap.bootstrapping(x,z)
    # cur = ap.sklearn_svm(x,z)
    # cur = ap.bagging(x,z, 10)

    # regr = ap.sklearn_svm(x,z)
    # print(regr.predict([[1,1]]))
    boosts_err = 0
    for i in range(len(z)):
        # boosts_err += ( regr.predict( [x[:,i]] ) - z[i])**2
        # boosts_err += ( ap.kernel[ap.method]( _x, r) - z[i])**2
        boosts_err += ( cur([x[:,i]]) - z[i])**2

    print(boosts_err)