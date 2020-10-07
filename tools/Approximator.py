import sys


import numpy as np
import pandas as pd

from numpy.linalg import matrix_rank, norm

from scipy.optimize import linprog
from scipy.optimize import minimize
from scipy.linalg import null_space

from sklearn import svm, linear_model, neural_network, preprocessing


import json
import pickle
import time
import copy 

import logging
# from imp import reload


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

        # reload(logging)
        logging.basicConfig(level=logging.INFO, filename='ap_result.log')

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

    def quadratic_random_matrix(self, x, z, setting):
        basis_size = setting['basis_size']
        seed = 0 if 'seed' not in setting else setting['seed']
        convex = setting['convex']

        data_size = len(z)
        x_size = len(x[:,0])

        np.random.seed(seed)

        # generate basis and z_bar
        Q_basis = {}
        b_basis = {}
        # z_bar are values for each ((Q,b),z) pair
        z_bar = np.zeros(shape=(data_size, basis_size))
        for i in range(data_size):
            for k in range(basis_size):
                Q_basis[k] = np.random.rand(x_size,x_size)
                Q_basis[k] = Q_basis[k].dot(Q_basis[k]) if convex else -Q_basis[k].dot(Q_basis[k])
                b_basis[k] = np.random.normal(0, 1, x_size)

                z_bar[i,k] = x[:,i].T.dot(Q_basis[k].dot(x[:,i])) + x[:,i].dot(b_basis[k])

        # calculate optimal beta for each data
        all_Q = np.array( [ Q_basis[i] for i in range(basis_size) ] ).T
        all_b = np.array( [ b_basis[i] for i in range(basis_size) ] ).T
        Q_hat = {}
        b_hat = {}
        for i in range(data_size):
            # linearly fit (z_bar,z) without b
            reg = linear_model.LinearRegression(fit_intercept=False)
            # fit: |[beta]_i * [z_bar]_i - z_i| for each i
            reg.fit([z_bar[i,:].T], [z[i]])
            beta = reg.coef_
            
            # optimal Q,b for each i: Q_hat = \sum_j beta_j*Q_j
            Q_hat[i] = all_Q.dot(beta).reshape((x_size,x_size))
            b_hat[i] = all_b.dot(beta)

        # calculate z_hat: values for each ((Q_hat,b_hat),z) pair
        z_hat = np.zeros(shape=(data_size,data_size))
        for i in range(data_size):
            for j in range(data_size):
                z_hat[i,j] = x[:,i].T.dot(Q_hat[j].dot(x[:,i])) + x[:,i].dot(b_hat[j])
        
        # linearly fit (z_hat,z) without b, z_hat is a matrix
        reg = linear_model.LinearRegression(fit_intercept=False)
        # fit: || alpha * z_hat -z ||
        reg.fit(z_hat, z)
        alpha = reg.coef_

        all_Q = np.array( [ Q_hat[i] for i in range(data_size) ] ).T
        all_b = np.array( [ b_hat[i] for i in range(data_size) ] ).T
        Q = all_Q.dot(alpha).reshape((x_size,x_size))
        b = all_b.dot(alpha)

        self.parameter['Q'] = Q
        self.parameter['b'] = b
        return copy.deepcopy(self.phi_quadratic)


    def bootstrapping(self, x, z):
        x_size = len(x[:,0])
        p_0 = np.zeros(shape=((x_size+1)*x_size,1))
        # print(p_0)
        data_size = len(x[0,:])

        def phi_dist_boosts(p):
            sum_dist = 0
            for k in range(data_size):
                self.theta[self.method](p, x_size)
                phi = self.kernel[self.method]( [x[:,k]] )
                
                sum_dist += (phi - z[k])**2
            
            return sum_dist/data_size
        
        result = minimize(phi_dist_boosts, p_0, method='CG')
        self.theta[self.method](result.x, x_size)
        return self.kernel[self.method]


    def bagging(self, x, z, setting):
        sets = setting['sets']
        xk = {}
        zk = {}
        data2sets = {}
        sets2data = {}
        for k in range(sets):
            xk[k] = []
            zk[k] = []
            sets2data[k] = []

        data_size = len(x[0,:])
        for i in range(data_size):
            k = np.random.choice(range(sets))
            xk[k].append(x[:,i])
            zk[k].append(z[i])
            data2sets[i] = k
            sets2data[k].append(i)

        cur = []
        for k in range(sets):
            _x = np.array(xk[k]).T
            _z = np.array(zk[k])
            cur.append( copy.deepcopy( self.bootstrapping(_x, _z) ) )

        z_hat = np.zeros(shape=(data_size, sets))
        for i in range(data_size):
            for j in range(data_size):
                k = data2sets[j]
                z_hat[i,k] = cur[k]( [x[:,i]] )

        reg = linear_model.LinearRegression(fit_intercept=False)
        reg.fit(z_hat, z)
        alpha = reg.coef_

        def phi_bagging(x):
            phi_hat = np.array( [cur[i](x) for i in range(sets)] )
            return alpha.dot( phi_hat )

        return phi_bagging


    def sklearn_svm(self, x, z, setting):
        regr = svm.SVR(
            kernel=setting['kernel'], 
            degree=setting['degree'], 
            gamma=setting['gamma'], 
            tol=setting['tol']
            )
        regr.fit(x.T, z)

        return regr.predict

    
    def sklearn_neutral(self, x, z, setting):
        clf = neural_network.MLPRegressor(
            solver=setting['solver'], 
            alpha=setting['alpha'],
            learning_rate=setting['learning_rate'],
            tol=setting['tol'],
            max_iter=setting['max_iter'],
            hidden_layer_sizes=setting['hidden_layer'], 
            random_state=setting['random_state'],
            activation=setting['activation']
            )
        # x_sacled = preprocessing.scale(x.T)
        # z_scaled = preprocessing.scale(z)
        clf.fit(x.T, z)
        return clf.predict


    def check(self, cur, x, z):
        z_approx = np.zeros(len(z))
        for i in range(len(z)):
            z_approx[i] = cur([x[:,i]])

        sigma = np.linalg.norm(z_approx-z)
        mu = sigma/np.mean(z)

        logging.info(f'mean {np.mean(z)}, 2norm: {sigma}, cov: {mu}')



def demo(X):
    np.random.seed(0)
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

    # setting = {
    #     'basis_size': 10,
    #     'convex': True,
    # }
    # cur = ap.quadratic_random_matrix(x,z, setting)

    # cur = ap.bootstrapping(x,z)

    # setting = {
    #     'kernel': 'poly',
    #     'degree': 2,
    #     'gamma': 'auto',
    #     'tol': 10e-6
    # }
    # cur = ap.sklearn_svm(x,z, setting=setting)
    
    setting = {
        'sets': 10
    }
    cur = ap.bagging(x,z, setting)



    boosts_err = 0
    for i in range(len(z)):
        # boosts_err += ( regr.predict( [x[:,i]] ) - z[i])**2
        # boosts_err += ( ap.kernel[ap.method]( _x, r) - z[i])**2
        boosts_err += ( cur([x[:,i]]) - z[i])**2

    print(boosts_err)


if __name__ == "__main__":
    demo(100)