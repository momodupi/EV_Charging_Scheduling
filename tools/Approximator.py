import enum
import sys


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from numpy.linalg import matrix_rank, norm

from scipy.optimize import linprog
from scipy.optimize import minimize
from scipy.linalg import null_space

from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
# from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor 

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable

import json
import pickle
import time
import copy 
import timeit

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
        # logging.basicConfig(level=logging.INFO, filename='ap_result.log')

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
        basis_size = 10 if 'basis_size' not in setting else setting['basis_size']
        seed = 0 if 'seed' not in setting else setting['seed']
        convex = False if 'convex' not in setting else setting['convex']
        intercept = False if 'b' not in setting else setting['b']

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

        beta = np.zeros(shape=(data_size, basis_size))

        for i in range(data_size):
            # linearly fit (z_bar,z) without b
            reg = linear_model.LinearRegression(fit_intercept=False)
            # fit: |[beta]_i * [z_bar]_i - z_i| for each i
            reg.fit([z_bar[i,:].T], [z[i]])
            beta[i,:] = reg.coef_
            
            # optimal Q,b for each i: Q_hat = \sum_j beta_j*Q_j
            Q_hat[i] = all_Q.dot(beta[i,:]).reshape((x_size,x_size))
            b_hat[i] = all_b.dot(beta[i,:])

        # calculate z_hat: values for each ((Q_hat,b_hat),z) pair
        z_hat = np.zeros(shape=(data_size,data_size))
        for i in range(data_size):
            for j in range(data_size):
                z_hat[i,j] = x[:,i].T.dot(Q_hat[j].dot(x[:,i])) + x[:,i].dot(b_hat[j])
        
        # linearly fit (z_hat,z) without b, z_hat is a matrix
        reg = linear_model.LinearRegression(fit_intercept=intercept)
        # fit: || alpha * z_hat -z ||
        reg.fit(z_hat, z)
        alpha = reg.coef_
        c = 0. if not intercept else reg.intercept_

        all_Q = np.array( [ Q_hat[i] for i in range(data_size) ] ).T
        all_b = np.array( [ b_hat[i] for i in range(data_size) ] ).T
        Q = all_Q.dot(alpha).reshape((x_size,x_size))
        b = all_b.dot(alpha)

        self.parameter['w'] = alpha.dot(beta)
        self.parameter['Q'] = Q
        self.parameter['b'] = b
        self.parameter['c'] = c

        def q_res(x):
            _x = x[0]
            return _x.T.dot(Q.dot(_x))+b.T.dot(_x)+c
        
        return copy.deepcopy(q_res)

    def quadratic_random_matrix_fast(self, x, z, setting):
        basis_size = 10 if 'basis_size' not in setting else setting['basis_size']
        seed = 0 if 'seed' not in setting else setting['seed']
        convex = False if 'convex' not in setting else setting['convex']
        intercept = False if 'b' not in setting else setting['b']

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
                _Q = np.random.rand(x_size,x_size)
                Q_basis[k] = _Q.dot(_Q) if convex else -_Q.dot(_Q)
                b_basis[k] = np.random.normal(0, 1, x_size)

                z_bar[i,k] = x[:,i].T.dot(Q_basis[k].dot(x[:,i])) + x[:,i].dot(b_basis[k])

        # linearly fit (z_hat,z) without b, z_hat is a matrix
        reg = linear_model.LinearRegression(fit_intercept=intercept)
        # fit: || alpha * z_hat -z ||
        reg.fit(z_bar, z)
        alpha = reg.coef_
        c = 0. if not intercept else reg.intercept_

        all_Q = np.array( [ Q_basis[i] for i in range(basis_size) ] ).T
        all_b = np.array( [ b_basis[i] for i in range(basis_size) ] ).T
        Q = all_Q.dot(alpha).reshape((x_size,x_size))
        b = all_b.dot(alpha)

        self.parameter['w'] = alpha
        self.parameter['Q'] = Q
        self.parameter['b'] = b
        self.parameter['c'] = c

        def q_res(x):
            _x = x[0]
            return _x.T.dot(Q.dot(_x))+b.T.dot(_x)+c
        
        return copy.deepcopy(q_res)
    

    def bregman(self, x, z, setting):
        basis_size = setting['basis_size']
        data_size = len(z)
        x_size = len(x[:,0])
        
        Q_basis = {}
        for k in range(basis_size):
            Q_basis[k] = np.random.rand(x_size,x_size)
        def strong_convex(x, cnt):
            # def sc_1(x):
            #     x += 1e-6
            #     return x.dot(x.T) + np.sum(-np.log(x))
            # def sc_2(x):
            #     x += 1e-6
            #     return x.dot(x.T) + x.dot( np.log(x) )
            # def sc_3(x):
            #     # p_Q = np.random.rand(len(x),len(x))
            #     p_Q = np.eye(len(x))
            #     return x.dot( np.eye(len(x)) + p_Q.dot(p_Q) ).dot(x.T)
            # st_conv = [ sc_1, sc_2, sc_3 ]
            # return st_conv[cnt](x)
            return x.dot( np.eye(x_size) + Q_basis[cnt].dot(Q_basis[cnt]) ).dot(x.T)

        np.random.seed(0)
        sets = setting['sets']
        xk = {}
        zk = {}
        data2sets = {}
        sets2data = {}
        for k in range(sets):
            xk[k] = []
            zk[k] = []
            # sets2data[k] = []

        data_size = len(x[0,:])
        for i in range(data_size):
            k = np.random.choice(range(sets))
            xk[k].append(x[:,i])
            zk[k].append(z[i])
            # data2sets[i] = k
            # sets2data[k].append(i)

        beta = np.zeros(shape=(basis_size, sets))
        # fit for each set
        for k in range(sets):
            zk_bar = np.zeros( shape=(basis_size, len(xk[k])) )
            for i,_xk in enumerate(xk[k]):
                zk_bar[:,i] = np.array( [ strong_convex(_xk,b_i) for b_i in range(basis_size) ] )

            reg = linear_model.LinearRegression(fit_intercept=False)
            # print( np.shape(zk_bar), len(zk[k]) )
            reg.fit(zk_bar.T, np.array(zk[k]))
            beta[:,k] = reg.coef_


        # z_bar are values for each ((Q,b),z) pair
        z_bar = np.zeros(shape=(data_size, sets))
        for i in range(data_size):
            for j in range(sets):
                _beta = beta[:,j]
                z_title = np.array( [ strong_convex(x[:,i],b_i) for b_i in range(basis_size) ] )
                z_bar[i,j] = z_title.dot(_beta)
        
        # linearly fit (z_hat,z) without b, z_hat is a matrix
        reg = linear_model.LinearRegression(fit_intercept=False)
        reg.fit(z_bar, z)
        alpha = reg.coef_
        
        w = alpha.dot(beta.T)
        def b_res(_x):
            z_title = np.array( [ strong_convex(_x[0],i) for i in range(basis_size) ] )
            return z_title.dot(w)

        self.parameter['w'] = w
        return copy.deepcopy(b_res)


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

        return copy.deepcopy(phi_bagging)


    def sklearn_svm(self, x, z, setting):
        regr = SVR(
            kernel=setting['kernel'], 
            degree=setting['degree'], 
            gamma=setting['gamma'], 
            tol=setting['tol']
            )
        regr.fit(x.T, z)

        return regr.predict

    
    def sklearn_neural(self, x, z, setting):
        clf = MLPRegressor(
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


    def PCA(self, x):
        # print(x.shape)
        pca = PCA(n_components=3)
        return pca.fit_transform(x.T).T


    def RandomForest(self, x, z, setting):
        n_estimators = setting['basis_size']
        clf = RandomForestRegressor (n_estimators=n_estimators)
        # print(x.T,z)
        clf.fit(x.T, z)
        return clf.predict


    def check(self, cur, x, z, visual=False):
        z_approx = np.zeros(len(z))
        for i in range(len(z)):
            z_approx[i] = cur([x[:,i]])

        self.check_res = {
            'MSE': metrics.mean_squared_error(z, z_approx),
            'MAE': metrics.mean_absolute_error(z, z_approx),
            'ME': metrics.max_error(z, z_approx),
            'R2': metrics.r2_score(z, z_approx),
            'z': z,
            'a_z': z_approx
        }
        
        logging.info(f"mean:z {np.mean(z)}, mean:z\' {np.mean(z_approx)}, " \
            f"MSE: {self.check_res['MSE']}, MAE: {self.check_res['MAE']}, ME: {self.check_res['MaxE']}, R2: {self.check_res['R2']}")

        if visual:
            t = np.arange(0, len(z), 1)
            plt.plot(t, z, 'ro', t, z_approx, 'b*', t, np.abs(z-z_approx), 'y--')
            plt.show()





def demo(X):
    logging.basicConfig(level=logging.INFO)

    def test_fun(x):
        z = np.zeros(len(x[0,:]))
        for i in range(len(z)):
            # z[i] = ( np.sum(x[:,i]) )**5
            # z[i] = ( np.sum(x[:,i]) )**2
            z[i] = x[:,i].dot(x[:,i])
            # z[i] = np.exp(np.sum(x[:,i]))
        return z

    np.random.seed(0)
    x = np.array(
        [
            np.random.uniform(0,10,X),
            np.random.uniform(0,10,X),
            np.random.uniform(0,10,X),
            np.random.uniform(0,10,X),
        ]
    )
    z = test_fun(x)

    test_x = np.array(
        [
            np.random.uniform(0,10,X),
            np.random.uniform(0,10,X),
            np.random.uniform(0,10,X),
            np.random.uniform(0,10,X),
        ]
    )
    test_z = test_fun(test_x)


    setting = {
        'basis_size': 100,
        'convex': True,
        'b': False
    }
    ap = Approximator()
    cur = ap.quadratic_random_matrix(x, z, setting)
    ap.check(cur, test_x, test_z)

    setting = {
        'basis_size': 100,
        'convex': True,
        'b': False
    }
    ap = Approximator()
    cur = ap.quadratic_random_matrix_fast(x, z, setting)
    ap.check(cur, test_x, test_z)

    setting = {
        'basis_size': 100,
        'sets': 10,
    }
    ap = Approximator()
    cur = ap.bregman(x, z, setting)
    ap.check(cur, test_x, test_z)


    # ap = Approximator()
    # setting = {'basis_size': 100}
    # cur = ap.RandomForest(x,z,setting)
    # ap.check(cur, x, z)
    
    # x_prj = ap.PCA(x)
    # print(x[:,0],x_prj[:,0])
    # cur = ap.quadratic_random_matrix(x_prj, z, setting)
    # ap.check(cur, x_prj, z)

    # cur = ap.bootstrapping(x,z)

    # setting = {
    #     'kernel': 'poly',
    #     'degree': 2,
    #     'gamma': 'auto',
    #     'tol': 10e-6
    # }
    # cur = ap.sklearn_svm(x,z, setting=setting)
    
    # setting = {
    #     'sets': 10
    # }
    # cur = ap.bagging(x,z, setting)

    # setting = {
    #     'alpha': 10e-6,
    #     'random_state': 1,
    #     'hidden_layer': (5,5),
    #     'solver': 'lbfgs',
    #     'activation': 'relu',
    #     'tol': 10e-3,
    #     'max_iter': 50000,
    #     'learning_rate': 'constant'
    # }

    # cur = ap.sklearn_neural(x,z, setting=setting)

    # boosts_err = 0
    # for i in range(len(z)):
    #     # boosts_err += ( regr.predict( [x[:,i]] ) - z[i])**2
    #     # boosts_err += ( ap.kernel[ap.method]( _x, r) - z[i])**2
    #     boosts_err += ( cur([x[:,i]]) - z[i])**2
    # print(boosts_err)


if __name__ == "__main__":
    demo(2000)