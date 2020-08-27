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
    p_0 = np.zeros(shape=((len(x[:,0])+1)*len(x[:,0]),1))
    # print(p_0)
    K = len(x[0,:])

    def phi_dist_boosts(p):
        _p = p.reshape( (len(x[:,0])+1, len(x[:,0])) )
        Q = _p[:-1,:]
        b = _p[-1,:]
        # print(Q, b)
        # return norm( (phi_quadratic(Q, b) - z), 2)
        # return (phi_quadratic(x, Q, b) - z)**2
        sum_dist = 0
        for k in range(K):
            sum_dist += (phi_quadratic(x[:,k], Q, b) - z[k])**2
        return sum_dist/K
    
    result = minimize(phi_dist_boosts, p_0, method='CG')
    # print(result)
    # return result.x
    res = result.x.reshape( (len(x)+1, len(x)) )
    Q = res[:-1,:]
    b = res[-1,:]
    return Q, b



# x = np.array( [[1,2,3,4,5], [5,6,7,8,9]] )

# x = np.array(
#     [
#         np.linspace(0,10,1000),
#         2*np.linspace(0,10,1000)
#     ]
# )
# z = np.zeros(len(x[0,:]))
# for i,_z in enumerate(z):
#     z[i] = ( np.sum(x[:,i]) )**1.3



# print(x,z)
# res = bootstrapping(x,z)
# res = res.reshape( (len(x)+1, len(x)) )
# Q = res[:-1,:]
# b = res[-1,:]

# # print(x.T.dot(Q.dot(x))+b.T.dot(x), z)
# for i in range(len(z)):
#     print(x[:,i].T.dot(Q.dot(x[:,i]))+b.T.dot(x[:,i]), z[i])


X = 1000

# x = np.array(
#     [
#         np.linspace(0,10,X),
#         2*np.linspace(0,10,X)
#     ]
# )
x = np.array(
    [
        np.random.uniform(0,10,X),
        np.random.uniform(0,20,X),
    ]
)
z = np.zeros(len(x[0,:]))
for i,_z in enumerate(z):
    z[i] = ( np.sum(x[:,i]) )**1.3

# bagging
K = 10
xk = {}
zk = {}
for k in range(K):
    xk[k] = []
    zk[k] = []

for i in range(X):
    k = np.random.choice(range(K))
    xk[k].append(x[:,i])
    zk[k].append(z[i])


Qk = {}
bk = {}
for k in range(K):
    _x = np.array(xk[k]).T
    _z = np.array(zk[k])
    Qk[k], bk[k] = bootstrapping(_x, _z)



def phi_quadratic_bagging(x, Q, b, alpha):
    k_size = len(alpha)
    phi_hat = np.zeros(k_size)
    for k in range(k_size):
        phi_hat[k] = phi_quadratic( x, Q[k], b[k] )
    return alpha.dot( phi_hat )

def bagging(x, z):
    alpha_0 = np.ones(K)
    
    def phi_dist_bagging(alpha):
        dist_sum = 0

        for i in range(X):
            _x = x[:,i]
            _z = z[i]
            dist_sum += (phi_quadratic_bagging(_x, Qk, bk, alpha) - _z)**2
            # print(dist_sum)
        return dist_sum

    result = minimize(phi_dist_bagging, alpha_0, method='CG')
    # print(result)
    return result.x

s_time = time.time()
alpha = bagging(x, z)
bagging_dur = time.time()-s_time

# for i in range(len(z)):
#     print(phi_quadratic_bagging(x[:,i], Qk, bk, alpha), z[i])

s_time = time.time()
Q,b = bootstrapping(x,z)
boosts_dur = time.time()-s_time

boosts_err = 0
bagging_err = 0
for i in range(len(z)):
    boosts_err += (phi_quadratic(x[:,i], Q, b) - z[i])**2
    bagging_err += (phi_quadratic_bagging(x[:,i], Qk, bk, alpha)-z[i])**2
    
    # print(phi_quadratic_bagging(x[:,i], Qk, bk, alpha) - z[i])

print(f'error: booststrapping: {boosts_err}, bagging: {bagging_err}')
print(f'time: booststrapping: {boosts_dur}, bagging: {bagging_dur}')