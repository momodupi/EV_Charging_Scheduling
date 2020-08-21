import sys


import numpy as np
import pandas as pd

from numpy.linalg import matrix_rank

from scipy.optimize import linprog
from scipy.linalg import null_space


from scipy.stats import gamma

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

import json
import pickle
import time

# import data
with open('ev.pickle', 'rb') as pickle_file:
    EV = pickle.load(pickle_file)


time_horizon = EV['info']['time_horizon']
time_arrival_horizon = EV['info']['time_arrival_horizon']
time_arrival_ratio = time_horizon/time_arrival_horizon

charge_rate = EV['info']['charge_rate']
time_unit = EV['info']['time_unit']
menu_m_step = EV['info']['m_step']
menu_n_step = EV['info']['n_step']
menu_m_size = EV['info']['m_size']
menu_n_size = EV['info']['n_size']

menu = {
    'm': np.array(EV['info']['m']),
    'n': np.array(EV['info']['n'])
}

time_unit_set = {
    'hour': 1,
    'min': 60,
    'sec': 3600,
    'halfh': 2
}

del EV['info']


with open('w.pickle', 'rb') as pickle_file:
    w = pickle.load(pickle_file) 

with open('c.pickle', 'rb') as pickle_file:
    c = pickle.load(pickle_file) 

with open('v.pickle', 'rb') as pickle_file:
    v = pickle.load(pickle_file) 

with open('d.pickle', 'rb') as pickle_file:
    d = pickle.load(pickle_file) 

with open('result.pickle', 'rb') as pickle_file:
    result = pickle.load(pickle_file) 

y = result['y']
e = result['e']

r = charge_rate

# update z
z = {}
for s in range(time_horizon):
    z[s] = {}
    for t in range(0,time_horizon):
        # print(t)
        z[s][t] = {}
        for mi,m in enumerate(menu['m']):
            z[s][t][mi] = {}
with open('z.json', 'w') as json_file:
    json.dump(z, json_file)

# for mi,m in enumerate(menu['m']):
#     for ni,n in enumerate(menu['n']):
#         for t in range(time_horizon):
#             z[t][t][mi][ni] = m*w[t][mi][ni]
#             if t+n+1 > time_horizon:
#                 continue
#             # print(t,mi,ni)
#             for s in range(t+1, t+n):
#                 # print(s,t,mi,ni)
#                 z[s][t][mi][ni] = z[s-1][t][mi][ni] - y[s-1][t][mi][ni]
#             for s in range(t+n+1, time_horizon):
#                 z[s][t][mi][ni] = 0

for mi,m in enumerate(menu['m']):
    for ni,n in enumerate(menu['n']):
        for t in range(time_horizon):
            z[t][t][mi][ni] = m*w[t][mi][ni]
            if t >= time_horizon:
                continue
            # print(t,mi,ni)
            for s in range(t+1, t+n):
                # print(s,t,mi,ni)
                if s >= time_horizon:
                    break
                z[s][t][mi][ni] = z[s-1][t][mi][ni] - y[s-1][t][mi][ni]
            for s in range(t+n, time_horizon):
                if s >= time_horizon:
                    break
                z[s][t][mi][ni] = 0

with open('z.json', 'w') as json_file:
    json.dump(z, json_file)


# do the dp

# set c_s
def get_v(s):
    _v = []
    cnt2tmn = []
    for t in range(s-menu_n_size, s):
        if t < 0:
            continue
        for mi,m in enumerate(menu['m']):
            for ni,n in enumerate(menu['n']):
                _v.append(v[s][t][mi][ni])
                cnt2tmn.append((t,mi,ni))
    return _v, len(_v), cnt2tmn



def dp_s(s):
    v_s, y_size, cnt2tmn = get_v(s)
    # print(v_s, c[s])
    c_lin = np.concatenate( (np.array(v_s), [c[s]]) )
    # print(c_lin)
    ye_size = y_size + 1 

    # c_lin = np.ones(ye_size)
    A_eq = np.ones(ye_size)
    A_eq[-1] = -1.0
    A_eq = np.matrix(A_eq)
    b_eq = 0
    A_ub = np.eye(ye_size)
    
    # set b_ub
    b_ub = np.zeros(ye_size)

    for i,tmn in enumerate(cnt2tmn):
        # print(tmn, s)
        # print(z[s][tmn[0]])
        # b_ub[i] = np.min( r*w[tmn[0]][tmn[1]][tmn[2]], z[s][tmn[0]][tmn[1]][tmn[2]] )
        b_ub[i] = min( r*w[tmn[0]][tmn[1]][tmn[2]], z[s][tmn[0]][tmn[1]][tmn[2]] )
        
    b_ub[-1] = d[s]
    result = linprog(
            c=-c_lin,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[0,None],
            method='interior-point'
        )
    return result, cnt2tmn

# run dp along optmial trajectory
result = {}
J = {}
ye = {}
for _s in range(1,time_horizon+1):
    s = time_horizon-_s
    result[s], cnt2tmn = dp_s(s)
    # print(cnt2tmn)
    J[s] = result[s].fun
    # ye[s] = list(result[s].x)
    ye[s] = {}
    for i,tmn in enumerate(cnt2tmn):
        t, mi, ni = tmn
        if t not in ye[s]:
            ye[s][t] = {}
        if mi not in ye[s][t]:
            ye[s][t][mi] = {}
        ye[s][t][mi][ni] = result[s].x[i]

with open('result_dp.json', 'w') as json_file:
    json.dump(ye, json_file)
with open('J_dp.json', 'w') as json_file:
    json.dump(J, json_file)