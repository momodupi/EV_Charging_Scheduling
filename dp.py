import sys


import numpy as np
import pandas as pd

from numpy.linalg import matrix_rank

from scipy.optimize import linprog
from scipy.linalg import null_space


from scipy.stats import gamma

from cvxopt import matrix, solvers

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

with open('result_static.pickle', 'rb') as pickle_file:
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
            
            z[t][t][mi][ni] = int(m*w[t][mi][ni])
            
            u_s = t+n if t+n<=time_horizon else time_horizon

            for s in range(t+1, u_s):
                z[s][t][mi][ni] = z[s-1][t][mi][ni] - y[s-1][t][mi][ni]
                # print(z[s][t][mi][ni])
            for s in range(u_s, time_horizon):
                z[s][t][mi][ni] = 0
                
with open('z.json', 'w') as json_file:
    json.dump(z, json_file)


class Result(object):
    def __init__(self, sol, c_lin):
        self.sol = sol
        self.x = np.array(sol['x'])
        # print(sol)
        self.fun = float(-c_lin.dot(self.x))
        self.mu_e = np.array(sol['y'])
        self.mu_ie = np.array(sol['z'])
        # print(self.mu_ie)

# set c_s
def get_v(s):
    _v = []
    cnt2tmn = []
    for t in range(s-menu_n_size+1, s+1):
        if t < 0:
            continue
        for mi,m in enumerate(menu['m']):
            for ni,n in enumerate(menu['n']):
                _v.append(v[s][t][mi][ni])
                cnt2tmn.append((t,mi,ni))
    return _v, len(_v), cnt2tmn


def get_c(s):
    return c[s]


def dp_s(s, lag_s):
    v_s, y_size, cnt2tmn = get_v(s)
    # print(v_s, c[s])
    c_lin = np.concatenate( (np.array(v_s), [get_c(s)]) )
    # print(c_lin)
    ye_size = y_size + 1 

    # if lag_s != None:
    #     for i,tmn in enumerate(cnt2tmn):
    #         t,mi,ni = tmn
    #         if r*w[t][mi][ni] >= z[s+1][t][mi][ni]:
    #             if t in lag_s and mi in lag_s[t] and ni in lag_s[t][mi]:
    #                 # print(t,m,n,i)
    #                 c_lin[i] = c_lin[i] + lag_s[t][mi][ni]
    #     c_lin[-1] = c_lin[-1] + lag_s['e']

    # print(c_lin)
    # c_lin = np.ones(ye_size)
    A_eq = np.ones(ye_size)
    A_eq[-1] = -1.0
    A_eq = np.matrix(A_eq)
    b_eq = 0.0

    A_ub_u = np.eye(ye_size)
    b_ub_u = np.zeros(ye_size)
    A_ub_l = -np.eye(ye_size)
    b_ub_l = np.zeros(ye_size)
    
    # set b_ub
    b_ub = np.zeros(ye_size)

    for i,tmn in enumerate(cnt2tmn):
        # print(tmn, s)
        # print(z[s][tmn[0]])
        # b_ub[i] = np.min( r*w[tmn[0]][tmn[1]][tmn[2]], z[s][tmn[0]][tmn[1]][tmn[2]] )
        b_ub_u[i] = min( r*w[tmn[0]][tmn[1]][tmn[2]], z[s][tmn[0]][tmn[1]][tmn[2]] )
        
    b_ub_u[-1] = d[s]

    A_ub = np.concatenate( (A_ub_u, A_ub_l) )
    b_ub = np.concatenate( (b_ub_u, b_ub_l) )
    # result = linprog(
    #         c=-c_lin,
    #         A_eq=A_eq,
    #         b_eq=b_eq,
    #         A_ub=A_ub,
    #         b_ub=b_ub,
    #         bounds=[0,None],
    #         method='interior-point'
    #     )
    '''
    use CVXOPT can directly get lagrange
    '''
    c = matrix(-c_lin)
    G = matrix(A_ub)
    h = matrix(b_ub)
    A = matrix(A_eq)
    b = matrix(b_eq)

    solvers.options['show_progress'] = False

    # s_time = time.time()
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    solvers.options['msg_lev'] = 'GLP_MSG_OFF'
    solvers.options['LP_K_MSGLEV'] = 0
    solvers.options['abstol'] = 1e-3
    solvers.options['reltol'] = 1e-2
    solvers.options['feastol'] = 1
    solvers.options['refinement'] = 1

    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver='glpk')
    # lp_dur = time.time() - s_time
            
    result = Result(sol, c_lin)
    
    return result, cnt2tmn

# run dp along optmial trajectory
result = {}
J = {}
ye = {}
lag = {}
cnt2tmn = {}


for _s in range(1,time_horizon+1):
    s = time_horizon-_s
    if s+1 in lag:
        lag_s = lag[s+1]
    else:
        lag_s = None
    
    # if s+1 in cnt2tmn:
    #     tmn2cnt_s = {}
    #     for i,tmn in enumerate(cnt2tmn[s+1]):
    #         t, m, n = tmn
    #         if t not in tmn2cnt_s:
    #             tmn2cnt_s[t] = {}
    #         if m not in tmn2cnt_s[t]:
    #             tmn2cnt_s[t][m] = {}
    #         tmn2cnt_s[t][m][n] = i
    #         # print(tmn2cnt_s)
    # else:
    #     tmn2cnt_s = None

    result[s], cnt2tmn = dp_s(s, lag_s)
    # print(cnt2tmn)
    J[s] = result[s].fun
    # ye[s] = list(result[s].x)
    ye[s] = {}
    lag[s] = {}
    for i,tmn in enumerate(cnt2tmn):
        t, mi, ni = tmn
        # print(tmn)
        if t not in ye[s]:
            ye[s][t] = {}
            lag[s][t] = {}
        if mi not in ye[s][t]:
            ye[s][t][mi] = {}
            lag[s][t][mi] = {}
        ye[s][t][mi][ni] = float(result[s].x[i])
        lag[s][t][mi][ni] = float(result[s].mu_ie[i])
        lag[s]['e'] = float(result[s].mu_ie[-1])

# print(ye)

with open('ye_dp.json', 'w') as json_file:
    json.dump(ye, json_file)
with open('lagrange_dp.json', 'w') as json_file:
    json.dump(lag, json_file)
with open('J_dp.json', 'w') as json_file:
    json.dump(J, json_file)

y_xlse = pd.ExcelWriter('y_dp.xlsx', engine='xlsxwriter')
# y_df = []
for s in ye:
    # y_df_s = {'arrive time': [t for t in range(time_horizon)]}
    y_df_s = {}
    for t in ye[s]:
        # y_df_s['arrive time'].append(int(t))
        for mi in ye[s][t]:
            for ni in ye[s][t][mi]:
                
                if ni in ye[s][t][mi]:
                    if f'({menu["m"][mi]},{menu["n"][ni]})' not in y_df_s:
                        y_df_s[f'({menu["m"][mi]},{menu["n"][ni]})'] = np.ones(time_horizon)*(-1)
                    y_df_s[f'({menu["m"][mi]},{menu["n"][ni]})'][t] = ye[s][t][mi][ni]

    y_df_s = pd.DataFrame(y_df_s)
    y_df_s.to_excel(y_xlse, sheet_name=f'time={s}')
y_xlse.save()