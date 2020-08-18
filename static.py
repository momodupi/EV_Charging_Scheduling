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

# import data
with open('ev.json') as json_file:
    EV = json.load(json_file)

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

# get w_s
w = {}
for t in range(time_horizon):
    w[t] = np.zeros(shape=(menu_m_size,menu_n_size))

for t in EV:
    # tran t to time
    s = int( int(t)*time_arrival_ratio )
    # print(s)
    # if s in w:
    #     w[s].append( EV[t] )
    # else:
    #     w[s] = [ EV[t] ] 
    # print( EV[t]['m'], EV[t]['n'] )
    w[s][ EV[t]['m'] ][ EV[t]['n'] ] += 1


def price_turbulance(s):
    s = s-int(s/24)*24 if s>24 else s
    s_p = s if s>8 else 24+s
    return gamma.pdf(s_p-10, 8)

# set price
def get_v(s,t,m,n):
    '''
    t is trivial\\
    fix s,n: linear with m\\
    fix s,m: linear with n\\
    fix m,n: gamma with (-5,10)
    '''
    # base_price = 0.12 # $/kWh
    base_price = 0.11 # $/kWh
    # return ((m*menu_m_step) * base_price)/(1+n/100)

    '''
    price = base price *  amount of electricty \\
            * price turbulance over time (gamma) 
            / time flexibility discount
    '''
    return (1+price_turbulance(s)) * base_price*(m+1)*menu_m_step  / (1+(n/menu_n_size))

def get_c(s):
    # base_price = 0.12 # $/kWh
    # return (1+price_turbulance(s)) * base_price
    # print(s/time_unit_set[time_unit])
    cur_time = int(s/time_unit_set[time_unit]) % 24
    return 9.341/100 if cur_time >= 11 and cur_time <= 19 else 0.948/100



def get_d(s=0):
    return 500 # kWh

# set all v,c,z
v = {}
c = []
z = {}
for s in range(time_horizon):
    c.append(get_c(s=s))
    v[s] = {}
    # z[s] = {}
    
    for t in range(s-menu_n_size, s+1):
        v[s][t] = np.zeros(shape=(menu_m_size,menu_n_size))

        if t >= s - menu_n_size and t >= 0:
            for mi, m in enumerate(menu['m']):
                for ni,n in enumerate(menu['n']):
                    v[s][t][mi][ni] = get_v(s=s, t=0, m=m, n=n)




# construct c_lin, Au_b, b_ub, A_eq, b_eq
r = charge_rate
c_lin = []

stmn2cnt = {}
cnt2stmn = {}

cnt = 0
for s in range(time_horizon):
    stmn2cnt[s] = {}
    for mi, m in enumerate(menu['m']):
        for ni,n in enumerate(menu['n']):
            for t in range(s-ni, s+1):
                if t < 0:
                    continue
                c_lin.append( v[s][t][mi][ni] )

                if t not in stmn2cnt[s]:
                    stmn2cnt[s][t] = {}
                if mi not in stmn2cnt[s][t]:
                    stmn2cnt[s][t][mi] = {}
                stmn2cnt[s][t][mi][ni] = cnt
                cnt2stmn[cnt] = (s,t,mi,ni)
                cnt += 1
# print(cnt)
# print(stmn2cnt)     
with open('stmn2cnt.json', 'w') as json_file:
    json.dump(stmn2cnt, json_file) 
with open('cnt2stmn.json', 'w') as json_file:
    json.dump(cnt2stmn, json_file) 

# add -c for e_s at end of c_lin
c_lin = np.concatenate(( np.array(c_lin), np.array(c)*(-1) ))
# print(c_lin)
# A_ub_buf = {}
# print(len(c_lin))
ye_size = len(c_lin)

A_eq_e = np.zeros(shape=(time_horizon, ye_size))
for s in range(time_horizon):
    A_eq_e[s][-(time_horizon-s)] = -1
    for mi, m in enumerate(menu['m']):
        for ni,n in enumerate(menu['n']):
            for t in range(s-ni, s+1):
                if t < 0:
                    continue
                A_eq_e[s][ stmn2cnt[s][t][mi][ni] ] = 1
b_eq_e = np.zeros(time_horizon)

# print(A_eq_e)
pd.DataFrame(A_eq_e).to_csv("A_eq_e.csv")

# print(A_ub_e)
b_eq_c = np.zeros(time_horizon*menu_m_size*menu_n_size)
A_eq_c = np.zeros(shape=(time_horizon*menu_m_size*menu_n_size, ye_size))
A_eq_c_cnt = 0
for t in range(0, time_horizon):
    for mi, m in enumerate(menu['m']):
        for ni,n in enumerate(menu['n']):
            # constraint
            for s in range(t, t+ni):
                # if stmn in the dic
                
                # if s in stmn2cnt and t in stmn2cnt[s] and mi in stmn2cnt[s][t] and ni in stmn2cnt[s][t][mi]:
                # if t in stmn2cnt[s]:
                #     if mi in stmn2cnt[s][t]:
                #         if ni in stmn2cnt[s][t][mi]:
                if s in stmn2cnt:
                    A_eq_c[A_eq_c_cnt][ stmn2cnt[s][t][mi][ni] ] = 1
                    b_eq_c[A_eq_c_cnt] = m*w[t][mi][ni]
                    # print(A_ub_c[A_ub_c_cnt])
        A_eq_c_cnt += 1

# np.set_printoptions(threshold=sys.maxsize)
pd.DataFrame(A_eq_c).to_csv("A_eq_c.csv")
pd.DataFrame(b_eq_c).to_csv("b_eq_c.csv")


A_eq = np.concatenate( (A_eq_e, A_eq_c) )
b_eq = np.concatenate( (b_eq_e, b_eq_c) )
# A_eq = A_eq_c
# b_eq = b_eq_c

pd.DataFrame(A_eq).to_csv("A_eq.csv")
pd.DataFrame(b_eq).to_csv("b_eq.csv")

A_ub_u = np.eye(ye_size)
b_ub_u = np.zeros(ye_size)
A_ub_l = -np.eye(ye_size)
b_ub_l = np.zeros(ye_size)

for s in range(time_horizon):
    for mi, m in enumerate(menu['m']):
        for ni,n in enumerate(menu['n']):
            for t in range(0, s):
                if s in stmn2cnt and t in stmn2cnt[s] and mi in stmn2cnt[s][t] and ni in stmn2cnt[s][t][mi]:
                    b_ub_u[ stmn2cnt[s][t][mi][ni] ] = r*w[t][mi][ni]

    # assign d
    b_ub_u[-time_horizon+s] = get_d(s)


A_ub = np.concatenate( (A_ub_u, A_ub_l) )
b_ub = np.concatenate( (b_ub_u, b_ub_l) )
# A_ub = A_ub_u
# b_ub = b_ub_u

pd.DataFrame(A_ub).to_csv("A_ub.csv")
pd.DataFrame(b_ub).to_csv("b_ub.csv")

# lambdas, V = np.linalg.eig(A_eq.T)
# print(np.shape(A_eq))
Q, R = np.linalg.qr(A_eq)
# print(np.shape(R))
# print(np.all(Q.dot(R)==A_eq))
rank_R = matrix_rank(R)
# print(np.shape(R))
# print(matrix_rank(R[0:rank_R]))

R_reduction = R[np.where(R.any(axis=1))]
# print(matrix_rank(R_reduction), np.shape(R_reduction))
R_non_zero = np.where(R.any(axis=1))[0]
Q_reduction = Q[:,R_non_zero]
# Q_reduction = Q_reduction[np.where(R.any(axis=1))]
# print('R:', np.shape(R_reduction))
# print('Q:', np.shape(Q_reduction))
eq_prj = np.linalg.pinv(Q_reduction)
print(eq_prj.dot(A_eq), eq_prj.dot(b_eq))
# print(matrix_rank(eq_prj.dot(A_eq)), np.shape(eq_prj.dot(A_eq)))


result_prj = linprog(
    c=-c_lin, 
    A_eq=eq_prj.dot(A_eq), 
    b_eq=eq_prj.dot(b_eq), 
    A_ub=A_ub, 
    b_ub=b_ub
    )
result = linprog(
    c=-c_lin, 
    A_eq=A_eq, 
    b_eq=b_eq, 
    A_ub=A_ub, 
    b_ub=b_ub,
    method='simplex'
    )
print(result.x)
print(-result.fun)
# print(result_prj.x)

# # construct z_s and the matrix of J^*
# z_s = {}
# for t in range(s):
#     z_s[t] = {}
#     for mi,m in enumerate(menu['m']):
#         z_s[t][mi] = {}
#         for ni,n in enumerate(menu['n']):
#             # print(w[t][mi][ni])
#             z_s[t][mi][ni] = np.arange(0, r*w[t][mi][ni], 1)

y = {}
for cnt in cnt2stmn:
    s, t, m, n = cnt2stmn[cnt]
    if s not in y:
        y[s] = {}
    if t not in y[s]:
        y[s][t] = {}
    if m not in y[s][t]:
        y[s][t][m] = {}
            
    y[s][t][m][n] = result.x[cnt]
e = result.x[-time_horizon:]
# print(y, e)
res = {
    'y': y,
    'e': e
}
with open('result.json', 'w') as json_file:
    json.dump(y, json_file) 
    
print(np.sum(e))




