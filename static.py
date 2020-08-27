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


np.set_printoptions(threshold=sys.maxsize)

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

# get w_s
w = {}
for t in range(time_horizon):
    # w[t] = np.zeros(shape=(menu_m_size,menu_n_size))
    w[t] = {}
    for mi,m in enumerate(menu['m']):
        w[t][mi] = {}
        for ni,n in enumerate(menu['n']):
            w[t][mi][ni] = 0

for index in EV:
    # tran t to time
    t = EV[index]['arrive']
    mi = EV[index]['m']
    ni = EV[index]['n']
    w[t][mi][ni] += 1

# print(w)
# reform w to csv
w_csv = { }
for mi,m in enumerate(menu['m']):
    for ni,n in enumerate(menu['n']):
        w_csv[f'({m},{n})'] = []
        for t in w:
            w_csv[f'({m},{n})'].append(int(w[t][mi][ni]))


pd.DataFrame(w_csv).to_csv("w.csv")


with open('w.pickle', 'wb') as pickle_file:
    pickle.dump(w, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 

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
    base_price = 0.11 # $/kWh
    # return ((m*menu_m_step) * base_price)/(1+n/100)

    '''
    price = base price *  amount of electricty \\
            * price turbulance over time (gamma) 
            / time flexibility discount
    '''
    # return (1+price_turbulance(s)) * base_price*m / n
    return base_price*m / (1+n/10)

def get_c(s):
    # base_price = 0.12 # $/kWh
    # return (1+price_turbulance(s)) * base_price
    # print(s/time_unit_set[time_unit])
    cur_time = int(s/time_unit_set[time_unit]) % 24
    return 9.341/100 if cur_time >= 11 and cur_time <= 19 else 0.948/100
    # return 0.09341


def get_d(s):
    return 2000. # kWh

# set all v,c,z
v = {}
c = []
z = {}
d = []
for s in range(time_horizon):
    c.append(get_c(s=s))
    d.append(get_d(s=s))
    v[s] = {}
    # z[s] = {}
    
    # for t in range(s-menu_n_size, s+1):
    for t in range(s-menu_n_size+1, s+1):
        v[s][t] = np.zeros(shape=(menu_m_size,menu_n_size))

        # if t >= s - menu_n_size and t >= 0:
        if t >= 0:
            for mi, m in enumerate(menu['m']):
                for ni,n in enumerate(menu['n']):
                    v[s][t][mi][ni] = get_v(s=s, t=0, m=m, n=n)

with open('v.pickle', 'wb') as pickle_file:
    pickle.dump(v, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 

with open('c.pickle', 'wb') as pickle_file:
    pickle.dump(c, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
with open('d.pickle', 'wb') as pickle_file:
    pickle.dump(d, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
# print(v)

# construct c_lin, Au_b, b_ub, A_eq, b_eq
r = charge_rate
c_lin = []

stmn2cnt = {}
cnt2stmn = {}


'''
Note: 
t in [s-n+1, s]
s in [t, t+n-1]
where n starting from 1 (not ni)
'''

cnt = 0
for s in range(time_horizon):
    stmn2cnt[s] = {}
    for mi, m in enumerate(menu['m']):
        for ni,n in enumerate(menu['n']):
            # this should be n
            # for t in range(s-n, s+1):
            for t in range(s-n+1, s+1):
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

with open('stmn2cnt.pickle', 'wb') as pickle_file:
    pickle.dump(stmn2cnt, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 
with open('cnt2stmn.pickle', 'wb') as pickle_file:
    pickle.dump(cnt2stmn, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 

# add -c for e_s at end of c_lin
c_lin = np.concatenate(( np.array(c_lin), np.array(c)*(-1) ))

pd.DataFrame(c_lin).to_csv("c_lin.csv")

# print(c_lin)
# A_ub_buf = {}
# print(len(c_lin))
ye_size = len(c_lin)

A_eq_e = np.zeros(shape=(time_horizon, ye_size))
for s in range(time_horizon):
    A_eq_e[s][-(time_horizon-s)] = -1
    for mi, m in enumerate(menu['m']):
        for ni,n in enumerate(menu['n']):
            # this should be n
            # for t in range(s-n, s+1):
            for t in range(s-n+1, s+1):
                if t < 0:
                    continue
                A_eq_e[s][ stmn2cnt[s][t][mi][ni] ] = 1
b_eq_e = np.zeros(time_horizon)
# print(b_eq_e)

# test_aeq = np.zeros(ye_size)
# for s in range(time_horizon):
#     test_aeq += A_eq_e[s]

# pd.DataFrame(test_aeq).to_csv("test_aeq.csv")


# print(A_eq_e)
pd.DataFrame(A_eq_e).to_csv("A_eq_e.csv")
pd.DataFrame(b_eq_e).to_csv("b_eq_e.csv")

# print(A_ub_e)
b_eq_c = np.zeros(time_horizon*menu_m_size*menu_n_size)
A_eq_c = np.zeros(shape=(time_horizon*menu_m_size*menu_n_size, ye_size))
A_eq_c_cnt = 0


for t in range(0, time_horizon):
    for mi,m in enumerate(menu['m']):
        for ni,n in enumerate(menu['n']):
            # constraint, this should be n

            # u_s = t+n+1 if t+n+1<=time_horizon else time_horizon
            u_s = t+n if t+n<=time_horizon else time_horizon

            for s in range(t, u_s):
                # if stmn in the dic
                # print(s,t,mi,ni)
                # if s in stmn2cnt and t in stmn2cnt[s] and mi in stmn2cnt[s][t] and ni in stmn2cnt[s][t][mi]:
                # if t in stmn2cnt[s]:
                #     if mi in stmn2cnt[s][t]:
                #         if ni in stmn2cnt[s][t][mi]:
                # if s in stmn2cnt:
                # if t==0:
                #     print(s,t,mi,ni, u_s, stmn2cnt[s][t][mi][ni])
                A_eq_c[ A_eq_c_cnt, stmn2cnt[s][t][mi][ni] ] = 1
                    # print(A_ub_c[A_ub_c_cnt])
            # m should be the multiple of r
            # m_ceil = (int(m/r))*r
            b_eq_c[A_eq_c_cnt] = m*w[t][mi][ni]
            A_eq_c_cnt += 1

total_demand = np.sum(b_eq_c)

# test_aeq = np.zeros(ye_size)
# for s in range(A_eq_c_cnt):
#     test_aeq += A_eq_c[s,:]

# pd.DataFrame(test_aeq).to_csv("test_aeq.csv")


# np.set_printoptions(threshold=sys.maxsize)
pd.DataFrame(A_eq_c).to_csv("A_eq_c.csv")
pd.DataFrame(b_eq_c).to_csv("b_eq_c.csv")


A_eq = np.concatenate( (A_eq_e, A_eq_c) )
b_eq = np.concatenate( (b_eq_e, b_eq_c) )

# A_eq = A_eq_e
# b_eq = b_eq_e
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
            for t in range(0, s+1):
                if s in stmn2cnt and t in stmn2cnt[s] and mi in stmn2cnt[s][t] and ni in stmn2cnt[s][t][mi]:
                    # print(w[t][mi][ni])
                    b_ub_u[ stmn2cnt[s][t][mi][ni] ] = r*w[t][mi][ni]

    # assign d
    # b_ub_u[-time_horizon+s] = get_d(s)
    b_ub_u[-time_horizon+s] = d[s]


A_ub = np.concatenate( (A_ub_u, A_ub_l) )
b_ub = np.concatenate( (b_ub_u, b_ub_l) )
# A_ub = A_ub_u
# b_ub = b_ub_u

pd.DataFrame(A_ub).to_csv("A_ub.csv")
pd.DataFrame(b_ub).to_csv("b_ub.csv")


''' QR decompositon'''
# print(np.shape(A_eq))
Q, R = np.linalg.qr(A_eq)
# print(np.shape(R))
# print(np.all(Q.dot(R)==A_eq))
rank_R = matrix_rank(R)
# print(np.shape(R))
# print(matrix_rank(R[0:rank_R]))

# dimension reduction of A_eq
R_reduction = R[np.where(R.any(axis=1))]
# print(matrix_rank(R_reduction), np.shape(R_reduction))
R_non_zero = np.where(R.any(axis=1))[0]
Q_reduction = Q[:,R_non_zero]
# Q_reduction = Q_reduction[np.where(R.any(axis=1))]
# print('R:', np.shape(R_reduction))
# print('Q:', np.shape(Q_reduction))
eq_prj = np.linalg.pinv(Q_reduction)
# print(eq_prj.dot(A_eq), eq_prj.dot(b_eq))

# print(matrix_rank(A_eq), np.shape(A_eq))
# print(matrix_rank(eq_prj.dot(A_eq)), np.shape(eq_prj.dot(A_eq)))


'''
projected LP
'''
# s_time = time.time()
# result = linprog(
#         c=-c_lin, 
#         A_eq=eq_prj.dot(A_eq), 
#         b_eq=eq_prj.dot(b_eq), 
#         # A_eq=A_eq, 
#         # b_eq=b_eq,
#         A_ub=A_ub, 
#         b_ub=b_ub,
#         # bounds=[0, None],
#         method='interior-point'
#         # method='simplex'
#     )
# # e_time = time.time()
# lp_dur = time.time() - s_time

# print(result_prj.x)

'''
LP
'''
s_time = time.time()
result = linprog(
        c=-c_lin, 
        A_eq=A_eq, 
        b_eq=b_eq, 
        A_ub=A_ub, 
        b_ub=b_ub,
        bounds=[0., None],
        method='interior-point',
        # method='simplex'
        options={
            'lstsq': True,
            'presolve': True
        }
    )
lp_dur = time.time() - s_time

'''
LP using CVXOPT
'''
# c = matrix(-c_lin)
# G = matrix(A_ub)
# h = matrix(b_ub)
# A = matrix(A_eq)
# b = matrix(b_eq)

# solvers.options['show_progress'] = False

# s_time = time.time()
# sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
# lp_dur = time.time() - s_time

# class Result(object):
#     def __init__(self, sol):
#         self.sol = sol
#         self.x = np.array(sol['x'])
#         # print(sol)
#         self.fun = -c_lin.dot(self.x)
        
# result = Result(sol)



print(f'lp time: {lp_dur}')

# print('y: ', result.x[:-time_horizon])
# print('e: ', result.x[-time_horizon:])

print(f'value: {-result.fun}')

print(f'test: y=e: {np.abs(np.sum(result.x[:-time_horizon])-np.sum(result.x[-time_horizon:]))<10e-4}' )
print(f'test: demand=supply: {np.abs(total_demand-np.sum(result.x[-time_horizon:]))<10e-4}' )
print(f'test: pd: {np.all((np.abs(A_eq_e.dot(result.x)-b_eq_e) <10e-4))}')
print(f'test: te: {np.all((np.abs(A_eq_c.dot(result.x)-b_eq_c) <10e-4))}')

pd.DataFrame(result.x).to_csv("result_static.csv")
with open('result_static.pickle', 'wb') as pickle_file:
    pickle.dump(result.x, pickle_file)


# print(result_prj.x)
# print(result_prj.x==result.x)

# print(-result_prj.fun)
# print(np.allclose(result.x, result_prj.x))
# print(t1, t2)
# print(-c_lin.dot(result.x))

# pd.DataFrame(np.array( [A_eq.dot(result.x), b_eq] )).to_csv("test_eq_c.csv")



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
    s, t, mi, ni = cnt2stmn[cnt]
    if s not in y:
        y[s] = {}
    if t not in y[s]:
        y[s][t] = {}
    if mi not in y[s][t]:
        y[s][t][mi] = {}
    y[s][t][mi][ni] = result.x[cnt]
    
e = result.x[-time_horizon:]

# varification
varify = True
for s in stmn2cnt:
    for t in stmn2cnt[s]:
        for mi in stmn2cnt[s][t]:
            for ni in stmn2cnt[s][t][mi]:
                if y[s][t][mi][ni] != result.x[ stmn2cnt[s][t][mi][ni] ]:
                    varify = False
print(f'test: output: {varify}')

# print(y, e)
result_json_output = {
    'y': y,
    'e': list(e)
}
result_pickle_output = {
    'y': y,
    'e': e
}

# with open('result_static.json', 'w') as json_file:
#     json.dump(result_json_output, json_file) 
    
with open('result_static.pickle', 'wb') as pickle_file:
    pickle.dump(result_pickle_output, pickle_file) 
    
# # convert to csv
# y_df = {}
# for s in y:
#     # y_df['time'] = [s for s in range(time_horizon)]
#     for t in y[s]:
#         for mi,m in enumerate(y[s][t]):
#             for ni,n in enumerate(y[s][t][mi]):
#                 if f'(t={t},{menu["m"][m]},{menu["n"][n]})' not in y_df:
#                     y_df[f'(t={t},{menu["m"][m]},{menu["n"][n]})'] = np.ones(time_horizon)*(-1)
#                 if ni in y[s][t][mi]:
#                     y_df[f'(t={t},{menu["m"][m]},{menu["n"][n]})'][t] = y[s][t][mi][ni]
#                 else:
#                     y_df[f'(t={t},{menu["m"][m]},{menu["n"][n]})'][t] = -1
# pd.DataFrame(y_df).to_csv('y_static.csv')

# convert to xlsx
y_xlse = pd.ExcelWriter('y_static.xlsx', engine='xlsxwriter')
# y_df = []
for s in y:
    # y_df_s = {'arrive time': [t for t in range(time_horizon)]}
    y_df_s = {}
    for t in y[s]:
        # y_df_s['arrive time'].append(int(t))
        for mi in y[s][t]:
            for ni in y[s][t][mi]:
                
                if ni in y[s][t][mi]:
                    if f'({menu["m"][mi]},{menu["n"][ni]})' not in y_df_s:
                        y_df_s[f'({menu["m"][mi]},{menu["n"][ni]})'] = np.ones(time_horizon)*(-1)
                    y_df_s[f'({menu["m"][mi]},{menu["n"][ni]})'][t] = y[s][t][mi][ni]
                # if ni in y[s][t][mi]:
                #     y_df_s[f'({menu["m"][m]},{menu["n"][n]})'][t] = y[s][t][mi][ni]
                # else:
                #     y_df_s[f'({menu["m"][m]},{menu["n"][n]})'][t] = -1
                # if ni in y[s][t][mi]:
                #     if f'({m},{n})' not in y_df_s:
                #         y_df_s[f'({m},{n})'] = []
                #     y_df_s[f'({m},{n})'].append(y[s][t][mi][ni])
                # else:
                #     y_df_s[f'({m},{n})'].append(0)
    # y_df.append(y_df_s)
    y_df_s = pd.DataFrame(y_df_s)
    y_df_s.to_excel(y_xlse, sheet_name=f'time={s}')
y_xlse.save()

pd.DataFrame(e).to_csv('e_static.csv')




