import sys


import numpy as np
import pandas as pd

from cvxopt import matrix, solvers


from tools.Parameter import Parameter
from tools.Squeeze import Squeeze
from tools.Result import Result
from tools.Arrivals import Arrivals
from tools.Approximator import Approximator

from Static_H import Static_H

import logging
import json
import pickle
import time




def CVXOPT_LP(s, pa, ye):
    pa_dic = pa.get_LP_dynamic(ye=ye, s=s)
    
    c = pa_dic['c']
    A_eq, b_eq = pa_dic['A_eq'], pa_dic['b_eq']
    A_ub, b_ub = pa_dic['A_ub'], pa_dic['b_ub']

    
    '''
    use CVXOPT can directly get lagrange
    '''
    solvers.options['show_progress'] = True

    # s_time = time.time()
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    solvers.options['msg_lev'] = 'GLP_MSG_OFF'
    solvers.options['LP_K_MSGLEV'] = 0
    solvers.options['abstol'] = 1e-4
    solvers.options['reltol'] = 1e-4
    solvers.options['feastol'] = 1
    solvers.options['refinement'] = 1

    result = solvers.lp(
        c=matrix(-c), 
        G=matrix(A_ub), 
        h=matrix(b_ub), 
        A=matrix(A_eq), 
        b=matrix(b_eq), 
        solver='glpk'
        )

    mu_e = result['y']
    mu_ie = result['z']
    pa.input_lagrange(s, mu_e, mu_ie)
    # result = Result(pa=pa, x=result['x'], dur=0)
    
    return np.squeeze(np.asarray(result['x']))




def Dynamic_H():
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(threshold=sys.maxsize)


    with open('cache/result_static.pickle', 'rb') as pickle_file:
        result_static = pickle.load(pickle_file) 

    with open('cache/cache_static.pickle', 'rb') as pickle_file:
        result_cache = pickle.load(pickle_file) 

    pa = result_cache['pa']

    ye = np.ones(len(pa.c_lin))*(-1)
    
    s_time = time.time()
    for _s in range(1,pa.time_horizon+1):
        s = pa.time_horizon-_s

        result = CVXOPT_LP(s, pa, result_static)
        # print(result, s)

        for i,tmn in enumerate(pa.cnt2tmn_s[s]):
            t, mi, ni = tmn
            # print(s,t,mi,ni)
            if s in pa.stmn2cnt and t in pa.stmn2cnt[s] and mi in pa.stmn2cnt[s][t] and ni in pa.stmn2cnt[s][t][mi]:
                ye[ pa.stmn2cnt[s][t][mi][ni] ] = float(result[i])

    dur = time.time()-s_time

    result = Result(pa=pa, x=ye, dur=dur)
    result.output(file='dynamic')



def Dynamic_Data(data_size=0):
    info = {
        'time_horizon': 48,
        'unit': 'hour',
        'arrival_rate': 20, #/hour
        'RoC': 10, #kw
        'BC': 50, #kwh
        'm': 4,
        'n': 4,
        'seed': 0
    }
    result = Static_H(info)
    pa = result.pa
    ye = result.result_output
    pa.get_z(ye)

    tmn2cnt = {}
    cnt2tme = []
    for mi in range(pa.menu_m_size):
        for ni in range(pa.menu_m_size):
            t = 0

    # formalized {train_x,train_z}
    s0 = 0
    
    value_s = 0
    z = np.zeros(pa.time_horizon)
    x = np.zeros(shape=(2*pa.menu_n_size*pa.menu_m_size*pa.menu_n_size, pa.time_horizon))

    for s in range(s0, pa.time_horizon):
        # x.append( pa.get_state(ye, s).flatten() )
        x[:,s] = pa.get_state(ye, s).flatten().T

        for t in ye['y'][s]:
            for mi in ye['y'][s][t]:
                for ni in ye['y'][s][t][mi]:
                    value_s += ye['y'][s][t][mi][ni]*pa.v[s][t][mi][ni]
        value_s -= pa.c[s]*ye['e'][s]
        z[s] = value_s

    ap = Approximator()
    # cur = ap.bootstrapping(x,z)
    setting = {
        'kernel': 'poly',
        'degree': 5,
        'gamma': 'auto',
        'tol': 10e-6
    }
    cur = ap.sklearn_svm(x, z, setting)
    ap.check(cur, x, z)


def main():
    # Dynamic_H()
    Dynamic_Data()


if __name__ == "__main__":
    main()