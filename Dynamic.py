import sys


import numpy as np
import pandas as pd

from cvxopt import matrix, solvers


from Parameter import Parameter
from Squeeze import Squeeze
from Result import Result

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
    solvers.options['show_progress'] = False

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





def main():
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

        for i,tmn in enumerate(pa.cnt2tmn_s[s]):
            t, mi, ni = tmn
            # print(s,t,mi,ni)
            if s in pa.stmn2cnt and t in pa.stmn2cnt[s] and mi in pa.stmn2cnt[s][t] and ni in pa.stmn2cnt[s][t][mi]:
                ye[ pa.stmn2cnt[s][t][mi][ni] ] = float(result[i])

    dur = time.time()-s_time

    result = Result(pa=pa, x=ye, dur=dur)
    result.output(file='dynamic')

if __name__ == "__main__":
    main()