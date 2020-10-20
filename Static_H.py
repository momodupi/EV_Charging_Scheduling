import sys


import numpy as np
import pandas as pd

from numpy.linalg import matrix_rank

from scipy.optimize import linprog
from scipy.linalg import null_space
from scipy.stats import gamma

from cvxopt import matrix, solvers

import logging
import json
import pickle
import time

from tools.Parameter import Parameter
from tools.Squeeze import Squeeze
from tools.Result import Result
from tools.Arrivals import Arrivals




def scipy_LP(c, A_eq, b_eq, A_ub, b_ub):
    logging.info('LP: solver: scipy interior-point')
    logging.info('LP: Start')
    result = linprog(
            c=c, 
            A_eq=A_eq, 
            b_eq=b_eq, 
            A_ub=A_ub, 
            b_ub=b_ub,
            # bounds=[0., None],
            method='interior-point',
            # method='simplex'
            options={
                'lstsq': True,
                'presolve': True
            }
        )
    logging.info('LP: Done')
    return result.x


def CVXOPT_LP(c, A_eq, b_eq, A_ub, b_ub):
    '''
    LP using CVXOPT
    '''
    logging.info('LP: solver: CVXOPT')
    solvers.options['show_progress'] = False
    logging.info('LP: Start')
    result = solvers.lp(
        c=matrix(c), 
        G=matrix(A_ub), 
        h=matrix(b_ub), 
        A=matrix(A_eq), 
        b=matrix(b_eq)
        )
    if result['x'] == None:
        logging.warning('CVXOPT bag solution')
        return np.zeros(len(c))
    logging.info('LP: Done')
    return np.squeeze(np.asarray(result['x']))


def Static_H(info):
    # logging.basicConfig(level=logging.INFO)
    # np.set_printoptions(threshold=sys.maxsize)

    ar = Arrivals(setting=info)
    # ar.EV_arrivals()

    arrival_matrix = {}
    for t in range(ar.time_horizon):
        arrival_matrix[t] = {}
        for mi,m in enumerate(ar.menu['m']):
            arrival_matrix[t][mi] = {}
            for ni,n in enumerate(ar.menu['n']):
                arrival_matrix[t][mi][ni] = 20
    ar.simple_arrivals(arrival_matrix)
    # print(ar.w)
    
    pa = Parameter(ar, readable=True)

    pa_dic = pa.get_LP_static()
    c = pa_dic['c']
    A_eq, b_eq = pa_dic['A_eq'], pa_dic['b_eq']
    A_ub, b_ub = pa_dic['A_ub'], pa_dic['b_ub']


    # dimentionality reduction
    # sq = Squeeze(A_eq=A_eq, b_eq=b_eq)

    # k = int(np.shape(A_eq)[0]/4)+1
    # A_eq, b_eq = sq.Random_Projection(k=k, method='')

    s_time = time.time()
    
    # choose different solver
    # x = scipy_LP(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)
    x = CVXOPT_LP(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)

    dur = time.time()-s_time
    result = Result(pa=pa, x=x, dur=dur)
    result.check()
    result.output('static')
    return result



def main():

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
    Static_H(info)

if __name__ == "__main__":
    main()
