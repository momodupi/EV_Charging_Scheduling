from copy import copy, deepcopy
from math import inf
import sys


import numpy as np
from numpy.core.defchararray import translate
import pandas as pd

from cvxopt import matrix, solvers


from tools.Parameter import Parameter
from tools.Squeeze import Squeeze
from tools.Result import Result
from tools.Arrivals import Arrivals
from tools.Approximator import Approximator

from Static_H import Static_H

import logging
import queue
import json
import copy
import pickle
import time
import multiprocessing
import os



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

def CVXOPT_OPT(s, pa, ye, ap_para):
    pa_dic = pa.get_LP_dynamic(ye=ye, s=s)
    
    c = pa_dic['c']
    A_eq, b_eq = pa_dic['A_eq'], pa_dic['b_eq']
    A_ub, b_ub = pa_dic['A_ub'], pa_dic['b_ub']

    
    '''
    use CVXOPT can directly get lagrange
    '''
    solvers.options['show_progress'] = True

    Q = 2*matrix([ [2, .5], [.5, 1] ])
    p = matrix([1.0, 1.0])
    G = matrix(A_ub)
    h = matrix(b_ub)
    A = matrix(A_eq)
    b = matrix(b_eq)
    result = solvers.qp(Q, p, G, h, A, b)
    return np.squeeze(np.asarray(result['x']))


def Dynamic_H():
    # logging.basicConfig(level=logging.INFO)
    # np.set_printoptions(threshold=sys.maxsize)


    with open('cache/result_static.pickle', 'rb') as pickle_file:
        result_static = pickle.load(pickle_file) 

    with open('cache/approx_rq_100.pickle', 'rb') as pickle_file:
        ap_s = pickle.load(pickle_file) 

    

    with open('cache/cache_static.pickle', 'rb') as pickle_file:
        result_cache = pickle.load(pickle_file) 

    pa = result_cache['pa']

    ye = np.ones(len(pa.c_lin))*(-1)
    
    s_time = time.time()
    # for _s in range(1,pa.time_horizon+1):
    #     s = pa.time_horizon-_s

    #     result = CVXOPT_LP(s, pa, result_static)
    #     # print(result, s)

    #     for i,tmn in enumerate(pa.cnt2tmn_s[s]):
    #         t, mi, ni = tmn
    #         # print(s,t,mi,ni)
    #         if s in pa.stmn2cnt and t in pa.stmn2cnt[s] and mi in pa.stmn2cnt[s][t] and ni in pa.stmn2cnt[s][t][mi]:
    #             ye[ pa.stmn2cnt[s][t][mi][ni] ] = float(result[i])



    for s in range(1,pa.time_horizon+1):
        ap = ap_s[s]
        Q = ap.parameter['Q']
        b = ap.parameter['b']
        c = ap.parameter['c']
        ap_para = {
            'Q': Q,
            'b': b,
            'c': c
        }

        result = CVXOPT_LP(s, pa, result_static, ap_para)

    dur = time.time()-s_time

    result = Result(pa=pa, x=ye, dur=dur)
    result.output(file='dynamic')



def Generators_single_process(info):
    result = Static_H(info)
    pa = result.pa
    ye = result.result_output
    pa.get_z(ye)
    pa.readable = False

    value_s = 0
    data = {}
    for i in range(info['time_horizon']):
        data[i] = []
    # terminal cost is useless: 0 ~ T-1
    for _s in range(1,pa.time_horizon+1):
        s = pa.time_horizon-_s
        # x_s = pa.get_state(ye, s).flatten().T
        x_s = pa.get_state(ye, s)
        for t in ye['y'][s]:
            for mi in ye['y'][s][t]:
                for ni in ye['y'][s][t][mi]:
                    value_s -= ye['y'][s][t][mi][ni]*pa.v[s][t][mi][ni]
            value_s += pa.c[s]*ye['e'][s]
        z_s = value_s
        data[s].append( (x_s,z_s) )
    return data



def Traning_Generators(data_size=0, info=None, mp=False):
    process_pool = []
    # result_queue = queue.Queue()
    # result_queue = multiprocessing.Manager().Queue()

    for cnt in range(data_size):
        p_info = copy.deepcopy(info)
        p_info['seed'] = cnt
        process_pool.append(p_info)

    with multiprocessing.Pool() as pool:
        data_list = pool.map( Generators_single_process, process_pool )
        
        # print(data_list)

    data = {}
    for i in range(info['time_horizon']):
        data[i] = []
    for s_data in data_list:
        for s in s_data:
            data[s].append(s_data[s][0])

    with open('cache/training_data.pickle', 'wb') as pickle_file:
        pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 
    # if not mp:
    #     data = {}
    #     # initialize x and z for each time
    #     for i in range(info['time_horizon']):
    #         data[i] = []

    #     for cnt in range(data_size):
    #         info['seed'] = cnt

    #         s_data = Generators_single_process(info)
        
    #         for s in s_data:
    #             data[s].append(s_data[s][0])
            
    #         # result = Static_H(info)
    #         # pa = result.pa
    #         # ye = result.result_output
    #         # pa.get_z(ye)
    #         # pa.readable = True

    #         # value_s = 0

    #         # # terminal cost is useless: 0 ~ T-1
    #         # for s in range(0, pa.time_horizon):
    #         #     x_s = pa.get_state(ye, s).flatten().T
    #         #     for t in ye['y'][s]:
    #         #         for mi in ye['y'][s][t]:
    #         #             for ni in ye['y'][s][t][mi]:
    #         #                 value_s -= ye['y'][s][t][mi][ni]*pa.v[s][t][mi][ni]
    #         #         value_s += pa.c[s]*ye['e'][s]
    #         #     z_s = value_s
    #         #     data[s].append( (x_s,z_s) )

    #     # with open('cache/training_data.pickle', 'wb') as pickle_file:
    #     #     pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 
    
    # else:
    #     # process_cnt = min(int(multiprocessing.cpu_count()/2), data_size)
    #     process_pool = []
    #     # result_queue = queue.Queue()
    #     # result_queue = multiprocessing.Manager().Queue()

    #     for cnt in range(data_size):
    #         p_info = copy.deepcopy(info)
    #         p_info['seed'] = cnt
    #         process_pool.append(p_info)

    #     with multiprocessing.Pool() as pool:
    #         data_list = pool.map( Generators_single_process, process_pool )
            
    #         # print(data_list)

    #     data = {}
    #     for i in range(info['time_horizon']):
    #         data[i] = []
    #     for s_data in data_list:
    #         for s in s_data:
    #             print(s_data[s][0])
    #             data[s].append(s_data[s][0])

    # with open('cache/training_data.pickle', 'wb') as pickle_file:
    #     pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 
    # # print(data)


def Fitting_single_process(args):
    training_data = args['d']
    testing_data = args['e']
    settings = args['m'].split('_')

    method = settings[0]

    for s in training_data:
        # print(training_data[s])
        data_size = len(training_data[s])
        x_size = len(training_data[s][0][0])

        x_train = np.zeros(shape=(x_size, data_size))
        z_train = np.zeros(data_size)

        x_test = np.zeros(shape=(x_size, data_size))
        z_test = np.zeros(data_size)
        
        for k,data_k in enumerate(training_data[s]):
            x_train[:,k] = data_k[0]
            z_train[k] = data_k[1]

            x_test[:,k] = testing_data[s][k][0]
            z_test[k] = testing_data[s][k][1]

        if 'p' in settings:
            sq = Squeeze()
            pca_setting = {
                'nc': int(settings[settings.index('p')+1]),
                'fit': False,
                'var_thrsd': 0.1
            }
            x_train = sq.PCA(x_train, pca_setting)

            pca_setting['fit'] = True
            x_test= sq.PCA(x_test, pca_setting)

        ap = Approximator()
        if method == 'rq':
            setting = {'convex': True, 'basis_size': int(settings[1])}
            cur = ap.quadratic_random_matrix( x_train, z_train, setting )
        elif method == 'br':
            setting = {'basis': settings[1], 'buckets': int(settings[2])}
            cur = ap.bregman_div( x_train, z_train, setting )
        else:
            setting = {'convex': True, 'basis_size': int(settings[1])}
            cur = ap.quadratic_random_matrix( x_train, z_train, setting )

        # method_set = {
        #     'rq': ap.quadratic_random_matrix,
        #     'br': ap.bregman_div,
        #     'scr': ap.strongly_convex_random_matrix
        # }
        # cur = method_set[method]( x_train, z_train, setting_set[method] )
        # cur = method_set[method]( x_train, z_train, setting )

        # print(f'training: s={s}')
        # ap.check(cur, x_train, z_train)

        ap.check(cur, x_test, z_test)
        # print(int(settings[settings.index('p')+1]), ap.runtime)
        ap_s = copy.deepcopy(ap)

        with open(f'cache/approx_{args["m"]}_{s}.pickle', 'wb') as pickle_file:
            pickle.dump(ap_s, pickle_file)

    ap_set = []
    for s in training_data:
        with open(f'cache/approx_{args["m"]}_{s}.pickle', 'rb') as pickle_file:
            ap_s = pickle.load(pickle_file)
        ap_set.append(ap_s)

    with open(f'cache/approx_{args["m"]}.pickle', 'wb') as pickle_file:
        pickle.dump(ap_set, pickle_file)
    
    for s in training_data:
        if os.path.exists(f'cache/approx_{args["m"]}_{s}.pickle'):
            os.remove(f'cache/approx_{args["m"]}_{s}.pickle')



def main():
    logging.basicConfig(level=logging.INFO, filename='ap_res.log')
    Dynamic_H()
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
    # Traning_Generators(2, info, True)

    # with open('cache/training_data.pickle', 'rb') as pickle_file:
    #     training_data = pickle.load(pickle_file)

    # with open('cache/training_data.pickle', 'rb') as pickle_file:
    #     testing_data = pickle.load(pickle_file)

    # with open('cache/training_data_1000.pickle', 'rb') as pickle_file:
    #     training_data = pickle.load(pickle_file)

    # with open('cache/testing_data_1000.pickle', 'rb') as pickle_file:
    #     testing_data = pickle.load(pickle_file)


    # process_pool = [
    #     {'d': copy.deepcopy(training_data), 'e': copy.deepcopy(testing_data), 'm': 'rq_100'}
    # ]

    # # process_pool = [
    # #     {'d': copy.deepcopy(training_data), 'e': copy.deepcopy(testing_data), 'm': 'rq_10'},
    # #     {'d': copy.deepcopy(training_data), 'e': copy.deepcopy(testing_data), 'm': 'rq_10'},
    # #     {'d': copy.deepcopy(training_data), 'e': copy.deepcopy(testing_data), 'm': 'rq_10'}
    # #  ]
    # # process_pool = []
    # # for i in range(1,50,1):
    # #     process_pool.append(
    # #         {
    # #             'd': copy.deepcopy(training_data), 
    # #             'e': copy.deepcopy(testing_data), 
    # #             'm': f'rq_10_p_{i+1}'
    # #         }
    # #     )

    # with multiprocessing.Pool() as pool:
    #     pool.map( Fitting_single_process, process_pool )

    # with open('cache/training_data_1000.pickle', 'rb') as pickle_file:
    #     training_data = pickle.load(pickle_file)


    # with open('cache/approx_rf_10000.pickle', 'rb') as pickle_file:
    #     ap = pickle.load(pickle_file)
    # print(ap[0].check_res)
    
    # with open('cache/test_data.pickle', 'rb') as pickle_file:
    #     test_data = pickle.load(pickle_file)

    # ap = Approximator()

    # setting = {
    #     'convex': True,
    #     'basis_size': 1,
    # }

    # approx = {}

    # for s in training_data:
    #     # print(training_data[s])
    #     data_size = len(training_data[s])
    #     x_size = len(training_data[s][0][0])

    #     x_train = np.zeros(shape=(x_size, data_size))
    #     z_train = np.zeros(data_size)
        
    #     for k,data_k in enumerate(training_data[s]):
    #         x_train[:,k] = data_k[0]
    #         z_train[k] = data_k[1]
        
    #     # pd.DataFrame(x_train).to_csv(f'cache/x_training_{s}.csv')
    #     # pd.DataFrame(z_train).to_csv(f'cache/z_training_{s}.csv')
    #     # print('saved')

    #     # print(np.shape(x_train))
    #     # x_prj = ap.PCA(x_train.T)
    #     # print(np.shape(x_prj))
    #     # cur = ap.quadratic_random_matrix(x_prj, z_train, setting)
    #     ap = Approximator()

    #     setting = {
    #         'convex': True,
    #         'basis_size': 1000,
    #     }
    #     cur = ap.quadratic_random_matrix(x_train, z_train, setting)
    #     # Q = ap.parameter['Q']
    #     # b = ap.parameter['b']

    #     # print( np.all(np.linalg.eigvals(Q) > 0) )

    #     # setting = {
    #     #     'kernel': 'poly',
    #     #     'degree': 2,
    #     #     'gamma': 'auto',
    #     #     'tol': 10e-6
    #     # }
    #     # cur = ap.sklearn_svm(x_train,z_train, setting=setting)

    #     # setting = {
    #     #     'alpha': 10e-6,
    #     #     'random_state': 1,
    #     #     'hidden_layer': (5,5),
    #     #     'solver': 'lbfgs',
    #     #     'activation': 'relu',
    #     #     'tol': 10e-3,
    #     #     'max_iter': 50000,
    #     #     'learning_rate': 'constant'
    #     # }

    #     # cur = ap.sklearn_neural(x_train,z_train, setting=setting)

    #     # cur = ap.pytorch_neural(x_train,z_train, setting=None)

    #     print(f'training: s={s}')
    #     ap.check(cur, x_train, z_train)

    #     # ap.check(cur, x_prj, z_train)

    #     approx[s] = copy.deepcopy(ap)
    # #     x_test = np.zeros(shape=(x_size, data_size))
    # #     z_test = np.zeros(data_size)
        
    # #     for k,data_k in enumerate(test_data[s]):
    # #         x_test[:,k] = data_k[0]
    # #         z_test[k] = data_k[1]
    # #     print(f'test: s={s}')
    # #     ap.check(cur, x_test, z_test)

    # with open('cache/approx.pickle', 'wb') as pickle_file:
    #     pickle.dump(approx, pickle_file)



if __name__ == "__main__":
    main()
