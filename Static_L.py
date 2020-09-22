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
import copy

from tools.Parameter import Parameter
from tools.Squeeze import Squeeze
from tools.Result import Result



def EV_setdetail(pa, subslots):
    busket = {}
    for t in pa.w:
        busket[t] = {}
        for mi in pa.w[t]:
            busket[t][mi] = {}
            for ni in pa.w[t][mi]:
                busket[t][mi][ni] = []

    EV = copy.deepcopy(pa.EV)

    for ev_id in EV:
        t = EV[ev_id]['arrive']
        mi = EV[ev_id]['m']
        ni = EV[ev_id]['n']
        del EV[ev_id]['m']
        del EV[ev_id]['n']
        EV[ev_id]['menu'] = f'({pa.menu["m"][mi]},{pa.menu["n"][ni]})'
        # print(pa.EV[ev_id]['menu'])
        busket[t][mi][ni].append(ev_id)

        l = EV[ev_id]['arrive'] + pa.menu['n'][ni]

        for s in range(subslots*pa.menu['n'][ni]):
        # for s in range(subslots*pa.menu['n'][-1]):
            sub_time = f'{ t+int(s/subslots) }:{int( (s%subslots)*60/subslots )}'
            # pa.EV[ev_id][sub_time] = 0 if s < subslots*pa.menu['n'][ni] else -1
            EV[ev_id][sub_time] = 0

    return busket, EV
    # print(busket)


def even_sch(setting):
    subslots = setting['subslots']
    busket = setting['busket']
    pa = setting['pa']
    y = copy.deepcopy(setting['y'])
    EV = setting['EV']

    for s in y:
        for t in y[s]:
            for mi in y[s][t]:
                for ni in y[s][t][mi]:
                    if pa.w[t][mi][ni] > 0:
                        y[s][t][mi][ni] = (y[s][t][mi][ni] / pa.r)*subslots/pa.w[t][mi][ni]
                    
                        u_s = int(y[s][t][mi][ni]+0.4999) if y[s][t][mi][ni]<subslots else subslots
                        
                        if pa.w[t][mi][ni] > 1:
                            first_line = []
                            last_line = []
                            half_busket= int(pa.w[t][mi][ni]/2)
                            # print(half_busket)
                            first_line = busket[t][mi][ni][:half_busket]
                            last_line = busket[t][mi][ni][half_busket:]

                            for ev_id in first_line:
                                for sub in range(u_s):
                                    sub_time = f'{ s }:{int( (sub%subslots)*60/subslots )}'
                                    # print(sub_time)
                                    EV[ev_id][sub_time] = 1

                            for ev_id in last_line:
                                for sub in range(u_s):
                                    sub_time = f'{ s }:{int( ( (subslots-1-sub)%subslots)*60/subslots )}'
                                    # print(sub_time)
                                    EV[ev_id][sub_time] = 1
                        else:                            
                            for ev_id in busket[t][mi][ni]:
                                for sub in range(u_s):
                                    sub_time = f'{ s }:{int( (sub%subslots)*60/subslots )}'
                                    # print(sub_time)
                                    EV[ev_id][sub_time] = 1
    return EV


def greedy_sch(setting):
    subslots = setting['subslots']
    busket = setting['busket']
    pa = setting['pa']
    y = copy.deepcopy(setting['y'])
    EV = setting['EV']

    # translate y to slots
    for s in y:
        for t in y[s]:
            for mi in y[s][t]:
                for ni in y[s][t][mi]:
                    if pa.w[t][mi][ni] > 0:
                        y[s][t][mi][ni] = int((y[s][t][mi][ni] / pa.r)*subslots+0.4999)
                    else:
                        y[s][t][mi][ni] = 0

                    # if pa.menu['m'][mi] == 24 and pa.menu['n'][ni] == 3:
                    #     print(f'time={s}, a time={t}: slots={y[s][t][mi][ni]}')

    for t in busket:
        for mi in busket[t]:
            for ni in busket[mi]:        
                n = pa.menu['n'][ni]

                if pa.w[t][mi][ni] > 0:
                    
                    # in each menu, do greedy
                    for ev_id in busket[t][mi][ni]:
                        # translate mi to slots
                        req_slots = int( (pa.menu['m'][mi] / pa.r)*subslots )
                        # print(f'{ev_id} requires {req_slots}')
                        s_ub = t+n if t+n<=pa.time_horizon else pa.time_horizon
                        # all time steps for each EV                   
                        for s in range(t, s_ub):   

                            charge_slots = min(y[s][t][mi][ni], req_slots, subslots)
                            for sub in range( charge_slots ):
                                sub_time = f'{ s }:{int( (sub%subslots)*60/subslots )}'
                                EV[ev_id][sub_time] = 1

                            req_slots -= charge_slots
                            y[s][t][mi][ni] -= charge_slots


    return EV    



def dp_sch(setting):
    subslots = setting['subslots']
    busket = setting['busket']
    pa = setting['pa']
    y = copy.deepcopy(setting['y'])
    EV = setting['EV']

    # translate y to slots
    q = copy.deepcopy(y)
    for s in y:
        for t in y[s]:
            for mi in y[s][t]:
                for ni in y[s][t][mi]:
                    if pa.w[t][mi][ni] > 0:
                        y[s][t][mi][ni] = int((y[s][t][mi][ni] / pa.r)*subslots+0.4999)
                    else:
                        y[s][t][mi][ni] = 0

                    # q is another version of z: splite z into hours
                    q[s][t][mi][ni] = np.zeros(pa.menu['n'][ni])
                    
    def value_f(state):
        # count the number of gaps
        gap = 0
        pre_st = state[0]
        for st in state[1:]:
            if pre_st < subslots and pre_st>0:
                if st == subslots:
                    pass
                elif st < subslots:
                    pre_st = 0
                    continue
            elif pre_st == 0:
                gap += 1
            else:
                pass
            pre_st = st

        return gap


    for t in busket:
        for mi in busket[t]:
            for ni in busket[mi]:        
                n = pa.menu['n'][ni]

                if pa.w[t][mi][ni] > 0:
                    
                    # in each menu, do greedy
                    for ev_id in busket[t][mi][ni]:
                        # translate mi to slots
                        req_slots = int( (pa.menu['m'][mi] / pa.r)*subslots )
                        # print(f'{ev_id} requires {req_slots}')
                        s_ub = t+n if t+n<=pa.time_horizon else pa.time_horizon
                        # all time steps for each EV                   
                        for s in range(t, s_ub):   

                            charge_slots = min(y[s][t][mi][ni], req_slots, subslots)
                            for sub in range( charge_slots ):
                                sub_time = f'{ s }:{int( (sub%subslots)*60/subslots )}'
                                EV[ev_id][sub_time] = 1

                            req_slots -= charge_slots
                            y[s][t][mi][ni] -= charge_slots

                                


    return EV    



def main():
    logging.basicConfig(level=logging.INFO)

    with open('cache/result_static.pickle', 'rb') as pickle_file:
        result_static = pickle.load(pickle_file) 

    with open('cache/cache_static.pickle', 'rb') as pickle_file:
        result_cache = pickle.load(pickle_file) 


    subsch = {
        'subslots': 10,
        'y': result_static['y'],
        'pa': result_cache['pa'],
    }
    subsch['busket'], subsch['EV'] =  EV_setdetail(result_cache['pa'], subsch['subslots'])
    
    EVsch = greedy_sch(subsch)

    with open('cache/ev_schedule.json', 'w') as json_file:
        json.dump(EVsch, json_file) 

    pd.DataFrame.from_dict(EVsch).T.to_csv('results/ev_schedule.csv')





if __name__ == "__main__":
    main()