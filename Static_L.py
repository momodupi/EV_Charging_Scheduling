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

    for ev_id in pa.EV:
        t = pa.EV[ev_id]['arrive']
        mi = pa.EV[ev_id]['m']
        ni = pa.EV[ev_id]['n']
        del pa.EV[ev_id]['m']
        del pa.EV[ev_id]['n']
        pa.EV[ev_id]['menu'] = f'({pa.menu["m"][mi]},{pa.menu["n"][ni]})'
        # print(pa.EV[ev_id]['menu'])
        busket[t][mi][ni].append(ev_id)

        l = pa.EV[ev_id]['arrive'] + pa.menu['n'][ni]

        for s in range(subslots*pa.menu['n'][ni]):
        # for s in range(subslots*pa.menu['n'][-1]):
            sub_time = f'{ t+int(s/subslots) }:{int( (s%subslots)*60/subslots )}'
            # pa.EV[ev_id][sub_time] = 0 if s < subslots*pa.menu['n'][ni] else -1
            pa.EV[ev_id][sub_time] = 0


    return busket
    # print(busket)



def main():
    logging.basicConfig(level=logging.INFO)

    with open('cache/result_static.pickle', 'rb') as pickle_file:
        result_static = pickle.load(pickle_file) 

    with open('cache/cache_static.pickle', 'rb') as pickle_file:
        result_cache = pickle.load(pickle_file) 

    pa = result_cache['pa']
    y = result_static['y']

    subslots = 10
    busket = EV_setdetail(pa, subslots)
    u = copy.deepcopy( y )
    for s in u:
        for t in u[s]:
            for mi in u[s][t]:
                for ni in u[s][t][mi]:
                    if pa.w[t][mi][ni] == 0:
                        u[s][t][mi][ni] = 0
                    else:
                        u[s][t][mi][ni] = (y[s][t][mi][ni] / pa.r)*subslots/pa.w[t][mi][ni]
                    
                        u_s = int(u[s][t][mi][ni]+0.4999) if u[s][t][mi][ni]<subslots else subslots
                        
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
                                    pa.EV[ev_id][sub_time] = 1

                            for ev_id in last_line:
                                for sub in range(u_s):
                                    sub_time = f'{ s }:{int( ( (subslots-1-sub)%subslots)*60/subslots )}'
                                    # print(sub_time)
                                    pa.EV[ev_id][sub_time] = 1
                        else:                            
                            for ev_id in busket[t][mi][ni]:
                                for sub in range(u_s):
                                    sub_time = f'{ s }:{int( (sub%subslots)*60/subslots )}'
                                    # print(sub_time)
                                    pa.EV[ev_id][sub_time] = 1

    with open('cache/ev_schedule.json', 'w') as json_file:
        json.dump(pa.EV, json_file) 

    pd.DataFrame.from_dict(pa.EV).T.to_csv('results/ev_schedule.csv')





if __name__ == "__main__":
    main()