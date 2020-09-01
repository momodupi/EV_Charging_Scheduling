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

from Parameter import Parameter
from Squeeze import Squeeze


class Result(object):
    def __init__(self, pa, x, dur):
        self.pa = pa
        self.x = x
        self.dur = dur
        logging.debug(self.x)
        logging.info(f'lp time: {dur} sec')

    def check(self):
        logging.info(f'value: {self.pa.c_lin.dot(self.x)}')

        y_e = np.abs(np.sum(self.x[:-self.pa.time_horizon])-np.sum(self.x[-self.pa.time_horizon:]))<10e-4
        d_s = np.abs(self.pa.total_demand-np.sum(self.x[-self.pa.time_horizon:]))<10e-4
        
        e_test = np.squeeze(np.asarray(self.pa.A_eq_e.dot(self.x)))
        pd = np.all((np.abs(e_test-self.pa.b_eq_e) <10e-4))
        c_test = np.squeeze(np.asarray(self.pa.A_eq_c.dot(self.x)))
        te = np.all((np.abs(c_test-self.pa.b_eq_c) <10e-4))

        if y_e and d_s and pd and te:
            logging.debug(f'test: y=e: {y_e}')
            logging.debug(f'test: demand=supply: {d_s}')
            logging.debug(f'test: pd: {pd}')
            logging.debug(f'test: te: {te}')
        else:
            logging.warning(f'test: y=e: {y_e}')
            logging.warning(f'test: demand=supply: {d_s}')
            logging.warning(f'test: pd: {pd}')
            logging.warning(f'test: te: {te}')


    def output(self):
        y = {}
        for cnt in self.pa.cnt2stmn:
            s, t, mi, ni = self.pa.cnt2stmn[cnt]
            if s not in y:
                y[s] = {}
            if t not in y[s]:
                y[s][t] = {}
            if mi not in y[s][t]:
                y[s][t][mi] = {}
            y[s][t][mi][ni] = self.x[cnt]
            
        e = self.x[-self.pa.time_horizon:]

        # varification
        y_form = True
        for s in self.pa.stmn2cnt:
            for t in self.pa.stmn2cnt[s]:
                for mi in self.pa.stmn2cnt[s][t]:
                    for ni in self.pa.stmn2cnt[s][t][mi]:
                        if y[s][t][mi][ni] != self.x[ self.pa.stmn2cnt[s][t][mi][ni] ]:
                            y_form = False
        if y_form:
            logging.debug(f'output reform: {y_form}')
        else:
            logging.warning(f'output reform: {y_form}')

        result_output = {
            'y': y,
            'e': e,
            'pa': self.pa,
            'duration': self.dur
        }

        with open('cache/result_static.pickle', 'wb') as pickle_file:
            pickle.dump(result_output, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 

        # convert to xlsx
        y_xlse = pd.ExcelWriter('cache/y_static.xlsx', engine='xlsxwriter')
        for s in y:
            y_df_s = {}
            for t in y[s]:
                for mi in y[s][t]:
                    for ni in y[s][t][mi]:
                        if ni in y[s][t][mi]:
                            if f'({self.pa.menu["m"][mi]},{self.pa.menu["n"][ni]})' not in y_df_s:
                                y_df_s[f'({self.pa.menu["m"][mi]},{self.pa.menu["n"][ni]})'] = np.ones(self.pa.time_horizon)*(-1)
                            y_df_s[f'({self.pa.menu["m"][mi]},{self.pa.menu["n"][ni]})'][t] = y[s][t][mi][ni]

            y_df_s = pd.DataFrame(y_df_s)
            y_df_s.to_excel(y_xlse, sheet_name=f'time={s}')
        y_xlse.save()

        pd.DataFrame(e).to_csv('cache/e_static.csv')
        logging.info(f'Finished!')



def scipy_LP(c, A_eq, b_eq, A_ub, b_ub):
    logging.info('LP: solver: scipy interior-point')
    logging.info('LP: Start')
    result = linprog(
            c=-c, 
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
        c=matrix(-c), 
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


def main():
    logging.basicConfig(level=logging.DEBUG)
    np.set_printoptions(threshold=sys.maxsize)

    pa = Parameter('cache/ev.pickle')

    pa_dic = pa.get_LP(s=0)
    c = pa_dic['c']
    A_eq, b_eq = pa_dic['A_eq'], pa_dic['b_eq']
    A_ub, b_ub = pa_dic['A_ub'], pa_dic['b_ub']


    # dimentionality reduction
    sq = Squeeze(A_eq=A_eq, b_eq=b_eq)

    # k = int(np.shape(A_eq)[0]/2)
    # A_eq, b_eq = sq.Random_Projection(k=k, method='')

    s_time = time.time()
    
    # choose different solver
    # x = scipy_LP(c=c_lin, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)
    x = CVXOPT_LP(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)

    dur = time.time()-s_time
    result = Result(pa=pa, x=x, dur=dur)
    result.check()
    result.output()


if __name__ == "__main__":
    main()