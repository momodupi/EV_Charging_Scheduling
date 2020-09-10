import numpy as np
import pandas as pd


import logging
import json
import pickle
import time


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


    def output(self, file=None):
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

        self.result_output = {
            'y': y,
            'e': e,
        }
        result_cache = {
            'pa': self.pa,
            'duration': self.dur
        }

        with open(f'cache/result_{file}.pickle', 'wb') as pickle_file:
            pickle.dump(self.result_output, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 
        with open(f'cache/cache_{file}.pickle', 'wb') as pickle_file:
            pickle.dump(result_cache, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 

        # convert to xlsx
        y_xlse = pd.ExcelWriter(f'results/y_{file}.xlsx', engine='xlsxwriter')
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
            # print(y_df_s)
            y_df_s.to_excel(y_xlse, sheet_name=f'time={s}')
        y_xlse.save()

        pd.DataFrame(e).to_csv(f'results/e_{file}.csv')
        logging.info(f'data saved to results/y_{file}.xlsx and results/e_{file}.csv!')