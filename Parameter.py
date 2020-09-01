import sys


import numpy as np
import pandas as pd

from numpy.linalg import matrix_rank

from scipy.linalg import null_space
from scipy.stats import gamma


import json
import pickle
import time
import logging


np.set_printoptions(threshold=sys.maxsize)


class Parameter(object):
    def __init__(self, ev_file, readable=False):
        # import data
        with open(ev_file, 'rb') as pickle_file:
            EV = pickle.load(pickle_file)


        self.time_horizon = EV['info']['time_horizon']
        self.time_arrival_horizon = EV['info']['time_arrival_horizon']
        self.time_arrival_ratio = self.time_horizon/ self.time_arrival_horizon

        self.charge_rate = EV['info']['charge_rate']
        self.time_unit = EV['info']['time_unit']
        self.menu_m_step = EV['info']['m_step']
        self.menu_n_step = EV['info']['n_step']
        self.menu_m_size = EV['info']['m_size']
        self.menu_n_size = EV['info']['n_size']

        self.menu = {
            'm': np.array(EV['info']['m']),
            'n': np.array(EV['info']['n'])
        }

        self.time_unit_set = {
            'hour': 1,
            'min': 60,
            'sec': 3600,
            'halfh': 2
        }

        self.r = self.charge_rate

        del EV['info']
        self.EV = EV


        self.stmn2cnt = {}
        self.cnt2stmn = {}
        '''
        Note: 
        t in [s-n+1, s]
        s in [t, t+n-1]
        where n starting from 1 (not ni)
        '''
        cnt = 0
        for s in range(self.time_horizon):
            self.stmn2cnt[s] = {}
            for mi, m in enumerate(self.menu['m']):
                for ni,n in enumerate(self.menu['n']):
                    # this should be n
                    # for t in range(s-n, s+1):
                    for t in range(s-n+1, s+1):
                        if t < 0:
                            continue
                        # c_lin.append( v[s][t][mi][ni] )

                        if t not in self.stmn2cnt[s]:
                            self.stmn2cnt[s][t] = {}
                        if mi not in self.stmn2cnt[s][t]:
                            self.stmn2cnt[s][t][mi] = {}
                        self.stmn2cnt[s][t][mi][ni] = cnt
                        self.cnt2stmn[cnt] = (s,t,mi,ni)
                        cnt += 1
        self.readable = readable
        if self.readable:
            with open('cache/stmn2cnt.json', 'w') as json_file:
                json.dump(self.stmn2cnt, json_file) 
            with open('cache/cnt2stmn.json', 'w') as json_file:
                json.dump(self.cnt2stmn, json_file) 
        

        with open('cache/stmn2cnt.pickle', 'wb') as pickle_file:
            pickle.dump(self.stmn2cnt, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 
        with open('cache/cnt2stmn.pickle', 'wb') as pickle_file:
            pickle.dump(self.cnt2stmn, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 

        self.total_demand = 0

        self.w = {}
        self.c = []
        self.d = []
        self.v = {}

        self.c_lin = None
        self.A_eq_e = None
        self.b_eq_e = None
        self.A_eq_c =None
        self.b_eq_c = None

        self.A_ub_u = None
        self.b_ub_u = None
        self.A_ub_l = None
        self.b_ub_l = None

        logging.debug('parameter: w')
        self.w = self.get_w()
        logging.debug('parameter: c')
        self.c = self.get_c()
        logging.debug('parameter: d')
        self.d = self.get_d()
        logging.debug('parameter: v')
        self.v = self.get_v()

        


    def get_w(self):
        w = {}
        # get w_s
        for t in range(self.time_horizon):
            # w[t] = np.zeros(shape=(menu_m_size,menu_n_size))
            w[t] = {}
            for mi,m in enumerate(self.menu['m']):
                w[t][mi] = {}
                for ni,n in enumerate(self.menu['n']):
                    w[t][mi][ni] = 0

        for index in self.EV:
            # tran t to time
            t = self.EV[index]['arrive']
            mi = self.EV[index]['m']
            ni = self.EV[index]['n']
            w[t][mi][ni] += 1

        # print(w)
        # reform w to csv
        w_csv = { }
        for mi,m in enumerate(self.menu['m']):
            for ni,n in enumerate(self.menu['n']):
                w_csv[f'({m},{n})'] = []
                for t in self.w:
                    w_csv[f'({m},{n})'].append(int(self.w[t][mi][ni]))

        if self.readable:
            d.DataFrame(w_csv).to_csv('cache/w.csv')
        with open('cache/w.pickle', 'wb') as pickle_file:
            pickle.dump(self.w, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 
        return w

    def price_turbulance(self, s):
        s = s-int(s/24)*24 if s>24 else s
        s_p = s if s>8 else 24+s
        return gamma.pdf(s_p-10, 8)



    def get_c(self):
        c = []
        def c_s(s):
            # base_price = 0.12 # $/kWh
            # return (1+price_turbulance(s)) * base_price
            # print(s/time_unit_set[time_unit])
            cur_time = int(s/self.time_unit_set[self.time_unit]) % 24
            return 9.341/100 if cur_time >= 11 and cur_time <= 19 else 0.948/100
            # return 0.09341

        
        for s in range(self.time_horizon):
            c.append(c_s(s=s))

        with open('cache/c.pickle', 'wb') as pickle_file:
            pickle.dump(c, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        return c


    def get_d(self):
        d = []
        def d_s(s):
            return 2000. # kWh
        for s in range(self.time_horizon):
            d.append(d_s(s=s))
        
        with open('cache/d.pickle', 'wb') as pickle_file:
            pickle.dump(d, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        return d

    def get_v(self):
        v = {}
        def v_s(s,t,m,n):
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

        # set all v,c,z
        for s in range(self.time_horizon):
            v[s] = {}
            
            # for t in range(s-menu_n_size, s+1):
            for t in range(s-self.menu_n_size+1, s+1):
                v[s][t] = np.zeros(shape=(self.menu_m_size,self.menu_n_size))

                # if t >= s - menu_n_size and t >= 0:
                if t >= 0:
                    for mi, m in enumerate(self.menu['m']):
                        for ni,n in enumerate(self.menu['n']):
                            v[s][t][mi][ni] = v_s(s=s, t=0, m=m, n=n)

        with open('cache/v.pickle', 'wb') as pickle_file:
            pickle.dump(v, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 
        return v


    def get_c_lin(self):
        self.c_lin = []
        for s in range(self.time_horizon):
            for mi, m in enumerate(self.menu['m']):
                for ni,n in enumerate(self.menu['n']):
                    # this should be n
                    # for t in range(s-n, s+1):
                    for t in range(s-n+1, s+1):
                        if t < 0:
                            continue
                        self.c_lin.append( self.v[s][t][mi][ni] )

        c_e = np.array( self.c[-self.time_horizon:] )*(-1)
        self.c_lin = np.concatenate(( np.array(self.c_lin), c_e ))
        
        if self.readable:
            pd.DataFrame(self.c_lin).to_csv('cache/c_lin.csv')
        return self.c_lin


    def get_eq(self):

        ye_size = len(self.c_lin)

        self.A_eq_e = np.zeros(shape=(self.time_horizon, ye_size))
        for s in range(self.time_horizon):
            self.A_eq_e[s][-(self.time_horizon-s)] = -1
            for mi, m in enumerate(self.menu['m']):
                for ni,n in enumerate(self.menu['n']):
                    # this should be n
                    # for t in range(s-n, s+1):
                    for t in range(s-n+1, s+1):
                        if t < 0:
                            continue
                        self.A_eq_e[s][ self.stmn2cnt[s][t][mi][ni] ] = 1
        self.b_eq_e = np.zeros(self.time_horizon)

        if self.readable:
            pd.DataFrame(self.A_eq_e).to_csv('cache/A_eq_e.csv')
            pd.DataFrame(self.b_eq_e).to_csv('cache/b_eq_e.csv')

        self.b_eq_c = np.zeros(self.time_horizon*self.menu_m_size*self.menu_n_size)
        self.A_eq_c = np.zeros(shape=(self.time_horizon*self.menu_m_size*self.menu_n_size, ye_size))
        A_eq_c_cnt = 0


        for t in range(self.time_horizon):
            for mi,m in enumerate(self.menu['m']):
                for ni,n in enumerate(self.menu['n']):
                    # constraint, this should be n

                    # u_s = t+n+1 if t+n+1<=time_horizon else time_horizon
                    u_s = t+n if t+n<=self.time_horizon else self.time_horizon

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
                        self.A_eq_c[ A_eq_c_cnt, self.stmn2cnt[s][t][mi][ni] ] = 1
                            # print(A_ub_c[A_ub_c_cnt])
                    # m should be the multiple of r
                    # m_ceil = (int(m/r))*r
                    self.b_eq_c[A_eq_c_cnt] = m*self.w[t][mi][ni]
                    A_eq_c_cnt += 1

        self.total_demand = np.sum(self.b_eq_c)

        if self.readable:
            pd.DataFrame(self.A_eq_c).to_csv('cache/A_eq_c.csv')
            pd.DataFrame(self.b_eq_c).to_csv('cache/b_eq_c.csv')

        A_eq = np.concatenate( (self.A_eq_e, self.A_eq_c) )
        b_eq = np.concatenate( (self.b_eq_e, self.b_eq_c) )

        
        if self.readable:
            pd.DataFrame(A_eq).to_csv('cache/A_eq.csv')
            pd.DataFrame(b_eq).to_csv('cache/b_eq.csv')

        return A_eq, b_eq


    def get_ineq(self):
        ye_size = len(self.c_lin)

        self.A_ub_u = np.eye(ye_size)
        self.b_ub_u = np.zeros(ye_size)
        self.A_ub_l = -np.eye(ye_size)
        self.b_ub_l = np.zeros(ye_size)

        for s in range(self.time_horizon):
            for mi, m in enumerate(self.menu['m']):
                for ni,n in enumerate(self.menu['n']):
                    for t in range(0, s+1):
                        if s in self.stmn2cnt and t in self.stmn2cnt[s] and mi in self.stmn2cnt[s][t] and ni in self.stmn2cnt[s][t][mi]:
                            # print(w[t][mi][ni])
                            self.b_ub_u[ self.stmn2cnt[s][t][mi][ni] ] = self.r*self.w[t][mi][ni]

            # assign d
            # b_ub_u[-time_horizon+s] = get_d(s)
            self.b_ub_u[-self.time_horizon+s] = self.d[s]


        A_ub = np.concatenate( (self.A_ub_u, self.A_ub_l) )
        b_ub = np.concatenate( (self.b_ub_u, self.b_ub_l) )

        if self.readable:
            pd.DataFrame(A_ub).to_csv('cache/A_ub.csv')
            pd.DataFrame(b_ub).to_csv('cache/b_ub.csv')

        return A_ub, b_ub



    def get_LP(self, s):
        if s==0:
            logging.debug('parameter: LP: c')
            c = self.get_c_lin()
            logging.debug('parameter: LP: eq')
            A_eq, b_eq = self.get_eq()
            logging.debug('parameter: LP: ineq')
            A_ub, b_ub = self.get_ineq()
            p_dic = {
                'c': c,
                'A_eq': A_eq,
                'b_eq': b_eq,
                'A_ub': A_ub,
                'b_ub': b_ub
            }
            return p_dic
        else:
            return {}