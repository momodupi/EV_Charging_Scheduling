import numpy as np
import pandas as pd

import json
import pickle


class Arrivals(object):
    def __init__(self, setting=None):

        # set the time_horizon horizon
        self.time_horizon = setting['time_horizon']
        self.time_unit_set = {
            'hour': 1,
            'min': 60,
            'sec': 3600,
            'halfh': 2
        }
        self.time_unit = setting['unit']
        # arrival horizon unit is smaller than time horizon
        self.time_arrival_horizon = self.time_horizon*self.time_unit_set['sec']
        self.time_arrival_ratio = self.time_horizon/self.time_arrival_horizon

        # EV set
        self.EV = {}

        # Poisson arrival
        self.arrival_rate = setting['arrival_rate']
        prob = 1 - np.exp( - self.arrival_rate*self.time_arrival_ratio ) 
        prob_set = np.array( [prob] * self.time_arrival_horizon )

        # generate passengers
        np.random.seed(setting['seed'])
        randomness = np.random.uniform(low=0, high=1, size=(self.time_arrival_horizon,))

        # data info
        self.charge_rate = setting['RoC'] # in kW
        self.bat_cap = setting['BC'] # in Kwh

        # set the size of menu
        self.menu_m_size = setting['m']
        self.menu_n_size = setting['n']
        self.menu_m_range = self.bat_cap
        # menu_n_range = time_arrival_horizon

        self.menu_m_step = int(self.menu_m_range/self.menu_m_size)
        # step should always be 1 unit of time_horizon
        self.menu_n_step = 1

        # menu = {
        #     'm': np.arange(menu_m_step, menu_m_range+menu_m_step, menu_m_step),
        #     'n': np.arange(1, menu_m_size, 1)
        # }
        self.menu = {
            'm': range(self.menu_m_step, self.menu_m_range+1, self.menu_m_step),
            'n': range(1, self.menu_n_size+1, self.menu_n_step)
        }


        self.EV['info'] = {
            'bat_cap': self.bat_cap,
            'time_horizon': self.time_horizon,
            'time_arrival_horizon': self.time_arrival_horizon,
            'time_unit': self.time_unit,
            'arrival_rate': self.arrival_rate,
            'charge_rate': self.charge_rate,
            'm': list(self.menu['m']),
            'n': list(self.menu['n']),
            'm_step': self.menu_m_step, 
            'n_step': self.menu_n_step,
            'm_size': self.menu_m_size,
            'n_size': self.menu_n_size
        }


        # toss = np.where( np.greater(prob_set, randomness) )[0]
        # print(randomness)
        # print(toss)
        ev_id = 0
        EV_csv = []

        # for index, t in enumerate(toss):
        for index,r in enumerate(randomness):
            # EV[int(t)] = {}
            # print(soc, flx)
            if r > prob:
                continue
            soc = float(np.random.uniform(low=0.3, high=0.8, size=1))
            demand = self.bat_cap * (1-soc)
            menu_m = int(demand / self.menu_m_step)

            # lax = int(np.random.choice(np.arange(0,time_arrival_horizon-t)))
            # menu_n = int(lax/time_unit_set['min']) / menu_n_step
            # for n=3, we assume that the laxity = [1,2,3]
            charge_time = int(demand/self.charge_rate + 1)
            # print(charge_time)
            if charge_time >= self.menu['n'][-1]:
                continue
            

            # arrive_time = int(index/time_unit_set['sec'])

            # if arrive_time+charge_time >= time_horizon:
            #     continue
            # leave_time = int(np.random.choice(range(arrive_time+charge_time, time_horizon, 1)))
            
            # menu_n = leave_time-arrive_time if leave_time-arrive_time<max(menu['n']) else max(menu['n'])-1
            
            
            menu_n = int(np.random.choice(range(charge_time, self.menu_n_size, 1)))
            # print(menu_n)

            arrive_time = int(index/self.time_unit_set['sec'])
            leave_time = arrive_time+self.menu['n'][menu_n]

            # print(demand, menu_m, lax, menu_n)
            if leave_time >= self.time_horizon:
                continue

            self.EV[ev_id] = {
                # 'id': ev_id,
                'soc': soc,
                # 'laxity': lax,
                'demand': demand,
                'arrive': arrive_time,
                'leave': leave_time,
                'm': menu_m,
                'n': menu_n
            }

            EV_csv.append(self.EV[ev_id])
            ev_id += 1


        # with open('cache/ev.json', 'w') as json_file:
        #     json.dump(EV, json_file) 

        pd.DataFrame.from_dict(EV_csv).to_csv('cache/EV.csv')

        with open('cache/ev.pickle', 'wb') as pickle_file:
            pickle.dump(self.EV, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 


