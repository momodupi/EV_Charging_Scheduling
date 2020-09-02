import numpy as np
import pandas as pd

import json
import pickle

# set the time_horizon horizon
time_horizon = 48
time_unit_set = {
    'hour': 1,
    'min': 60,
    'sec': 3600,
    'halfh': 2
}
time_unit = 'hour'
# arrival horizon unit is smaller than time horizon
time_arrival_horizon = time_horizon*time_unit_set['sec']
time_arrival_ratio = time_horizon/time_arrival_horizon

# set the seed
seed = 0

# EV set
EV = {}

# Poisson arrival
arrival_rate = 20
prob = 1 - np.exp( - arrival_rate*time_arrival_ratio ) 
prob_set = np.array( [prob] * time_arrival_horizon )

# generate passengers
np.random.seed(seed)
randomness = np.random.uniform(low=0, high=1, size=(time_arrival_horizon,))

# data info
charge_rate = 10 # in kW
bat_cap = 50 # in Kwh

# set the size of menu
menu_m_size = 6
menu_n_size = 6
menu_m_range = bat_cap
# menu_n_range = time_arrival_horizon

menu_m_step = int(menu_m_range/menu_m_size)
# step should always be 1 unit of time_horizon
menu_n_step = 1

# menu = {
#     'm': np.arange(menu_m_step, menu_m_range+menu_m_step, menu_m_step),
#     'n': np.arange(1, menu_m_size, 1)
# }
menu = {
    'm': range(menu_m_step, menu_m_range+1, menu_m_step),
    'n': range(1, menu_n_size+1, menu_n_step)
}


EV['info'] = {
    'bat_cap': bat_cap,
    'time_horizon': time_horizon,
    'time_arrival_horizon': time_arrival_horizon,
    'time_unit': time_unit,
    'arrival_rate': arrival_rate,
    'charge_rate': charge_rate,
    'm': list(menu['m']),
    'n': list(menu['n']),
    'm_step': menu_m_step, 
    'n_step': menu_n_step,
    'm_size': menu_m_size,
    'n_size': menu_n_size
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
    soc = float(np.random.uniform(low=0.2, high=0.8, size=1))
    demand = bat_cap * (1-soc)
    menu_m = int(demand / menu_m_step)

    # lax = int(np.random.choice(np.arange(0,time_arrival_horizon-t)))
    # menu_n = int(lax/time_unit_set['min']) / menu_n_step
    # for n=3, we assume that the laxity = [1,2,3]
    charge_time = int(demand/charge_rate + 1)
    # print(charge_time)
    if charge_time >= menu['n'][-1]:
        continue
    menu_n = int(np.random.choice(range(charge_time, menu_n_size, 1)))
    # print(menu_n)

    arrive_time = int(index/time_unit_set['sec'])
    leave_time = arrive_time+menu['n'][menu_n]

    # print(demand, menu_m, lax, menu_n)
    if leave_time >= time_horizon:
        continue

    EV[index] = {
        'id': ev_id,
        'soc': soc,
        # 'laxity': lax,
        'demand': demand,
        'arrive': arrive_time,
        'leave': leave_time,
        'm': menu_m,
        'n': menu_n
    }

    EV_csv.append(EV[index])
    ev_id += 1


with open('cache/ev.json', 'w') as json_file:
    json.dump(EV, json_file) 

pd.DataFrame.from_dict(EV_csv).to_csv('cache/EV.csv')

with open('cache/ev.pickle', 'wb') as pickle_file:
    pickle.dump(EV, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 


