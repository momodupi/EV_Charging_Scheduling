import numpy as np
import pandas as pd

import json

# set the time_horizon horizon
time_horizon = 24
time_unit_set = {
    'hour': 1,
    'min': 60,
    'sec': 3600,
    'halfh': 2
}
time_unit = 'hour'
# arrival horizon unit is smaller than time horizon
time_arrival_horizon = time_horizon*time_unit_set['min']
time_arrival_ratio = time_horizon/time_arrival_horizon

# set the seed
seed = 0

# EV set
EV = {}

# Poisson arrival
arrival_rate = 20
prob = 1 - np.exp( - arrival_rate/time_unit_set[time_unit] ) 
prob_set = [prob] * time_arrival_horizon
# print(prob_set)

# generate passengers
# np.random.seed(seed)
randomness = np.random.uniform(low=0, high=1, size=(time_arrival_horizon,))

# data info
charge_rate = 10 # in kW
bat_cap = 50 # in Kwh

# set the size of menu
menu_m_size = 3
menu_n_size = 3
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
    'm': range(menu_m_step, menu_m_range, menu_m_step),
    'n': range(1, menu_m_size+1, menu_n_step)
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


toss = np.where( np.greater(prob_set, randomness) )[0]
# print(randomness)
# print(toss)
ev_id = 0
for t in toss:
    # EV[int(t)] = {}
    # print(soc, flx)
    soc = float(np.random.uniform(low=0.3, high=0.9, size=1))
    demand = bat_cap * (1-soc)
    menu_m = int(demand / menu_m_step)

    # lax = int(np.random.choice(np.arange(0,time_arrival_horizon-t)))
    # menu_n = int(lax/time_unit_set['min']) / menu_n_step
    # for n=3, we assume that the laxity = [1,2,3]
    charge_time = int(demand/charge_rate + 1)
    # print(charge_time)
    if charge_time > menu['n'][-1]:
        continue
    menu_n = int(np.random.choice(range(charge_time-1, menu_n_size, 1)))
    # print(menu_n)

    arrive_time = int(t/time_unit_set['min'])
    leave_time = arrive_time+menu['n'][menu_n]

    # print(demand, menu_m, lax, menu_n)
    if leave_time >= time_horizon:
        continue

    EV[int(t)] = {
        'id': ev_id,
        'soc': soc,
        # 'laxity': lax,
        'demand': demand,
        'arrive': arrive_time,
        'leave': leave_time,
        'm': menu_m,
        'n': menu_n
    }
    ev_id += 1


with open('ev.json', 'w') as json_file:
    json.dump(EV, json_file) 