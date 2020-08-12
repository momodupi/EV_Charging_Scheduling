import numpy as np
import pandas as pd

import json

# set the time horizon
time_input = 240
time_unit_set = {
    'hour': 1,
    'min': 60,
    'sec': 3600
}
time_unit = 'min'
time_horizon = time_input*time_unit_set[time_unit]

# set the seed
seed = 0

# EV set
EV = {}

# Poisson arrival
arrival_rate = 10
prob = 1 - np.exp( - arrival_rate/time_unit_set[time_unit] ) 
prob_set = [prob] * time_horizon
# print(prob_set)

# generate passengers
# np.random.seed(seed)
randomness = np.random.uniform(low=0, high=1, size=(time_horizon,))

toss = np.where( np.greater(prob_set, randomness) )[0]
# print(randomness)
# print(toss)
ev_id = 0
for t in toss:
    EV[int(t)] = {}
    # print(soc, flx)
    EV[int(t)] = {
        'id': ev_id,
        'soc': float(np.random.uniform(low=0, high=0.8, size=1)),
        'flx': int(np.random.choice(np.arange(t,time_horizon)))
    }
    ev_id += 1

with open('ev.json', 'w') as json_file:
    json.dump(EV, json_file) 