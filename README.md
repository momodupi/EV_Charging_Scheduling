# EV_Charging_Scheduling

## Prerequisite
### Download
```shell
git clone https://code.osu.edu/shao.367/EV_Charging_Scheduling.git
```

### Install the required packages
We recommend install the packages with [conda](https://docs.conda.io/en/latest/):
```shell
conda install numpy scipy pandas cvxopt scikit-learn xlsxwriter
```

## Arrivals
`Arrivals.py` provides a class that simulates the arrivals of all EVs. The initialization of this class is given by
```python
ar = Arrivals(setting=info)
```
where `info` is the basic setting of this simulation
```python
info = {
        'time_horizon': 48,   # simulation time steps
        'unit': 'hour',       # unit of time steps 
        'arrival_rate': 20,   # arrivals per time step
        'RoC': 10,            # rate of charge, in kw
        'BC': 50,             # battery capacity, in kwh
        'm': 4,               # size of the charging menu
        'n': 4,               # size of the time menu
        'seed': 0             # random seed
    }
```

The initialization of `Arrivals` will automatically generate `/cache/EV.csv` and `/cache/ev.pickle` as the results. `/cache/EV.csv` is a readable file that is used for reviewing the arrivals of EVs. `/cache/ev.pickle` is a binary file that is used by other parts of the simulation.

### Model
#### Arrival process
We here consider only one charging station. The arrival process of EVs is considered to be a Poisson process with a specific rate, i.e., 20 arrivals per hour. To discretize the arrival process, we will use a binomial approximation to Poisson process.

Let the Poisson process be $\{N(t)\}_{t\in\mathbb{R}_+}$, and the rate of Poisson be $\lambda$. This implies that there are $\lambda$ arrivals during each time interval in average. If a time interval represents one hour, then for each second, the probability that an EV will arrive at each second is given by:
$$p=\mathbb{P}\left(N(t+\Delta t)-N(t)\geq 1\right)=e^{-\lambda \Delta t} + o\left(\Delta t\right)$$
where $\Delta t=1/3600$ is one second.

Then we have the discrete arrival process: $\{A(t)\}_{t\in\mathbb N}$, where
$$ A(t)=
\begin{cases}
1 &\text{ w.p. } p\\
0 &\text{ w.p. } 1-p
\end{cases}
$$
and approximately, ${N(t)}\approx A(3600t)$ for every $t\in\mathbb{N}$.

With this approximation, we separate each hour into seconds, and generate $\{A(t)\}_t$ based on the outcomes of `numpy.random.uniform`.

#### Demands
For each EV arrive at the charging station, we assign a random SoC which is uniformly distributed, i.e.,

`np.random.uniform(low=0.3, high=0.8, size=1)`

where the lowest and highest SoC are supposed to be 0.3 and 0.8. We also assume a fixed battery capacity for each EV, i.e., 50 kwh. This implies that the demands of each EV is

$$\text{demands}=(1-\text{SoC})\times50\text{kwh}.$$


#### Flexibility
The flexibility (or the choice of time menu) of an EV is also a uniformly distributed random variable determined by its demand. For instance, if an EV requires 30 kwh electricity, and the charging rate is upper bounded by 10 kw, then the flexibility should always larger than 3 hours. Moreover, the flexibility is also upper bounded by the size of the menu, i.e., no EV can wait for 6 hours if the longest waiting provided by the charging station is 5 hours. Therefor, the choice of time menu is generated by

`np.random.choice(range(charge_time, menu_n_size, 1))`


## Parameters in LP
`Parameter.py` provides a class that generates all the objective functions, equality constraints and inequality constraints that are used in the LP. The initialization of this class is given by
```python
pa = Parameter(EV=ar.EV, readable=True)
```
where `readable` sets whether to generate readable files showing `c`, `A_eq`, `b_eq`, `A_ub`, `b_ub` in the LP.

For the static scheduling high level LP, one can call
```python
pa_dic = pa.get_LP_static()
c = pa_dic['c']
A_eq, b_eq = pa_dic['A_eq'], pa_dic['b_eq']
A_ub, b_ub = pa_dic['A_ub'], pa_dic['b_ub']
```
where the return of `pa.get_LP_static()` is a dictionary with the following structure
```python
pa_dic = {
    'c':    c,
    'A_eq': A_eq,
    'b_eq': b_eq,
    'A_ub': A_ub,
    'b_ub': b_ub
}
```

The process of generating the LP parameters in dynamic LP is similar with the above, by replacing `pa.get_LP_static()` into `pa.get_LP_dynamic()`.


## Static High Level Scheduling
In `Static_H.py`, it provides two solvers to solve the LP: `scipy_LP` and `CVXOPT_LP`.

`scipy_LP` is the solver provided by `scipy`, which is flexible to choose the algorithm of solving the LP, i.e., `simplex`, `interor-point`, etc.

`CVXOPT_LP` is the solver provided by `CVXOPT`, which is a faster solver for convex optimization.

Here we highly recommend to use `CVXOPT_LP` because of its advantages in both speed and stability.

One can directly run
```shell
python Static_H.py
```
to start the simulation. All the initializations of `Arrivals` and `Parameters` are completes automatically. The results will be saved in `results` folder.

To change the setting of the simulation, one can edit the `info` dictionary in the `main()`.


## Dynamic High Level Scheduling
In `Dynamic_H.py`, we will solve the LP in backward. Note that the dynamic LP is computing along the optimal trajectory. One should run `Static_H.py` __*before*__ running `Dynamic_H.py`.


One can directly run
```shell
python Dynamic_H.py
```
to start the simulation. The results will be saved in `results` folder.


## Results
`Results.py` provides a class that store the results of LP. The initialization of this class is given by
```python
result = Result(pa=pa, x=x, dur=dur)
```
where `x` is the solution of the LP, `dur` is the running time of the LP.

By calling
```python
result.check()
```
one can check the result by validating the constraints of the LP. Due to the accuracy of the solver, the tolerance of the equality constraints is given by 10<sup>-4</sup>.

The results of the LP contain two parts: the allocation scheme $y$ (how much electricity should be allocated to each menu) and the purchased electricity $e$ (how much electricity should be purchased from the grid). Both $y$ and $e$ will be reorganized and stored into readable files.

By calling 
```python
result.output('static')    # or result.output('dynamic') for the dynamic LP
```
the $y$ will be saved to `results/y_static.xlsx` (or `results/y_dynamic.xlsx`), and the $e$
will be saved to `results/e_static.csv` (or `results/e_dynamic.csv`).


## Squeeze
...


## Low Level Scheduling
...


## Approximator
...


