import enum
from time import sleep
import numpy as np
import itertools as it

from numpy.lib.function_base import vectorize

class FVI(object):
    def __init__(self) -> None:
        super().__init__()


    def state_space(self, s_r, step, dim) -> None:
        self.s_arange = np.arange(s_r[0], s_r[1], step)
        md = [ self.s_arange for i in range(dim) ]
        self.s_space = np.array(np.meshgrid(*md)).T.reshape(-1,dim)


    def action_space(self, a_r, step, dim) -> None:
        self.a_arange = np.arange(a_r[0], a_r[1], step)
        md = [ self.a_arange for i in range(dim) ]
        self.a_space = np.array(np.meshgrid(*md)).T.reshape(-1,dim)


    def prob_space(self, w_r, step) -> None:
        self.w_space = np.arange(w_r[0], w_r[1], step) 

    def set_space(self, s_r, a_r, w_r, step, dim):
        '''
        s_r,a_r,w_r: (lb, ub) -> positve
        dim: dimension -> int
        step: step size -> less than ub
        '''
        assert s_r[1] >= s_r[0]
        assert a_r[1] >= a_r[0]
        assert w_r[1] >= w_r[0]
        self.s_r = s_r
        self.a_r = a_r
        self.w_r = w_r
        self.dim = dim
        self.step = step

        self.state_space(s_r, step, dim)
        self.action_space(a_r, step, dim)
        self.prob_space(w_r, step)
        
        self.decimal = 1
        while step <= 1:
            step *= 10
            self.decimal += 1

    def P_sa(self, s0, a) -> np.array:
        # transition probability
        # p = np.zeros(len(self.s_space))

        # for w in self.w_space:
        #     s1 = self.update(s0, a, w)
        #     assert self.assert_space(s1, self.s_arange)

        #     pos_s1 = list(self.s_space).index(s1)
        #     p[pos_s1] += self.uniform(s0, a, s1)

        # return p
        p = np.zeros(len(self.s_space))

        s1_range = np.zeros(shape=(self.dim, len(self.w_space)))
        for w_i,w in enumerate(self.w_space):
            s1_range[:,w_i] = self.update(s0, a, w)
            
            if not self.assert_space(s1_range[:,w_i], self.s_arange):
                print(s1_range[:,w_i])
                assert self.assert_space(s1_range[:,w_i], self.s_arange)

        for s1_i,s1 in enumerate(self.s_space):
            for w_i,w in enumerate(self.w_space):
                if np.array_equal( s1, s1_range[:,w_i] ):
                    p[s1_i] += self.uniform(s0, a, s1)
                # else:
                #     print(self.update(s0, a, w))

        # print(p)
        if np.sum(p) != 1:
            print(p, )
            assert np.isclose( np.sum(p), 1)
        return p


    def uniform(self, s0, a, s1):
        # unif_prob = 1/len(self.w_space) * np.ones(len(self.w_space))
        # w = np.random.choice(self.w_space,size=1,p=unif_prob)
        # return w, list(self.w_space).index(w)
        return 1/len(self.w_space)


    def update(self, s, a, w):
        # s1 = np.around(s-a+w, decimals=self.decimal)
        s1 = s-a+w
        # print(s1)
        
        # inside state space
        for i in range(self.dim):
            if s1[i] not in self.s_arange:
                k = np.array(self.s_arange)
                ki = np.searchsorted(k, s1[i], side="left")
                # s1[i] = max( min(s1[i], self.s_arange[-1]),self.s_arange[0] )
                ki = len(self.s_arange)-1 if ki > len(self.s_arange)-1 else ki
                s1[i] = k[ki]
        return s1



    def assert_space(self, x, x_arange):
        for xi in x:
            if xi not in x_arange:
                assert xi in x_arange
                return False
        return True


    def cost(self, s0, a, s1):
        # step cost function
        return np.sum(s0)+np.sum(a)+np.sum(s1)

    def cost_vec(self, s0, a):
        s1_vec = np.zeros(len(self.s_space))
        for s1_i,s1 in enumerate(self.s_space):
            s1_vec[s1_i] = self.cost(s0, a, s1)
        return s1_vec



    def V(self, s0, v0, gamma):
        v_a = np.zeros(len(self.a_space))
        assert len(v0) == len(self.s_space)

        for a_i,a in enumerate(self.a_space):
            # action space
            # print(self.P_sa(s0, a))
            v_a[a_i] = (self.cost_vec(s0, a) + gamma*v0).dot(self.P_sa(s0, a))

        return np.min(v_a)

    
    def value_interation(self, iter_steps, gamma, tol):
        v0 = np.zeros(len(self.s_space))
        v = np.zeros(len(self.s_space))

        for k in range(iter_steps):
            print('v:', v0)
            for s_i,s in enumerate(self.s_space):
                v[s_i] = self.V(s, v0, gamma)
            
            error = np.linalg.norm(v-v0, 2)
            print(error)
            if error < tol:
                return v
            else:
                v0 = np.array(v)
        return v


f = FVI()
f.set_space( (0,1), (0,1), (0,1), 0.3, 3)
v = f.value_interation(iter_steps=100, gamma=0.6, tol=0.001)
print('final', v)


