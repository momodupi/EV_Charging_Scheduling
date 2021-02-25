import numpy as np
from numpy.lib.function_base import select
from cvxopt import matrix, solvers

class DP(object):
    def __init__(self, T, r) -> None:
        super().__init__()

        self.T = T

        self.tmn2idx_dic = {}
        self.idx2tmn_dic = {}
        self.r = r


        
    def set_menu(self, menu_m, menu_n) -> None:
        self.menu_m = menu_m
        self.menu_n = menu_n

        self.M, self.N = len(menu_m), len(menu_n)

        self.y_size = self.z_size = self.u_size = self.N*self.M*self.N
        self.x_size = self.y_size + self.z_size
        self.w_size = self.M*self.N


        cnt = 0
        for t in range(self.N):
            self.tmn2idx_dic[t] = {}
            for m in range(self.M):
                self.tmn2idx_dic[t][m] = {}
                for n in range(self.N):
                    self.tmn2idx_dic[t][m][n] = cnt
                    self.idx2tmn_dic[cnt] = (t,m,n)
                    cnt += 1
        
        self.mn2idx_dic = {}
        self.idx2mn_dic = {}
        cnt = 0
        for m in range(self.M):
            self.mn2idx_dic[m] = {}
            for n in range(self.N):
                self.mn2idx_dic[m][n] = cnt
                self.idx2mn_dic[cnt] = (m,n)
                cnt += 1


    def update_y(self, s, y, w) -> np.array:
        assert len(y) == self.y_size
        assert len(w) == self.w_size

        Y_shift = np.roll(np.eye(self.y_size), -self.w_size, axis=0)
        Y_shift = np.concatenate((
            Y_shift[:self.y_size-self.w_size], 
            np.zeros(shape=(self.w_size,self.y_size))
            ))

        W_shift = np.concatenate((
            np.zeros(shape=(self.y_size-self.w_size, self.w_size)), 
            np.eye(self.w_size)
            )).T
        return Y_shift.dot(y) + w.dot(W_shift)


    def update_z(self, s, z, u, w) -> np.array:
        assert len(z) == self.z_size
        assert len(u) == self.u_size
        assert len(w) == self.w_size

        z_new = z - u
        z_new = np.roll(z_new, -self.w_size)
        for w_idx, wi in enumerate(w):
            (pm,pn) = self.idx2mn_dic[w_idx]
            z_new[self.z_size-self.w_size+w_idx] = self.menu_m[pm]*wi
        return z_new

        


    def update(self, s, x, u, w) -> np.array:
        # shift matrices
        assert len(x) == self.x_size

        y = x[:self.y_size]
        z = x[self.y_size:]

        y_new = self.update_y(s,y,w)
        z_new = self.update_z(s,z,u,w)
        # print(y_new, z_new)
        return np.concatenate((y_new, z_new))



    def state_space(self, s, x, step):
        y = x[:self.y_size]
        z = x[self.y_size:]

        # md = [ np.arange(0, min(self.r*y[i], z[i]), step) for i in range(self.u_size) ]
        # return np.array(np.meshgrid(*md)).T.reshape(-1,self.u_size)
        # u_space = self.action_space(s, x, step)
        # w_space = self.prob_space(s, 10, step)

        # x_space = []
        # x_space_prob = []
        # for u in u_space:
        #     for w in w_space:
        #         x_space.append( self.update(x, u, w) )
        #         x_space_prob.append( 1/len(w_space) )
        # return x_space, x_space_prob


    def prob_space(self, s, ps_size) -> np.array:
        # sample from probability space
        w_space = np.zeros(shape=(ps_size, self.w_size))
        for i in range(ps_size):
            w_space[i,:] = np.random.poisson(1, self.w_size)
        return w_space
        

    def action_space(self, s, x, step):
        y = x[:self.y_size]
        z = x[self.y_size:]

        md = [ np.arange(0, min(self.r*y[i], z[i]), step) for i in range(self.u_size) ]
        return np.array(np.meshgrid(*md)).T.reshape(-1,self.u_size)


    def step(self, s, x, w, V):
        x_space, x_space_prob = self.state_space(s, x, w)
        
        Q_vec = np.zeros(len)
        V_vec = np.zeros(len(x_space))
        for xi,x in enumerate(x_space):
            V_vec[xi] = V(x)
        Q = V_vec.dot(x_space_prob)



            
            


        

dp = DP(10, 1)
dp.set_menu(menu_m=[10,20,30], menu_n=[1,2,3])

x = 30*np.ones(27*2)
x[:27] = np.random.poisson(1, 27)
s = 0
u = 10*np.ones(27)





