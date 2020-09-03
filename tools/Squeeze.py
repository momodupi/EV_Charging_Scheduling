import numpy as np
import pandas as pd

from numpy.linalg import matrix_rank

import time

class Squeeze(object):
    def __init__(self, A_eq, b_eq):
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.m, self.n = np.shape(self.A_eq)

    def QR_projection(self):
        ''' QR decompositon'''
        s_time = time.time()
        # print(np.shape(A_eq))
        Q, R = np.linalg.qr(self.A_eq)
        # print(np.shape(R))
        # print(np.all(Q.dot(R)==A_eq))
        # rank_R = matrix_rank(R)
        print('squeeze: QR decompositon')
        print(f'squeeze: original: ({np.shape(self.A_eq)}), rank: {matrix_rank(self.A_eq)}')
        # print(np.shape(R))
        # print(matrix_rank(R[0:rank_R]))

        # dimension reduction of A_eq
        R_reduction = R[np.where(R.any(axis=1))]
        # print(matrix_rank(R_reduction), np.shape(R_reduction))
        R_non_zero = np.where(R.any(axis=1))[0]
        Q_reduction = Q[:,R_non_zero]
        # Q_reduction = Q_reduction[np.where(R.any(axis=1))]
        # print('R:', np.shape(R_reduction))
        # print('Q:', np.shape(Q_reduction))
        prj = np.linalg.pinv(Q_reduction)
        # print(eq_prj.dot(A_eq), eq_prj.dot(b_eq))

        # print(matrix_rank(A_eq), np.shape(A_eq))
        # print(matrix_rank(eq_prj.dot(A_eq)), np.shape(eq_prj.dot(A_eq)))
        prj_A_eq = prj.dot(self.A_eq)
        prj_b_eq = prj.dot(self.b_eq)
        print(f'squeeze: projected: ({np.shape(prj_A_eq)}), rank: {matrix_rank(prj_A_eq)}')
        print(f'squeeze time: {time.time()-s_time} sec')
        return prj_A_eq, prj_b_eq

    def PCA(self):
        return self.A_eq, self.b_eq

    def Random_Projection(self, k, method):
        ''' Ranodm Projection'''
        s_time = time.time()
        prj = np.zeros(shape=(k, self.m))

        print('squeeze: Random Projection')
        print(f'squeeze: original: ({np.shape(self.A_eq)}), rank: {matrix_rank(self.A_eq)}')
        prj_set = [-1, 0, 1]
        for i in range(k):
            # for j in range(self.n):
            prj[i,:] = np.random.choice(prj_set, self.m, p=[1/6, 2/3, 1/6])
        
        prj_A_eq = prj.dot(self.A_eq)
        prj_b_eq = prj.dot(self.b_eq)
        print(f'squeeze: projected: ({np.shape(prj_A_eq)}), rank: {matrix_rank(prj_A_eq)}')
        print(f'squeeze time: {time.time()-s_time} sec')
        return prj_A_eq, prj_b_eq

