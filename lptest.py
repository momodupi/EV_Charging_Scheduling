from cvxopt import matrix, solvers
import numpy as np


c = matrix([2., -2.])

# A = matrix([1., 1.])
# b = matrix([1.])

A = matrix(np.array([1.,1.]), (1,2))
b = matrix(np.array([1.]))



G = matrix([[1., 0., -1., 0.], [0., 1., 0., -1.]])
h = matrix([3., 3., 0., 0.])

sol = solvers.lp(c, G, h, A, b)
print(sol['y'])