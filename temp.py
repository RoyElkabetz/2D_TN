import numpy as np
import copy as cp
import simple_update_algorithm as su
from random import shuffle

d = 2
p = 3
D_max = d ** 2

T0 = np.random.rand(p, d, d, d, d)
T1 = np.random.rand(p, d, d, d, d)
T2 = np.random.rand(p, d, d, d, d)
T3 = np.random.rand(p, d, d, d, d)

TT = [T0, T1, T2, T3]

imat = np.array([[1, 1, 1, 0, 1, 0, 0, 0],
                 [1, 0, 1, 1, 0, 1, 0, 0],
                 [0, 1, 0, 0, 1, 0, 1, 1],
                 [0, 0, 0, 1, 0, 1, 1, 1]])

smat = np.array([[1, 2, 3, 0, 4, 0, 0, 0],
                 [3, 0, 1, 2, 0, 4, 0, 0],
                 [0, 4, 0, 0, 2, 0, 1, 3],
                 [0, 0, 0, 4, 0, 2, 3, 1]])

LL = []
for i in range(8):
    LL.append(np.ones((d), dtype=float) * i / 10)

Uij = np.random.rand(d, d)


TT, LL = su.simple_update(TT, LL, Uij, imat, smat, D_max)


