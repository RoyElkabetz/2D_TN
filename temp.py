import numpy as np
import copy as cp
import simple_update_algorithm as su
from scipy import linalg
from random import shuffle
import matplotlib.pyplot as plt

d = 2
p = 3
D_max = 2

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
    LL.append(np.ones(d, dtype=float) / d)

pauli_z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1.]])
pauli_y = np.array([[0, -1j, 0.], [1j, 0, -1j], [0, 1j, 0.]]) / np.sqrt(2)
pauli_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1., 0]]) / np.sqrt(2)

t_list = np.linspace(0, 0.1, 100)
Uij = np.random.rand(d, d)
H_term = np.kron(pauli_x, pauli_x) + np.kron(pauli_y, pauli_y) + np.kron(pauli_z, pauli_z)
hij = H_term.reshape(p, p, p, p)
hij, _, _ = su.permshape(hij, [0, 2, 1, 3], [p ** 2, p ** 2])
unitary = [np.reshape(linalg.expm(-t * hij), [p, p, p, p]) for t in range(len(t_list))]

'''
# check unitary values
usum = []
for j in range(len(unitary)):
    usum.append(np.sum(np.real(unitary[j])))
plt.figure()
plt.plot(range(len(usum[0:80])), usum[0:80])
plt.show()
'''


for i in range(len(t_list)):
    print('i = ', i)
    TT, LL = su.simple_update(TT, LL, unitary[i], imat, smat, D_max)


