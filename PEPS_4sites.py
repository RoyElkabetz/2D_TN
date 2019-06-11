import numpy as np
import copy as cp
import simple_update_algorithm as su
from scipy import linalg
from random import shuffle
import matplotlib.pyplot as plt

d = 2
p = 3
D_max = d
J = 1.

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

sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1.]])
sy = np.array([[0, -1j, 0.], [1j, 0, -1j], [0, 1j, 0.]]) / np.sqrt(2)
sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1., 0]]) / np.sqrt(2)

t_list = np.exp(np.array(np.linspace(-1, -10, 100)))
heisenberg = J * np.real(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
hij = np.reshape(heisenberg, (p, p, p, p))
hij_perm = [0, 2, 1, 3]
hij_energy_term = cp.deepcopy(hij)
hij = np.transpose(hij, hij_perm)
hij = np.reshape(hij, [p ** 2, p ** 2])
unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]


iterations = 1
energy = []
save_data = np.zeros((D_max, len(t_list)), dtype=float)
T0_it_time = np.zeros((len(np.ravel(TT[0])), len(t_list) * iterations))
k = 0
for i in range(len(t_list)):
    for j in range(iterations):
        print('i, j = ', i, j)
        TT, LL = su.simple_update(TT, LL, unitary[i], imat, smat, D_max)
        T0_it_time[:, k] = np.ravel(TT[0])
        k += 1
        energy.append(su.energy_per_site(TT, LL, imat, smat, hij_energy_term))
        save_data[:, i] = LL[0]


plt.figure()
plt.title('lambda0 values')
for k in range(save_data.shape[0]):
    plt.plot(range(len(t_list) * iterations), save_data[k, :], 'o')
plt.grid()
plt.show()


plt.figure()
plt.title('energy values')
plt.plot(range(len(t_list) * iterations), energy, 'o')
plt.grid()
plt.show()

plt.figure()
plt.title('tensors entries values')
for k in range(len(np.ravel(T0))):
    plt.plot(range(len(t_list) * iterations), T0_it_time[k, :], 'o')
plt.grid()
plt.show()
