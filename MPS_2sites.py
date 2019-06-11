import numpy as np
import copy as cp
import simple_update_algorithm as su
from scipy import linalg
import matplotlib.pyplot as plt

d = 2
p = 2
D_max = d
J = 1





T0 = np.random.rand(p, d, d)
T1 = np.random.rand(p, d, d)
T2 = np.random.rand(p, d, d)
T3 = np.random.rand(p, d, d)
TT = [T0, T1, T2, T3]
imat = np.array([[1, 0, 0, 1],
                 [1, 1, 0, 0],
                 [0, 1, 1, 0],
                 [0, 0, 1, 1]])

smat = np.array([[2, 0, 0, 1],
                 [1, 2, 0, 0],
                 [0, 1, 2, 0],
                 [0, 0, 1, 2]])

"""
T0 = np.random.rand(p, d, d)
T1 = np.random.rand(p, d, d)
T2 = np.random.rand(p, d, d)
TT = [T0, T1, T2]
imat = np.array([[1, 0, 1],
                 [1, 1, 0],
                 [0, 1, 1]])

smat = np.array([[2, 0, 1],
                 [1, 2, 0],
                 [0, 1, 2]])
"""
"""
T0 = np.random.rand(p, d, d)
T1 = np.random.rand(p, d, d)
TT = [T0, T1]
imat = np.array([[1, 1],
                 [1, 1]])

smat = np.array([[2, 1],
                 [2, 1]])
"""
LL = []
for i in range(imat.shape[1]):
    LL.append(np.ones(d, dtype=float) / d)

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

t_list = np.exp(np.array(np.linspace(-1, -10, 1000)))
heisenberg = -J * np.real(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
hij = np.reshape(heisenberg, (p, p, p, p))
hij_perm = [0, 2, 1, 3]
hij_energy_term = cp.deepcopy(hij)
hij = np.transpose(hij, hij_perm)
hij = np.reshape(hij, [p ** 2, p ** 2])
unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]


iterations = 1
energy = []
LL_in_time = np.zeros((len(LL), D_max, len(t_list) * iterations), dtype=float)
TT_in_time = np.zeros((len(TT), len(np.ravel(TT[0])), len(t_list) * iterations))
counter = 0
for i in range(len(t_list)):
    for j in range(iterations):
        print('t, iters = ', i, j)
        for k in range(len(LL)):
            LL_in_time[k, :, counter] = LL[k]
        for l in range(len(TT)):
            TT_in_time[l, :, counter] = np.ravel(TT[l])
        TT, LL = su.simple_update(TT, LL, unitary[i], imat, smat, D_max)
        #TT, LL = su.gauge_fix1(TT, LL, imat, smat)
        energy.append(su.energy_per_site(TT, LL, imat, smat, hij_energy_term))
        counter += 1

for k in range(len(LL)):
    plt.figure()
    plt.title('lambda' + str(k) + ' values in time')
    plt.xlabel('t')
    for s in range(D_max):
        plt.plot(range(len(t_list) * iterations), LL_in_time[k, s, :], 'o')
    plt.grid()
    plt.show()

plt.figure()
plt.title('energy values')
plt.xlabel('t')
plt.plot(range(len(t_list) * iterations), energy, 'o')
plt.grid()
plt.show()

for k in range(len(TT)):
    plt.figure()
    plt.title('T' + str(k) + ' entries values in time')
    plt.xlabel('t')
    for s in range(len(np.ravel(T0))):
        plt.plot(range(len(t_list) * iterations), TT_in_time[k, s, :], 'o')
    plt.grid()
    plt.show()

