import numpy as np
import copy as cp
import simple_update_algorithm as su
from scipy import linalg
from random import shuffle
import matplotlib.pyplot as plt

d = 4
p = 3
D_max = 4

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

t_list = np.linspace(1e-1, 1e-5, 100)
Uij = np.random.rand(d, d)
H_term = np.kron(pauli_x, pauli_x) + np.kron(pauli_y, pauli_y) + np.kron(pauli_z, pauli_z)
hij = H_term.reshape(p, p, p, p)
hij_energy_term, _, _ = su.permshape(hij, [0, 2, 1, 3], [p, p, p, p])
hij, _, _ = su.permshape(hij, [0, 2, 1, 3], [p ** 2, p ** 2])
unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]

'''
# check unitary values
usum = []
for j in range(len(unitary)):
    usum.append(np.sum(np.real(unitary[j])))
plt.figure()
plt.plot(range(len(usum[0:80])), usum[0:80])
plt.show()
'''
'''
energy_per_site_per_bond = []
save_data = np.zeros((D_max, len(t_list)), dtype=float)
for i in range(len(t_list)):
    for j in range(2):
        print('i, j = ', i, j)
        TT, LL = su.simple_update(TT, LL, unitary[i], imat, smat, D_max)
        energy_per_site_per_bond.append(su.cacl_energy_per_site(TT, LL, imat, smat, hij_energy_term))
        save_data[:, i] = LL[4]


plt.figure()
plt.plot(range(len(t_list)), save_data[0, :], 'o')
#plt.plot(range(len(t_list)), save_data[1, :], 'o')
plt.grid()
plt.show()

plt.figure()
plt.plot(range(len(t_list) * 2), energy_per_site_per_bond, 'o')
plt.grid()
plt.show()
'''
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
        energy.append(su.cacl_energy_per_site(TT, LL, imat, smat, hij_energy_term))
        save_data[:, i] = LL[0]


plt.figure()
plt.plot(range(len(t_list)), save_data[0, :], 'o')
plt.plot(range(len(t_list)), save_data[1, :], 'o')
plt.grid()
plt.show()


plt.figure()
plt.plot(range(len(t_list) * iterations), energy, 'o')
plt.grid()
plt.show()

plt.figure()
plt.plot(range(len(t_list) * iterations), T0_it_time[0], 'o')
plt.grid()
plt.show()

