import numpy as np
import copy as cp
import simple_update_algorithm as su
from scipy import linalg
import matplotlib.pyplot as plt

d = 2
p = 2
D_max = 2

T0 = np.random.rand(p, d, d)
T1 = np.random.rand(p, d, d)


TT = [T0, T1]

imat = np.array([[1, 1],
                 [1, 1]])

smat = np.array([[1, 2],
                 [2, 1]])

LL = []
for i in range(2):
    LL.append(np.ones(d, dtype=float) / d)

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

#pauli_z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1.]])
#pauli_y = np.array([[0, -1j, 0.], [1j, 0, -1j], [0, 1j, 0.]]) / np.sqrt(2)
#pauli_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1., 0]]) / np.sqrt(2)

t_list = np.linspace(1e-1, 1e-5, 100)
H_term = np.kron(pauli_x, pauli_x) + np.kron(pauli_y, pauli_y) + np.kron(pauli_z, pauli_z)
hij = H_term.reshape(p, p, p, p)
hij_perm = [0, 2, 1, 3]
hij_energy_term, _, _ = su.permshape(hij, hij_perm, [p, p, p, p])
hij, reverse_perm, old_shape = su.permshape(hij, hij_perm, [p ** 2, p ** 2])

unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]


# check unitary values
'''
usum = []
for j in range(len(unitary)):
    usum.append(np.sum(np.real(unitary[j])))
plt.figure()
plt.plot(range(len(usum)), usum)
plt.show()
'''
iterations = 10
energy = []
save_data = np.zeros((D_max, len(t_list)), dtype=float)
for i in range(len(t_list)):
    for j in range(iterations):
        print('i, j = ', i, j)
        TT, LL = su.simple_update(TT, LL, unitary[i], imat, smat, D_max)
        energy.append(su.cacl_energy_per_site(TT, LL, imat, smat, hij_energy_term))
        save_data[:, i] = LL[0]


plt.figure()
plt.plot(range(len(t_list)), save_data[0, :], 'o')
plt.plot(range(len(t_list)), save_data[1, :], 'o')
plt.show()

plt.figure()
plt.plot(range(len(t_list) * iterations), energy, 'o')
plt.show()

