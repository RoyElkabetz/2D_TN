import numpy as np
import copy as cp
import simple_update_algorithm as su
from scipy import linalg
import matplotlib.pyplot as plt

d = 20
p = 2
D_max = d

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

LL = []
for i in range(3):
    LL.append(np.ones(d, dtype=float) / d)

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

#pauli_z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1.]])
#pauli_y = np.array([[0, -1j, 0.], [1j, 0, -1j], [0, 1j, 0.]]) / np.sqrt(2)
#pauli_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1., 0]]) / np.sqrt(2)

t_list = np.exp(np.array(range(-1, -70, -1)) / 10.)
heisenberg = np.real(np.kron(pauli_x, pauli_x) + np.kron(pauli_y, pauli_y) + np.kron(pauli_z, pauli_z))
hij = np.reshape(heisenberg, (p, p, p, p))
hij_perm = [0, 2, 1, 3]
hij_energy_term = np.transpose(hij, hij_perm)
hij = np.reshape(hij_energy_term, [p ** 2, p ** 2])
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

iterations = 3
energy = []
save_data = np.zeros((D_max, len(t_list) * iterations), dtype=float)
T0_in_time = np.zeros((len(np.ravel(TT[0])), len(t_list) * iterations))
k = 0
for i in range(len(t_list)):
    for j in range(iterations):
        print('t, iters = ', i, j)
        save_data[:, k] = LL[0]
        TT, LL = su.simple_update(TT, LL, unitary[i], imat, smat, D_max)
        T0_in_time[:, k] = np.ravel(TT[0])
        energy.append(su.energy_per_site(TT, LL, imat, smat, hij_energy_term))
        k += 1


plt.figure()
plt.title('lambda0 values')
plt.plot(range(len(t_list) * iterations), save_data[0, :], 'o')
plt.plot(range(len(t_list) * iterations), save_data[1, :], 'o')
plt.plot(range(len(t_list) * iterations), save_data[2, :], 'o')
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
    plt.plot(range(len(t_list) * iterations), T0_in_time[k], 'o')
plt.grid()
plt.show()

