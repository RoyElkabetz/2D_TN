import numpy as np
import copy as cp
import simple_update_algorithm2 as su
from scipy import linalg
import matplotlib.pyplot as plt

d_vec = range(2, 30)
E = []
d_and_t = np.zeros((2, len(d_vec)))

d = 2
p = 2
J = 1


imat = np.array([[1, 1],
                 [1, 1]])

smat = np.array([[2, 1],
                 [2, 1]])



pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

# t_list = [0.1, 0.05, 0.01, 0.001]
iterations = 1
#t_list = np.ones((100)) * 1e-1
t_list = np.exp(np.concatenate((np.linspace(-1, -2, 20), np.linspace(-3, -8, 20))))
heisenberg = -J * np.real(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
hij = np.reshape(heisenberg, (p, p, p, p))
hij_perm = [0, 2, 1, 3]
hij_energy_term = cp.deepcopy(hij)
hij = np.transpose(hij, hij_perm)
hij = np.reshape(hij, [p ** 2, p ** 2])
unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]

for ss in range(len(d_vec)):

    T0 = np.random.rand(p, d, d)
    T1 = np.random.rand(p, d, d)
    TT = [T0, T1]

    LL = []
    for i in range(imat.shape[1]):
        #LL.append(np.ones(d, dtype=float) / d)
        LL.append(np.random.rand(d))

    D_max = d_vec[ss]
    print('\n')
    print('D_max = ', D_max)
    print('\n')

    counter = 0
    for i in range(0, len(t_list) - 1, 2):
        print('t, iters = ', i)
        TT1, LL1 = su.simple_update(cp.deepcopy(TT), cp.deepcopy(LL), unitary[i], imat, smat, D_max)
        TT2, LL2 = su.simple_update(cp.deepcopy(TT1), cp.deepcopy(LL1), unitary[i + 1], imat, smat, D_max)




        energy1 = su.energy_per_site(TT1, LL1, imat, smat, hij_energy_term)
        print(energy1)
        energy2 = su.energy_per_site(TT2, LL2, imat, smat, hij_energy_term)
        print(energy2)

        if np.abs(energy1 - energy2) < 1e-5:
            break
        else:
            TT = cp.deepcopy(TT2)
            LL = cp.deepcopy(LL2)
        print('\n')
    d_and_t[:, ss] = np.array([D_max, i])
    E.append(su.energy_per_site(TT, LL, imat, smat, hij_energy_term))

print(d_and_t)

dd = np.zeros(len(d_vec))
for i in range(len(dd)):
    dd[i] = np.float(d_vec[i]) ** (-1)

plt.figure()
plt.plot(d_vec, E, 'o')
plt.xlabel('D_max')
plt.ylabel('normalized energy')
plt.grid()
plt.show()

plt.figure()
plt.plot(d_and_t[0, :], d_and_t[1, :], 'o')
plt.xlabel('D_max')
plt.ylabel('# of ITE iters until converged')
plt.grid()
plt.show()