import numpy as np
import copy as cp
import simple_update3 as su
from scipy import linalg
import matplotlib.pyplot as plt

D_max = 2
h = np.linspace(0., 4., num=100)
E = []
mx = []
mz = []

d = 2
p = 2
J = 1


imat = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 0, 0, 1, 1],
                 [0, 0, 0, 0, 1, 1, 1, 1]])

smat = np.array([[1, 2, 3, 4, 0, 0, 0, 0],
                 [1, 2, 0, 0, 3, 4, 0, 0],
                 [0, 0, 1, 2, 0, 0, 3, 4],
                 [0, 0, 0, 0, 1, 2, 3, 4]])

T0 = np.random.rand(p, d, d, d, d)
T1 = np.random.rand(p, d, d, d, d)
T2 = np.random.rand(p, d, d, d, d)
T3 = np.random.rand(p, d, d, d, d)

TT = [T0, T1, T2, T3]

LL = []
for i in range(imat.shape[1]):
    LL.append(np.ones(d, dtype=float) / d)

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

t_list = [0.1, 0.01, 0.001, 0.0001]
iterations = 50



for ss in range(h.shape[0]):
    hij = -J * np.kron(pauli_z, pauli_z) - 0.25 * h[ss] * (np.kron(np.eye(p), pauli_x) + np.kron(pauli_x, np.eye(p)))
    hij_energy_operator = np.reshape(cp.deepcopy(hij), (p, p, p, p))
    hij = np.reshape(hij, [p ** 2, p ** 2])
    unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]

    counter = 0
    for i in range(len(t_list)):
        flag = 0
        for j in range(iterations):
            counter += 1
            print('h, i, j = ', h[ss], ss, i, j)
            TT1, LL1 = su.simple_update(cp.deepcopy(TT), cp.deepcopy(LL), unitary[i], imat, smat, D_max)
            TT2, LL2 = su.simple_update(cp.deepcopy(TT1), cp.deepcopy(LL1), unitary[i], imat, smat, D_max)

            energy1 = su.energy_per_site(TT1, LL1, imat, smat, hij_energy_operator)
            energy2 = su.energy_per_site(TT2, LL2, imat, smat, hij_energy_operator)

            if np.abs(energy1 - energy2) < 1e-8:
                flag = 1
                break
            else:
                TT = cp.deepcopy(TT2)
                LL = cp.deepcopy(LL2)
        if flag:
            flag = 0
            break
    z_magnetization = 0
    x_magnetization = 0

    for k in range(len(TT)):
        z_magnetization += su.single_tensor_expectation(k, TT, LL, imat, smat, pauli_z)
        x_magnetization += su.single_tensor_expectation(k, TT, LL, imat, smat, pauli_x)

    mx.append(x_magnetization / len(TT))
    mz.append(z_magnetization / len(TT))
    E.append(su.energy_per_site(TT, LL, imat, smat, hij_energy_operator))
    print('E, Mx, Mz: ', E[ss], mx[ss], mz[ss])



plt.figure()
plt.plot(h, E, 'o')
plt.xlabel('h')
plt.ylabel('Energy')
plt.grid()
plt.show()

plt.figure()
plt.plot(h, mx, 'o')
plt.plot(h, mz, 'o')
plt.xlabel('h')
plt.ylabel('Magnetization')
plt.legend(['mx', 'mz'])
plt.grid()
plt.show()

