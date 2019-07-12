import numpy as np
import copy as cp
import gPEPS as su
from scipy import linalg
import matplotlib.pyplot as plt

D_max = 7
h = np.linspace(0., 4., num=500)
time_to_converge = np.zeros((len(h)))
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
    T0 = np.random.rand(p, d, d, d, d)
    T1 = np.random.rand(p, d, d, d, d)
    T2 = np.random.rand(p, d, d, d, d)
    T3 = np.random.rand(p, d, d, d, d)

    TT = [T0, T1, T2, T3]

    LL = []
    for i in range(imat.shape[1]):
        LL.append(np.ones(d, dtype=float) / d)

    hij = -J * np.kron(pauli_z, pauli_z) - 0.25 * h[ss] * (np.kron(np.eye(p), pauli_x) + np.kron(pauli_x, np.eye(p)))
    hij_energy_operator = np.reshape(cp.deepcopy(hij), (p, p, p, p))
    hij = np.reshape(hij, [p ** 2, p ** 2])
    unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]

    counter = 0
    for i in range(len(t_list)):
        flag = 0
        for j in range(iterations):
            counter += 2
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
            time_to_converge[ss] = counter
            break

    mx.append(su.magnetization(TT, LL, imat, smat, pauli_x))
    mz.append(su.magnetization(TT, LL, imat, smat, pauli_z))
    E.append(su.energy_per_site(TT, LL, imat, smat, hij_energy_operator))
    print('E, Mx, Mz: ', E[ss], mx[ss], mz[ss])



plt.figure()
plt.title('2D Quantum Ising Model in a transverse field')
plt.subplot()
color = 'tab:red'
plt.xlabel('h')
plt.ylabel('Energy per site', color=color)
plt.plot(h, E, color=color)
plt.grid()
plt.tick_params(axis='y', labelcolor=color)
plt.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
plt.ylabel('# of iterations for energy convergence', color=color)  # we already handled the x-label with ax1
plt.plot(h, time_to_converge, color=color)
plt.tick_params(axis='y', labelcolor=color)
plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.show()
'''
plt.figure()
plt.plot(h, E, 'o')
plt.xlabel('h')
plt.ylabel('Energy')
plt.grid()
plt.show()
'''
plt.figure()
plt.plot(h, np.log10(np.array(mx)), 'o')
#plt.plot(h, mz, 'o')
plt.xlabel('h')
plt.ylabel('Magnetization')
#plt.legend(['mx', 'mz'])
plt.grid()
plt.show()

