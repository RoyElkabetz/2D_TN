import numpy as np
import copy as cp
import gPEPS as su
from scipy import linalg
import matplotlib.pyplot as plt


d = 8
p = 2
D_max = d
J = -1.

TT = []
LL = []
energy = []


imat = np.zeros((25, 50), dtype=int)
smat = np.zeros((25, 50), dtype=int)
n, m = imat.shape
for i in range(n):
    imat[i, 2 * i] = 1
    imat[i, 2 * i + 1] = 1
    imat[i, 2 * np.mod(i + 1, 5) + 10 * np.int(np.floor(np.float(i) / 5.))] = 1
    imat[i, 2 * np.mod(i + 5, 25) + 1] = 1

    smat[i, 2 * i] = 1
    smat[i, 2 * i + 1] = 2
    smat[i, 2 * np.mod(i + 1, 5) + 10 * np.int(np.floor(np.float(i) / 5.))] = 3
    smat[i, 2 * np.mod(i + 5, 25) + 1] = 4

    TT.append(np.random.rand(p, d, d, d, d))

for i in range(m):
    LL.append(np.ones(d, dtype=float) / d)

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

t_list = [0.1]
iterations = 200
heisenberg = -J * np.real(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
hij = np.reshape(heisenberg, (p, p, p, p))
hij_energy_term = cp.deepcopy(hij)
hij = np.reshape(hij, [p ** 2, p ** 2])
unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]

counter = 0


for i in range(len(t_list)):
    for j in range(iterations):
        print('t, iters = ', i, j)

        TT_new, LL_new = su.simple_update(TT, LL, unitary[i], imat, smat, D_max)
        energy.append(su.energy_per_site(TT, LL, imat, smat, hij_energy_term))
        counter += 1
        TT = cp.deepcopy(TT_new)
        LL = cp.deepcopy(LL_new)


plt.figure()
plt.title('energy values')
plt.xlabel('t')
plt.plot(range(counter), energy[:counter], 'o')
plt.grid()
plt.show()
