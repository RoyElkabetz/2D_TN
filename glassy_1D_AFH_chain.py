import numpy as np
import copy as cp
import BPupdate_MPS_openBC as su
from scipy import linalg
import matplotlib.pyplot as plt

np.random.seed(seed=18)

N = 2

EE_exact = []
EE_gpeps = []
EE_bp = []

d_vec = [2, 3, 4, 5, 6, 10]
t_max = 100
epsilon = 1e-5
dumping = 0.1
d_and_t = np.zeros((2, len(d_vec)))

d = 2
p = 2
h = 0

imat = np.zeros((N, N), dtype=int)
smat = np.zeros((N, N), dtype=int)
for i in range(N):
    imat[i][i] = 1
    imat[np.mod(i + 1, N)][i] = 1
    smat[i][i] = 2
    smat[np.mod(i + 1, N)][i] = 1

J = [1.] * smat.shape[1]


pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

t_list = [0.1]
iterations = 200
Aij = np.real(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
Bij = 0
for ss in range(len(d_vec)):
    D_max = d_vec[ss]
    TT = []
    for i in range(smat.shape[0]):
        TT.append(np.random.rand(p, d, d))

    LL = []
    for i in range(imat.shape[1]):
        LL.append(np.ones(d, dtype=float) / d)

    counter = 0
    for i in range(len(t_list)):
        dt = t_list[i]
        flag = 0
        for j in range(iterations):
            counter += 2
            print('N, D_max, ss, i, j = ', N, D_max, ss, i, j)
            TT1, LL1 = su.PEPS_BPupdate(TT, LL, dt, J, h, Aij, Bij, imat, smat, D_max)
            #TT1, LL1 = su.BPupdate(TT1, LL1, smat, imat, t_max, epsilon, dumping, D_max)
            TT2, LL2 = su.PEPS_BPupdate(TT1, LL1, dt, J, h, Aij, Bij, imat, smat, D_max)
            #TT2, LL2 = su.BPupdate(TT2, LL2, smat, imat, t_max, epsilon, dumping, D_max)

            #energy1 = su.energy_per_site(TT1, LL1, imat, smat, J, h, Aij, Bij)
            #energy2 = su.energy_per_site(TT2, LL2, imat, smat, J, h, Aij, Bij)
            energy1 = su.exact_energy_per_site(TT1, LL1, smat, J, h, Aij, Bij)
            energy2 = su.exact_energy_per_site(TT2, LL2, smat, J, h, Aij, Bij)
            print(energy1)
            print(energy2)
            print('\n')

            if np.abs(energy1 - energy2) < 1e-5:
                flag = 1
                break
            else:
                TT = cp.deepcopy(TT2)
                LL = cp.deepcopy(LL2)
        if flag:
            flag = 0
            TT = cp.deepcopy(TT2)
            LL = cp.deepcopy(LL2)
            break
    d_and_t[:, ss] = np.array([D_max, counter])
    EE_exact.append(su.exact_energy_per_site(TT, LL, smat, J, h, Aij, Bij))
    EE_gpeps.append(su.energy_per_site(TT, LL, imat, smat, J, h, Aij, Bij))
    #EE_bp.append(su.BP_energy_per_site(TT, LL, smat, J, h, Aij, Bij))
    print('exact: ', EE_exact[ss])
    print('gPEPS: ', EE_gpeps[ss])
    #print('BP: ', EE_bp[ss])
'''
dE = np.array(EE_gpeps) - np.array(EE_exact)

plt.figure()
plt.title('dE')
plt.xlabel('D_max')
plt.ylabel('dE')
plt.plot(d_vec, dE, 'o')
plt.grid()
plt.show()


plt.figure()
plt.title('glassy PBC AFH chain energy smart truncation')
plt.subplot()
color = 'tab:red'
plt.xlabel('D_max')
plt.ylabel('Energy per site', color=color)
plt.plot(d_vec, EE_gpeps, 'o', color=color)
plt.plot(d_vec, EE_exact, 'v', color=color)

plt.ylim([-0.45, -0.30])
plt.grid()
plt.tick_params(axis='y', labelcolor=color)
plt.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
plt.ylabel('# of iterations for energy convergence dE < 1e-5', color=color)  # we already handled the x-label with ax1
plt.plot(d_vec, d_and_t[1, :], '.', color=color)
plt.tick_params(axis='y', labelcolor=color)
plt.ylabel('# iterations')
plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.show()
'''