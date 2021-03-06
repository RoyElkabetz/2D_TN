import numpy as np
import copy as cp
import BPupdate_MPS_openBC as su
from scipy import linalg
import matplotlib.pyplot as plt

np.random.seed(seed=18)
E = []
E_exact = []
num_of_iteractions = []
NN = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]

for N in NN:
    EE_exact = []
    EE_gpeps = []
    EE_bp = []

    d_vec = [2]
    t_max = 100
    epsilon = 1e-5
    dumping = 0.1
    d_and_t = np.zeros((2, len(d_vec)))

    d = 2
    p = 2
    h = 0

    imat = np.zeros((N, N - 1), dtype=int)
    smat = np.zeros((N, N - 1), dtype=int)
    for i in range(N - 1):
        imat[i][i] = 1
        imat[np.mod(i + 1, N)][i] = 1
        smat[i][i] = 2
        smat[np.mod(i + 1, N)][i] = 1
        if i == 0:
            smat[i][i] = 1

    J = [1.] * smat.shape[1]

    TT = [np.random.rand(p, d)]
    for i in range(smat.shape[0] - 2):
        TT.append(np.random.rand(p, d, d))
    TT.append(np.random.rand(p, d))

    LL = []
    for i in range(imat.shape[1]):
        LL.append(np.ones(d, dtype=float) / d)

    pauli_z = np.array([[1, 0], [0, -1]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_x = np.array([[0, 1], [1, 0]])

    sz = 0.5 * pauli_z
    sy = 0.5 * pauli_y
    sx = 0.5 * pauli_x

    t_list = [0.1]
    iterations = 500
    Aij = np.real(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
    Bij = 0
    for ss in range(len(d_vec)):
        D_max = d_vec[ss]

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
        #print('exact: ', EE_exact[ss])
        print('gPEPS: ', EE_gpeps[ss])
        #print('BP: ', EE_bp[ss])
    num_of_iteractions.append(counter)
    E.append(EE_gpeps[-1])
    E_exact.append(EE_exact[-1])

dE = np.array(E) - np.array(E_exact)

plt.figure()
plt.title('dE')
plt.xlabel('N')
plt.ylabel('dE')
plt.plot(NN, dE, 'o')
plt.grid()
plt.show()
'''
dd = np.zeros(len(d_vec))
for i in range(len(dd)):
    dd[i] = np.float(d_vec[i]) ** (-1)

plt.figure()
plt.title('glassy AFH chain energy')
plt.plot(dd, EE_gpeps, 'o')
plt.xlabel('D_max')
plt.ylabel('normalized energy')
plt.ylim([-0.45, -0.30])
plt.grid()
plt.show()

plt.figure()
plt.title('glassy AFH chain iterations')
plt.plot(d_and_t[0, :], d_and_t[1, :], 'o')
plt.xlabel('D_max')
plt.ylabel('# of ITE iters until converged')
plt.grid()
#plt.show()
'''


plt.figure()
plt.title('energy of N 1/2 spins (AFH MPS) gPEPS')
plt.subplot()
color = 'tab:red'
plt.xlabel('N')
plt.ylabel('Energy per site', color=color)
plt.plot(NN, E, 'o', color=color)
plt.plot(NN, E_exact, 'v', color=color)
plt.legend(['SU', 'EXACT'])
plt.ylim([-0.45, -0.30])
plt.grid()
plt.tick_params(axis='y', labelcolor=color)
plt.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
plt.ylabel('# of iterations for energy convergence dE < 1e-5', color=color)  # we already handled the x-label with ax1
plt.plot(NN, num_of_iteractions, 'o', color=color)
plt.tick_params(axis='y', labelcolor=color)
plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()