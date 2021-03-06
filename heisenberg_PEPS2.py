import numpy as np
import copy as cp
import BPupdate_PEPS_smart_trancation2 as su
from scipy import linalg
import matplotlib.pyplot as plt
import ncon_lists_generator as nlg
import ncon
import DEnFG as fg

#date = '2019.09.09_'
#experiment_num = '_2_'

np.random.seed(seed=15)


#---------------------- Tensor Network paramas ------------------

N = 4 # number of spins
L = np.int(np.sqrt(N))

t_max = 100
epsilon = 1e-15
dumping = 0.1

d = 2  # virtual bond dimension
p = 2  # physical bond dimension
D_max = 2  # maximal virtual bond dimension
J = 1  # Hamiltonian: interaction coeff
h = [0]  # Hamiltonian: magnetic field coeff

mu = -1
sigma = 0
Jk = np.random.normal(mu, sigma, (2 * N))
#Jk = np.ones((2 * N))
print('Jk = ', Jk)
J_prop = 'J = N(' + str(mu) + ',' + str(sigma) + ')_'


time_to_converge = np.zeros((len(h)))
mz_matrix_TN = np.zeros((p, p, len(h)))

E = []
E_exact = []
mx = []
mz = []
mx_exact = []
mz_exact = []
mx_graph = []
mz_graph = []
sum_of_trace_distance_exact_graph = []
sum_of_trace_distance_exact_gPEPS = []
sum_of_trace_distance_gPEPS_graph = []
trace_distance_exact_graph = np.zeros((len(h), L, L), dtype=complex)
trace_distance_exact_gPEPS = np.zeros((len(h), L, L), dtype=complex)
trace_distance_gPEPS_graph = np.zeros((len(h), L, L), dtype=complex)
reduced_dm_gPEPS = np.zeros((len(h), L, L, p, p), dtype=complex)
reduced_dm_exact = np.zeros((len(h), L, L, p, p), dtype=complex)


mx_mat = np.zeros((len(h), L, L), dtype=complex)
mz_mat = np.zeros((len(h), L, L), dtype=complex)
mx_mat_exact = np.zeros((len(h), L, L), dtype=complex)
mz_mat_exact = np.zeros((len(h), L, L), dtype=complex)


pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

t_list = [0.1]  # imaginary time evolution time steps list
iterations = 100


Opi = [sx, sy, sz]
Opj = [sx, sy, sz]
Op_field = np.eye(p)

#------------- generating the finite PEPS structure matrix------------------
imat = np.zeros((N, 2 * N), dtype=int)
smat = np.zeros((N, 2 * N), dtype=int)
n, m = imat.shape
for i in range(n):
    imat[i, 2 * i] = 1
    imat[i, 2 * i + 1] = 1
    imat[i, 2 * np.mod(i + 1, L) + 2 * L * np.int(np.floor(np.float(i) / np.float(L)))] = 1
    imat[i, 2 * np.mod(i + L, N) + 1] = 1

    smat[i, 2 * i] = 1
    smat[i, 2 * i + 1] = 2
    smat[i, 2 * np.mod(i + 1, L) + 2 * L * np.int(np.floor(np.float(i) / np.float(L)))] = 3
    smat[i, 2 * np.mod(i + L, N) + 1] = 4


# ------------- generating tensors and bond vectors for each magnetic field ---------------------------

TT = []
for ii in range(n):
    TT.append(np.random.rand(p, d, d, d, d) + 1j *np.random.rand(p, d, d, d, d))
LL = []
for i in range(imat.shape[1]):
    LL.append(np.ones(d, dtype=float) / d)
for ss in range(len(h)):

    counter = 0

    # --------------------------------- iterating the gPEPS algorithm -------------------------------------
    for dt in t_list:
        flag = 0

        for j in range(iterations):
            counter += 2
            print('h, h_idx, t, j = ', h[ss], ss, dt, j)
            TT1, LL1 = su.PEPS_BPupdate(TT, LL, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max)
            TT2, LL2 = su.PEPS_BPupdate(TT1, LL1, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max)

            energy1 = su.energy_per_site(TT1, LL1, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
            energy2 = su.energy_per_site(TT2, LL2, imat,  smat, Jk, h[ss], Opi, Opj, Op_field)
            print(energy1, energy2)
            #energy1 = su.exact_energy_per_site(TT1, LL1, smat, Jk, h[ss], Opi, Opj, Op_field)
            #energy2 = su.exact_energy_per_site(TT2, LL2, smat, Jk, h[ss], Opi, Opj, Op_field)
            #print(energy1, energy2)
            print('\n')


            if np.abs(energy1 - energy2) < 1e-5:
                flag = 1
                TT = TT2
                LL = LL2
                break
            else:
                TT = TT2
                LL = LL2
        if flag:
            flag = 0
            break


    # --------------------------------- calculating magnetization matrices -------------------------------
    for l in range(L):
        for ll in range(L):
            spin_index = np.int(L * l + ll)
            #T_list_n, idx_list_n = nlg.ncon_list_generator(TT, LL, smat, np.eye(p), spin_index)
            #T_listz, idx_listz = nlg.ncon_list_generator(TT, LL, smat, pauli_z, spin_index)
            #mz_mat_exact[ss, l, ll] = ncon.ncon(T_listz, idx_listz) / ncon.ncon(T_list_n, idx_list_n)

            #T_listx, idx_listx = nlg.ncon_list_generator(TT, LL, smat, pauli_x, spin_index)
            #mx_mat_exact[ss, l, ll] = ncon.ncon(T_listx, idx_listx) / ncon.ncon(T_list_n, idx_list_n)

            mz_mat[ss, l, ll] = su.single_tensor_expectation(spin_index, TT, LL, imat, smat, pauli_z)
            mx_mat[ss, l, ll] = su.single_tensor_expectation(spin_index, TT, LL, imat, smat, pauli_x)

            # ------------------------ Trace distances of every spin reduced density matrix results ---------------

            reduced_dm_gPEPS[ss, l, ll, :, :] = su.tensor_reduced_dm(spin_index, TT, LL, smat, imat)
            #tensors_reduced_dm_list, indices_reduced_dm_list = nlg.ncon_list_generator_reduced_dm(TT, LL, smat, spin_index)
            #tensors_reduced_dm_listn, indices_reduced_dm_listn = nlg.ncon_list_generator(TT, LL, smat, np.eye(p), spin_index)
            #reduced_dm_exact[ss, l, ll, :, :] = ncon.ncon(tensors_reduced_dm_list, indices_reduced_dm_list) / ncon.ncon(tensors_reduced_dm_listn, indices_reduced_dm_listn)
            #trace_distance_exact_gPEPS[ss, l, ll] = su.trace_distance(reduced_dm_exact[ss, l, ll, :, :], reduced_dm_gPEPS[ss, l, ll, :, :])


    # ------------------ calculating total magnetization, energy and time to converge -------------------
    mz.append(np.sum(mz_mat[ss, :, :]) / n)
    mx.append(np.sum(mx_mat[ss, :, :]) / n)
    #mz_exact.append(np.sum(mz_mat_exact[ss, :, :]) / n)
    #mx_exact.append(np.sum(mx_mat_exact[ss, :, :]) / n)
    time_to_converge[ss] = counter
    E.append(su.energy_per_site(TT, LL, imat, smat, Jk, h[ss], Opi, Opj, Op_field))
    #E_exact.append(su.exact_energy_per_site(TT, LL, smat, Jk, h[ss], Opi, Opj, Op_field))

    #sum_of_trace_distance_exact_gPEPS.append(trace_distance_exact_gPEPS[ss, :, :].sum())
    #print('Mx_exact, Mz_exact', mx_exact[ss], mz_exact[ss])
    print('E, Mx, Mz: ', E[ss], mx[ss], mz[ss])
    print('\n')
    #print('d(exact, gPEPS) = ', sum_of_trace_distance_exact_gPEPS[ss])

LLL = cp.deepcopy(LL)
TTT = cp.deepcopy(TT)
'''
plt.figure()
plt.title('Heisenberg Model exact x magnetization')
plt.imshow(np.real(mx_mat_exact[0, :, :]))
plt.colorbar()
plt.clim(-1, 1)
plt.show()

plt.figure()
plt.title('Heisenberg Model exact z magnetization')
plt.imshow(np.real(mz_mat_exact[0, :, :]))
plt.colorbar()
plt.clim(-1, 1)
plt.show()

plt.figure()
plt.title('Heisenberg Model BP x magnetization')
plt.imshow(np.real(mx_mat[0, :, :]))
plt.colorbar()
plt.clim(-1, 1)
plt.show()

plt.figure()
plt.title('Heisenberg Model z magnetization')
plt.imshow(np.real(mz_mat[0, :, :]))
plt.colorbar()
plt.clim(-1, 1)
plt.show()
'''
