import numpy as np
import copy as cp
import BPupdate_PEPS_smart_trancation as su
from scipy import linalg
import matplotlib.pyplot as plt
import ncon_lists_generator as nlg
import virtual_DEFG as defg
import ncon
import time
import Tensor_Network_functions as tnf


np.random.seed(seed=15)


#---------------------- Tensor Network paramas ------------------

N = 4 # number of spins
L = np.int(np.sqrt(N))

t_max = 1000  # BP maximal iterations
epsilon = 1e-5 # BP convergence error
dumping = 0.1 # BP dumping

d = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]  # virtual bond dimension
p = 2  # physical bond dimension
D_max = 2  # maximal virtual bond dimension
J = 1  # Hamiltonian: interaction coeff
h = [0]  # Hamiltonian: magnetic field coeffs

mu = -1
sigma = 0
Jk = np.random.normal(mu, sigma, (2 * N)) # interaction constant list
print('Jk = ', Jk)
J_prop = 'J = N(' + str(mu) + ',' + str(sigma) + ')_'


time_to_converge_BP = np.zeros((len(h)))
mz_matrix_TN = np.zeros((p, p, len(h)))

E_BP = []
E_exact_BP = []
mx_BP = []
mz_BP = []
mx_exact_BP = []
mz_exact_BP = []
mx_graph = []
mz_graph = []


mx_mat_BP = np.zeros((len(h), L, L), dtype=complex)
mz_mat_BP = np.zeros((len(h), L, L), dtype=complex)
mx_mat_exact_BP = np.zeros((len(h), L, L), dtype=complex)
mz_mat_exact_BP = np.zeros((len(h), L, L), dtype=complex)
reduced_dm_BP_gPEPS = np.zeros((len(h), L, L, p, p), dtype=complex)


pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

t_list = [0.1, 0.05, 0.01, 0.005, 0.]  # imaginary time evolution time steps list
iterations = 100


Opi = [sx, sy, sz]
Opj = [sx, sy, sz]
Op_field = np.eye(p)

#------------- generating the finite PEPS structure matrix------------------

#smat, imat = tnf.PEPS_smat_imat_gen(N)
#smat, imat = tnf.PEPS_OBC_smat_imat(N)

smat = np.array([[1, 2, 3, 0, 0, 4, 0, 0], [0, 0, 1, 2, 3, 0, 0, 4], [0, 0, 0, 4, 1, 2, 3, 0], [3, 4, 0, 0, 0, 0, 1, 2]])
n, m = smat.shape

# ------------- generating tensors and bond vectors ---------------------------

#TT, LL = tnf.random_tn_gen(smat, p, d)
TT, LL = tnf.PEPS_OBC_random_tn_gen(smat, p, d)

# ------------- generating the double-edge factor graph (defg) of the tensor network ---------------------------

graph = defg.Graph()
graph = su.PEPStoDEnFG_transform(graph, TT, LL, smat)

for ss in range(len(h)):

    counter = 0

    # --------------------------------- iterating the gPEPS and BP algorithms -------------------------------------
    for dt in t_list:
        flag = 0

        for j in range(iterations):
            counter += 2
            print('h, h_idx, t, j = ', h[ss], ss, dt, j)
            TT1, LL1 = su.PEPS_BP_update(TT, LL, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max, graph)
            TT2, LL2 = su.PEPS_BP_update(TT1, LL1, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max, graph)

            energy1 = su.energy_per_site(TT1, LL1, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
            energy2 = su.energy_per_site(TT2, LL2, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
            print(energy1, energy2)
            #energy1 = su.exact_energy_per_site(TT1, LL1, smat, Jk, h[ss], Opi, Opj, Op_field)
            #energy2 = su.exact_energy_per_site(TT2, LL2, smat, Jk, h[ss], Opi, Opj, Op_field)
            #print(energy1, energy2)
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

    # ---------------------------------- calculating reduced density matrices using DEFG ----------------------------

    graph.sum_product(t_max, epsilon, dumping)
    graph.calc_rdm_belief()


    # --------------------------------- calculating magnetization matrices -------------------------------
    for l in range(L):
        for ll in range(L):
            spin_index = np.int(L * l + ll)
            #T_list_n, idx_list_n = nlg.ncon_list_generator(TT, LL, smat, np.eye(p), spin_index)
            #T_listz, idx_listz = nlg.ncon_list_generator(TT, LL, smat, pauli_z, spin_index)
            #mz_mat_exact_BP[ss, l, ll] = ncon.ncon(T_listz, idx_listz) / ncon.ncon(T_list_n, idx_list_n)

            #T_listx, idx_listx = nlg.ncon_list_generator(TT, LL, smat, pauli_x, spin_index)
            #mx_mat_exact_BP[ss, l, ll] = ncon.ncon(T_listx, idx_listx) / ncon.ncon(T_list_n, idx_list_n)

            mz_mat_BP[ss, l, ll] = su.single_tensor_expectation(spin_index, TT, LL, imat, smat, pauli_z)
            mx_mat_BP[ss, l, ll] = su.single_tensor_expectation(spin_index, TT, LL, imat, smat, pauli_x)

            # ------------------------ Trace distances of every spin reduced density matrix results ---------------

            reduced_dm_BP_gPEPS[ss, l, ll, :, :] = su.tensor_reduced_dm(spin_index, TT, LL, smat, imat)
            #tensors_reduced_dm_list, indices_reduced_dm_list = nlg.ncon_list_generator_reduced_dm(TT, LL, smat, spin_index)
            #tensors_reduced_dm_listn, indices_reduced_dm_listn = nlg.ncon_list_generator(TT, LL, smat, np.eye(p), spin_index)
            #reduced_dm_BP_exact = ncon.ncon(tensors_reduced_dm_list, indices_reduced_dm_list) / ncon.ncon(tensors_reduced_dm_listn, indices_reduced_dm_listn)

            #trace_distance_BPexact_graph[ss, l, ll] = su.trace_distance(reduced_dm_BP_exact, graph.rdm_belief[spin_index])
            #trace_distance_BPexact_BPgPEPS[ss, l, ll] = su.trace_distance(reduced_dm_BP_exact, reduced_dm_BP_gPEPS)
            trace_distance_BPgPEPS_graph[ss, l, ll] = su.trace_distance(reduced_dm_BP_gPEPS[ss, l, ll, :, :], graph.rdm_belief[spin_index])
            trace_distance_gPEPS_graph[ss, l, ll] = su.trace_distance(reduced_dm_gPEPS[ss, l, ll, :, :], graph.rdm_belief[spin_index])
            #trace_distance_gPEPS_BPexact[ss, l, ll] = su.trace_distance(reduced_dm_gPEPS[ss, l, ll, :, :], reduced_dm_BP_exact)
            trace_distance_gPEPS_BPgPEPS[ss, l, ll] = su.trace_distance(reduced_dm_gPEPS[ss, l, ll, :, :], reduced_dm_BP_gPEPS[ss, l, ll, :, :])
            #trace_distance_exact_graph[ss, l, ll] = su.trace_distance(reduced_dm_exact[ss, l, ll, :, :], graph.rdm_belief[spin_index])
            #trace_distance_exact_BPgPEPS[ss, l, ll] = su.trace_distance(reduced_dm_exact[ss, l, ll, :, :], reduced_dm_BP_gPEPS)
            #trace_distance_exact_BPexact[ss, l, ll] = su.trace_distance(reduced_dm_exact[ss, l, ll, :, :], reduced_dm_BP_exact)
            #trace_distance_exact_gPEPS[ss, l, ll] = su.trace_distance(reduced_dm_exact[ss, l, ll, :, :], reduced_dm_gPEPS[ss, l, ll, :, :])

    # ------------------ calculating total magnetization, energy and time to converge -------------------
    mz_BP.append(np.sum(mz_mat_BP[ss, :, :]) / n)
    mx_BP.append(np.sum(mx_mat_BP[ss, :, :]) / n)
    #mz_exact_BP.append(np.sum(mz_mat_exact_BP[ss, :, :]) / n)
    #mx_exact_BP.append(np.sum(mx_mat_exact_BP[ss, :, :]) / n)
    time_to_converge_BP[ss] = counter
    E_BP.append(su.energy_per_site(TT, LL, imat, smat, Jk, h[ss], Opi, Opj, Op_field))
    #E_exact_BP.append(su.exact_energy_per_site(TT, LL, smat, Jk, h[ss], Opi, Opj, Op_field))

    #sum_of_trace_distance_BPexact_BPgPEPS.append(trace_distance_BPexact_BPgPEPS[ss, :, :].sum())
    #sum_of_trace_distance_BPexact_graph.append(trace_distance_BPexact_graph[ss, :, :].sum())
    sum_of_trace_distance_BPgPEPS_graph.append(trace_distance_BPgPEPS_graph[ss, :, :].sum())
    sum_of_trace_distance_gPEPS_graph.append(trace_distance_gPEPS_graph[ss, :, :].sum())
    #sum_of_trace_distance_gPEPS_BPexact.append(trace_distance_gPEPS_BPexact[ss, :, :].sum())
    sum_of_trace_distance_gPEPS_BPgPEPS.append(trace_distance_gPEPS_BPgPEPS[ss, :, :].sum())
    #sum_of_trace_distance_exact_graph.append(trace_distance_exact_graph[ss, :, :].sum())
    #sum_of_trace_distance_exact_BPgPEPS.append(trace_distance_exact_BPgPEPS[ss, :, :].sum())
    #sum_of_trace_distance_exact_BPexact.append(trace_distance_exact_BPexact[ss, :, :].sum())
    #sum_of_trace_distance_exact_gPEPS.append(trace_distance_exact_gPEPS[ss, :, :].sum())

    #print('Mx_exact, Mz_exact', mx_exact_BP[ss], mz_exact_BP[ss])
    print('E, Mx, Mz: ', E_BP[ss], mx_BP[ss], mz_BP[ss])
    print('\n')
    #print('d(exact, gPEPS) = ', sum_of_trace_distance_BPexact_BPgPEPS[ss])

e2 = time.time()
run_time_of_BPupdate = e2 - s2

# ------------------------------------- plotting results ----------------------------------------------
file_name_energy = date + 'experiment_#' + experiment_num + 'Energy_' + 'glassy_PEPS_BPupdate_'+ J_prop + str(L) + 'x' + str(L) + '_d-' + str(D_max) +'.pdf'
file_name_magnetization = date + 'experiment_#' + experiment_num + 'Magnetization_' + 'glassy_PEPS_BPupdate_'+ J_prop + str(L) + 'x' + str(L) + '_d-' + str(D_max) +'.pdf'
file_name_TD = date + 'experiment_#' + experiment_num + 'Trace_Distance_' + 'glassy_PEPS_BPupdate_'+ J_prop + str(L) + 'x' + str(L) + '_d-' + str(D_max) +'.pdf'



plt.figure()
plt.title('Heisenberg Model gPEPS x magnetization')
plt.imshow(np.real(mx_mat[0, :, :]))
plt.colorbar()
plt.clim(-1, 1)
plt.show()

plt.figure()
plt.title('Heisenberg Model gPEPS z magnetization')
plt.imshow(np.real(mz_mat[0, :, :]))
plt.colorbar()
plt.clim(-1, 1)
plt.show()

plt.figure()
plt.title('Heisenberg Model BP x magnetization')
plt.imshow(np.real(mx_mat_BP[0, :, :]))
plt.colorbar()
plt.clim(-1, 1)
plt.show()

plt.figure()
plt.title('Heisenberg Model BP z magnetization')
plt.imshow(np.real(mz_mat_BP[0, :, :]))
plt.colorbar()
plt.clim(-1, 1)
plt.show()
