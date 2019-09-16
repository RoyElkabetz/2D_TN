import numpy as np
import copy as cp
import BPupdate_PEPS_smart_trancation as su
from scipy import linalg
import matplotlib.pyplot as plt
import ncon_lists_generator as nlg
import virtual_DEFG as defg
import ncon
import time
#from glassyPEPS_BPupdate_experiment2 import E, E_exact, mz, mx, mz_exact, mx_exact, time_to_converge, mx_mat, mx_mat_exact, mz_mat, mz_mat_exact, reduced_dm_gPEPS, reduced_dm_exact
#from glassyPEPS_BPupdate_experiment2 import E, mz, mx, time_to_converge, mx_mat, mz_mat, reduced_dm_gPEPS, LLL, TTT

date = '2019.16.09_'
experiment_num = '_1_'

np.random.seed(seed=16)


#---------------------- Tensor Network paramas ------------------

N = 16 # number of spins
L = np.int(np.sqrt(N))

t_max = 100  # BP maximal iterations
epsilon = 1e-5 # BP convergence error
dumping = 0.1 # BP dumping

d = 2  # virtual bond dimension
p = 2  # physical bond dimension
D_max = 2  # maximal virtual bond dimension
J = 1  # Hamiltonian: interaction coeff
h = np.linspace(0.1, 5., num=10)  # Hamiltonian: magnetic field coeffs

mu = 1
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
sum_of_trace_distance_BPexact_graph = []
sum_of_trace_distance_BPexact_BPgPEPS = []
sum_of_trace_distance_BPgPEPS_graph = []
sum_of_trace_distance_gPEPS_graph = []
sum_of_trace_distance_gPEPS_BPexact = []
sum_of_trace_distance_gPEPS_BPgPEPS = []
sum_of_trace_distance_exact_graph = []
sum_of_trace_distance_exact_BPgPEPS = []
sum_of_trace_distance_exact_BPexact = []
sum_of_trace_distance_exact_gPEPS = []

trace_distance_BPexact_graph = np.zeros((len(h), L, L))
trace_distance_BPexact_BPgPEPS = np.zeros((len(h), L, L))
trace_distance_BPgPEPS_graph = np.zeros((len(h), L, L))
trace_distance_gPEPS_graph = np.zeros((len(h), L, L))
trace_distance_gPEPS_BPexact = np.zeros((len(h), L, L))
trace_distance_gPEPS_BPgPEPS = np.zeros((len(h), L, L))
trace_distance_exact_graph = np.zeros((len(h), L, L))
trace_distance_exact_BPgPEPS = np.zeros((len(h), L, L))
trace_distance_exact_BPexact = np.zeros((len(h), L, L))
trace_distance_exact_gPEPS = np.zeros((len(h), L, L))

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

t_list = [0.1] # imaginary time evolution time steps list
iterations = 40

Opi = pauli_z
Opj = pauli_z
Op_field = pauli_x

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


# ------------- generating tensors and bond vectors ---------------------------

TT = []
for ii in range(n):
    TT.append(np.random.rand(p, d, d, d, d) + 1j *np.random.rand(p, d, d, d, d))
LL = []
for i in range(imat.shape[1]):
    LL.append(np.ones(d, dtype=float) / d)

graph = defg.Graph()
graph = su.PEPStoDEnFG_transform(graph, TT, LL, smat)

for ss in range(h.shape[0]):

    counter = 0

    # --------------------------------- iterating the gPEPS and BP algorithms -------------------------------------
    for dt in t_list:
        flag = 0

        for j in range(iterations):
            counter += 2
            print('h, h_idx, t, j = ', h[ss], ss, dt, j)
            TT1, LL1 = su.PEPS_BPupdate(TT, LL, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max, graph)
            #TT1, LL1 = su.BPupdate(TT1, LL1, smat, imat, t_max, epsilon, dumping, D_max)
            TT2, LL2 = su.PEPS_BPupdate(TT1, LL1, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max, graph)
            #TT2, LL2 = su.BPupdate(TT2, LL2, smat, imat, t_max, epsilon, dumping, D_max)

            energy1 = su.energy_per_site(TT1, LL1, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
            energy2 = su.energy_per_site(TT2, LL2, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
            #energy1 = su.exact_energy_per_site(TT1, LL1, smat, Jk, h[ss], Opi, Opj, Op_field)
            #energy2 = su.exact_energy_per_site(TT2, LL2, smat, Jk, h[ss], Opi, Opj, Op_field)
            print('energy1 = ', energy1)
            print('energy2 = ', energy2)
            if np.abs(energy1 - energy2) < 1e-4:
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
            T_list_n, idx_list_n = nlg.ncon_list_generator(TT, LL, smat, np.eye(p), spin_index)
            T_listz, idx_listz = nlg.ncon_list_generator(TT, LL, smat, pauli_z, spin_index)
            mz_mat_exact_BP[ss, l, ll] = ncon.ncon(T_listz, idx_listz) / ncon.ncon(T_list_n, idx_list_n)

            T_listx, idx_listx = nlg.ncon_list_generator(TT, LL, smat, pauli_x, spin_index)
            mx_mat_exact_BP[ss, l, ll] = ncon.ncon(T_listx, idx_listx) / ncon.ncon(T_list_n, idx_list_n)

            mz_mat_BP[ss, l, ll] = su.single_tensor_expectation(spin_index, TT, LL, imat, smat, pauli_z)
            mx_mat_BP[ss, l, ll] = su.single_tensor_expectation(spin_index, TT, LL, imat, smat, pauli_x)

            # ------------------------ Trace distances of every spin reduced density matrix results ---------------

            #reduced_dm_BP_gPEPS[ss, l, ll, :, :] = su.tensor_reduced_dm(spin_index, TT, LL, smat, imat)
            #tensors_reduced_dm_list, indices_reduced_dm_list = nlg.ncon_list_generator_reduced_dm(TT, LL, smat, spin_index)
            #tensors_reduced_dm_listn, indices_reduced_dm_listn = nlg.ncon_list_generator(TT, LL, smat, np.eye(p), spin_index)
            #reduced_dm_BP_exact = ncon.ncon(tensors_reduced_dm_list, indices_reduced_dm_list) / ncon.ncon(tensors_reduced_dm_listn, indices_reduced_dm_listn)

            #trace_distance_BPexact_graph[ss, l, ll] = su.trace_distance(reduced_dm_BP_exact, graph.rdm_belief[spin_index])
            #trace_distance_BPexact_BPgPEPS[ss, l, ll] = su.trace_distance(reduced_dm_BP_exact, reduced_dm_BP_gPEPS)
            #trace_distance_BPgPEPS_graph[ss, l, ll] = su.trace_distance(reduced_dm_BP_gPEPS[ss, l, ll, :, :], graph.rdm_belief[spin_index])
            #trace_distance_gPEPS_graph[ss, l, ll] = su.trace_distance(reduced_dm_gPEPS[ss, l, ll, :, :], graph.rdm_belief[spin_index])
            #trace_distance_gPEPS_BPexact[ss, l, ll] = su.trace_distance(reduced_dm_gPEPS[ss, l, ll, :, :], reduced_dm_BP_exact)
            #trace_distance_gPEPS_BPgPEPS[ss, l, ll] = su.trace_distance(reduced_dm_gPEPS[ss, l, ll, :, :], reduced_dm_BP_gPEPS[ss, l, ll, :, :])
            #trace_distance_exact_graph[ss, l, ll] = su.trace_distance(reduced_dm_exact[ss, l, ll, :, :], graph.rdm_belief[spin_index])
            #trace_distance_exact_BPgPEPS[ss, l, ll] = su.trace_distance(reduced_dm_exact[ss, l, ll, :, :], reduced_dm_BP_gPEPS)
            #trace_distance_exact_BPexact[ss, l, ll] = su.trace_distance(reduced_dm_exact[ss, l, ll, :, :], reduced_dm_BP_exact)
            #trace_distance_exact_gPEPS[ss, l, ll] = su.trace_distance(reduced_dm_exact[ss, l, ll, :, :], reduced_dm_gPEPS[ss, l, ll, :, :])

    # ------------------ calculating total magnetization, energy and time to converge -------------------
    mz_BP.append(np.sum(mz_mat_BP[ss, :, :]) / n)
    mx_BP.append(np.sum(mx_mat_BP[ss, :, :]) / n)
    mz_exact_BP.append(np.sum(mz_mat_exact_BP[ss, :, :]) / n)
    mx_exact_BP.append(np.sum(mx_mat_exact_BP[ss, :, :]) / n)
    time_to_converge_BP[ss] = counter
    E_BP.append(su.energy_per_site(TT, LL, imat, smat, Jk, h[ss], Opi, Opj, Op_field))
    E_exact_BP.append(su.exact_energy_per_site(TT, LL, smat, Jk, h[ss], Opi, Opj, Op_field))

    #sum_of_trace_distance_BPexact_BPgPEPS.append(trace_distance_BPexact_BPgPEPS[ss, :, :].sum())
    #sum_of_trace_distance_BPexact_graph.append(trace_distance_BPexact_graph[ss, :, :].sum())
    #sum_of_trace_distance_BPgPEPS_graph.append(trace_distance_BPgPEPS_graph[ss, :, :].sum())
    #sum_of_trace_distance_gPEPS_graph.append(trace_distance_gPEPS_graph[ss, :, :].sum())
    #sum_of_trace_distance_gPEPS_BPexact.append(trace_distance_gPEPS_BPexact[ss, :, :].sum())
    #sum_of_trace_distance_gPEPS_BPgPEPS.append(trace_distance_gPEPS_BPgPEPS[ss, :, :].sum())
    #sum_of_trace_distance_exact_graph.append(trace_distance_exact_graph[ss, :, :].sum())
    #sum_of_trace_distance_exact_BPgPEPS.append(trace_distance_exact_BPgPEPS[ss, :, :].sum())
    #sum_of_trace_distance_exact_BPexact.append(trace_distance_exact_BPexact[ss, :, :].sum())
    #sum_of_trace_distance_exact_gPEPS.append(trace_distance_exact_gPEPS[ss, :, :].sum())

    print('Mx_exact, Mz_exact', mx_exact_BP[ss], mz_exact_BP[ss])
    print('E, Mx, Mz: ', E_BP[ss], mx_BP[ss], mz_BP[ss])
    print('\n')
    #print('d(exact, gPEPS) = ', sum_of_trace_distance_BPexact_BPgPEPS[ss])

# ------------------------------------- plotting results ----------------------------------------------
file_name_energy = date + 'experiment_#' + experiment_num + 'Energy_' + 'glassy_PEPS_BPupdate_'+ J_prop + str(L) + 'x' + str(L) + '_d-' + str(D_max) +'.pdf'
file_name_magnetization = date + 'experiment_#' + experiment_num + 'Magnetization_' + 'glassy_PEPS_BPupdate_'+ J_prop + str(L) + 'x' + str(L) + '_d-' + str(D_max) +'.pdf'
file_name_TD = date + 'experiment_#' + experiment_num + 'Trace_Distance_' + 'glassy_PEPS_BPupdate_'+ J_prop + str(L) + 'x' + str(L) + '_d-' + str(D_max) +'.pdf'



plt.figure()
plt.title(str(N) + ' spins 2D glassy PEPS BP update (' + J_prop + ') Quantum Ising Model with \n a transverse field and maximal bond dimension d = ' + str(D_max))
plt.subplot()
color = 'tab:red'
plt.xlabel('h')
plt.ylabel('Energy per site', color=color)
plt.plot(h, E_BP, '-.', color=color)
#plt.plot(h, E, linewidth=2, color=color)
plt.plot(h, E_exact_BP, 'o', markersize=3, color=color)
#plt.plot(h, E_exact, '+', markersize=5, color=color)
plt.grid()
plt.tick_params(axis='y', labelcolor=color)
#plt.legend(['BP gPEPS', 'gPEPS', 'BP exact', 'gPEPS exact'])
plt.legend(['BP gPEPS', 'BP exact'])
plt.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
plt.ylabel('# of gPEPS iterations until convergence', color=color)  # we already handled the x-label with ax1
plt.plot(h, time_to_converge_BP, '-.', color=color)
#plt.plot(h, time_to_converge, linewidth=2, color=color)
plt.tick_params(axis='y', labelcolor=color)
plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.savefig(file_name_energy, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(h, mx_BP, 'go', markersize=4)
#plt.plot(h, mx, 'go', markersize=3)
plt.plot(h, np.abs(np.array(mz_BP)), 'bo', markersize=4)
#plt.plot(h, np.abs(np.array(mz)), 'bo', markersize=3)
plt.plot(h, mx_exact_BP, 'r-.', linewidth=3)
#plt.plot(h, mx_exact, 'r-', linewidth=2)
plt.plot(h, np.abs(np.array(mz_exact_BP)), 'y-.', linewidth=3)
#plt.plot(h, np.abs(np.array(mz_exact)), 'y-', linewidth=2)
plt.title('Averaged magnetization vs h at d = ' + str(D_max) + ' in a ' + str(L) + 'x' + str(L) + ' PEPS BP update ')
plt.xlabel('h')
plt.ylabel('Magnetization')
plt.legend(['BP mx ', 'BP |mz|', 'BP mx exact', 'BP |mz| exact'])
#plt.legend(['BP mx ', 'gPEPS mx', 'BP |mz|', 'gPEPS |mz|'])
plt.grid()
plt.savefig(file_name_magnetization, bbox_inches='tight')
plt.show()

'''
#legend = ['D(BPe,BPg)', 'D(BPe,BP)', 'D(BPg,BP)', 'D(g,BP)', 'D(BPe,g)', 'D(g,BPg)', 'D(e,BP)', 'D(e,BPg)', 'D(BPe,e)', 'D(e,g)']
legend = ['D(BPg,BP)', 'D(g,BP)', 'D(g,BPg)']
plt.figure()
#plt.plot(h, sum_of_trace_distance_BPexact_BPgPEPS)
#plt.plot(h, sum_of_trace_distance_BPexact_graph)
plt.plot(h, sum_of_trace_distance_BPgPEPS_graph)
plt.plot(h, sum_of_trace_distance_gPEPS_graph)
#plt.plot(h, sum_of_trace_distance_gPEPS_BPexact)
plt.plot(h, sum_of_trace_distance_gPEPS_BPgPEPS)
#plt.plot(h, sum_of_trace_distance_exact_graph)
#plt.plot(h, sum_of_trace_distance_exact_BPgPEPS)
#plt.plot(h, sum_of_trace_distance_exact_BPexact)
#plt.plot(h, sum_of_trace_distance_exact_gPEPS)
plt.title('Total Trace Distance comparison of all particles rdms \n in a ' + str(L) + 'x' + str(L) + ' gPEPS update and BP update')
plt.xlabel('h')
plt.ylabel('Trace distance')
plt.legend(legend)
plt.ylim([0, 0.5])
plt.grid()
plt.savefig(file_name_TD, bbox_inches='tight')
plt.show()
'''
