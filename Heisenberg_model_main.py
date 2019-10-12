

import numpy as np
import copy as cp
from scipy import linalg
import matplotlib.pyplot as plt
import ncon
import time
import pickle
import pandas as pd



import BPupdate_PEPS_smart_trancation as BP
import BPupdate_PEPS_smart_trancation2 as gPEPS
import ncon_lists_generator as nlg
import virtual_DEFG as defg
import Tensor_Network_functions as tnf
import Heisenberg_model_function as hmf
import bmpslib as bmps


#
#################################################    MAIN    ###########################################################
#
flag_run_new_experiment = 0
flag_save_variables = 0
flag_save_xlsx = 1
flag_load_data = 1
flag_calculating_expectations = 1
flag_plot = 1

#
############################################    EXPERIMENT PARAMETERS    ###############################################
#

np.random.seed(seed=14)

N, M = 10, 10

bc = 'open'
dE = 1e-4
t_max = 200
dumping = 0.2
epsilon = 1e-5
D_max = [1, 2, 3, 4]
mu = -1
sigma = 0
Jk = np.random.normal(mu, sigma, np.int((N - 1) * M + (M - 1) * N))
dt = [0.5, 0.1, 0.05]
iterations = 100

if bc == 'open':
    smat, imat = tnf.PEPS_OBC_smat_imat(N, M)
elif bc == 'periodic':
    smat, imat = tnf.PEPS_smat_imat_gen(N * M)

#Dp = [1, 2, 4, 8, 16, 32]
Dp = [1]
p = 2
h = 0
environment_size = [0, 1]

#
############################################  RUN AND COLLECT DATA  ####################################################
#
if flag_run_new_experiment:

    BP_data = []
    gPEPS_data = []

    for D in D_max:
        b = hmf.Heisenberg_PEPS_gPEPS(N, M, Jk, dE, D, bc, dt, iterations)
        a = hmf.Heisenberg_PEPS_BP(N, M, Jk, dE, D, t_max, epsilon, dumping, bc, dt, iterations)
        BP_data.append(a)
        gPEPS_data.append(b)


#
#################################################  SAVING VARIABLES  ###################################################
#
if flag_save_variables:

    parameters = [['N, M', [N, M]], ['dE', dE], ['t_max', t_max], ['dumping', dumping], ['epsilon', epsilon], ['D_max', D_max]]
    file_name = "2019_10_11_1_100_OBC_Antiferomagnetic_Heisenberg_lattice.p"
    pickle.dump(parameters, open(file_name + '_parameters.p', "wb"))
    pickle.dump(BP_data, open(file_name + '_BP.p', "wb"))
    pickle.dump(gPEPS_data, open(file_name + '_gPEPS.p', "wb"))



#
#################################################   LOADING DATA   #####################################################
#
if flag_load_data:

    file_name = "2019_10_10_1_100_OBC_Antiferomagnetic_Heisenberg_lattice.p"
    file_name1 = "2019_10_10_1_100_OBC_Antiferomagnetic_Heisenberg_lattice.p_parameters.p"

    data = pickle.load(open(file_name, "rb"))
    data_params = pickle.load(open(file_name1, "rb"))


E_gPEPS = []
E_BP = []
E_BP_factor_belief = []
E_BP_bmps = []
E_gPEPS_bmps = []

#
############################################  CALCULATING EXPECTATIONS  ################################################
#
for ii in [1, 2, 3]:
    if flag_calculating_expectations:
        graph, TT_BP, LL_BP, TT_gPEPS, LL_gPEPS, BP_energy, gPEPS_energy = data[ii]
        TT_BP_bmps = cp.deepcopy(TT_BP)
        TT_gPEPS_bmps = cp.deepcopy(TT_gPEPS)



    #
    ######### PARAMETERS ########
    #


        # pauli matrices
        pauli_z = np.array([[1, 0], [0, -1]])
        pauli_y = np.array([[0, -1j], [1j, 0]])
        pauli_x = np.array([[0, 1], [1, 0]])
        sz = 0.5 * pauli_z
        sy = 0.5 * pauli_y
        sx = 0.5 * pauli_x
        Opi = [sx, sy, sz]
        Opj = [sx, sy, sz]
        Op_field = np.eye(p)
        hij = np.zeros((p * p, p * p), dtype=complex)
        for i in range(len(Opi)):
            hij += np.kron(Opi[i], Opj[i])
        hij = hij.reshape(p, p, p, p)



    #
    ######### CALCULATING ENERGIES  ########
    #
        for e in environment_size:
            E_gPEPS.append(np.real(BP.energy_per_site_with_environment([N, M], e, TT_gPEPS, LL_gPEPS, smat, Jk, h, Opi, Opj, Op_field)))
            E_BP.append(np.real(BP.energy_per_site_with_environment([N, M], e, TT_BP, LL_BP, smat, Jk, h, Opi, Opj, Op_field)))
            E_BP_factor_belief.append(np.real(BP.BP_energy_per_site_using_factor_belief_with_environment(graph, e, [N, M], smat, Jk, h, Opi, Opj, Op_field)))


        TT_BP_bmps = BP.absorb_all_sqrt_bond_vectors(TT_BP_bmps, LL_BP, smat)
        TT_BP_bmps = tnf.PEPS_OBC_broadcast_to_Itai(TT_BP_bmps, [N, M], p, ii + 1)
        BP_peps = bmps.peps(N, M)
        for t, T in enumerate(TT_BP_bmps):
            i, j = np.unravel_index(t, [N, M])
            BP_peps.set_site(T, i, j)
        for dp in [1, 2, 4, 8]:
            print(dp)
            rho_BP_bmps = bmps.calculate_PEPS_2RDM(BP_peps, dp)
            rho_BP_bmps_sum = cp.deepcopy(rho_BP_bmps[0])
            for i in range(1, len(rho_BP_bmps)):
                rho_BP_bmps_sum += rho_BP_bmps[i]
            E_BP_bmps.append(np.real(np.einsum(rho_BP_bmps_sum, [0, 1, 2, 3], hij, [0, 2, 1, 3]) / (N * M)))

        TT_gPEPS_bmps = BP.absorb_all_sqrt_bond_vectors(TT_gPEPS_bmps, LL_gPEPS, smat)
        TT_gPEPS_bmps = tnf.PEPS_OBC_broadcast_to_Itai(TT_gPEPS_bmps, [N, M], p, ii + 1)
        gPEPS_peps = bmps.peps(N, M)
        for t, T in enumerate(TT_gPEPS_bmps):
            i, j = np.unravel_index(t, [N, M])
            gPEPS_peps.set_site(T, i, j)
        for dp in [1, 2, 4, 8]:
            print(dp)
            rho_gPEPS_bmps = bmps.calculate_PEPS_2RDM(gPEPS_peps, dp)
            rho_gPEPS_bmps_sum = cp.deepcopy(rho_gPEPS_bmps[0])
            for i in range(1, len(rho_gPEPS_bmps)):
                rho_gPEPS_bmps_sum += rho_gPEPS_bmps[i]
            E_gPEPS_bmps.append(np.real(np.einsum(rho_gPEPS_bmps_sum, [0, 1, 2, 3], hij, [0, 2, 1, 3]) / (N * M)))
    print('\n')
    print('E BP ------------------>', np.real(np.array(E_BP)))
    print('E BP factor belief ---->', np.real(np.array(E_BP_factor_belief)))
    print('E gPEPS --------------->', np.real(np.array(E_gPEPS)))
    print('E BP bmps ------------->', np.real(np.array(E_BP_bmps)))
    print('E gPEPS bmps ---------->', np.real(np.array(E_gPEPS_bmps)))
    print('\n')
    #
    ###################################################  PLOTTING DATA  ####################################################
    #

    if flag_plot:

        plt.figure()
        plt.title('BP and gPEPS convergence comparison')
        plt.plot(range(len(BP_energy)), BP_energy, 'o')
        plt.plot(range(len(gPEPS_energy)), gPEPS_energy, 'o')
        plt.ylim([-0.61, -0.56])
        plt.ylabel('energy per site')
        plt.xlabel('# iterations')
        plt.legend(['BP', 'gPEPS'])
        plt.grid()
        plt.show()


'''
plt.figure()
plt.subplot()
color = 'tab:red'
plt.xlabel('D')
#plt.ylabel('Energy per site', color=color)
#plt.plot(D_max, E_exact_gPEPS, '.', color=color)
#plt.plot(D_max, E_exact_BP, '+', color=color)
#plt.plot(D_max, E_article, '.-', color=color)
plt.ylabel('Energy per site')
plt.plot(D_max, E_exact_gPEPS, 'o')
plt.plot(D_max, E_exact_BP, 'o')

plt.grid()
plt.tick_params(axis='y', labelcolor=color)
plt.legend(['exact gPEPS', 'exact BP'])
plt.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
plt.ylabel('dE BP gPEPS', color=color)  # we already handled the x-label with ax1
plt.plot(D_max, E_dif, '+', color=color)
plt.tick_params(axis='y', labelcolor=color)
plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.show()
'''


#
#################################################  SAVING DATA TO XLSX  ################################################
#

if flag_save_xlsx:
    save_list = [E_BP, E_BP_factor_belief, E_gPEPS, E_BP_bmps, E_gPEPS_bmps]
    df = pd.DataFrame(save_list, columns=range(12), index=['E BP', 'E BP factor belief', 'E gPEPS', 'E BP bmps', 'E gPEPS bmps'])
    filepath = 'energies.xlsx'
    df.to_excel(filepath, index=True)








