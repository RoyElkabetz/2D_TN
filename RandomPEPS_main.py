

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
import RandomPEPS as hmf
import bmpslib as bmps


#
#################################################    MAIN    ###########################################################
#
flag_run_new_experiment = 1
flag_save_variables = 1
flag_load_data = 1
flag_calculating_expectations = 1
flag_plot = 0
flag_save_xlsx = 0


#
############################################    EXPERIMENT PARAMETERS    ###############################################
#

np.random.seed(seed=8)

N, M = 4, 4

bc = 'open'
dE = 1e-5
t_max = 200
dumping = 0.2
epsilon = 1e-5
D_max = [2]
mu = -1
sigma = 0
Jk = np.random.normal(mu, sigma, np.int((N - 1) * M + (M - 1) * N))
#Jk = np.random.normal(mu, sigma, np.int(2 * (N * M)))
dt = [0.5, 0.1, 0.05, 0.01, 0.005]
iterations = 100


if bc == 'open':
    smat, imat = tnf.PEPS_OBC_smat_imat(N, M)
elif bc == 'periodic':
    smat, imat = tnf.PEPS_smat_imat_gen(N * M)


Dp = [2, 4, 8, 16, 32, 64, 100]
p = 2
h = 0
environment_size = [0, 1, 2]

#
############################################  RUN AND COLLECT DATA  ####################################################
#
if flag_run_new_experiment:

    BP_data = []
    SU_data = []

    for D in D_max:
        b = hmf.RandomPEPS_SU(N, M, Jk, dE, D, bc, dt, iterations)
        TT0, LL0 = b[0], b[1]
        a = hmf.RandomPEPS_BP(N, M, Jk, dE, D, t_max, epsilon, dumping, bc, dt, iterations, [TT0, LL0])
        BP_data.append(a)
        SU_data.append(b)


#
#################################################  SAVING VARIABLES  ###################################################
#
if flag_save_variables:

    parameters = [['N, M', [N, M]], ['dE', dE], ['t_max', t_max], ['dumping', dumping], ['epsilon', epsilon], ['D_max', D_max]]
    file_name = "2019_02_14_1_16_OBC_Random_PEPS"
    pickle.dump(parameters, open(file_name + '_parameters.p', "wb"))
    pickle.dump(BP_data, open(file_name + '_BP.p', "wb"))
    pickle.dump(SU_data, open(file_name + '_gPEPS.p', "wb"))



#
#################################################   LOADING DATA   #####################################################
#
if flag_load_data:

    file_name_bp = "2019_02_14_1_16_OBC_Random_PEPS_BP.p"
    file_name_gpeps = "2019_02_14_1_16_OBC_Random_PEPS_gPEPS.p"
    file_name1 = "2019_02_14_1_16_OBC_Random_PEPS_parameters.p"

    data_bp = pickle.load(open(file_name_bp, "rb"))
    data_su = pickle.load(open(file_name_gpeps, "rb"))
    data_params = pickle.load(open(file_name1, "rb"))


rho_SU = []
rho_BP = []
rho_BP_factor_belief = []
rho_BP_bmps = []
rho_SU_bmps = []
traceDistance = []

#
############################################  CALCULATING EXPECTATIONS  ################################################
#
if flag_calculating_expectations:
    for ii in range(len(data_params[5][1])):

        graph, TT_BP, LL_BP = data_bp[ii]
        TT_SU, LL_SU = data_su[ii][2], data_su[ii][3]
        TT_BP_bmps = cp.deepcopy(TT_BP)
        TT_SU_bmps = cp.deepcopy(TT_SU)

    #
    ######### CALCULATING REDUCED DENSITY MATRICES  ########
    #

        for i in range(len(TT_BP)):
            rho_SU.append(BP.tensor_reduced_dm(i, TT_SU, LL_SU, smat))
            rho_BP.append(BP.tensor_reduced_dm(i, TT_BP, LL_BP, smat))
        rho_BP_factor_belief.append(graph.rdm_using_factors())


        TT_BP_bmps = BP.absorb_all_sqrt_bond_vectors(TT_BP_bmps, LL_BP, smat)
        TT_BP_bmps = tnf.PEPS_OBC_broadcast_to_Itai(TT_BP_bmps, [N, M], p, data_params[5][1][ii])
        BP_peps = bmps.peps(N, M)
        for t, T in enumerate(TT_BP_bmps):
            i, j = np.unravel_index(t, [N, M])
            BP_peps.set_site(T, i, j)
        for dp in Dp:
            print('D, Dp = ',data_params[5][1][ii], dp)
            rho_BP_bmps.append(bmps.calculate_PEPS_2RDM(BP_peps, dp))


        TT_SU_bmps = BP.absorb_all_sqrt_bond_vectors(TT_SU_bmps, LL_SU, smat)
        TT_SU_bmps = tnf.PEPS_OBC_broadcast_to_Itai(TT_SU_bmps, [N, M], p, data_params[5][1][ii])
        SU_peps = bmps.peps(N, M)
        for t, T in enumerate(TT_SU_bmps):
            i, j = np.unravel_index(t, [N, M])
            SU_peps.set_site(T, i, j)
        for dp in Dp:
            print(dp)
            rho_SU_bmps.append(bmps.calculate_PEPS_2RDM(SU_peps, dp))


    #
    ###################################################  PLOTTING DATA  ####################################################
    #




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
    '''
    save_list = [E_BP, E_BP_factor_belief, E_gPEPS, E_BP_bmps, E_gPEPS_bmps]
    df = pd.DataFrame(save_list, columns=range(len(Dp) * (len(data_params[5][1]) - 5)), index=['E BP', 'E BP factor belief', 'E gPEPS', 'E BP bmps', 'E gPEPS bmps'])
    filepath = 'energies16AFH_D7_64.xlsx'
    df.to_excel(filepath, index=True)
    '''








