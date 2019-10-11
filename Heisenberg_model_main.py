

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


#
############################################    EXPERIMENT PARAMETERS    ###############################################
#

np.random.seed(seed=14)

N, M = 4, 4

bc = 'open'
dE = 1e-4
t_max = 200
dumping = 0.2
epsilon = 1e-5
D_max = [2]
mu = -1
sigma = 0
Jk = np.random.normal(mu, sigma, np.int((N - 1) * M + (M - 1) * N))
dt = [0.5, 0.1, 0.05]
iterations = 100

#
############################################  RUN AND COLLECT DATA  ####################################################
#

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

#parameters = [['N, M', [N, M]], ['dE', dE], ['t_max', t_max], ['dumping', dumping], ['epsilon', epsilon], ['D_max', D_max]]
#file_name = "2019_10_11_1_100_OBC_Antiferomagnetic_Heisenberg_lattice.p"
#pickle.dump(parameters, open(file_name + '_parameters.p', "wb"))
#pickle.dump(BP_data, open(file_name + '_BP.p', "wb"))
#pickle.dump(gPEPS_data, open(file_name + '_gPEPS.p', "wb"))


#
#################################################  SAVING DATA TO XLSX  ################################################
#

#save_list = []
#df = pd.DataFrame(E, columns=env_size)
#filepath = 'my_excel_file4x4AFH.xlsx'
#df.to_excel(filepath, index=True)


#
#################################################   LOADING DATA   #####################################################
#

#file_name = "2019_09_22_1_Antiferomagnetic_Heisenberg_lattice_single_100spins"
#BP_data = pickle.load(open(file_name + '_BP.p', "rb"))
#gPEPS_data = pickle.load(open(file_name + '_gPEPS.p', "rb"))



#
############################################  CALCULATING EXPECTATIONS  ################################################
#
graph, TT_BP, LL_BP, BP_energy = BP_data[0]
TT_gPEPS, LL_gPEPS, gPEPS_energy = gPEPS_data[0]


#
######### PARAMETERS ########
#
if bc == 'open':
    smat, imat = tnf.PEPS_OBC_smat_imat(N, M)
elif bc == 'periodic':
    smat, imat = tnf.PEPS_smat_imat_gen(N * M)

Dp = 10
p = 2
h = 0
environment_size = [0, 1]

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

E_gPEPS = []
E_BP = []
E_BP_factor_belief = []


for e in environment_size:
    E_gPEPS.append(BP.energy_per_site_with_environment([N, M], e, TT_gPEPS, LL_gPEPS, smat, Jk, h, Opi, Opj, Op_field))
    E_BP.append(BP.energy_per_site_with_environment([N, M], e, TT_BP, LL_BP, smat, Jk, h, Opi, Opj, Op_field))
    E_BP_factor_belief.append(BP.BP_energy_per_site_using_factor_belief_with_environment(graph, e, [N, M], smat, Jk, h, Opi, Opj, Op_field))


E_bmps = []
TT_bmps = cp.deepcopy(TT_BP)
TT_bmps = BP.absorb_all_sqrt_bond_vectors(TT_bmps, LL_BP, smat)
TT_bmps = tnf.PEPS_OBC_broadcast_to_Itai(TT_bmps, [N, M], p, D_max[0])
peps = bmps.peps(N, M)
for t, T in enumerate(TT_bmps):
    i, j = np.unravel_index(t, [N, M])
    peps.set_site(T, i, j)
rho_bmps = bmps.calculate_PEPS_2RDM(peps, Dp)
rho_bmps_sum = cp.deepcopy(rho_bmps[0])
for i in range(1, len(rho_bmps)):
    rho_bmps_sum += rho_bmps[i]
E_bmps.append(np.einsum(rho_bmps_sum, [0, 1, 2, 3], hij, [0, 2, 1, 3]) / (N * M))




'''
plt.figure()
plt.title('BP and gPEPS convergence comparison')
plt.plot(range(len(BP_energy)), BP_energy, 'o')
plt.plot(range(len(gPEPS_energy)), gPEPS_energy, 'o')
plt.ylim([-0.54, -0.50])
plt.ylabel('energy per site')
plt.xlabel('# iterations')
plt.legend(['BP', 'gPEPS'])
plt.grid()
plt.show()
'''




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
