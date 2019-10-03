import numpy as np
import copy as cp
import BPupdate_PEPS_smart_trancation as BP
import BPupdate_PEPS_smart_trancation2 as gPEPS
from scipy import linalg
import matplotlib.pyplot as plt
import ncon_lists_generator as nlg
import virtual_DEFG as defg
import ncon
import time
import Tensor_Network_functions as tnf
import pickle
import Heisenberg_model_function as hmf

#------------------------- main ----------------------------
'''
start = time.time()
file_name = "2019_09_22_4_OBC_Antiferomagnetic_Heisenberg_lattice"
file_name_single = "2019_09_22_4_OBC_Antiferomagnetic_Heisenberg_lattice_single_"

N = [100]
bc = 'open'
dE = 1e-6
t_max = 100
dumping = 0.2
epsilon = 1e-6
D_max = 2
mu = -1
sigma = 0
Jk = np.random.normal(mu, sigma, np.int(2 * N[0] - 2 * np.sqrt(N[0]))) # interaction constant list
print('Jk = ', Jk)
parameters = [['N', N], ['dE', dE], ['t_max', t_max], ['dumping', dumping], ['epsilon', epsilon], ['D_max', D_max]]
BP_data = []
gPEPS_data = []
#pickle.dump(parameters, open(file_name + '_parameters.p', "wb"))



for n in range(len(N)):
    
    b = hmf.Heisenberg_PEPS_gPEPS(N[n], Jk, dE, D_max, bc)
    a = hmf.Heisenberg_PEPS_BP(N[n], Jk, dE, D_max, t_max, epsilon, dumping, bc, b[6], b[7])
    #pickle.dump(a, open(file_name_single + str(N[n]) + 'spins_BP.p', "wb"))
    #pickle.dump(b, open(file_name_single + str(N[n]) + 'spins_gPEPS.p', "wb"))
    BP_data.append(a)
    gPEPS_data.append(b)
    print('\n')
    print('run time in seconds = ', time.time() - start)
    print('\n')

#pickle.dump(BP_data, open(file_name + '_BP.p', "wb"))
#pickle.dump(gPEPS_data, open(file_name + '_gPEPS.p', "wb"))


EBP = []
EgPEPS = []
iterationsBP = []
iterationsPEPS = []
timeBP = []
timegPEPS = []
stag_magnetization_zBP = []
stag_magnetization_zgPEPS = []
stag_magnetization_xBP = []
stag_magnetization_xgPEPS = []


for n in range(len(N)):
    EBP.append(BP_data[n][0])
    EgPEPS.append(gPEPS_data[n][0])
    iterationsBP.append(BP_data[n][1])
    iterationsPEPS.append(gPEPS_data[n][1])
    timeBP.append(BP_data[n][4] / 60)
    timegPEPS.append(gPEPS_data[n][4] / 60)
    AzBP = 0
    AxBP = 0
    AzgPEPS = 0
    AxgPEPS = 0
    for i in range(np.int(np.sqrt(N[n]))):
        for j in range(np.int(np.sqrt(N[n]))):
            AzBP += ((-1.) ** (i + j)) * BP_data[n][2][i, j]
            AxBP += ((-1.) ** (i + j)) * BP_data[n][3][i, j]
            AzgPEPS += ((-1.) ** (i + j)) * gPEPS_data[n][2][i, j]
            AxgPEPS += ((-1.) ** (i + j)) * gPEPS_data[n][3][i, j]
    stag_magnetization_zBP.append(AzBP / N[n])
    stag_magnetization_xBP.append(AxBP / N[n])
    stag_magnetization_zgPEPS.append(AzgPEPS / N[n])
    stag_magnetization_xgPEPS.append(AxgPEPS / N[n])



names = ['BP', 'gPEPS']

plt.figure()
plt.title('Energy')
plt.plot(N, EBP, 'o')
plt.plot(N, EgPEPS, 'o')
plt.xlabel('nXn gPEPS')
plt.xticks(N)
plt.ylabel('Energy per site')
plt.legend(names)
plt.grid()
plt.show()

plt.figure()
plt.title('Iterations')
plt.plot(N, iterationsBP, 'o')
plt.plot(N, iterationsPEPS, 'o')
plt.xlabel('nXn gPEPS')
plt.xticks(N)
plt.ylabel('# of iterations')
plt.legend(names)
plt.grid()
plt.show()

plt.figure()
plt.title('Time')
plt.plot(N, timeBP, 'o')
plt.plot(N, timegPEPS, 'o')
plt.xlabel('nXn gPEPS')
plt.xticks(N)
plt.ylabel('time [min]')
plt.legend(names)
plt.grid()
plt.show()

plt.figure()
plt.title('x staggered magnetization')
plt.plot(N, stag_magnetization_xBP, 'o')
plt.plot(N, stag_magnetization_xgPEPS, 'o')
plt.xlabel('nXn gPEPS')
plt.xticks(N)
plt.ylabel('magnetization in x')
plt.legend(names)
plt.grid()
plt.show()

plt.figure()
plt.title('z staggered magnetization')
plt.plot(N, stag_magnetization_zBP, 'o')
plt.plot(N, stag_magnetization_zgPEPS, 'o')
plt.xlabel('nXn gPEPS')
plt.xticks(N)
plt.ylabel('magnetization in z')
plt.legend(names)
plt.grid()
plt.show()

'''
# ---------------------------------- BP and gPEPS comparison --------------------------------------

np.random.seed(seed=14)

N, M = 10, 10


bc = 'open'
dE = 1e-6
t_max = 100
dumping = 0.2
epsilon = 1e-5
D_max = [2, 3, 4, 5]
mu = -1
sigma = 0
Jk = np.random.normal(mu, sigma, np.int((N - 1) * M + (M - 1) * N)) # interaction constant list
#Jk = np.random.normal(mu, sigma, np.int(2 * N[0])) # interaction constant list
print('Jk = ', Jk)



BP_data = []
gPEPS_data = []
E_gPEPS = []
E_BP = []
E_BP_rdm_belief = []
E_BP_factor_belief = []
E_exact_BP = []
E_exact_gPEPS = []
E_article = [-0.54557, -0.55481, -0.56317, -0.56660, -0.56714, -0.56715]
D = []


for n in range(len(D_max)):
    b = hmf.Heisenberg_PEPS_gPEPS(N, M, Jk, dE, D_max[n], bc)
    TT, LL = cp.deepcopy(b[7]), cp.deepcopy(b[8])
    a = hmf.Heisenberg_PEPS_BP(N, M, Jk, dE, D_max[n], t_max, epsilon, dumping, bc, TT, LL)

    E_gPEPS.append(b[0])
    E_exact_gPEPS.append(b[1])
    E_BP.append(a[0])
    E_exact_BP.append(a[1])
    E_BP_rdm_belief.append(a[2])
    E_BP_factor_belief.append(a[3])
    BP_data.append(a)
    gPEPS_data.append(b)
    #E_BP.append(a[0])
    #print('\n')
    #print('gPEPS, BP = ', E_gPEPS[n], E_BP[n])
    #print('\n')


plt.figure()
plt.plot(D_max, E_gPEPS, 'o')
#plt.plot(D_max, E_exact_gPEPS, 'o')
plt.plot(D_max, E_BP, 'o')
#plt.plot(D_max, E_exact_BP, 'o')
#plt.plot(D_max, E_BP_rdm_belief, 'o')
plt.plot(D_max, E_BP_factor_belief, 'o')

#plt.legend(['gPEPS', 'exact gPEPS', 'BP gPEPS', 'exact BP', 'BP factor'])
plt.legend(['gPEPS', 'BP gPEPS', 'BP factor'])
plt.show()

E_dif = -(np.array(E_exact_gPEPS) - np.array(E_exact_BP))

plt.figure()
plt.plot(D_max, E_gPEPS, 'o')
plt.plot(D_max, E_BP_factor_belief, 'o')
#plt.plot(D_max, E_article, 'o')
plt.legend(['gPEPS','BP'])
plt.show()

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

[5, 6, 7, 14, 15, 16]
plt.imshow(np.real(BP_data[3][7]))
plt.colorbar()
plt.show()

#parameters = [['N, M', [N, M]], ['dE', dE], ['t_max', t_max], ['dumping', dumping], ['epsilon', epsilon], ['D_max', D_max]]
#file_name = "2019_10_2_1_100_OBC_glassy_Antiferomagnetic_Heisenberg_lattice"
#file_name_single = "2019_10_2_1_100_OBC_glassy_Antiferomagnetic_Heisenberg_lattice_single_"
#pickle.dump(parameters, open(file_name + '_parameters.p', "wb"))
#pickle.dump(BP_data, open(file_name + '_BP.p', "wb"))
#pickle.dump(gPEPS_data, open(file_name + '_gPEPS.p', "wb"))


# ---------------------------------- BP, gPEPS and exact rdm's comparison --------------------------------------
'''
np.random.seed(seed=14)

N = [16]
bc = 'open'
dE = 1e-8
t_max = 100
dumping = 0.2
epsilon = 1e-10
D_max = [4]
mu = -1
sigma = 0
Jk = np.random.normal(mu, sigma, np.int(2 * N[0] - 2 * np.sqrt(N[0]))) # interaction constant list
print('Jk = ', Jk)

parameters = [['N', N], ['dE', dE], ['t_max', t_max], ['dumping', dumping], ['epsilon', epsilon], ['D_max', D_max]]

BP_data = []
gPEPS_data = []

for n in range(len(D_max)):
    b = hmf.Heisenberg_PEPS_gPEPS(N[0], Jk, dE, D_max[n], bc)
    TT, LL = cp.deepcopy(b[7]), cp.deepcopy(b[8])
    a = hmf.Heisenberg_PEPS_BP(N[0], Jk, dE, D_max[n], t_max, epsilon, dumping, bc, TT, LL)
    BP_rdm = a[12]
    gPEPS_rdm = a[11]
    exact_rdm = a[13]
    graph = a[14]

d_BP_gPEPS_rdm = []
d_BP_exact_rdm = []
d_exact_gPEPS_rdm = []

d_BP_gPEPS_rdm_trace_dis = []
d_BP_exact_rdm_trace_dis = []
d_exact_gPEPS_rdm_trace_dis = []
for i in range(N[0]):
    d_BP_gPEPS_rdm.append(np.abs(np.array(BP_rdm[i]) - np.array(gPEPS_rdm[i])))
    d_BP_exact_rdm.append(np.abs(np.array(BP_rdm[i]) - np.array(exact_rdm[i])))
    d_exact_gPEPS_rdm.append(np.abs(np.array(exact_rdm[i]) - np.array(gPEPS_rdm[i])))

    d_BP_gPEPS_rdm_trace_dis.append(BP.trace_distance(np.array(BP_rdm[i]), np.array(gPEPS_rdm[i])))
    d_BP_exact_rdm_trace_dis.append(BP.trace_distance(np.array(BP_rdm[i]), np.array(exact_rdm[i])))
    d_exact_gPEPS_rdm_trace_dis.append(BP.trace_distance(np.array(exact_rdm[i]), np.array(gPEPS_rdm[i])))

print('BP-gPEPS: ', np.sum(d_BP_gPEPS_rdm))
print('BP-exact: ', np.sum(d_BP_exact_rdm))
print('exact-gPEPS: ', np.sum(d_exact_gPEPS_rdm))
print('\n')
print('Total trance distance -> BP-gPEPS: ', np.sum(d_BP_gPEPS_rdm_trace_dis))
print('Total trance distance -> BP-exact: ', np.sum(d_BP_exact_rdm_trace_dis))
print('Total trance distance -> exact-gPEPS: ', np.sum(d_exact_gPEPS_rdm_trace_dis))

#file_name = "2019_09_25_1_16_3_OBC_glassy_Antiferomagnetic_Heisenberg_lattice"
#file_name_single = "2019_09_25_1_16_3_OBC_glassy_Antiferomagnetic_Heisenberg_lattice_single_"
#pickle.dump(parameters, open(file_name + '_parameters.p', "wb"))
#pickle.dump(BP_data, open(file_name + '_BP.p', "wb"))
#pickle.dump(gPEPS_data, open(file_name + '_gPEPS.p', "wb"))

# using absorb edges for graph
#('Total trance distance -> BP-gPEPS: ', '0.00019925973838922696')
#('Total trance distance -> BP-exact: ', '0.5931748480390501')
#('Total trance distance -> exact-gPEPS: ', '0.5929756546259999')

# using absorb edges
#('Total trance distance -> BP-gPEPS: ', '0.00019954609285328088')
#('Total trance distance -> BP-exact: ', '0.5957141951533694')
#('Total trance distance -> exact-gPEPS: ', '0.5955147163308192')


#[[0.274192  -5.17266778e-18j 0.05903992-4.85335739e-02j]
# [0.05903992+4.85335739e-02j 0.725808  +5.17266778e-18j]]

#[[ 0.72308918-1.03568695e-18j -0.05347533+4.48482259e-02j]
# [-0.05347533-4.48482259e-02j  0.27691082+1.03568695e-18j]]

#[[ 0.72262374-3.84747751e-18j -0.05336337+4.47538292e-02j]
# [-0.05336337-4.47538292e-02j  0.27737626+3.84747751e-18j]]

#[[0.28337498+5.57622080e-18j 0.05644523-4.64361896e-02j]
# [0.05644523+4.64361896e-02j 0.71662502-5.57622080e-18j]]



#[[0.29501479-2.12958749e-20j 0.08086003-6.72047305e-02j]
# [0.08086003+6.72047305e-02j 0.70498521+2.12958749e-20j]]

#[[ 0.70135976+3.30985433e-20j -0.07563283+6.37027882e-02j]
# [-0.07563283-6.37027882e-02j  0.29864024-3.30985433e-20j]]

#[[ 0.70129775+1.27811712e-20j -0.07550553+6.36205586e-02j]
# [-0.07550553-6.36205586e-02j  0.29870225-1.27811712e-20j]]

#[[0.30733656-9.28415934e-21j 0.0790835 -6.56153418e-02j]
# [0.0790835 +6.56153418e-02j 0.69266344+9.28415934e-21j]]
'''
'''
file_name = "2019_09_22_1_Antiferomagnetic_Heisenberg_lattice_single_100spins"
#file_name_single = "2019_09_24_1_16_OBC_glassy_Antiferomagnetic_Heisenberg_lattice_single_"
a = pickle.load(open(file_name + '_BP.p', "rb"))
b = pickle.load(open(file_name + '_gPEPS.p', "rb"))

plt.imshow(np.real(b[2]))
plt.colorbar()
plt.show()
'''