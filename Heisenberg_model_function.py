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

def Heisenberg_PEPS_BP(N, dE, D_max, t_max, epsilon, dumping):
    np.random.seed(seed=14)
    s2 = time.time()
    #---------------------- Tensor Network paramas ------------------

    L = np.int(np.sqrt(N))
    d = 2  # virtual bond dimension
    p = 2  # physical bond dimension
    h = [0]  # Hamiltonian: magnetic field coeffs
    mu = -1
    sigma = 0
    Jk = np.random.normal(mu, sigma, (2 * N)) # interaction constant list
    print('Jk = ', Jk)

    time_to_converge_BP = []
    E_BP = []
    mx_BP = []
    mz_BP = []
    mx_mat_BP = np.zeros((len(h), L, L), dtype=complex)
    mz_mat_BP = np.zeros((len(h), L, L), dtype=complex)

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

    #smat, imat = tnf.PEPS_smat_imat_gen(N)
    smat, imat = tnf.PEPS_OBC_smat_imat(N)
    n, m = smat.shape

    # ------------- generating tensors and bond vectors ---------------------------

    TT, LL = tnf.random_tn_gen(smat, p, d)

    # ------------- generating the double-edge factor graph (defg) of the tensor network ---------------------------

    graph = defg.Graph()
    graph = BP.PEPStoDEnFG_transform(graph, TT, LL, smat)

    for ss in range(len(h)):
        counter = 0
        # --------------------- finding initial condition approximated ground state for BP using gPEPS -----------------

        for dt in t_list:
            flag = 0
            for j in range(iterations):
                print('N, dt, j = ', N, dt, j)
                TT1, LL1 = gPEPS.PEPS_BPupdate(TT, LL, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max)
                TT2, LL2 = gPEPS.PEPS_BPupdate(TT1, LL1, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max)
                energy1 = BP.energy_per_site(TT1, LL1, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
                energy2 = BP.energy_per_site(TT2, LL2, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
                print(energy1, energy2)

                if np.abs(energy1 - energy2) < dE:
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

        # --------------------------------- iterating the gPEPS and BP algorithms -------------------------------------
        for dt in t_list:
            flag = 0
            for j in range(iterations):
                counter += 2
                print('N, dt, j = ', N, dt, j)
                TT1, LL1 = BP.PEPS_BPupdate(TT, LL, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max, graph)
                TT2, LL2 = BP.PEPS_BPupdate(TT1, LL1, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max, graph)
                energy1 = BP.energy_per_site(TT1, LL1, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
                energy2 = BP.energy_per_site(TT2, LL2, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
                print(energy1, energy2)

                if np.abs(energy1 - energy2) < dE:
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
                mz_mat_BP[ss, l, ll] = BP.single_tensor_expectation(spin_index, TT, LL, imat, smat, pauli_z)
                mx_mat_BP[ss, l, ll] = BP.single_tensor_expectation(spin_index, TT, LL, imat, smat, pauli_x)

        # ------------------ calculating total magnetization, energy and time to converge -------------------
        mz_BP.append(np.sum(mz_mat_BP[ss, :, :]) / n)
        mx_BP.append(np.sum(mx_mat_BP[ss, :, :]) / n)
        time_to_converge_BP.append(counter)
        E_BP.append(BP.energy_per_site(TT, LL, imat, smat, Jk, h[ss], Opi, Opj, Op_field))
        print('E, Mx, Mz: ', E_BP[ss], mx_BP[ss], mz_BP[ss])
    e2 = time.time()
    run_time_of_BPupdate = e2 - s2
    return [E_BP[0], time_to_converge_BP[0], mz_mat_BP[0, :, :], mx_mat_BP[0, :, :], run_time_of_BPupdate, TT, LL]


def Heisenberg_PEPS_gPEPS(N, dE, D_max):
    np.random.seed(seed=14)
    s2 = time.time()
    # ---------------------- Tensor Network paramas ------------------

    L = np.int(np.sqrt(N))
    d = 2  # virtual bond dimension
    p = 2  # physical bond dimension
    h = [0]  # Hamiltonian: magnetic field coeffs
    mu = -1
    sigma = 0
    Jk = np.random.normal(mu, sigma, (2 * N))  # interaction constant list
    print('Jk = ', Jk)

    time_to_converge = []
    E = []
    mx = []
    mz = []
    mx_mat = np.zeros((len(h), L, L), dtype=complex)
    mz_mat = np.zeros((len(h), L, L), dtype=complex)

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

    # ------------- generating the finite PEPS structure matrix------------------

    #smat, imat = tnf.PEPS_smat_imat_gen(N)
    smat, imat = tnf.PEPS_OBC_smat_imat(N)
    n, m = smat.shape

    # ------------- generating tensors and bond vectors ---------------------------

    TT, LL = tnf.random_tn_gen(smat, p, d)

    for ss in range(len(h)):
        counter = 0
        # --------------------------------- iterating the gPEPS and BP algorithms -------------------------------------
        for dt in t_list:
            flag = 0
            for j in range(iterations):
                counter += 2
                print('N, dt, j = ', N, dt, j)
                TT1, LL1 = gPEPS.PEPS_BPupdate(TT, LL, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max)
                TT2, LL2 = gPEPS.PEPS_BPupdate(TT1, LL1, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max)
                energy1 = gPEPS.energy_per_site(TT1, LL1, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
                energy2 = gPEPS.energy_per_site(TT2, LL2, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
                print(energy1, energy2)

                if np.abs(energy1 - energy2) < dE:
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
                mz_mat[ss, l, ll] = BP.single_tensor_expectation(spin_index, TT, LL, imat, smat, pauli_z)
                mx_mat[ss, l, ll] = BP.single_tensor_expectation(spin_index, TT, LL, imat, smat, pauli_x)

        # ------------------ calculating total magnetization, energy and time to converge -------------------
        mz.append(np.sum(mz_mat[ss, :, :]) / n)
        mx.append(np.sum(mx_mat[ss, :, :]) / n)
        time_to_converge.append(counter)
        E.append(BP.energy_per_site(TT, LL, imat, smat, Jk, h[ss], Opi, Opj, Op_field))
        print('E, Mx, Mz: ', E[ss], mx[ss], mz[ss])
    e2 = time.time()
    run_time_of_gPEPS = e2 - s2
    return [E[0], time_to_converge[0], mz_mat[0, :, :], mx_mat[0, :, :], run_time_of_gPEPS, TT, LL]



#------------------------- main ----------------------------
start = time.time()
file_name = "2019_09_22_2_Antiferomagnetic_Heisenberg_lattice"
file_name_single = "2019_09_22_2_Antiferomagnetic_Heisenberg_lattice_single_"

N = [4, 16, 36, 64]
dE = 1e-4
t_max = 100
dumping = 0.
epsilon = 1e-4
D_max = 2
parameters = [['N', N], ['dE', dE], ['t_max', t_max], ['dumping', dumping], ['epsilon', epsilon], ['D_max', D_max]]
BP_data = []
gPEPS_data = []
pickle.dump(parameters, open(file_name + '_parameters.p', "wb"))



for n in range(len(N)):
    a = Heisenberg_PEPS_BP(N[n], dE, D_max, t_max, epsilon, dumping)
    b = Heisenberg_PEPS_gPEPS(N[n], dE, D_max)
    pickle.dump(a, open(file_name_single + str(N[n]) + 'spins_BP.p', "wb"))
    pickle.dump(b, open(file_name_single + str(N[n]) + 'spins_gPEPS.p', "wb"))
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