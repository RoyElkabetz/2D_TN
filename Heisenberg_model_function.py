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

def Heisenberg_PEPS_BP(N, dE, D_max, t_max, epsilon, dumping):
    np.random.seed(seed=15)
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

    smat, imat = tnf.PEPS_smat_imat_gen(N)
    n, m = smat.shape

    # ------------- generating tensors and bond vectors ---------------------------

    TT, LL = tnf.random_tn_gen(smat, p, d)

    # ------------- generating the double-edge factor graph (defg) of the tensor network ---------------------------

    graph = defg.Graph()
    graph = BP.PEPStoDEnFG_transform(graph, TT, LL, smat)

    for ss in range(len(h)):
        counter = 0
        # --------------------------------- iterating the gPEPS and BP algorithms -------------------------------------
        for dt in t_list:
            flag = 0
            for j in range(iterations):
                counter += 2
                print('h, h_idx, t, j = ', h[ss], ss, dt, j)
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
        print('\n')
    e2 = time.time()
    run_time_of_BPupdate = e2 - s2
    return [E_BP[0], time_to_converge_BP[0], mz_mat_BP[0, :, :], mx_mat_BP[0, :, :], run_time_of_BPupdate]


def Heisenberg_PEPS_gPEPS(N, dE, D_max):
    np.random.seed(seed=15)
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

    smat, imat = tnf.PEPS_smat_imat_gen(N)
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
                print('h, h_idx, t, j = ', h[ss], ss, dt, j)
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
        print('\n')
    e2 = time.time()
    run_time_of_gPEPS = e2 - s2
    return [E[0], time_to_converge[0], mz_mat[0, :, :], mx_mat[0, :, :], run_time_of_gPEPS]



#------------------------- main ----------------------------

N = [4, 9, 16, 25, 36, 49, 64]
dE = 1e-4
t_max = 100
dumping = 0.
epsilon = 1e-4
D_max = 2
BP_data = []
gPEPS_data = []

for n in N:
    BP_data.append(Heisenberg_PEPS_BP(n, dE, D_max, t_max, epsilon, dumping))
    gPEPS_data.append(Heisenberg_PEPS_gPEPS(n, dE, D_max))
