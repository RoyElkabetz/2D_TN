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

def Heisenberg_PEPS_BP(N, M, Jk, dE, D_max, t_max, epsilon, dumping, bc, TT, LL):
    np.random.seed(seed=13)
    s2 = time.time()
    #---------------------- Tensor Network paramas ------------------

    d = 2  # virtual bond dimension
    p = 2  # physical bond dimension
    h = [0]  # Hamiltonian: magnetic field coeffs
    mu = -1
    sigma = 0
    #Jk = np.random.normal(mu, sigma, (2 * N)) # interaction constant list
    print('Jk = ', Jk)

    time_to_converge_BP = []
    E_BP = []
    E_BP_rdm_belief = []
    E_BP_factor_belief = []
    E_BP_exact = []
    mx_BP = []
    mz_BP = []
    my_BP = []

    mx_mat_BP = np.zeros((len(h), N, M), dtype=complex)
    mz_mat_BP = np.zeros((len(h), N, M), dtype=complex)
    my_mat_BP = np.zeros((len(h), N, M), dtype=complex)
    mx_mat_exact = np.zeros((len(h), N, M), dtype=complex)
    mz_mat_exact = np.zeros((len(h), N, M), dtype=complex)
    my_mat_exact = np.zeros((len(h), N, M), dtype=complex)

    gPEPS_rdm = []
    gPEPS1_rdm = []
    exact_rdm = []


    pauli_z = np.array([[1, 0], [0, -1]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_x = np.array([[0, 1], [1, 0]])
    sz = 0.5 * pauli_z
    sy = 0.5 * pauli_y
    sx = 0.5 * pauli_x

    t_list = [1, 0.5, 0.1, 0.05, 0.01]  # imaginary time evolution time steps list
    iterations = 100
    Opi = [sx, sy, sz]
    Opj = [sx, sy, sz]
    Op_field = np.eye(p)

    #------------- generating the finite PEPS structure matrix------------------

    if bc == 'open':
        smat, imat = tnf.PEPS_OBC_smat_imat(N, M)
    if bc == 'periodic':
        smat, imat = tnf.PEPS_smat_imat_gen(N * M)

    n, m = smat.shape



    # ------------- generating tensors and bond vectors ---------------------------
    '''
    if bc == 'open':
        TT, LL = tnf.PEPS_OBC_random_tn_gen(smat, p, d)
    if bc == 'periodic':
        TT, LL = tnf.random_tn_gen(smat, p, d)
    '''

    for ss in range(len(h)):

        counter = 0
        # --------------------- finding initial condition approximated ground state for BP using gPEPS -----------------
        '''
        for dt in t_list:
            flag = 0
            for j in range(iterations):
                print('N, D max, dt, j = ', N * M, D_max, dt, j)
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
        
        # --------------------------------- calculating magnetization matrices -------------------------------
        for l in range(N):
            for ll in range(M):
                spin_index = np.int(M * l + ll)

                gPEPS1_rdm.append(BP.tensor_reduced_dm(spin_index, TT, LL, smat, imat))
        '''

        # ------------- generating the double-edge factor graph (defg) of the tensor network ---------------------------

        graph = defg.Graph()
        graph = BP.PEPStoDEnFG_transform(graph, TT, LL, smat)
        graph.sum_product(t_max, epsilon, dumping)


        # --------------------------------- iterating the gPEPS and BP algorithms -------------------------------------
        for dt in t_list:
            flag = 0
            for j in range(iterations):
                counter += 2
                print('BP_N, D max, dt, j = ', N, D_max, dt, j)
                TT1, LL1 = BP.PEPS_BPupdate(TT, LL, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max, graph, t_max, epsilon, dumping)
                graph.sum_product(t_max, epsilon, dumping, 'init_with_old_messages')
                energy1 = BP.BP_energy_per_site_using_factor_belief(graph, smat, imat, Jk, h[0], Opi, Opj, Op_field)
                TT2, LL2 = BP.PEPS_BPupdate(TT1, LL1, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max, graph, t_max, epsilon, dumping)
                graph.sum_product(t_max, epsilon, dumping, 'init_with_old_messages')
                energy2 = BP.BP_energy_per_site_using_factor_belief(graph, smat, imat, Jk, h[0], Opi, Opj, Op_field)
                #energy1 = BP.energy_per_site(TT1, LL1, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
                #energy2 = BP.energy_per_site(TT2, LL2, imat, smat, Jk, h[ss], Opi, Opj, Op_field)

                print(energy1, energy2)

                if np.abs(energy1 - energy2) < dE or energy1 < energy2:
                    flag = 1
                    TT = TT2
                    LL = LL2
                    break
                else:
                    TT = TT2
                    LL = LL2


        # ---------------------------------- calculating reduced density matrices using DEFG ----------------------------

        graph.sum_product(t_max, epsilon, dumping, 'init_with_old_messages')
        graph.calc_rdm_belief()
        #graph.calc_factor_belief()
        #graph.rdm_comparison()
        #rdm = graph.rdm_using_factors()


        # --------------------------------- calculating magnetization matrices -------------------------------
        for l in range(N):
            for ll in range(M):
                spin_index = np.int(M * l + ll)
                mz_mat_BP[ss, l, ll] = BP.single_tensor_expectation(spin_index, TT, LL, imat, smat, sz)
                mx_mat_BP[ss, l, ll] = BP.single_tensor_expectation(spin_index, TT, LL, imat, smat, sx)
                my_mat_BP[ss, l, ll] = BP.single_tensor_expectation(spin_index, TT, LL, imat, smat, sy)

                gPEPS_rdm.append(BP.tensor_reduced_dm(spin_index, TT, LL, smat, imat))

                rdm_t_list, rdm_i_list = nlg.ncon_list_generator_reduced_dm(TT, LL, smat, spin_index)
                T_list_n, idx_list_n = nlg.ncon_list_generator(TT, LL, smat, np.eye(p), spin_index)
                exact_rdm.append(ncon.ncon(rdm_t_list, rdm_i_list) / ncon.ncon(T_list_n, idx_list_n))

                mz_mat_exact[ss, l, ll] = np.trace(np.matmul(exact_rdm[spin_index], sz))
                mx_mat_exact[ss, l, ll] = np.trace(np.matmul(exact_rdm[spin_index], sx))
                my_mat_exact[ss, l, ll] = np.trace(np.matmul(exact_rdm[spin_index], sy))

        ## trace distance comparison
        #error_gPEPS1 = 0
        #error_BP = 0
        #error_gPEPS = 0
        #error = []
        #for i in range(n):
        #    error_BP += BP.trace_distance(rdm[i], exact_rdm[i])
        #    error_gPEPS += BP.trace_distance(gPEPS_rdm[i], exact_rdm[i])
        #    error_gPEPS1 += BP.trace_distance(gPEPS1_rdm[i], exact_rdm[i])
        #    error.append(BP.trace_distance(rdm[i], exact_rdm[i]))
        #print('rdm BP error = ', error_BP)
        #print('rdm gPEPS error = ', error_gPEPS)
        #print('rdm gPEPS1 error = ', error_gPEPS1)

        # ------------------ calculating total magnetization, energy and time to converge -------------------
        mz_BP.append(np.sum(mz_mat_BP[ss, :, :]) / n)
        mx_BP.append(np.sum(mx_mat_BP[ss, :, :]) / n)
        my_BP.append(np.sum(my_mat_BP[ss, :, :]) / n)
        time_to_converge_BP.append(counter)
        E_BP_rdm_belief.append(BP.BP_energy_per_site_using_rdm_belief(graph, smat, imat, Jk, h[0], Opi, Opj, Op_field))
        E_BP_factor_belief.append(BP.BP_energy_per_site_using_factor_belief(graph, smat, imat, Jk, h[0], Opi, Opj, Op_field))
        E_BP.append(BP.energy_per_site(TT, LL, imat, smat, Jk, h[ss], Opi, Opj, Op_field))
        E_BP_exact.append(BP.exact_energy_per_site(TT, LL, smat, Jk, h[ss], Opi, Opj, Op_field))
        print('E, E_exact E_BP_rdm_belief, E_BP_factor_belief = ', E_BP[ss], E_BP_exact[ss], E_BP_rdm_belief[ss], E_BP_factor_belief[ss])
    e2 = time.time()
    run_time_of_BPupdate = e2 - s2

    return [E_BP[0], E_BP_exact[0], E_BP_rdm_belief[0], E_BP_factor_belief[0], time_to_converge_BP[0],
            mz_mat_BP[0, :, :], mx_mat_BP[0, :, :], my_mat_BP[0, :, :], run_time_of_BPupdate, TT, LL,
            gPEPS_rdm, graph.rdm_belief, exact_rdm, mz_mat_exact[0, :, :], mx_mat_exact[0, :, :], my_mat_exact[0, :, :]]


def Heisenberg_PEPS_gPEPS(N, M, Jk, dE, D_max, bc):
    np.random.seed(seed=13)
    s2 = time.time()
    # ---------------------- Tensor Network paramas ------------------


    d = 2  # virtual bond dimension
    p = 2  # physical bond dimension
    h = [0]  # Hamiltonian: magnetic field coeffs
    mu = -1
    sigma = 0
    #Jk = np.random.normal(mu, sigma, (2 * N))  # interaction constant list
    print('Jk = ', Jk)

    time_to_converge = []
    E = []
    E_exact = []
    mx = []
    my = []
    mz = []
    mx_mat = np.zeros((len(h), N, M), dtype=complex)
    my_mat = np.zeros((len(h), N, M), dtype=complex)
    mz_mat = np.zeros((len(h), N, M), dtype=complex)

    pauli_z = np.array([[1, 0], [0, -1]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_x = np.array([[0, 1], [1, 0]])
    sz = 0.5 * pauli_z
    sy = 0.5 * pauli_y
    sx = 0.5 * pauli_x
    t_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]  # imaginary time evolution time steps list
    iterations = 100
    Opi = [sx, sy, sz]
    Opj = [sx, sy, sz]
    Op_field = np.eye(p)

    # ------------- generating the finite PEPS structure matrix------------------

    if bc == 'open':
        smat, imat = tnf.PEPS_OBC_smat_imat(N, M)
    if bc == 'periodic':
        smat, imat = tnf.PEPS_smat_imat_gen(N)

    n, m = smat.shape

    # ------------- generating tensors and bond vectors ---------------------------

    if bc == 'open':
        TT, LL = tnf.PEPS_OBC_random_tn_gen(smat, p, d)
    if bc == 'periodic':
        TT, LL = tnf.random_tn_gen(smat, p, d)


    for ss in range(len(h)):
        counter = 0
        # --------------------------------- iterating the gPEPS and BP algorithms -------------------------------------
        for dt in t_list:
            flag = 0
            for j in range(iterations):
                counter += 2
                print('N, D max, dt, j = ', N * M, D_max, dt, j)
                TT1, LL1 = gPEPS.PEPS_BPupdate(TT, LL, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max)
                TT2, LL2 = gPEPS.PEPS_BPupdate(TT1, LL1, dt, Jk, h[ss], Opi, Opj, Op_field, imat, smat, D_max)
                energy1 = gPEPS.energy_per_site(TT1, LL1, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
                energy2 = gPEPS.energy_per_site(TT2, LL2, imat, smat, Jk, h[ss], Opi, Opj, Op_field)
                #energy1 = gPEPS.exact_energy_per_site(TT1, LL1, smat, Jk, h[ss], Opi, Opj, Op_field)
                #energy2 = gPEPS.exact_energy_per_site(TT2, LL2, smat, Jk, h[ss], Opi, Opj, Op_field)

                print(energy1, energy2)

                if np.abs(energy1 - energy2) < dE:
                    flag = 1
                    TT = TT2
                    LL = LL2
                    break
                else:
                    TT = TT2
                    LL = LL2


        # --------------------------------- calculating magnetization matrices -------------------------------
        for l in range(N):
            for ll in range(M):
                spin_index = np.int(M * l + ll)
                mz_mat[ss, l, ll] = BP.single_tensor_expectation(spin_index, TT, LL, imat, smat, sz)
                mx_mat[ss, l, ll] = BP.single_tensor_expectation(spin_index, TT, LL, imat, smat, sx)
                my_mat[ss, l, ll] = BP.single_tensor_expectation(spin_index, TT, LL, imat, smat, sy)

        # ------------------ calculating total magnetization, energy and time to converge -------------------
        mz.append(np.sum(mz_mat[ss, :, :]) / n)
        mx.append(np.sum(mx_mat[ss, :, :]) / n)
        my.append(np.sum(my_mat[ss, :, :]) / n)
        time_to_converge.append(counter)
        E.append(gPEPS.energy_per_site(TT, LL, imat, smat, Jk, h[ss], Opi, Opj, Op_field))
        #E_exact.append(gPEPS.exact_energy_per_site(TT, LL, smat, Jk, h[ss], Opi, Opj, Op_field))
        #print('E, E exact', E[ss], E_exact[ss])

    e2 = time.time()
    run_time_of_gPEPS = e2 - s2
    return [E[0], E_exact, time_to_converge[0], mz_mat[0, :, :], mx_mat[0, :, :], my_mat[0, :, :], run_time_of_gPEPS, TT, LL]



