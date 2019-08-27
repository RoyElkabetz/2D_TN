import numpy as np
import copy as cp


def ncon_list_generator(TT, LL, smat, O, spin):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)
    Oidx = [spins_idx[spin], spins_idx[-1] + 1]

    for i in range(n):

        ## pick a tensor T, T*
        T = cp.copy(TT[i])
        Tstar = np.conj(cp.copy(TT[i]))
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        Tidx = np.zeros((len(T.shape)), dtype=int)
        Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)

        ## creat T, T* indices
        if i == spin:
            Tidx[0] = Oidx[0]
            Tstaridx[0] = Oidx[1]
        else:
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = spins_idx[i]

        ## absorb its environment
        for j in range(len(edges)):
            T = np.einsum(T, range(len(T.shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(T.shape)))
            Tstar = np.einsum(Tstar, range(len(Tstar.shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(Tstar.shape)))
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m

        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))

        if i == spin:
            T_list.append(cp.copy(O))
            idx_list.append(cp.copy(Oidx))

        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))

    return T_list, idx_list


def ncon_list_generator_reduced_dm(TT, LL, smat, spin):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)

    #Oidx = [spins_idx[spin], spins_idx[-1] + 1]

    for i in range(n):

        ## pick a tensor T, T*
        T = cp.copy(TT[i])
        Tstar = np.conj(cp.copy(TT[i]))
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        Tidx = np.zeros((len(T.shape)), dtype=int)
        Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)


        ## creat T, T* indices
        if i == spin:
            Tidx[0] = -1
            Tstaridx[0] = -2
        else:
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = spins_idx[i]

        ## absorb its environment
        for j in range(len(edges)):
            T = np.einsum(T, range(len(T.shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(T.shape)))
            Tstar = np.einsum(Tstar, range(len(Tstar.shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(Tstar.shape)))
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m

        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))

    return T_list, idx_list


def ncon_list_generator_for_BPerror(TT1, LL1, TT2, LL2, smat):
    TT1 = cp.deepcopy(TT1)
    LL1 = cp.deepcopy(LL1)
    TT2 = cp.deepcopy(TT2)
    LL2 = cp.deepcopy(LL2)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)
    for i in range(n):

        ## pick a tensor T, T*
        T = cp.copy(TT1[i])
        Tstar = np.conj(cp.copy(TT2[i]))
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        Tidx = np.zeros((len(T.shape)), dtype=int)
        Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
        Tidx[0] = spins_idx[i]
        Tstaridx[0] = spins_idx[i]

        ## absorb its environment
        for j in range(len(edges)):
            T = np.einsum(T, range(len(T.shape)), np.sqrt(LL1[edges[j]]), [legs[j]], range(len(T.shape)))
            Tstar = np.einsum(Tstar, range(len(Tstar.shape)), np.sqrt(LL2[edges[j]]), [legs[j]], range(len(Tstar.shape)))
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m


        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))
    return T_list, idx_list


def ncon_list_generator_two_site_exact_expectation_peps(TT, TTstar, smat, edge, operator):
    TT = cp.deepcopy(TT)
    TTstar = cp.deepcopy(TTstar)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)

    ## fix operator legs
    tensors_indices = np.nonzero(smat[:, edge])[0]
    operator_idx = [spins_idx[tensors_indices[0]], spins_idx[tensors_indices[1]], 1000, 1001]  # [i, j, i', j']

    for i in range(n):
        if i == tensors_indices[0]:

            ## pick a tensor T, T*
            T = cp.copy(TT[i])
            Tstar = cp.copy(TTstar[i])
            edges = np.nonzero(smat[i, :])[0]
            legs = smat[i, edges]
            Tidx = np.zeros((len(T.shape)), dtype=int)
            Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = operator_idx[2]

        elif i == tensors_indices[1]:
            ## pick a tensor T, T*
            T = cp.copy(TT[i])
            Tstar = cp.copy(TTstar[i])
            edges = np.nonzero(smat[i, :])[0]
            legs = smat[i, edges]
            Tidx = np.zeros((len(T.shape)), dtype=int)
            Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = operator_idx[3]

        else:
            ## pick a tensor T, T*
            T = cp.copy(TT[i])
            Tstar = cp.copy(TTstar[i])
            edges = np.nonzero(smat[i, :])[0]
            legs = smat[i, edges]
            Tidx = np.zeros((len(T.shape)), dtype=int)
            Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = spins_idx[i]

        ## arange legs indices
        for j in range(len(edges)):
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m

        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))

    ## add operator to list
    T_list.append(operator)
    idx_list.append(operator_idx)

    return T_list, idx_list


def ncon_list_generator_braket_peps(TT1, TT2, smat):
    TT1 = cp.deepcopy(TT1)
    TT2 = cp.deepcopy(TT2)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)
    for i in range(n):

        ## pick a tensor T, T*
        T = cp.copy(TT1[i])
        Tstar = cp.copy(TT2[i])
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        Tidx = np.zeros((len(T.shape)), dtype=int)
        Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
        Tidx[0] = spins_idx[i]
        Tstaridx[0] = spins_idx[i]

        ## arange legs indices
        for j in range(len(edges)):
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m


        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))
    return T_list, idx_list


def ncon_list_generator_two_site_exact_expectation_mps(TT, TTstar, smat, edge, operator):
    TT = cp.deepcopy(TT)
    TTstar = cp.deepcopy(TTstar)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)

    ## fix operator legs
    tensors_indices = np.nonzero(smat[:, edge])[0]
    if edge == (m - 1):
        tensors_indices = np.flip(tensors_indices, axis=0)
    operator_idx = [spins_idx[tensors_indices[0]], spins_idx[tensors_indices[1]], 1000, 1001]  # [i, j, i', j']

    for i in range(n):
        if i == tensors_indices[0]:

            ## pick a tensor T, T*
            T = cp.copy(TT[i])
            Tstar = cp.copy(TTstar[i])
            edges = np.nonzero(smat[i, :])[0]
            legs = smat[i, edges]
            Tidx = np.zeros((len(T.shape)), dtype=int)
            Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = operator_idx[2]

        elif i == tensors_indices[1]:
            ## pick a tensor T, T*
            T = cp.copy(TT[i])
            Tstar = cp.copy(TTstar[i])
            edges = np.nonzero(smat[i, :])[0]
            legs = smat[i, edges]
            Tidx = np.zeros((len(T.shape)), dtype=int)
            Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = operator_idx[3]

        else:
            ## pick a tensor T, T*
            T = cp.copy(TT[i])
            Tstar = cp.copy(TTstar[i])
            edges = np.nonzero(smat[i, :])[0]
            legs = smat[i, edges]
            Tidx = np.zeros((len(T.shape)), dtype=int)
            Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
            Tidx[0] = spins_idx[i]
            Tstaridx[0] = spins_idx[i]

        ## arange legs indices
        for j in range(len(edges)):
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m

        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))

    ## add operator to list
    T_list.append(operator)
    idx_list.append(operator_idx)

    return T_list, idx_list

def ncon_list_generator_braket_mps(TT, TTstar, smat):
    TT = cp.deepcopy(TT)
    TTstar = cp.deepcopy(TTstar)

    T_list = []
    idx_list = []
    n, m = smat.shape
    spins_idx = range(2 * m + 1, 2 * m + 1 + n)
    for i in range(n):

        ## pick a tensor T, T*
        T = cp.copy(TT[i])
        Tstar = cp.copy(TTstar[i])
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        Tidx = np.zeros((len(T.shape)), dtype=int)
        Tstaridx = np.zeros((len(Tstar.shape)), dtype=int)
        Tidx[0] = spins_idx[i]
        Tstaridx[0] = spins_idx[i]

        ## arange legs indices
        for j in range(len(edges)):
            Tidx[legs[j]] = edges[j] + 1
            Tstaridx[legs[j]] = edges[j] + 1 + m


        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))
    return T_list, idx_list