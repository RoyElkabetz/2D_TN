import numpy as np
import copy as cp


def ncon_list_generator(TT, LL, smat, O, spin):

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

        ## creat T, T* indices
        if i == spin:
            Tidx = [Oidx[0]]
            Tstaridx = [Oidx[1]]
        else:
            Tidx = [spins_idx[i]]
            Tstaridx = [spins_idx[i]]

        ## absorb its environment
        for j in range(len(edges)):
            T = np.einsum(T, range(len(T.shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(T.shape)))
            Tstar = np.einsum(Tstar, range(len(Tstar.shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(Tstar.shape)))
            Tidx.append(edges[j] + 1)
            Tstaridx.append(edges[j] + 1 + m)

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

        ## creat T, T* indices
        if i == spin:
            Tidx = [-1]
            Tstaridx = [-2]
        else:
            Tidx = [spins_idx[i]]
            Tstaridx = [spins_idx[i]]

        ## absorb its environment
        for j in range(len(edges)):
            T = np.einsum(T, range(len(T.shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(T.shape)))
            Tstar = np.einsum(Tstar, range(len(Tstar.shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(Tstar.shape)))
            Tidx.append(edges[j] + 1)
            Tstaridx.append(edges[j] + 1 + m)

        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))

    return T_list, idx_list


def ncon_list_generator_for_BPerror(TT1, LL1, TT2, LL2, smat):
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
        Tidx = [spins_idx[i]]
        Tstaridx = [spins_idx[i]]

        ## absorb its environment
        for j in range(len(edges)):
            T = np.einsum(T, range(len(T.shape)), np.sqrt(LL1[edges[j]]), [legs[j]], range(len(T.shape)))
            Tstar = np.einsum(Tstar, range(len(Tstar.shape)), np.sqrt(LL2[edges[j]]), [legs[j]], range(len(Tstar.shape)))
            Tidx.append(edges[j] + 1)
            Tstaridx.append(edges[j] + 1 + m)

        ## add to lists
        T_list.append(cp.copy(T))
        idx_list.append(cp.copy(Tidx))
        T_list.append(cp.copy(Tstar))
        idx_list.append(cp.copy(Tstaridx))
    return T_list, idx_list
