import numpy as np


def random_tn_gen(smat, p, d):
    # generating random complex tensors and random virtual bond dimension vectors for a tensor network
    # smat: structure matrix
    # p: physical bond dimension
    # d: initial virtual bond dimension
    n, m = smat.shape
    TT = []
    for ii in range(n):
        TT.append(np.random.rand(p, d, d, d, d) + 1j * np.random.rand(p, d, d, d, d))
    LL = []
    for i in range(m):
        LL.append(np.ones(d, dtype=float) / d)
    return TT, LL


def PEPS_smat_imat_gen(N):
    # generating the smat (structure matrix) and imat (incidence matrix) of a 2D Square lattice tensor network
    # with periodic boundary conditions
    L = np.int(np.sqrt(N))
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
    return smat, imat

