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


def PEPS_OBC_smat_imat(N):
    L = np.int(np.sqrt(N))
    n = L
    m = L

    # edge = (node_a i, node_a j, node_a l, node_b i, node_b j, node_b l)
    edge_list = []
    for i in range(n):
        for j in range(m):
            if i < n - 1:
                edge_list.append((i, j, 4, i + 1, j, 2))
            if j < m - 1:
                edge_list.append((i, j, 3, i, j + 1, 1))

    smat = np.zeros(shape=[n * m, len(edge_list)], dtype=np.int)
    imat = np.zeros(shape=[n * m, len(edge_list)], dtype=np.int)

    for edge_idx, edge in enumerate(edge_list):
        noda_a_idx = np.ravel_multi_index([edge[0], edge[1]], (n, m))
        noda_b_idx = np.ravel_multi_index([edge[3], edge[4]], (n, m))
        smat[noda_a_idx, edge_idx] = edge[2]
        smat[noda_b_idx, edge_idx] = edge[5]
        imat[noda_a_idx, edge_idx] = 1
        imat[noda_b_idx, edge_idx] = 1

    for i in range(smat.shape[0]):
        row = smat[i, np.nonzero(smat[i, :])[0]]
        new_row = np.array(range(1, len(row) + 1))
        order = np.argsort(row)
        new_row = new_row[order]
        smat[i, np.nonzero(smat[i, :])[0]] = new_row
    return smat, imat