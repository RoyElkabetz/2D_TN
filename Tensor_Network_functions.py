import numpy as np


def random_tn_gen(smat, p, d):
    # generating random complex tensors and random virtual bond dimension vectors for a tensor network
    # smat: structure matrix
    # p: physical bond dimension
    # d: initial virtual bond dimension
    n, m = smat.shape
    TT = []
    for ii in range(n):
        TT.append(np.random.rand(p, d, d, d, d))
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


def PEPS_OBC_smat_imat(n, m):

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


def PEPS_OBC_random_tn_gen(smat, p, d):
    # generating random complex tensors and random virtual bond dimension vectors for a tensor network
    # smat: structure matrix
    # p: physical bond dimension
    # d: initial virtual bond dimension
    n, m = smat.shape
    TT = []
    for ii in range(n):
        shape = [p]
        shape = shape + [d] * np.nonzero(smat[ii, :])[0].shape[0]
        TT.append(np.random.random(shape) + 1j * np.random.random(shape))
    LL = []
    for i in range(m):
        LL.append(np.ones(d, dtype=float) / d)
    return TT, LL


def PEPS_OBC_edge_rect_env(Ek, smat, h_w, env_size):
    # given an edge 'Ek' and a structure matrix 'smat' (tensors, edges) of an obc PEPS this function finds the
    # rectangular environment (single step)
    # h_w = [height, width] of PEPS lattice
    n, m = smat.shape
    omat = np.arange(n).reshape((h_w[0], h_w[1]))
    Ek_tens = np.nonzero(smat[:, Ek])[0]
    i, j = np.unravel_index(Ek_tens, omat.shape)
    mat_j = np.mod(omat, h_w[1])
    mat_i = (omat - mat_j) / h_w[1]
    Ti_idx = [i[0], j[0]]
    Tj_idx = [i[1], j[1]]
    bmatTi = (mat_i <= Ti_idx[0] + env_size) & (mat_i >= Ti_idx[0] - env_size) & (mat_j <= Ti_idx[1] + env_size) & (mat_j >= Ti_idx[1] - env_size)
    bmatTj = (mat_i <= Tj_idx[0] + env_size) & (mat_i >= Tj_idx[0] - env_size) & (mat_j <= Tj_idx[1] + env_size) & (mat_j >= Tj_idx[1] - env_size)
    condition = bmatTi | bmatTj
    emat = np.where(condition, 1, -1)
    return emat


def PEPS_OBC_divide_edge_regions(emat, smat):
    omat = np.arange(smat.shape[0]).reshape(emat.shape)
    tensors = omat[np.nonzero(emat > -1)]
    edges = np.nonzero(smat[tensors, :])
    unique, indices, counts = np.unique(edges[1], return_index=True, return_counts=True)
    inside = unique[np.nonzero(np.where(counts == 2, unique, -1) > -1)[0]]
    outside = unique[np.nonzero(np.where(counts == 1, unique, -1) > -1)[0]]
    return inside, outside


def PEPS_OBC_edge_environment_sub_order_matrix(emat):
    n, m = emat.shape
    omat = np.arange(n * m).reshape(emat.shape)
    tensors = omat[np.nonzero(emat > -1)]
    for i in range(n):
        if len(np.nonzero(emat[i, :] > -1)[0]):
            env_ud = len(np.nonzero(emat[i, :] > -1)[0])
            break
    for j in range(m):
        if len(np.nonzero(emat[:, j] > -1)[0]):
            env_lr = len(np.nonzero(emat[:, j] > -1)[0])
            break
    sub_omat = np.array(tensors).reshape(env_lr, env_ud)
    return sub_omat


def PEPS_OBC_broadcast_to_Itai(TT, PEPS_shape, p, d):

    new_TT = []
    new_order = [0, 1, 3, 2, 4]
    N, M = PEPS_shape
    for t, T in enumerate(TT):
        i, j = np.unravel_index(t, PEPS_shape)
        Dup = d
        Ddown = d
        Dleft = d
        Dright = d

        if i == 0:
            Dup = 1
        if i == N - 1:
            Ddown = 1
        if j == 0:
            Dleft = 1
        if j == M - 1:
            Dright = 1

        tensor = np.transpose(TT[t].reshape(p, Dleft, Dup, Dright, Ddown), new_order)
        norm = np.einsum(tensor, [0, 1, 2, 3, 4], np.conj(tensor), [0, 1, 2, 3, 4])
        new_TT.append(tensor / norm)
    return new_TT

