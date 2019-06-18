from scipy import linalg
import numpy as np
import Tensor_Network_contraction as tnc
import copy as cp


"""
    A module for the function simple_update which preform the simple update algorithm over a given Tensor Network 
    as specified in the papper https://arxiv.org/abs/1808.00680 by Roman Orus.
"""


def simple_update(TT, LL, Uij, imat, smat, D_max):

    """
    :param TT: list of tensors in the tensor network TT = [T1, T2, T3, ..., Tp]
    :param LL: list of lists of the lambdas LL = [L1, L2, ..., Ls]
    :param hij: the interaction matrix which assumed to be the same for all interactions
    :param imat: The index matrix which indicates which tensor connect to which edge (as indicated in the papper)
    :param smat: The structure matrix which indicates which leg of every tensor is connected to which edge
    :param D_max: maximal virtual dimension
    :return: updated tensors list TTu and updated lambda tuple of lists LLu
    """
    #[0, 6, 2, 7, 4, 5, 1, 3]
    n, m = np.shape(imat)
    TT_new = cp.deepcopy(TT)
    LL_new = cp.deepcopy(LL)
    for Ek in range(m):

        ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
        tidx = np.nonzero(imat[:, Ek])[0]
        tdim = smat[tidx, Ek]
        Ti = [TT[tidx[0]], tdim[0]]
        Tj = [TT[tidx[1]], tdim[1]]
        lamda_k = LL[Ek]

        # collecting all neighboring (edges, legs)
        Ti_alldim = [list(np.nonzero(imat[tidx[0], :])[0]), list(smat[tidx[0], np.nonzero(imat[tidx[0], :])[0]])]
        Tj_alldim = [list(np.nonzero(imat[tidx[1], :])[0]), list(smat[tidx[1], np.nonzero(imat[tidx[1], :])[0]])]

        # removing the Ek edge and leg
        Ti_alldim[0].remove(Ek)
        Ti_alldim[1].remove(smat[tidx[0], Ek])
        Tj_alldim[0].remove(Ek)
        Tj_alldim[1].remove(smat[tidx[1], Ek])

        ## (b) Absorb bond matrices (lambdas) to all Em != Ek of Ti, Tj tensors
        for k in range(len(Ti_alldim[0])):
            Em = Ti_alldim[0][k]
            leg = Ti_alldim[1][k]
            Ti_index = range(len(Ti[0].shape))
            Ti_final_index = range(len(Ti[0].shape))
            Ti_final_index[leg] = len(Ti[0].shape)
            Ti[0] = np.einsum(Ti[0], Ti_index, np.diag(LL[Em]), [leg, Ti_final_index[leg]], Ti_final_index)

        for k in range(len(Tj_alldim[0])):
            Em = Tj_alldim[0][k]
            leg = Tj_alldim[1][k]
            Tj_index = range(len(Tj[0].shape))
            Tj_final_index = range(len(Tj[0].shape))
            Tj_final_index[leg] = len(Tj[0].shape)
            Tj[0] = np.einsum(Tj[0], Tj_index, np.diag(LL[Em]), [leg, Tj_final_index[leg]], Tj_final_index)

        # Gauge fix
        #Ti[0], Tj[0], lamda_k = gauge_fix1(TT[tidx[0]], TT[tidx[1]], Ti[0], Tj[0], tdim[0], tdim[1], lamda_k)
        #print('lamda diff = ', np.sum(np.abs(LL[Ek] - lamda_k)))
        #print('\n')


        ## (c) Group all virtual legs m!=Ek to form Pl, Pr MPS tensors
        i_perm = np.array(range(len(Ti[0].shape)))
        j_perm = np.array(range(len(Tj[0].shape)))

        # swapping the k leg with the element in the 1 place
        i_perm[[1, Ti[1]]] = i_perm[[Ti[1], 1]]
        j_perm[[1, Tj[1]]] = j_perm[[Tj[1], 1]]

        # reshaping Ti, Tj
        i_prod = np.delete(np.array(Ti[0].shape), [0, Ti[1]])
        j_prod = np.delete(np.array(Tj[0].shape), [0, Tj[1]])
        i_shape = [Ti[0].shape[0], Ti[0].shape[Ti[1]], np.prod(i_prod)]
        j_shape = [Tj[0].shape[0], Tj[0].shape[Tj[1]], np.prod(j_prod)]
        Pl = np.reshape(np.transpose(Ti[0], i_perm), i_shape)
        Pr = np.reshape(np.transpose(Tj[0], j_perm), j_shape)
        Pl = np.transpose(Pl, [1, 0, 2])
        Pr = np.transpose(Pr, [1, 0, 2])


        ## (d) QR/LQ decompose Pl, Pr to obtain Q1, R and L, Q2 sub-tensors, respectively
        Pl = np.transpose(np.reshape(Pl, (i_shape[0] * i_shape[1], i_shape[2])))
        Pr = np.transpose(np.reshape(Pr, (j_shape[0] * j_shape[1], j_shape[2])))
        Q1, R = np.linalg.qr(Pl, 'complete')
        Q2, L = np.linalg.qr(Pr, 'complete')
        R = np.reshape(R, (R.shape[0], i_shape[0], R.shape[1] / i_shape[0]))
        L = np.reshape(L, (L.shape[0], j_shape[0], L.shape[1] / j_shape[0]))

        ## (e) Contract the ITE gate Ui,Ek , with R, L and lambda_k to form theta tensor.
        theta = imaginary_time_evolution(R, L, lamda_k, Uij)

        # reshaping theta into a matrix
        new_shape = [theta.shape[0] * theta.shape[1], theta.shape[2] * theta.shape[3]]
        theta = np.reshape(theta, new_shape)

        ## (f) Obtain R', L', lambda'_k tensors by applying an SVD to theta and truncating
        ##     the tensors by keeping the D largest singular values.
        Rtild, lamda_ktild, Ltild = np.linalg.svd(theta)
        Ltild = np.transpose(Ltild)
        #lamda_ktild /= np.sum(lamda_ktild)

        # reshaping R' and L' to rank(3) tensors
        Rtild = np.reshape(Rtild, (Rtild.shape[0] / i_shape[0], i_shape[0], Rtild.shape[1]))
        Ltild = np.reshape(Ltild, (Ltild.shape[0] / j_shape[0], j_shape[0], Ltild.shape[1]))

        # trancating the SVD results up to D_max eigenvalues
        Rtild = Rtild[:, :, 0:D_max] if D_max < len(lamda_ktild) else Rtild
        Ltild = Ltild[:, :, 0:D_max] if D_max < len(lamda_ktild) else Ltild
        lamda_ktild = lamda_ktild[0:D_max] if D_max < len(lamda_ktild) else lamda_ktild


        ## (g) Glue back the R', L', sub-tensors to Q1, Q2, respectively, to form updated tensors P'l, P'r.
        Pltild = np.einsum('ij,jkl->kli', Q1, Rtild)
        Prtild = np.einsum('ij,jkl->kli', Q2, Ltild)

        ## (h) Reshape back the P`l, P`r to the original rank-(z + 1) tensors T'_i, T'_j
        Pltild = np.reshape(Pltild, Ti[0].shape)
        Prtild = np.reshape(Prtild, Tj[0].shape)
        Titild = np.transpose(Pltild, i_perm)
        Tjtild = np.transpose(Prtild, j_perm)

        ## (i) Remove bond matrices lambda_m from virtual legs m != Ek to obtain the updated tensors Ti~, Tj~.
        for k in range(len(Ti_alldim[0])):
            Em = Ti_alldim[0][k]
            leg = Ti_alldim[1][k]
            Ti_index = range(len(Ti[0].shape))
            Ti_final_index = range(len(Ti[0].shape))
            Ti_final_index[leg] = len(Ti[0].shape)
            Titild = np.einsum(Titild, Ti_index, np.diag(LL[Em] ** (-1)), [leg, Ti_final_index[leg]], Ti_final_index)

        for k in range(len(Tj_alldim[0])):
            Em = Tj_alldim[0][k]
            leg = Tj_alldim[1][k]
            Tj_index = range(len(Tj[0].shape))
            Tj_final_index = range(len(Tj[0].shape))
            Tj_final_index[leg] = len(Tj[0].shape)
            Tjtild = np.einsum(Tjtild, Tj_index, np.diag(LL[Em] ** (-1)), [leg, Tj_final_index[leg]], Tj_final_index)

        # Normalize and save new Ti Tj and lambda_k
        TT_new[tidx[0]] = cp.copy(Titild / np.sum(Titild))
        TT_new[tidx[1]] = cp.copy(Tjtild / np.sum(Tjtild))
        LL_new[Ek] = cp.copy(lamda_ktild / np.sum(lamda_ktild))
    return TT_new, LL_new



def imaginary_time_evolution(R, L, eigen_k, unitary):
    eig_k_mat = np.diag(eigen_k)
    tensor_list = [R, L, eig_k_mat, unitary]
    indices = ([-1, 1, 2], [-4, 3, 4], [2, 4], [1, -2, 3, -3])
    order = [1, 2, 3, 4]
    forder = [-1, -2, -3, -4]
    theta = tnc.scon(tensor_list, indices, order, forder)
    return theta


def gauge_fix(Ti, Tj, Ti_absorbed_lamdas, Tj_absorbed_lamdas, i_leg, j_leg, lamda_k):
    # i environment
    i_idx = range(len(Ti_absorbed_lamdas.shape))
    i_conj_idx = range(len(Ti_absorbed_lamdas.shape))
    i_conj_idx[i_leg] = len(Ti_absorbed_lamdas.shape)
    Mi = np.einsum(Ti_absorbed_lamdas, i_idx, np.conj(Ti_absorbed_lamdas), i_conj_idx, [i_idx[i_leg], i_conj_idx[i_leg]])

    # j environment
    j_idx = range(len(Tj_absorbed_lamdas.shape))
    j_conj_idx = range(len(Tj_absorbed_lamdas.shape))
    j_conj_idx[j_leg] = len(Tj_absorbed_lamdas.shape)
    Mj = np.einsum(Tj_absorbed_lamdas, j_idx, np.conj(Tj_absorbed_lamdas), j_conj_idx, [j_idx[j_leg], j_conj_idx[j_leg]])

    # Environment diagonalization
    di, ui = np.linalg.eig(Mi)
    dj, uj = np.linalg.eig(Mj)

    # contruction
    lamda_k_prime = np.matmul(np.matmul(np.conj(np.transpose(ui)), np.diag(lamda_k)), uj)
    lamda_k_prime = np.matmul(np.diag(np.sqrt(di)), np.matmul(lamda_k_prime, np.diag(np.sqrt(dj))))

    # SVD
    wi, lamda_k_tild, wj = np.linalg.svd(lamda_k_prime)
    lamda_k_tild /= np.sum(lamda_k_tild)

    # x and y construction
    x = np.matmul(np.matmul(np.conj(np.transpose(wi)), np.diag(np.sqrt(di))), np.conj(np.transpose(ui)))
    y = np.matmul(np.matmul(uj, np.diag(np.sqrt(dj))), wj)

    # fixing Ti and Tj
    Ti_idx_old = range(len(Ti.shape))
    Ti_idx_new = range(len(Ti.shape))
    Ti_idx_new[i_leg] = len(Ti_idx_old)
    Tj_idx_old = range(len(Tj.shape))
    Tj_idx_new = range(len(Tj.shape))
    Tj_idx_new[j_leg] = len(Tj_idx_old)
    Ti = np.einsum(Ti, Ti_idx_old, np.linalg.pinv(x), [Ti_idx_old[i_leg], len(Ti_idx_old)], Ti_idx_new)
    Tj = np.einsum(Tj, Tj_idx_old, np.linalg.pinv(y), [Tj_idx_old[j_leg], len(Tj_idx_old)], Tj_idx_new)
    return Ti, Tj, lamda_k_tild

def energy_per_site(TT, LL, imat, smat, Oij):
    energy_per_site = 0
    n, m = np.shape(imat)
    for Ek in range(m):

        ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
        tidx = np.nonzero(imat[:, Ek])[0]
        tdim = smat[tidx, Ek]
        Ti = [TT[tidx[0]], tdim[0]]
        Tj = [TT[tidx[1]], tdim[1]]
        Ti_conj = [np.conj(TT[tidx[0]]), tdim[0]]
        Tj_conj = [np.conj(TT[tidx[1]]), tdim[1]]
        lamda_k = LL[Ek]

        # collecting all neighboring (edges, legs)
        Ti_alldim = [list(np.nonzero(imat[tidx[0], :])[0]), list(smat[tidx[0], np.nonzero(imat[tidx[0], :])[0]])]
        Tj_alldim = [list(np.nonzero(imat[tidx[1], :])[0]), list(smat[tidx[1], np.nonzero(imat[tidx[1], :])[0]])]

        # removing the Ek edge and leg
        Ti_alldim[0].remove(Ek)
        Ti_alldim[1].remove(smat[tidx[0], Ek])
        Tj_alldim[0].remove(Ek)
        Tj_alldim[1].remove(smat[tidx[1], Ek])

        ## (b) Absorb bond matrices (lambdas) to all Em != Ek of Ti, Tj tensors
        for k in range(len(Ti_alldim[0])):
            Em = Ti_alldim[0][k]
            leg = Ti_alldim[1][k]
            Ti_index = range(len(Ti[0].shape))
            Ti_final_index = range(len(Ti[0].shape))
            Ti_final_index[leg] = len(Ti[0].shape)
            Ti[0] = np.einsum(Ti[0], Ti_index, np.diag(LL[Em]), [leg, Ti_final_index[leg]], Ti_final_index)
            Ti_conj[0] = np.einsum(Ti_conj[0], Ti_index, np.diag(LL[Em]), [leg, Ti_final_index[leg]], Ti_final_index)


        for k in range(len(Tj_alldim[0])):
            Em = Tj_alldim[0][k]
            leg = Tj_alldim[1][k]
            Tj_index = range(len(Tj[0].shape))
            Tj_final_index = range(len(Tj[0].shape))
            Tj_final_index[leg] = len(Tj[0].shape)
            Tj[0] = np.einsum(Tj[0], Tj_index, np.diag(LL[Em]), [leg, Tj_final_index[leg]], Tj_final_index)
            Tj_conj[0] = np.einsum(Tj_conj[0], Tj_index, np.diag(LL[Em]), [leg, Tj_final_index[leg]], Tj_final_index)


        ## prepering list of tensors and indices for scon function
        s = 1000
        t = 2000
        lamda_k_idx = [t, t + 1]
        lamda_k_conj_idx = [t + 2, t + 3]
        Oij_idx = range(s, s + 4)

        Ti_idx = range(len(Ti[0].shape))
        Ti_conj_idx = range(len(Ti_conj[0].shape))
        Ti_idx[0] = Oij_idx[0]
        Ti_conj_idx[0] = Oij_idx[1]
        Ti_idx[tdim[0]] = lamda_k_idx[0]
        Ti_conj_idx[tdim[0]] = lamda_k_conj_idx[0]

        Tj_idx = range(len(Ti[0].shape) + 1, len(Ti[0].shape) + 1 + len(Tj[0].shape))
        Tj_conj_idx = range(len(Ti_conj[0].shape) + 1, len(Ti_conj[0].shape) + 1 + len(Tj_conj[0].shape))
        Tj_idx[0] = Oij_idx[3]
        Tj_conj_idx[0] = Oij_idx[2]
        Tj_idx[tdim[1]] = lamda_k_idx[1]
        Tj_conj_idx[tdim[1]] = lamda_k_conj_idx[1]

        # two site energy calculation
        tensors = [Ti[0], np.conj(Ti[0]), Tj[0], np.conj(Tj[0]), Oij, np.diag(lamda_k), np.diag(lamda_k)]
        indices = [Ti_idx, Ti_conj_idx, Tj_idx, Tj_conj_idx, Oij_idx, lamda_k_idx, lamda_k_conj_idx]
        two_site_energy = tnc.scon(tensors, indices)

        ## prepering list of tensors and indices for two site normalization
        l = 3000
        Ti_idx[0] = l
        Ti_conj_idx[0] = l
        Tj_idx[0] = l + 1
        Tj_conj_idx[0] = l + 1

        tensors = [Ti[0], np.conj(Ti[0]), Tj[0], np.conj(Tj[0]), np.diag(lamda_k), np.diag(lamda_k)]
        indices = [Ti_idx, Ti_conj_idx, Tj_idx, Tj_conj_idx, lamda_k_idx, lamda_k_conj_idx]
        two_site_norm = tnc.scon(tensors, indices)
        #print('\n')
        #print('two site energy = ', two_site_energy)
        #print('two site norm = ', two_site_norm)

        two_site_energy /= two_site_norm
        print('two site normalized energy = ', two_site_energy)

        energy_per_site += two_site_energy
    energy_per_site /= n
    return energy_per_site


def gauge_fix1(TT, LL, imat, smat):
    n, m = np.shape(imat)
    TT_new = cp.deepcopy(TT)
    LL_new = cp.deepcopy(LL)
    for Ek in range(m):

        ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
        tidx = np.nonzero(imat[:, Ek])[0]
        tdim = smat[tidx, Ek]
        Ti = [TT[tidx[0]], tdim[0]]
        Tj = [TT[tidx[1]], tdim[1]]
        lamda_k = LL[Ek]

        # collecting all neighboring (edges, legs)
        Ti_alldim = [list(np.nonzero(imat[tidx[0], :])[0]), list(smat[tidx[0], np.nonzero(imat[tidx[0], :])[0]])]
        Tj_alldim = [list(np.nonzero(imat[tidx[1], :])[0]), list(smat[tidx[1], np.nonzero(imat[tidx[1], :])[0]])]

        # removing the Ek edge and leg
        Ti_alldim[0].remove(Ek)
        Ti_alldim[1].remove(smat[tidx[0], Ek])
        Tj_alldim[0].remove(Ek)
        Tj_alldim[1].remove(smat[tidx[1], Ek])

        ## (b) Absorb bond matrices (lambdas) to all Em != Ek of Ti, Tj tensors
        for k in range(len(Ti_alldim[0])):
            Em = Ti_alldim[0][k]
            leg = Ti_alldim[1][k]
            Ti_index = range(len(Ti[0].shape))
            Ti_final_index = range(len(Ti[0].shape))
            Ti_final_index[leg] = len(Ti[0].shape)
            Ti[0] = np.einsum(Ti[0], Ti_index, np.diag(LL[Em]), [leg, Ti_final_index[leg]], Ti_final_index)

        for k in range(len(Tj_alldim[0])):
            Em = Tj_alldim[0][k]
            leg = Tj_alldim[1][k]
            Tj_index = range(len(Tj[0].shape))
            Tj_final_index = range(len(Tj[0].shape))
            Tj_final_index[leg] = len(Tj[0].shape)
            Tj[0] = np.einsum(Tj[0], Tj_index, np.diag(LL[Em]), [leg, Tj_final_index[leg]], Tj_final_index)

        ## Gauge fixing
        # i environment
        i_leg = tdim[0]
        j_leg = tdim[1]
        i_idx = range(len(Ti[0].shape))
        i_conj_idx = range(len(Ti[0].shape))
        i_conj_idx[i_leg] = len(Ti[0].shape)
        Mi = np.einsum(Ti[0], i_idx, np.conj(Ti[0]), i_conj_idx,
                       [i_idx[i_leg], i_conj_idx[i_leg]])

        # j environment
        j_idx = range(len(Tj[0].shape))
        j_conj_idx = range(len(Tj[0].shape))
        j_conj_idx[j_leg] = len(Tj[0].shape)
        Mj = np.einsum(Tj[0], j_idx, np.conj(Tj[0]), j_conj_idx,
                       [j_idx[j_leg], j_conj_idx[j_leg]])

        # Environment diagonalization
        di, ui = np.linalg.eig(Mi)
        dj, uj = np.linalg.eig(Mj)

        # contruction
        lamda_k_prime = np.matmul(np.matmul(np.conj(np.transpose(ui)), np.diag(lamda_k)), uj)
        lamda_k_prime = np.matmul(np.diag(np.sqrt(di)), np.matmul(lamda_k_prime, np.diag(np.sqrt(dj))))

        # SVD
        wi, lamda_k_tild, wj = np.linalg.svd(lamda_k_prime)
        #lamda_k_tild /= np.sum(lamda_k_tild)

        # x and y construction
        x = np.matmul(np.matmul(np.conj(np.transpose(wi)), np.diag(np.sqrt(di))), np.conj(np.transpose(ui)))
        y = np.matmul(np.matmul(uj, np.diag(np.sqrt(dj))), np.transpose(wj))

        # fixing Ti and Tj
        Ti_idx_old = range(len(TT[tidx[0]].shape))
        Ti_idx_new = range(len(TT[tidx[0]].shape))
        Ti_idx_new[i_leg] = len(Ti_idx_old)
        Tj_idx_old = range(len(TT[tidx[1]].shape))
        Tj_idx_new = range(len(TT[tidx[1]].shape))
        Tj_idx_new[j_leg] = len(Tj_idx_old)
        Ti_fixed = np.einsum(TT[tidx[0]], Ti_idx_old, np.linalg.pinv(x), [Ti_idx_old[i_leg], len(Ti_idx_old)], Ti_idx_new)
        Tj_fixed = np.einsum(TT[tidx[1]], Tj_idx_old, np.linalg.pinv(y), [Tj_idx_old[j_leg], len(Tj_idx_old)], Tj_idx_new)

        TT_new[tidx[0]] = cp.copy(Ti_fixed)
        TT_new[tidx[1]] = cp.copy(Tj_fixed)
        LL_new[Ek] = cp.copy(lamda_k_tild / np.sum(lamda_k_tild))

    return TT_new, LL_new


