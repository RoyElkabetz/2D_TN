import numpy as np
import copy as cp


"""
    A module for the function simple_update which preform the simple update algorithm over a given Tensor Network 
    as specified in the papper https://arxiv.org/abs/1808.00680 by Roman Orus.
"""


def simple_update(TT, LL, Uij, imat, smat):

    """
    :param TT: list of tensors in the tensor network TT = [T1, T2, T3, ..., Tp]
    :param LL: list of lists of the lambdas LL = [L1, L2, ..., Ls]
    :param Uij: the interaction matrix which assumed to be the same for all interactions
    :param imat: The index matrix which indicates which tensor connect to which edge (as indicated in the papper)
    :param smat: The structure matrix which indicates which leg of every tensor is connected to which edge
    :return: updated tensors list TTu and updated lambda tuple of lists LLu
    """

    n, m = np.shape(imat)
    for Ek in range(m):

        ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
        tidx = np.nonzero(imat[:, Ek])[0]
        tdim = smat[tidx, Ek]
        Ti = [TT[tidx[0]], tdim[0]]
        Tj = [TT[tidx[1]], tdim[1]]
        lamda = LL[Ek]

        # collecting all neighboring (edges, legs)
        Ti_alldim = [list(np.nonzero(imat[tidx[0], :])[0]), list(smat[tidx[0], np.nonzero(imat[tidx[0], :])[0]])]
        Tj_alldim = [list(np.nonzero(imat[tidx[1], :])[0]), list(smat[tidx[1], np.nonzero(imat[tidx[1], :])[0]])]

        # removing the Ek edge and leg
        Ti_alldim[0].remove(Ek)
        Ti_alldim[1].remove(smat[tidx[0], Ek])
        Tj_alldim[0].remove(Ek)
        Tj_alldim[1].remove(smat[tidx[1], Ek])

        # collecting neighboring lamdas
        Ti_lamdas = [LL[Em] for Em in Ti_alldim[0]]
        Tj_lamdas = [LL[Em] for Em in Tj_alldim[0]]

        ## (b) Absorb bond matrices (lambdas) to all Em != Ek of Ti, Tj tensors
        for Em in range(len(Ti_lamdas)):
            Ti[0] = np.einsum(Ti[0], range(len(Ti[0].shape)), Ti_lamdas[Em], [Ti_alldim[1][Em]], range(len(Ti[0].shape)))
            Tj[0] = np.einsum(Tj[0], range(len(Tj[0].shape)), Tj_lamdas[Em], [Tj_alldim[1][Em]], range(len(Tj[0].shape)))

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
        Pl, i_revperm, i_revshape = permshape(Ti[0], i_perm, i_shape)
        Pr, j_revperm, j_revshape = permshape(Tj[0], j_perm, j_shape)

        ## (d) QR/LQ decompose Pl, Pr to obtain Q1, R and L, Q2 sub-tensors, respectively
        Pl = np.transpose(np.reshape(Pl, (i_shape[0] * i_shape[1], i_shape[2])))
        Pr = np.transpose(np.reshape(Pr, (j_shape[0] * j_shape[1], j_shape[2])))
        Q1, R = np.linalg.qr(Pl, 'complete')
        Q2, L = np.linalg.qr(Pr, 'complete')
        R = np.reshape(R, (R.shape[0], i_shape[0], R.shape[1] / i_shape[0]))
        L = np.reshape(L, (L.shape[0], j_shape[0], L.shape[1] / j_shape[0]))

        ## (e) Contract the ITE gate Ui,Ek , with R, L and lambda_k to form theta tensor.


        ## (f) Obtain R', L', lambda'_k tensors by applying an SVD to theta and truncating
        ##     the tensors by keeping the D largest singular values.


        ## (g) Glue back the R', L', sub-tensors to Q1, Q2, respectively, to form updated tensors P'l, P'r.


        ## (h) Reshape back the P`l, P`r to the original rank-(z + 1) tensors T'_i, T'_j


        ## (i) Remove bond matrices lambda_m from virtual legs m != Ek to obtain the updated tensors Ti~, Tj~.


def permshape(T, perm, shp, order=None):

    # permuting and then reshaping a tensor

    if order == None:
        T = np.transpose(T, perm)
        old_shape = T.shape
        T = np.reshape(T, shp)
        reverse_perm = np.argsort(perm)
        return T, reverse_perm, old_shape
    if order == 'reverse':
        T = np.reshape(T, shp)
        old_shape = T.shape
        T = np.transpose(T, perm)
        reverse_perm = np.argsort(perm)
        return T, reverse_perm, old_shape

