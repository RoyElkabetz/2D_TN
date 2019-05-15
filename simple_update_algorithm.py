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
        ## (a) Find tensors Ti, Tj and their corresponding dimensions connected along edge Ek.

        tidx = np.nonzero(imat[:, Ek])[0]
        tdim = smat[tidx, Ek]
        Ti = [TT[tidx[0]], tdim[0]]
        Tj = [TT[tidx[1]], tdim[1]]
        lamda = LL[Ek]

        # collecting all neighboring (edges, legs_dim)
        Ti_alldim = [list(np.nonzero(imat[tidx[0], :])[0]), list(smat[tidx[0], np.nonzero(imat[tidx[0], :])[0]])]
        Tj_alldim = [list(np.nonzero(imat[tidx[1], :])[0]), list(smat[tidx[1], np.nonzero(imat[tidx[1], :])[0]])]

        # removing the Ek edge and leg_dim
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


        ## (d) QR/LQ decompose Pl, Pr to obtain Q1, R and L, Q2 sub-tensors, respectively


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

