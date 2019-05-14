import numpy as np
import copy as cp


"""
    A module for the function simple_update which preform the simple update algorithm over a given Tensor Network 
    as specified in the papper https://arxiv.org/abs/1808.00680 by Roman Orus.
"""


def simple_update(TT, EE, LL, Uij, imat, smat):

    """
    :param TT: list of tensors in the tensor network TT = [T1, T2, T3, ..., Tp]
    :param EE: list of edges in the Tensor Network EE = [E1, E2, E3, ..., Es]
    :param LL: tuple of lists of the lambdas LL = (L1, L2, ..., Ls)
    :param Uij: the interaction matrix which assumed to be the same for all interactions
    :param imat: The index matrix which indicates which tensor connect to which edge (as indicated in the papper)
    :param smat: The structure matrix which indicates which leg of every tensor is connected to which edge
    :return: updated tensors list TTu and updated lambda tuple of lists LLu
    """

    n, m = np.shape(imat)
    for j in range(m):
        # (a) Find tensors Ti, Tj and their corresponding dimensions connected along edge Ek.

        tidx = np.nonzero(imat[:, j])[0]
        tdim = np.nonzero(smat[:, j])[0]
        Ti = (TT[tidx[0]], tdim[0])
        Tj = (TT[tidx[1]], tdim[1])
        lamda = LL[j]
        Ti_legs = np.nonzero(imat[tidx[0], :])[0]
        Tj_legs = np.nonzero(imat[tidx[1], :])[0]
        Ti_lamdas = np.nonzero()






        # (b) Absorb bond matrices (lambdas) to all m != k of Ti, Tj tensors


        # (c) Group all virtual legs m!=k to form Pl, Pr MPS tensors


        # (d) QR/LQ decompose Pl, Pr to obtain Q1, R and L, Q2 sub-tensors, respectively


        # (e) Contract the ITE gate Ui,j , with R, L and lambda_k to form theta tensor.


        # (f) Obtain R', L', lambda'_k tensors by applying an SVD to theta and truncating
        #     the tensors by keeping the D largest singular values.


        # (g) Glue back the R', L', sub-tensors to Q1, Q2, respectively, to form updated tensors P'l, P'r.


        # (h) Reshape back the P`l, P`r to the original rank-(z + 1) tensors T'_i, T'_j


        # (i) Remove bond matrices lambda_m from virtual legs m != k to obtain the updated tensors Ti~, Tj~.
