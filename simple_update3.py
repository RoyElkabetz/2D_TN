from scipy import linalg
import numpy as np
import Tensor_Network_contraction as tnc
import ncon as ncon
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
    n, m = np.shape(imat)
    for Ek in range(m):
        lamda_k = cp.deepcopy(LL[Ek])

        ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
        Ti, Tj = get_tensors(Ek, TT, smat, imat)

        # collecting all neighboring (edges, dimensions) without the Ek (edge, dimension)
        i_dim, j_dim = get_edges(Ek, smat, imat)

        ## (b) Absorb bond vectors (lambdas) to all Em != Ek of Ti, Tj tensors
        Ti = absorb_edges(cp.deepcopy(Ti), i_dim, LL)
        Tj = absorb_edges(cp.deepcopy(Tj), j_dim, LL)

        # permuting the Ek leg of tensors i and j into the 1'st dimension
        Ti = dim_perm(cp.deepcopy(Ti))
        Tj = dim_perm(cp.deepcopy(Tj))

        ## (c) Group all virtual legs Em!=Ek to form Pl, Pr MPS tensors
        Pl = rankN_to_rank3(cp.deepcopy(Ti[0]))
        Pr = rankN_to_rank3(cp.deepcopy(Tj[0]))

        '''
        ##### experimenting with QR instead of SVD
        Q1, R = np.linalg.qr(np.transpose(np.reshape(Pl, [Pl.shape[0] * Pl.shape[1], np.prod(Pl.shape[2:])])))
        Q2, L = np.linalg.qr(np.transpose(np.reshape(Pr, [Pr.shape[0] * Pr.shape[1], np.prod(Pr.shape[2:])])))
        Q1 = np.transpose(Q1)
        Q2 = np.transpose(Q2)
        R = np.transpose(R)
        L = np.transpose(L)
        '''

        ## (d) SVD decomposing of Pl, Pr to obtain Q1, R and Q2, L sub-tensors, respectively
        R, sr, Q1 = svd(Pl, [0, 1], [2], keep_s='yes')
        L, sl, Q2 = svd(Pr, [0, 1], [2], keep_s='yes')

        R = R.dot(np.diag(sr))
        L = L.dot(np.diag(sl))

        #Q1 = np.diag(sr).dot(Q1)
        #Q2 = np.diag(sl).dot(Q2)

        #R = R.dot(np.diag(np.sqrt(sr)))  # (i * Ek, Q1)
        #L = L.dot(np.diag(np.sqrt(sl)))  # (j * Ek, Q2)
        #Q1 = np.diag(np.sqrt(sr)).dot(Q1)
        #Q2 = np.diag(np.sqrt(sl)).dot(Q2)


        # reshaping R and L into rank 3 tensors with shape (physical_dim, Ek_dim, Q(1/2).shape[0])
        i_physical_dim = Ti[0].shape[0]
        j_physical_dim = Tj[0].shape[0]
        R = rank2_to_rank3(R, i_physical_dim)  # (i, Ek, Q1)
        L = rank2_to_rank3(L, j_physical_dim)  # (j, Ek, Q2)

        ## (e) Contract the ITE gate Uij, with R, L, and lambda_k to form theta tensor.
        theta = imaginary_time_evolution(R, L, lamda_k, Uij) # (Q1, i', j', Q2)

        ## (f) Obtain R', L', lambda'_k tensors by applying an SVD to theta
        # and trancating R', lamda_k', L' and keeping only D_max eigenvalues and eigenvectors
        R_tild, lamda_k_tild, L_tild = svd(theta, [0, 1], [2, 3], keep_s='yes', max_eigen_num=D_max)
        # (Q1 * i', D_max) # (D_max, D_max) # (D_max, j' * Q2)

        # reshaping R_tild and L_tild back to rank 3 tensor
        R_tild = np.reshape(R_tild, (Q1.shape[0], i_physical_dim, R_tild.shape[1]))  # (Q1, i', D_max)
        R_tild = np.transpose(R_tild, [1, 2, 0])  # (i', D_max, Q1)
        L_tild = np.reshape(L_tild, (L_tild.shape[0], j_physical_dim, Q2.shape[0]))  # (D_max, j', Q2)
        L_tild = np.transpose(L_tild, [1, 0, 2])  # (j', D_max, Q2)


        ## (g) Glue back the R', L', sub-tensors to Q1, Q2, respectively, to form updated tensors P'l, P'r.
        Pl_prime = np.einsum('ijk,kl->ijl', R_tild, Q1)
        Pr_prime = np.einsum('ijk,kl->ijl', L_tild, Q2)

        ## (h) Reshape back the P`l, P`r to the original rank-(z + 1) tensors Ti, Tj
        Ti_new_shape = list(Ti[0].shape)
        Ti_new_shape[1] = len(lamda_k_tild)
        Tj_new_shape = list(Tj[0].shape)
        Tj_new_shape[1] = len(lamda_k_tild)
        Ti[0] = rank3_to_rankN(cp.deepcopy(Pl_prime), Ti_new_shape)
        Tj[0] = rank3_to_rankN(cp.deepcopy(Pr_prime), Tj_new_shape)

        # permuting back the legs of Ti and Tj
        Ti = dim_perm(cp.deepcopy(Ti))
        Tj = dim_perm(cp.deepcopy(Tj))

        ## (i) Remove bond matrices lambda_m from virtual legs m != Ek to obtain the updated tensors Ti~, Tj~.
        Ti = remove_edges(cp.deepcopy(Ti), i_dim, LL)
        Tj = remove_edges(cp.deepcopy(Tj), j_dim, LL)

        # Normalize and save new Ti Tj and lambda_k
        '''
        TT[Ti[1][0]] = cp.deepcopy(Ti[0] / np.sum(Ti[0]))
        TT[Tj[1][0]] = cp.deepcopy(Tj[0] / np.sum(Tj[0]))
        LL[Ek] = cp.deepcopy(lamda_k_tild / np.sum(lamda_k_tild))
        '''
        TT[Ti[1][0]] = cp.deepcopy(Ti[0] / np.sum(Ti[0]))
        TT[Tj[1][0]] = cp.deepcopy(Tj[0] / np.sum(Tj[0]))
        LL[Ek] = cp.deepcopy(lamda_k_tild / np.sum(lamda_k_tild))
    return cp.deepcopy(TT), cp.deepcopy(LL)


def get_tensors(edge, tensors, structure_matrix, incidence_matrix):
    tidx = np.nonzero(incidence_matrix[:, edge])[0]
    tdim = structure_matrix[tidx, edge]
    Ti = [cp.deepcopy(tensors[tidx[0]]), [tidx[0], 'tensor_number'], [tdim[0], 'tensor_Ek_leg']]
    Tj = [cp.deepcopy(tensors[tidx[1]]), [tidx[1], 'tensor_number'], [tdim[1], 'tensor_Ek_leg']]
    return Ti, Tj


def get_edges(edge, structure_matrix, incidence_matrix):
    tidx = np.nonzero(incidence_matrix[:, edge])[0]
    i_dim = [list(np.nonzero(incidence_matrix[tidx[0], :])[0]), list(structure_matrix[tidx[0], np.nonzero(incidence_matrix[tidx[0], :])[0]])]
    j_dim = [list(np.nonzero(incidence_matrix[tidx[1], :])[0]), list(structure_matrix[tidx[1], np.nonzero(incidence_matrix[tidx[1], :])[0]])]
    # removing the Ek edge and leg
    i_dim[0].remove(edge)
    i_dim[1].remove(structure_matrix[tidx[0], edge])
    j_dim[0].remove(edge)
    j_dim[1].remove(structure_matrix[tidx[1], edge])
    return i_dim, j_dim


def absorb_edges(tensor, edges_dim, bond_vectors):
    for i in range(len(edges_dim[0])):
        tensor[0] = np.einsum(tensor[0], range(len(tensor[0].shape)), bond_vectors[edges_dim[0][i]], [edges_dim[1][i]], range(len(tensor[0].shape)))
    return tensor


def remove_edges(tensor, edges_dim, bond_vectors):
    for i in range(len(edges_dim[0])):
        tensor[0] = np.einsum(tensor[0], range(len(tensor[0].shape)), bond_vectors[edges_dim[0][i]] ** (-1), [edges_dim[1][i]], range(len(tensor[0].shape)))
    return tensor


def dim_perm(tensor):
    # swapping the k leg with the element in the 1 place
    permutation = np.array(range(len(tensor[0].shape)))
    permutation[[1, tensor[2][0]]] = permutation[[tensor[2][0], 1]]
    tensor[0] = np.transpose(tensor[0], permutation)
    return tensor


def rankN_to_rank3(tensor):
    # taking a rank N>=3 tensor and make it a rank 3 tensor by grouping all dimensions [2, 3, ..., N]
    if len(tensor.shape) < 3:
        rank = len(tensor.shape)
        raise IndexError('expecting tensor rank N>=3. instead got tensor of rank = ' + str(rank))
    shape = np.array(cp.copy(tensor.shape))
    new_shape = [shape[0], shape[1], np.prod(shape[2:])]
    tensor = tensor.reshape(new_shape)
    return tensor


def rank2_to_rank3(tensor, physical_dim):
    # taking a rank N=2 tensor and make it a rank 3 tensor where the physical dimension and the Ek dimension are [0, 1] respectively
    if len(tensor.shape) is not 2:
        raise IndexError('expecting tensor rank N=2. instead got tensor of rank=', len(tensor.shape))
    new_tensor = np.reshape(tensor, [physical_dim, tensor.shape[0] / physical_dim, tensor.shape[1]])
    return new_tensor


def rank3_to_rankN(tensor, old_shape):
    new_tensor = np.reshape(tensor, old_shape)
    return new_tensor


def svd(tensor, left_legs, right_legs, keep_s=None, max_eigen_num=None):
    shape = np.array(cp.copy(tensor.shape))
    left_dim = np.prod(shape[[left_legs]])
    right_dim = np.prod(shape[[right_legs]])
    if keep_s == 'yes':
        u, s, vh = np.linalg.svd(tensor.reshape(left_dim, right_dim), full_matrices=False)
        if max_eigen_num is not None:
            u = u[:, 0:max_eigen_num]
            s = s[0:max_eigen_num]
            vh = vh[0:max_eigen_num, :]
        return u, s, vh
    else:
        u, s, vh = np.linalg.svd(tensor.reshape(left_dim, right_dim), full_matrices=False)
        if max_eigen_num is not None:
            u = u[:, 0:max_eigen_num]
            s = s[0:max_eigen_num]
            vh = vh[0:max_eigen_num, :]
        u = np.einsum(u, [0, 1], np.sqrt(s), [1], [0, 1])
        vh = np.einsum(np.sqrt(s), [0], vh, [0, 1], [0, 1])
    return u, vh

def imaginary_time_evolution(left_tensor, right_tensor, bond_vector, unitary_time_op):
    # applying ITE and returning a rank 4 tensor with physical dimensions, i' and j' at (Q1, i', j', Q2)
    # the indices of the unitary_time_op should be (i, j, i', j')
    bond_matrix = np.diag(bond_vector)
    A = np.einsum(left_tensor, [0, 1, 2], bond_matrix, [1, 3], [0, 3, 2])  # (i, Ek, Q1)
    A = np.einsum(A, [0, 1, 2], right_tensor, [3, 1, 4], [2, 0, 3, 4])  # (Q1, i, j, Q2)
    theta = np.einsum(A, [0, 1, 2, 3], unitary_time_op, [1, 2, 4, 5], [0, 4, 5, 3])  # (Q1, i', j', Q2)
    return theta


def check_convergence(bond_vectors_old, bond_vectors_new, max_error):
    bond_vectors_new = np.array(bond_vectors_new)
    bond_vectors_old = np.array(bond_vectors_old)
    error = np.sum(np.abs(bond_vectors_new - bond_vectors_old))
    if error < max_error:
        return 'converged'
    else:
        return 'did not converged'


def energy_per_site(TT, LL, imat, smat, Oij):
    energy_per_site = 0
    n, m = np.shape(imat)
    for Ek in range(m):
        lamda_k = cp.deepcopy(LL[Ek])

        ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
        Ti, Tj = get_tensors(Ek, TT, smat, imat)
        Ti_conj, Tj_conj = get_tensors(Ek, TT, smat, imat)

        # collecting all neighboring (edges, dimensions) without the Ek (edge, dimension)
        i_dim, j_dim = get_edges(Ek, smat, imat)

        ## (b) Absorb bond vectors (lambdas) to all Em != Ek of Ti, Tj tensors
        Ti = absorb_edges(cp.deepcopy(Ti), i_dim, LL)
        Tj = absorb_edges(cp.deepcopy(Tj), j_dim, LL)
        Ti_conj = absorb_edges(cp.deepcopy(Ti_conj), i_dim, LL)
        Tj_conj = absorb_edges(cp.deepcopy(Tj_conj), j_dim, LL)


        ## prepering list of tensors and indices for scon function
        s = 1000
        t = 2000
        lamda_k_idx = [t, t + 1]
        lamda_k_conj_idx = [t + 2, t + 3]
        Oij_idx = [s, s + 1, s + 2, s + 3]  # (i, j, i', j')

        Ti_idx = range(len(Ti[0].shape))
        Ti_conj_idx = range(len(Ti_conj[0].shape))
        Ti_idx[0] = Oij_idx[0]  # i
        Ti_conj_idx[0] = Oij_idx[2]  # i'
        Ti_idx[Ti[2][0]] = lamda_k_idx[0]
        Ti_conj_idx[Ti_conj[2][0]] = lamda_k_conj_idx[0]

        Tj_idx = range(len(Ti[0].shape) + 1, len(Ti[0].shape) + 1 + len(Tj[0].shape))
        Tj_conj_idx = range(len(Ti_conj[0].shape) + 1, len(Ti_conj[0].shape) + 1 + len(Tj_conj[0].shape))
        Tj_idx[0] = Oij_idx[1]  # j
        Tj_conj_idx[0] = Oij_idx[3]  # j'
        Tj_idx[Tj[2][0]] = lamda_k_idx[1]
        Tj_conj_idx[Tj_conj[2][0]] = lamda_k_conj_idx[1]

        # two site energy calculation
        tensors = [Ti[0], Ti_conj[0], Tj[0], Tj_conj[0], Oij, np.diag(lamda_k), np.diag(lamda_k)]
        indices = [Ti_idx, Ti_conj_idx, Tj_idx, Tj_conj_idx, Oij_idx, lamda_k_idx, lamda_k_conj_idx]
        two_site_energy = tnc.scon(tensors, indices)

        ## prepering list of tensors and indices for two site normalization
        p = Ti[0].shape[0]
        eye = np.reshape(np.eye(p * p), (p, p, p, p))
        eye_idx = Oij_idx

        tensors = [Ti[0], Ti_conj[0], Tj[0], Tj_conj[0], eye, np.diag(lamda_k), np.diag(lamda_k)]
        indices = [Ti_idx, Ti_conj_idx, Tj_idx, Tj_conj_idx, eye_idx, lamda_k_idx, lamda_k_conj_idx]
        two_site_norm = ncon.ncon(tensors, indices)
        #print('\n')
        #print('two site energy = ', two_site_energy)
        #print('two site norm = ', two_site_norm)
        two_site_energy /= two_site_norm
        #print('two site normalized energy = ', two_site_energy)

        energy_per_site += two_site_energy
    energy_per_site /= n
    #print(energy_per_site)
    return energy_per_site




