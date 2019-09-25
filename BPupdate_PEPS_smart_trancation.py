import numpy as np
import ncon as ncon
import copy as cp
from scipy import linalg
import virtual_DEFG as denfg
from numba import jit
import time
import matplotlib.pyplot as plt
import ncon_lists_generator as nlg

"""
    A module for the function PEPS_BPupdate which preform the BPupdate algorithm over a given PEPS Tensor Network 

"""


def PEPS_BPupdate(TT, LL, dt, Jk, h, Opi, Opj, Op_field, imat, smat, D_max, graph, t_max, epsilon, dumping):
    """
    :param TT: list of tensors in the tensor network TT = [T1, T2, T3, ..., Tp]
    :param LL: list of lists of the lambdas LL = [L1, L2, ..., Ls]
    :param dt: time step for imaginary time evolution
    :param Jk: list of interaction constants for each edge Jk = [J1, J2, J3, ..., Js]
    :param h: field value
    :param Opi, Opj: operators of nearest neighbors interactions <i, j> (for ZZX hamiltonian Opi = Opj = z )
    :param Op_field: the operator of the field (for ZZX hamiltonian Op_filed = x )
    :param imat: The incidence matrix which indicates which tensor connect to which edge (as indicated in the paper)
    :param smat: The structure matrix which indicates which leg of every tensor is connected to which edge
    :param D_max: maximal virtual dimension

    """
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)

    n, m = np.shape(imat)
    for Ek in range(m):
        lamda_k = LL[Ek]

        ## Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
        Ti, Tj = get_tensors(Ek, TT, smat, imat)

        ## collect edges and remove the Ek edge from both lists
        iedges = list(np.nonzero(smat[Ti[1][0], :])[0])
        ilegs = list(smat[Ti[1][0], iedges])
        jedges = list(np.nonzero(smat[Tj[1][0], :])[0])
        jlegs = list(smat[Tj[1][0], jedges])
        iedges.remove(Ek)
        ilegs.remove(smat[Ti[1][0], Ek])
        jedges.remove(Ek)
        jlegs.remove(smat[Tj[1][0], Ek])

        ## absorb lamda vectros into tensors
        for ii in range(len(iedges)):
            Ti[0] = np.einsum(Ti[0], range(len(Ti[0].shape)), LL[iedges[ii]], [ilegs[ii]], range(len(Ti[0].shape)))
        for ii in range(len(jedges)):
            Tj[0] = np.einsum(Tj[0], range(len(Tj[0].shape)), LL[jedges[ii]], [jlegs[ii]], range(len(Tj[0].shape)))

        # permuting the Ek leg of tensors i and j into the 1'st dimension
        Ti = dim_perm(Ti)
        Tj = dim_perm(Tj)

        ## Group all virtual legs Em!=Ek to form Pl, Pr MPS tensors
        Pl = rankN_to_rank3(Ti[0])
        Pr = rankN_to_rank3(Tj[0])

        ## SVD decomposing of Pl, Pr to obtain Q1, R and Q2, L sub-tensors, respectively
        R, sr, Q1 = svd(Pl, [0, 1], [2], keep_s='yes')
        L, sl, Q2 = svd(Pr, [0, 1], [2], keep_s='yes')
        R = R.dot(np.diag(sr))
        L = L.dot(np.diag(sl))

        # reshaping R and L into rank 3 tensors with shape (physical_dim, Ek_dim, Q(1/2).shape[0])
        i_physical_dim = Ti[0].shape[0]
        j_physical_dim = Tj[0].shape[0]
        R = rank2_to_rank3(R, i_physical_dim)  # (i, Ek, Q1) (following the dimensions)
        L = rank2_to_rank3(L, j_physical_dim)  # (j, Ek, Q2)

        ## Contract the ITE gate with R, L, and lambda_k to form theta tensor.
        theta = imaginary_time_evolution(R, L, lamda_k, Ek, dt, Jk, h, Opi, Opj, Op_field)  # (Q1, i', j', Q2)

        ## Obtain R', L', lambda'_k tensors by applying an SVD to theta
        #R_tild, lamda_k_tild, L_tild = svd(theta, [0, 1], [2, 3], keep_s='yes', max_eigen_num=D_max) # with truncation
        R_tild, lamda_k_tild, L_tild = svd(theta, [0, 1], [2, 3], keep_s='yes') # without truncation
        # (Q1 * i', D') # (D', D') # (D', j' * Q2)

        # reshaping R_tild and L_tild back to rank 3 tensor
        R_tild = np.reshape(R_tild, (Q1.shape[0], i_physical_dim, R_tild.shape[1]))  # (Q1, i', D')
        R_tild = np.transpose(R_tild, [1, 2, 0])  # (i', D', Q1)
        L_tild = np.reshape(L_tild, (L_tild.shape[0], j_physical_dim, Q2.shape[0]))  # (D', j', Q2)
        L_tild = np.transpose(L_tild, [1, 0, 2])  # (j', D', Q2)

        ## Glue back the R', L', sub-tensors to Q1, Q2, respectively, to form updated tensors P'l, P'r.
        Pl_prime = np.einsum('ijk,kl->ijl', R_tild, Q1)
        Pr_prime = np.einsum('ijk,kl->ijl', L_tild, Q2)

        ## Reshape back the P`l, P`r to the original rank-(z + 1) tensors Ti, Tj
        Ti_new_shape = list(Ti[0].shape)
        Ti_new_shape[1] = len(lamda_k_tild)
        Tj_new_shape = list(Tj[0].shape)
        Tj_new_shape[1] = len(lamda_k_tild)
        Ti[0] = rank3_to_rankN(Pl_prime, Ti_new_shape)
        Tj[0] = rank3_to_rankN(Pr_prime, Tj_new_shape)

        # permuting back the legs of Ti and Tj
        Ti = dim_perm(Ti)
        Tj = dim_perm(Tj)

        ## Remove bond matrices lambda_m from virtual legs m != Ek to obtain the updated tensors Ti~, Tj~.
        for ii in range(len(iedges)):
            Ti[0] = np.einsum(Ti[0], range(len(Ti[0].shape)), LL[iedges[ii]] ** (-1), [ilegs[ii]], range(len(Ti[0].shape)))
        for ii in range(len(jedges)):
            Tj[0] = np.einsum(Tj[0], range(len(Tj[0].shape)), LL[jedges[ii]] ** (-1), [jlegs[ii]], range(len(Tj[0].shape)))

        # Normalize and save new Ti Tj and lambda_k
        TT[Ti[1][0]] = Ti[0] / tensor_normalization(Ti[0])
        TT[Tj[1][0]] = Tj[0] / tensor_normalization(Tj[0])
        LL[Ek] = lamda_k_tild / np.sum(lamda_k_tild)


        ## single edge BP update (uncomment for single edge BP implemintation)
        TT, LL = BPupdate_single_edge(TT, LL, smat, imat, t_max, epsilon, dumping, D_max, Ek, graph)

    return TT, LL


# ---------------------------------- gPEPS functions ---------------------------------

def get_tensors(edge, tensors, structure_matrix, incidence_matrix):
    # given an edge collect neighboring tensors
    tidx = np.nonzero(incidence_matrix[:, edge])[0]
    tdim = structure_matrix[tidx, edge]
    Ti = [tensors[tidx[0]], [tidx[0], 'tensor_number'], [tdim[0], 'tensor_Ek_leg']]
    Tj = [tensors[tidx[1]], [tidx[1], 'tensor_number'], [tdim[1], 'tensor_Ek_leg']]
    return Ti, Tj


def get_conjugate_tensors(edge, tensors, structure_matrix, incidence_matrix):
    # given an edge collect neighboring conjugate tensors
    tidx = np.nonzero(incidence_matrix[:, edge])[0]
    tdim = structure_matrix[tidx, edge]
    Ti = [cp.deepcopy(np.conj(tensors[tidx[0]])), [tidx[0], 'tensor_number'], [tdim[0], 'tensor_Ek_leg']]
    Tj = [cp.deepcopy(np.conj(tensors[tidx[1]])), [tidx[1], 'tensor_number'], [tdim[1], 'tensor_Ek_leg']]
    return Ti, Tj


def get_edges(edge, structure_matrix, incidence_matrix):
    # given an edge collect neighboring tensors edges and legs
    tidx = np.nonzero(incidence_matrix[:, edge])[0]
    i_dim = [list(np.nonzero(incidence_matrix[tidx[0], :])[0]), list(structure_matrix[tidx[0], np.nonzero(incidence_matrix[tidx[0], :])[0]])] # [edges, legs]
    j_dim = [list(np.nonzero(incidence_matrix[tidx[1], :])[0]), list(structure_matrix[tidx[1], np.nonzero(incidence_matrix[tidx[1], :])[0]])]
    # removing the Ek edge and leg
    i_dim[0].remove(edge)
    i_dim[1].remove(structure_matrix[tidx[0], edge])
    j_dim[0].remove(edge)
    j_dim[1].remove(structure_matrix[tidx[1], edge])
    return i_dim, j_dim


def get_all_edges(edge, structure_matrix, incidence_matrix):
    # given an edge collect neighboring tensors edges and legs
    tidx = np.nonzero(incidence_matrix[:, edge])[0]
    i_dim = [list(np.nonzero(incidence_matrix[tidx[0], :])[0]), list(structure_matrix[tidx[0], np.nonzero(incidence_matrix[tidx[0], :])[0]])]
    j_dim = [list(np.nonzero(incidence_matrix[tidx[1], :])[0]), list(structure_matrix[tidx[1], np.nonzero(incidence_matrix[tidx[1], :])[0]])]
    return i_dim, j_dim


def absorb_edges(tensor, edges_dim, bond_vectors):
    # absorb tensor edges
    for i in range(len(edges_dim[0])):
        tensor[0] = np.einsum(tensor[0], range(len(tensor[0].shape)), bond_vectors[edges_dim[0][i]], [edges_dim[1][i]], range(len(tensor[0].shape)))
    return tensor


def absorb_edges_for_graph(tensor, edges_dim, bond_vectors):
    # absorb tensor edges
    for i in range(len(edges_dim[0])):
        tensor[0] = np.einsum(tensor[0], range(len(tensor[0].shape)), np.sqrt(bond_vectors[edges_dim[0][i]]), [edges_dim[1][i]], range(len(tensor[0].shape)))
    return tensor


def remove_edges(tensor, edges_dim, bond_vectors):
    # remove tensor edges
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
    Pi = np.reshape(tensor, new_shape)
    return Pi


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
    shape = np.array(tensor.shape)
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


def imaginary_time_evolution(left_tensor, right_tensor, bond_vector, Ek, dt, Jk, h, Opi, Opj, Op_field):
    # applying ITE and returning a rank 4 tensor with physical dimensions, i' and j' at (Q1, i', j', Q2)
    # the indices of the unitary_time_op should be (i, j, i', j')
    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    hij = -Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p)))
    unitary_time_op = np.reshape(linalg.expm(-dt * hij), [p, p, p, p])
    bond_matrix = np.diag(bond_vector)
    A = np.einsum(left_tensor, [0, 1, 2], bond_matrix, [1, 3], [0, 3, 2])  # (i, Ek, Q1)
    A = np.einsum(A, [0, 1, 2], right_tensor, [3, 1, 4], [2, 0, 3, 4])  # (Q1, i, j, Q2)
    theta = np.einsum(A, [0, 1, 2, 3], unitary_time_op, [1, 2, 4, 5], [0, 4, 5, 3])  # (Q1, i', j', Q2)
    return theta


def tensor_normalization(T):
    T_conj = np.conj(cp.deepcopy(T))
    idx = range(len(T.shape))
    norm = np.einsum(T, idx, T_conj, idx)
    return np.sqrt(norm)


def graph_update(Ek, TT, LL, smat, imat, graph):
    fi, fj = get_tensors(Ek, cp.deepcopy(TT), smat, imat)
    i_dim, j_dim = get_all_edges(Ek, smat, imat)
    fi = absorb_edges_for_graph(fi, i_dim, LL)
    fj = absorb_edges_for_graph(fj, j_dim, LL)
    graph.factors['f' + str(fi[1][0])][1] = fi[0]
    graph.factors['f' + str(fj[1][0])][1] = fj[0]
    graph.nodes['n' + str(Ek)][0] = len(LL[Ek])


# ---------------------------------- gPEPS expectations and exact expectation functions ---------------------------------


def single_tensor_expectation(tensor_idx, TT, LL, imat, smat, Oi):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    normalization = site_norm(tensor_idx, TT, LL, imat, smat)

    env_edges = np.nonzero(imat[tensor_idx, :])[0]
    env_legs = smat[tensor_idx, env_edges]
    T = TT[tensor_idx]
    T_conj = np.conj(TT[tensor_idx])

    ## absorb its environment
    for j in range(len(env_edges)):
        T = np.einsum(T, range(len(T.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T.shape)))
        T_conj = np.einsum(T_conj, range(len(T_conj.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T_conj.shape)))

    T_idx = range(len(T.shape))
    T_conj_idx = range(len(T_conj.shape))
    T_conj_idx[0] = len(T_conj.shape)
    operator_idx = [T_conj_idx[0], T_idx[0]]
    expectation = ncon.ncon([T, T_conj, Oi], [T_idx, T_conj_idx, operator_idx])
    return expectation / normalization


def magnetization(TT, LL, imat, smat, Oi):
    # calculating the average magnetization per site
    magnetization = 0
    tensors_indices = range(len(TT))
    for i in tensors_indices:
        magnetization += single_tensor_expectation(i, TT, LL, imat, smat, Oi)
    magnetization /= len(TT)
    return magnetization


def site_norm(tensor_idx, TT, LL, imat, smat):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    env_edges = np.nonzero(imat[tensor_idx, :])[0]
    env_legs = smat[tensor_idx, env_edges]
    T = TT[tensor_idx]
    T_conj = np.conj(TT[tensor_idx])

    ## absorb its environment
    for j in range(len(env_edges)):
        T = np.einsum(T, range(len(T.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T.shape)))
        T_conj = np.einsum(T_conj, range(len(T_conj.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T_conj.shape)))

    T_idx = range(len(T.shape))
    T_conj_idx = range(len(T_conj.shape))
    normalization = np.einsum(T, T_idx, T_conj, T_conj_idx)
    return normalization


def two_site_expectation(Ek, TT, LL, imat, smat, Oij):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)

    # calculating the two site normalized expectation given a mutual edge Ek of those two sites (tensors) and the operator Oij
    lamda_k = cp.copy(LL[Ek])

    ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
    Ti, Tj = get_tensors(Ek, TT, smat, imat)
    Ti_conj, Tj_conj = get_conjugate_tensors(Ek, TT, smat, imat)

    # collecting all neighboring (edges, dimensions) without the Ek (edge, dimension)
    i_dim, j_dim = get_edges(Ek, smat, imat)

    ## (b) Absorb bond vectors (lambdas) to all Em != Ek of Ti, Tj tensors
    Ti = absorb_edges(Ti, i_dim, LL)
    Tj = absorb_edges(Tj, j_dim, LL)
    Ti_conj = absorb_edges(Ti_conj, i_dim, LL)
    Tj_conj = absorb_edges(Tj_conj, j_dim, LL)

    ## preparing list of tensors and indices for ncon function
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

    # two site expectation calculation
    tensors = [Ti[0], Ti_conj[0], Tj[0], Tj_conj[0], Oij, np.diag(lamda_k), np.diag(lamda_k)]
    indices = [Ti_idx, Ti_conj_idx, Tj_idx, Tj_conj_idx, Oij_idx, lamda_k_idx, lamda_k_conj_idx]
    two_site_expec = ncon.ncon(tensors, indices)

    ## prepering list of tensors and indices for two site normalization
    p = Ti[0].shape[0]
    eye = np.reshape(np.eye(p * p), (p, p, p, p))
    eye_idx = Oij_idx

    tensors = [Ti[0], Ti_conj[0], Tj[0], Tj_conj[0], eye, np.diag(lamda_k), np.diag(lamda_k)]
    indices = [Ti_idx, Ti_conj_idx, Tj_idx, Tj_conj_idx, eye_idx, lamda_k_idx, lamda_k_conj_idx]
    two_site_norm = ncon.ncon(tensors, indices)
    two_site_expec /= two_site_norm
    return two_site_expec


def two_site_exact_expectation(TT, LL, smat, edge, operator):
    TTstar = conjTN(TT)
    TT_tilde = absorb_all_bond_vectors(TT, LL, smat)
    TTstar_tilde = absorb_all_bond_vectors(TTstar, LL, smat)
    T_list, idx_list = nlg.ncon_list_generator_two_site_exact_expectation_peps(TT_tilde, TTstar_tilde, smat, edge, operator)
    T_list_norm, idx_list_norm = nlg.ncon_list_generator_braket_peps(TT_tilde, TTstar_tilde, smat)
    exact_expectation = ncon.ncon(T_list, idx_list) / ncon.ncon(T_list_norm, idx_list_norm)
    return exact_expectation


def conjTN(TT):
    TTconj = []
    for i in range(len(TT)):
        TTconj.append(np.conj(TT[i]))
    return TTconj


def energy_per_site(TT, LL, imat, smat, Jk, h, Opi, Opj, Op_field):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    # calculating the normalized energy per site(tensor)
    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    energy = 0
    n, m = np.shape(imat)
    for Ek in range(m):
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        energy += two_site_expectation(Ek, TT, LL, imat, smat, Oij)
    energy /= n
    return energy


def exact_energy_per_site(TT, LL, smat, Jk, h, Opi, Opj, Op_field):
    # calculating the normalized exact energy per site(tensor)
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    energy = 0
    n, m = np.shape(smat)
    for Ek in range(m):
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        energy += two_site_exact_expectation(TT, LL, smat, Ek, Oij)
    energy /= n
    return energy


def BP_energy_per_site_ising_factor_belief(graph, smat, imat, Jk, h, Opi, Opj, Op_field):
    # calculating the normalized exact energy per site(tensor)
    if graph.factor_belief == None:
        raise IndexError('First calculate factor beliefs')
    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    energy = 0
    n, m = np.shape(smat)
    for Ek in range(m):
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        tensors = np.nonzero(smat[:, Ek])[0]
        fi_belief = graph.factor_belief['f' + str(tensors[0])]
        fj_belief = graph.factor_belief['f' + str(tensors[1])]
        fi_idx = range(len(fi_belief.shape))
        fj_idx = range(len(fi_belief.shape), len(fi_belief.shape) + len(fj_belief.shape))
        Oij_idx = [1000, 1001, 1002, 1003]
        fi_idx[0] = Oij_idx[0]
        fi_idx[1] = Oij_idx[2]
        fj_idx[0] = Oij_idx[1]
        fj_idx[1] = Oij_idx[3]
        iedges, jedges = get_edges(Ek, smat, imat)
        for leg_idx, leg in enumerate(iedges[1]):
            fi_idx[2 * leg + 1] = fi_idx[2 * leg]
        for leg_idx, leg in enumerate(jedges[1]):
            fj_idx[2 * leg + 1] = fj_idx[2 * leg]
        Ek_legs = smat[np.nonzero(smat[:, Ek])[0], Ek]
        fi_idx[2 * Ek_legs[0]] = fj_idx[2 * Ek_legs[1]]
        fi_idx[2 * Ek_legs[0] + 1] = fj_idx[2 * Ek_legs[1] + 1]
        E = ncon.ncon([fi_belief, fj_belief, Oij], [fi_idx, fj_idx, Oij_idx])
        norm = ncon.ncon([fi_belief, fj_belief, np.eye(p ** 2).reshape((p, p, p, p))], [fi_idx, fj_idx, Oij_idx])
        E_normalized = E / norm
        energy += E_normalized
    energy /= n
    return energy


def BP_energy_per_site_ising_rdm_belief(graph, smat, imat, Jk, h, Opi, Opj, Op_field):
    # calculating the normalized exact energy per site(tensor)
    if graph.rdm_belief == None:
        raise IndexError('First calculate rdm beliefs')
    p = Opi[0].shape[0]
    Aij = np.zeros((p ** 2, p ** 2), dtype=complex)
    for i in range(len(Opi)):
        Aij += np.kron(Opi[i], Opj[i])
    energy = 0
    n, m = np.shape(smat)
    for Ek in range(m):
        Oij = np.reshape(-Jk[Ek] * Aij - 0.25 * h * (np.kron(np.eye(p), Op_field) + np.kron(Op_field, np.eye(p))), (p, p, p, p))
        tensors = np.nonzero(smat[:, Ek])[0]
        fi_belief = graph.rdm_belief[tensors[0]]
        fj_belief = graph.rdm_belief[tensors[1]]
        fij = np.einsum(fi_belief, [0, 1], fj_belief, [2, 3], [0, 2, 1, 3])
        Oij_idx = [0, 1, 2, 3]
        E = np.einsum(fij, [0, 1, 2, 3], Oij, Oij_idx)
        norm = np.einsum(fij, [0, 1, 0, 1])
        E_normalized = E / norm
        energy += E_normalized
    energy /= n
    return energy


def trace_distance(a, b):
    # returns the trace distance between the two density matrices a & b
    # d = 0.5 * norm(a - b)
    eigenvalues = np.linalg.eigvals(a - b)
    d = 0.5 * np.sum(np.abs(eigenvalues))
    return d


def tensor_reduced_dm(tensor_idx, TT, LL, smat, imat):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    normalization = site_norm(tensor_idx, TT, LL, imat, smat)
    env_edges = np.nonzero(smat[tensor_idx, :])[0]
    env_legs = smat[tensor_idx, env_edges]
    T = cp.deepcopy(TT[tensor_idx])
    T_conj = cp.deepcopy(np.conj(TT[tensor_idx]))

    ## absorb its environment
    for j in range(len(env_edges)):
        T = np.einsum(T, range(len(T.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T.shape)))
        T_conj = np.einsum(T_conj, range(len(T_conj.shape)), LL[env_edges[j]], [env_legs[j]], range(len(T_conj.shape)))

    T_idx = range(len(T.shape))
    T_idx[0] = -1
    T_conj_idx = range(len(T_conj.shape))
    T_conj_idx[0] = -2
    reduced_dm = ncon.ncon([T, T_conj], [T_idx, T_conj_idx])

    return reduced_dm / normalization


def absorb_all_bond_vectors(TT, LL, smat):
    TT = cp.deepcopy(TT)
    LL = cp.deepcopy(LL)
    n = len(TT)
    for i in range(n):
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        for j in range(len(edges)):
            TT[i] = np.einsum(TT[i], range(len(TT[i].shape)), np.sqrt(LL[edges[j]]), [legs[j]], range(len(TT[i].shape)))
    return TT


# ---------------------------------- BP truncation  ---------------------------------


'''
def BPupdate(graph, TT, LL, smat, imat, Dmax):
    ## this BP truncation is implemented on all edges

    # run over all edges

    for Ek in range(len(LL)):
        the_node = 'n' + str(Ek)
        Ti, Tj = get_tensors(Ek, TT, smat, imat)
        i_dim, j_dim = get_all_edges(Ek, smat, imat)

        fi = absorb_edges_for_graph(cp.deepcopy(Ti), i_dim, LL)
        fj = absorb_edges_for_graph(cp.deepcopy(Tj), j_dim, LL)

        A, B = AnB_calculation(graph, fi, fj, the_node)
        P = find_P(A, B, Dmax)
        TT, LL = smart_truncation(TT, LL, P, Ek, smat, imat, Dmax)
        graph_update(Ek, TT, LL, smat, imat, graph)
    return TT, LL
'''

def BPupdate_single_edge(TT, LL, smat, imat, t_max, epsilon, dumping, Dmax, Ek, graph):
    ## this BP truncation is implemented on a single edge Ek

    # run BP on graph
    graph.sum_product(t_max, epsilon, dumping, 'init_with_old_messages')

    the_node = 'n' + str(Ek)
    Ti, Tj = get_tensors(Ek, TT, smat, imat)
    i_dim, j_dim = get_all_edges(Ek, smat, imat)

    fi = absorb_edges_for_graph(cp.deepcopy(Ti), i_dim, LL)
    fj = absorb_edges_for_graph(cp.deepcopy(Tj), j_dim, LL)

    A, B = AnB_calculation(graph, fi, fj, the_node)
    P = find_P(A, B, Dmax)
    TT, LL = smart_truncation(TT, LL, P, Ek, smat, imat, Dmax)
    graph_update(Ek, TT, LL, smat, imat, graph)

    return TT, LL


def PEPStoDEnFG_transform(graph, TT, LL, smat):
    # generate the double edge factor graph from PEPS
    factors_list = absorb_all_bond_vectors(TT, LL, smat)

    # Adding virtual nodes
    n, m = np.shape(smat)
    for i in range(m):
        graph.add_node(len(LL[i]), 'n' + str(graph.node_count))
    # Adding factors
    for i in range(n):
        # generating the neighboring nodes of the i'th factor
        neighbor_nodes = {}
        edges = np.nonzero(smat[i, :])[0]
        legs = smat[i, edges]
        for j in range(len(edges)):
            neighbor_nodes['n' + str(edges[j])] = legs[j]
        graph.add_factor(neighbor_nodes, np.array(factors_list[i], dtype=complex))
    return graph


def find_P(A, B, D_max):
    A_sqrt = linalg.sqrtm(A)
    B_sqrt = linalg.sqrtm(np.transpose(B))

    ##  Calculate the environment matrix C and its SVD
    C = np.matmul(B_sqrt, A_sqrt)
    u_env, s_env, vh_env = np.linalg.svd(C, full_matrices=False)

    ##  Define P2
    new_s_env = cp.copy(s_env)
    new_s_env[D_max:] = 0
    P2 = np.zeros((len(s_env), len(s_env)))
    np.fill_diagonal(P2, new_s_env)
    P2 /= np.sum(new_s_env)

    ##  Calculating P = A^(-1/2) * V * P2 * U^(dagger) * B^(-1/2)
    P = np.matmul(np.linalg.inv(A_sqrt), np.matmul(np.transpose(np.conj(vh_env)), np.matmul(P2, np.matmul(np.transpose(np.conj(u_env)), np.linalg.inv(B_sqrt)))))
    return P


def smart_truncation(TT1, LL1, P, edge, smat, imat, D_max):
    iedges, jedges = get_edges(edge, smat, imat)
    Ti, Tj = get_tensors(edge, TT1, smat, imat)
    Ti = absorb_edges(Ti, iedges, LL1)
    Tj = absorb_edges(Tj, jedges, LL1)

    # absorb the mutual edge
    Ti[0] = np.einsum(Ti[0], range(len(Ti[0].shape)), np.sqrt(LL1[edge]), [Ti[2][0]], range(len(Ti[0].shape)))
    Tj[0] = np.einsum(Tj[0], range(len(Tj[0].shape)), np.sqrt(LL1[edge]), [Tj[2][0]], range(len(Tj[0].shape)))

    # reshaping
    Ti = dim_perm(Ti)
    Tj = dim_perm(Tj)
    i_old_shape = cp.copy(list(Ti[0].shape))
    j_old_shape = cp.copy(list(Tj[0].shape))
    Ti[0] = rankN_to_rank3(Ti[0])
    Tj[0] = rankN_to_rank3(Tj[0])

    # contracting P with Ti and Tj and then using SVD to generate Ti_tilde and Tj_tilde and lamda_tilde
    Ti, Tj, lamda_edge = Accordion(Ti, Tj, P, D_max)

    # reshaping back
    i_old_shape[1] = D_max
    j_old_shape[1] = D_max
    Ti[0] = rank3_to_rankN(Ti[0], i_old_shape)
    Tj[0] = rank3_to_rankN(Tj[0], j_old_shape)
    Ti = dim_perm(Ti)
    Tj = dim_perm(Tj)
    Ti = remove_edges(Ti, iedges, LL1)
    Tj = remove_edges(Tj, jedges, LL1)

    # saving tensors and lamda
    TT1[Ti[1][0]] = cp.deepcopy(Ti[0] / tensor_normalization(Ti[0]))
    TT1[Tj[1][0]] = cp.deepcopy(Tj[0] / tensor_normalization(Tj[0]))
    LL1[edge] = lamda_edge / np.sum(lamda_edge)
    return TT1, LL1

'''
def BPupdate_error(TT, LL, TT_old, LL_old, smat):
    psipsi_T_list, psipsi_idx_list = nlg.ncon_list_generator_for_BPerror(cp.deepcopy(TT), cp.deepcopy(LL), cp.deepcopy(TT), cp.deepcopy(LL), smat)
    psiphi_T_list, psiphi_idx_list = nlg.ncon_list_generator_for_BPerror(cp.deepcopy(TT), cp.deepcopy(LL), cp.deepcopy(TT_old), cp.deepcopy(LL_old), smat)
    phiphi_T_list, phiphi_idx_list = nlg.ncon_list_generator_for_BPerror(cp.deepcopy(TT_old), cp.deepcopy(LL_old), cp.deepcopy(TT_old), cp.deepcopy(LL_old), smat)
    phipsi_T_list, phipsi_idx_list = nlg.ncon_list_generator_for_BPerror(cp.deepcopy(TT_old), cp.deepcopy(LL_old), cp.deepcopy(TT), cp.deepcopy(LL), smat)

    psipsi = ncon.ncon(psipsi_T_list, psipsi_idx_list)
    psiphi = ncon.ncon(psiphi_T_list, psiphi_idx_list)
    phipsi = ncon.ncon(phipsi_T_list, phipsi_idx_list)
    phiphi = ncon.ncon(phiphi_T_list, phiphi_idx_list)

    psi_norm = np.sqrt(psipsi)
    phi_norm = np.sqrt(phiphi)
    # print('overlap_exact = ', psiphi / psi_norm / phi_norm)
    error = 2 - psiphi / psi_norm / phi_norm - phipsi / psi_norm / phi_norm
    return error
'''

def Accordion(Ti, Tj, P, D_max):
    # contracting two tensors i, j with P and SVD (with truncation) back
    L = cp.deepcopy(Ti[0])
    R = cp.deepcopy(Tj[0])

    A = np.einsum(L, [0, 1, 2], P, [1, 3], [0, 3, 2])  # (i, Ek, Q1)
    theta = np.einsum(A, [0, 1, 2], R, [3, 1, 4], [2, 0, 3, 4])  # (Q1, i, j, Q2)

    R_tild, lamda_k, L_tild = svd(theta, [0, 1], [2, 3], keep_s='yes', max_eigen_num=D_max)

    # reshaping R_tild and L_tild back
    R_tild_new_shape = [Ti[0].shape[2], Ti[0].shape[0], R_tild.shape[1]]  # (d, i, D')
    R_transpose = [1, 2, 0]
    L_tild_new_shape = [L_tild.shape[0], Tj[0].shape[0], Tj[0].shape[2]]  # (D', j, d)
    L_transpose = [1, 0, 2]

    R_tild = np.reshape(R_tild, R_tild_new_shape)
    Ti[0] = np.transpose(R_tild, R_transpose)  # (i, D', ...)
    L_tild = np.reshape(L_tild, L_tild_new_shape)
    Tj[0] = np.transpose(L_tild, L_transpose)  # (j, D', ...)

    return Ti, Tj, lamda_k


def AnB_calculation(graph, Ti, Tj, node_Ek):
    A = graph.f2n_message_chnaged_factor('f' + str(Ti[1][0]), node_Ek, graph.messages_n2f, cp.deepcopy(Ti[0]))
    B = graph.f2n_message_chnaged_factor('f' + str(Tj[1][0]), node_Ek, graph.messages_n2f, cp.deepcopy(Tj[0]))

    #A = graph.f2n_message_chnaged_factor_without_matching_dof('f' + str(Ti[1][0]), node_Ek, graph.messages_n2f, cp.deepcopy(Ti[0]))
    #B = graph.f2n_message_chnaged_factor_without_matching_dof('f' + str(Tj[1][0]), node_Ek, graph.messages_n2f, cp.deepcopy(Tj[0]))
    #print('A, A1', np.sum(np.abs(A - A1)) / np.sum(A) / np.sum(A1))
    #print('B, B1', np.sum(np.abs(B - B1)) / np.sum(B) / np.sum(B1))


    return A, B



