import numpy as np
import simple_update_algorithm2 as su
import copy as cp
from scipy import linalg

d = 2
p = 2
D_max = 4
J = 1.

#T0 = np.random.rand(p, d, d)
T0 = np.arange(p * d * d).reshape(p, d, d)
#T1 = np.random.rand(p, d, d)
T1 = np.arange(p * d * d).reshape(p, d, d)

TT = [T0, T1]
imat = np.array([[1, 1],
                 [1, 1]])

smat = np.array([[2, 1],
                 [2, 1]])

LL = []
for i in range(imat.shape[1]):
    LL.append(np.ones(d, dtype=float) / d)
    #LL.append(np.random.rand(d))

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

t_list = np.exp(np.concatenate((np.linspace(-1, -3, 20), np.linspace(-3, -5, 20))))
heisenberg = -J * np.real(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
hij = np.reshape(cp.deepcopy(heisenberg), (p, p, p, p))
hij_perm = [0, 1, 2, 3]
hij_energy_term = cp.deepcopy(hij)
hij = np.transpose(hij, hij_perm)
hij = np.reshape(hij, [p ** 2, p ** 2])
unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]
eye = np.eye(4).reshape(2, 2, 2, 2)

n, m = np.shape(imat)

TT_old = cp.deepcopy(TT)
LL_old = cp.deepcopy(LL)


for t in range(len(t_list)):
    for Ek in range(m):
        lamda_k = cp.deepcopy(LL[Ek])

        ## (a) Find tensors Ti, Tj and their corresponding legs connected along edge Ek.
        Ti, Tj = su.get_tensors(Ek, TT, smat, imat)

        # collecting all neighboring (edges, dimensions) without the Ek (edge, dimension)
        i_dim, j_dim = su.get_edges(Ek, smat, imat)

        ## (b) Absorb bond vectors (lambdas) to all Em != Ek of Ti, Tj tensors
        Ti = su.absorb_edges(cp.deepcopy(Ti), i_dim, LL)
        Tj = su.absorb_edges(cp.deepcopy(Tj), j_dim, LL)

        # permuting the Ek leg of tensors i and j into the 1'st dimension
        Ti = su.dim_perm(cp.deepcopy(Ti))
        Tj = su.dim_perm(cp.deepcopy(Tj))

        ## (c) Group all virtual legs Em!=Ek to form Pl, Pr MPS tensors
        Pl = su.rankN_to_rank3(cp.deepcopy(Ti[0]))
        Pr = su.rankN_to_rank3(cp.deepcopy(Tj[0]))

        ## (d) SVD decomposing of Pl, Pr to obtain Q1, R and Q2, L sub-tensors, respectively
        R, Q1 = su.svd(Pl, [0, 1], range(len(Pl.shape))[2:])
        L, Q2 = su.svd(Pr, [0, 1], range(len(Pr.shape))[2:])

        # reshaping R and L into rank 3 tensors with shape (physical_dim, Ek_dim, Q(1/2).shape[0])
        i_physical_dim = Ti[0].shape[0]
        j_physical_dim = Tj[0].shape[0]
        R = su.rank2_to_rank3(R, i_physical_dim)
        L = su.rank2_to_rank3(L, j_physical_dim)

        # (e) Contract the ITE gate Uij, with R, L, and lambda_k to form theta tensor.
        A = np.einsum(R, [0, 1, 2], np.diag(lamda_k), [1, 3], [0, 3, 2])
        A = np.einsum(A, [0, 1, 2], L, [3, 1, 4], [2, 0, 3, 4])  # (Q1 i, j, Q2)
        theta = np.einsum(A, [0, 1, 2, 3], unitary[t], [1, 2, 4, 5], [0, 4, 5, 3])  # (Q1, i', j', Q2)
        theta = np.transpose(theta, [1, 0, 2, 3])  # (i', Q1, j', Q2)

        ## (f) Obtain R', L', lambda'_k tensors by applying an SVD to theta
        # and trancating R', lamda_k', L' and keeping only D_max eigenvalues and eigenvectors
        R_tild, lamda_k_tild, L_tild = su.svd(theta, [0, 1], [2, 3], keep_s='yes', max_eigen_num=D_max)


        # reshaping R_tild and L_tild back to rank 3 tensor
        R_tild = su.rank2_to_rank3(R_tild, i_physical_dim)
        L_tild = su.rank2_to_rank3(np.transpose(L_tild), j_physical_dim)

        # permuting back to shape (physical_dim, Ek_dim, Q(1/2).shape[0])
        R_tild = np.transpose(R_tild, [0, 2, 1])
        L_tild = np.transpose(L_tild, [0, 2, 1])

        """
        R_tild = np.reshape(R_tild, (Q1.shape[0], i_physical_dim, R_tild.shape[1]))
        R_tild = np.transpose(R_tild, [1, 2, 0])
        L_tild = np.reshape(L_tild, (L_tild.shape[0], j_physical_dim, Q2.shape[0]))
        L_tild = np.transpose(L_tild, [1, 0, 2])
        """

        ## (g) Glue back the R', L', sub-tensors to Q1, Q2, respectively, to form updated tensors P'l, P'r.
        Pl_prime = np.einsum('ijk,kl->ijl', R_tild, Q1)
        Pr_prime = np.einsum('ijk,kl->ijl', L_tild, Q2)

        ## (h) Reshape back the P`l, P`r to the original rank-(z + 1) tensors Ti, Tj
        Ti_new_shape = list(cp.deepcopy(Ti[0].shape))
        Ti_new_shape[1] = len(lamda_k_tild)
        Tj_new_shape = list(cp.deepcopy(Tj[0].shape))
        Tj_new_shape[1] = len(lamda_k_tild)
        Ti[0] = su.rank3_to_rankN(cp.deepcopy(Pl_prime), Ti_new_shape)
        Tj[0] = su.rank3_to_rankN(cp.deepcopy(Pr_prime), Tj_new_shape)

        # permuting back the legs of Ti and Tj
        Ti = su.dim_perm(cp.deepcopy(Ti))
        Tj = su.dim_perm(cp.deepcopy(Tj))

        ## (i) Remove bond matrices lambda_m from virtual legs m != Ek to obtain the updated tensors Ti~, Tj~.
        Ti = su.remove_edges(cp.deepcopy(Ti), i_dim, LL)
        Tj = su.remove_edges(cp.deepcopy(Tj), j_dim, LL)

        # Normalize and save new Ti Tj and lambda_k
        TT[Ti[1][0]] = cp.deepcopy(Ti[0] / np.sum(Ti[0]))
        TT[Tj[1][0]] = cp.deepcopy(Tj[0] / np.sum(Tj[0]))
        LL[Ek] = cp.deepcopy(lamda_k_tild / np.sum(lamda_k_tild))

        print(su.energy_per_site(TT, LL, imat, smat, hij_energy_term))
        #error = (su.energy_per_site(TT, LL, imat, smat, hij_energy_term) - su.energy_per_site(TT_old, LL_old, imat, smat, hij_energy_term))
    #error = np.max(np.abs(TT[0] - TT_old[0]) + np.abs(TT[1] - TT_old[1]))
    #print(error)
    TT_old = cp.deepcopy(TT)
    LL_old = cp.deepcopy(LL)





'''
    1) remove_edges and absorb_edges works.
    2) rankN_to_rank3 and rank3_to_rankN works.
    3) dim_perm works.
    4) svd works
    5) if taking (Rtild, Ltild, lamda_k_tild) = (R, L, lamda_k) the su does have any effect of tensor and lamdas
    
'''


