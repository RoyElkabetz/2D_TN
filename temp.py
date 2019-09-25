from numba import jit
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import copy as cp
import Tensor_Network_functions as tnf
import virtual_DEFG as defg
import BPupdate_PEPS_smart_trancation as su

'''
@jit(nopython=True)
def monte_carlo_pi(nsamples, string):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
            string.append(acc)
    return 4.0 * acc / nsamples, string

@jit(nopython=True)
def print_value(i):
    #acc = 0
    for j in range(i):
        a = 1
        print(np.einsum(np.array([0, 1]), [0], np.array([0, 1]), [0]))
        #acc += st[j]
    #print(acc)

#@jit(nopython=True)
def PEPSfill(N, L):
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

N = 16
L = np.int(np.sqrt(N))
smat, imat = PEPSfill(N, L)


pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

Opi = [sx, sy, sz]
Opj = [sx, sy, sz]

Aij = np.kron(Opi, Opj)
p = Opj[0].shape[0]
AAij = np.zeros((p ** 2, p ** 2), dtype=complex)
for i in range(len(Opi)):
    AAij += np.kron(Opi[i], Opj[i])
'''
'''
t_max = 10  # BP maximal iterations
epsilon = 1e-10  # BP convergence error
dumping = 0.0  # BP dumping

d = 2  # virtual bond dimension
p = 2  # physical bond dimension

time_vs_model_size = []


NN = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144] # number of spins
for N in NN:
    L = np.int(np.sqrt(N))

    #------------- generating the finite PEPS structure matrix------------------

    smat, imat = tnf.PEPS_smat_imat_gen(N)
    n, m = smat.shape

    # ------------- generating tensors and bond vectors ---------------------------

    TT, LL = tnf.random_tn_gen(smat, p, d)

    # ------------- generating the double-edge factor graph (defg) of the tensor network ---------------------------

    graph = defg.Graph()
    graph = su.PEPStoDEnFG_transform(graph, TT, LL, smat)
    s = time.time()
    graph.sum_product(t_max, epsilon, dumping)
    e = time.time()
    print(e - s)
    time_vs_model_size.append((e - s) / t_max)


for i in range(len(time_vs_model_size)):
    time_vs_model_size[i] = time_vs_model_size[i] * 35 * NN[i] * 2 * 80 / 60 / 60
plt.figure()
plt.plot(NN, time_vs_model_size, 'o')
plt.xlabel('NxN PEPS')
plt.ylabel('single BP iteration time [hours]')
plt.xticks(NN)
plt.grid()
plt.show()
'''
'''
import pickle
import numpy as np
#favorite_color = [{"lion": "yellow", "kitty": "red"}, np.array([[1, 2, 3], [4, 5, 6]]), [0, 1, '456']]
#pickle.dump(favorite_color, open("save.p", "wb"))


cc = pickle.load(open("save.p", "rb"))
'''

'''
N = 16
# generating the smat (structure matrix) and imat (incidence matrix) of a 2D Square lattice tensor network
    # with open boundary conditions
L = np.int(np.sqrt(N))
imat = np.zeros((N, 2 * N - 2 * np.int(np.sqrt(N))), dtype=int)
smat = np.zeros((N, 2 * N - 2 * np.int(np.sqrt(N))), dtype=int)
n, m = imat.shape
for i in range(n):

    if i < L * (L - 1):
        imat[i, i + L * (L - 1)] = 1
        imat[i + L, i + L * (L - 1)] = 1
    if np.mod(i, L) == 1:
        imat[i, i - np.int(np.floor(np.float(i) / L))] = 1
        imat[i, i - np.int(np.floor(np.float(i) / L)) - 1] = 1
        imat[i - 1, i - np.int(np.floor(np.float(i) / L)) - 1] = 1
        imat[i + 1, i - np.int(np.floor(np.float(i) / L))] = 1

    #imat[i, 2 * i + 1] = 1
    #imat[i, 2 * np.mod(i + 1, L) + 2 * L * np.int(np.floor(np.float(i) / np.float(L)))] = 1
    #imat[i, 2 * np.mod(i + L, N) + 1] = 1

    #smat[i, 2 * i] = 1
    #smat[i, 2 * i + 1] = 2
    #smat[i, 2 * np.mod(i + 1, L) + 2 * L * np.int(np.floor(np.float(i) / np.float(L)))] = 3
    #smat[i, 2 * np.mod(i + L, N) + 1] = 4
'''

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])
sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x


x = np.kron(sx, sx)
y = np.kron(sy, sy)
z = np.kron(sz, sz)
i = np.eye(2)
h01 = np.kron(np.kron(np.kron(sx, sx), i), i) + np.kron(np.kron(np.kron(sy, sy), i), i) + np.kron(np.kron(np.kron(sz, sz), i), i)
h02 = np.kron(np.kron(np.kron(sx, i), sx), i) + np.kron(np.kron(np.kron(sy, i), sy), i) + np.kron(np.kron(np.kron(sz, i), sz), i)
h13 = np.kron(np.kron(np.kron(i, sx), i), sx) + np.kron(np.kron(np.kron(i, sy), i), sy) + np.kron(np.kron(np.kron(i, sz), i), sz)
h23 = np.kron(np.kron(np.kron(i, i), sx), sx) + np.kron(np.kron(np.kron(i, i), sy), sy) + np.kron(np.kron(np.kron(i, i), sz), sz)
H = - h01 - h02 - h13 - h23
lamda = np.linalg.eigvals(H)
print(lamda)

