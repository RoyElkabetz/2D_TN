from numba import jit
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import copy as cp

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
