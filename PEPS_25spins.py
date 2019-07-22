import numpy as np
import copy as cp
import gPEPS as su
from scipy import linalg
import matplotlib.pyplot as plt
import DEnFG as fg
import ncon_lists_generator as nlg
import ncon

#----------------------------- Simple Update Calculations --------------------------------
N = 4 # number of spins
L = np.int(np.sqrt(N))

d = 3
p = 2
D_max = d
J = 1.

h = np.linspace(0., 4., num=50)
time_to_converge = np.zeros((len(h)))

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0, 1], [1, 0]])

sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x

TT = []
LL = []
energy = []


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

    TT.append(np.random.rand(p, d, d, d, d))

for i in range(m):
    LL.append(np.ones(d, dtype=float) / d)

'''
plt.figure()
plt.matshow(smat)
plt.figure()
plt.matshow(imat)
'''



t_list = [0.1, 0.01, 0.001]
iterations = 30
hij = -J * np.kron(pauli_z, pauli_z)
hij_energy_operator = np.reshape(cp.deepcopy(hij), (p, p, p, p))
hij = np.reshape(hij, [p ** 2, p ** 2])
unitary = [np.reshape(linalg.expm(-t_list[t] * hij), [p, p, p, p]) for t in range(len(t_list))]
counter = 0


for i in range(len(t_list)):
    flag = 0
    for j in range(iterations):
        print('i, j = ', i, j)
        TT1, LL1 = su.simple_update(cp.deepcopy(TT), cp.deepcopy(LL), unitary[i], imat, smat, D_max)
        TT2, LL2 = su.simple_update(cp.deepcopy(TT1), cp.deepcopy(LL1), unitary[i], imat, smat, D_max)

        energy1 = su.energy_per_site(TT1, LL1, imat, smat, hij_energy_operator)
        energy2 = su.energy_per_site(TT2, LL2, imat, smat, hij_energy_operator)
        energy.append(su.energy_per_site(TT2, LL2, imat, smat, hij_energy_operator))
        counter += 1
        if np.abs(energy1 - energy2) < 1e-10:
            flag = 1
            break
        else:
            TT = cp.deepcopy(TT2)
            LL = cp.deepcopy(LL2)
    if flag:
        flag = 0
        break

mz_matrix_exact = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        print('i, j = ', i, j)
        T_list, idx_list = nlg.ncon_list_generator(TT2, LL2, smat, pauli_z, np.int(L * i + j))
        T_list_n, idx_list_n = nlg.ncon_list_generator(TT2, LL2, smat, np.eye(p), np.int(L * i + j))
        mz_matrix_exact[i, j] = ncon.ncon(T_list, idx_list) / ncon.ncon(T_list_n, idx_list_n)

plt.figure()
plt.title('energy values')
plt.xlabel('t')
plt.plot(range(counter), energy[:counter], 'o')
plt.grid()
plt.show()

#----------------------------- Double-Edge Calculations --------------------------------
sqrt_LL = []
for i in range(len(LL)):
    sqrt_LL.append(np.sqrt(LL[i]))

# Graph initialization
graph = fg.Graph()

# Adding virtual nodes
for i in range(m):
    graph.add_node(D_max, 'n' + str(graph.node_count))

# Adding physical nodes
for i in range(n):
    graph.add_node(p, 'n' + str(graph.node_count))

# Adding factors
for i in range(n):
    neighbor_nodes = {}
    edges = np.nonzero(smat[i, :])[0]
    legs = smat[i, edges]
    neighbor_nodes['n' + str(i + 2 * N)] = 0
    for j in range(len(edges)):
        neighbor_nodes['n' + str(legs[j])] = legs[j]

    graph.add_factor(neighbor_nodes, cp.deepcopy(TT[i]) / np.max(TT[i]))

# BP parameters
t_max = 200
graph.sum_product(t_max)
graph.calc_node_belief()

# Calculating expectations
mz_matrix_graph = np.zeros((L, L))
mz_matrix_TN = np.zeros((L, L))
k = 0
for i in range(L):
    for j in range(L):
        print('i, j = ', i, j)
        mz_matrix_TN[i, j] = su.single_tensor_expectation(np.int(L * i + j), TT, LL, imat, smat, pauli_z)
        mz_matrix_graph[i, j] = np.real(np.trace(np.matmul(pauli_z, graph.node_belief['n' + str(2 * N + k)])))
        T_list, idx_list = nlg.ncon_list_generator(TT2, LL2, smat, pauli_z, np.int(L * i + j))
        T_list_n, idx_list_n = nlg.ncon_list_generator(TT2, LL2, smat, np.eye(p), np.int(L * i + j))
        mz_matrix_exact[i, j] = ncon.ncon(T_list, idx_list) / ncon.ncon(T_list_n, idx_list_n)
        k += 1


print('error matrix = ', np.abs(mz_matrix_TN - mz_matrix_graph))
#for i in range(n):
#    print(np.linalg.eigvals(np.real(graph.node_belief['n' + str(2 * N + i)])))