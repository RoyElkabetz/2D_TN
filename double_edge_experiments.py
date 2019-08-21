
import numpy as np
import DEnFG as fg
import copy as cp
import matplotlib.pyplot as plt
#import pylustrator
#pylustrator.start()
'''

# ---------------------------------- 1D DEnFG PBC-----------------------------------
fac = np.array([[1, 0, 0, 3], [2, 0, 1, 0]])
# parameters
n = 3
alphabet = 4
d = 2
t_max = 200
epsilon = 1e-5

# saving data
node_marginals = np.zeros((alphabet, n, t_max))

# generate the graph
g = fg.Graph()

# add physical nodes
for i in range(n):
    g.add_node(alphabet, 'n' + str(i))

# add virtual nodes
for i in range(n):
    g.add_node(d, 'n' + str(g.node_count))

# add factors
for i in range(1, n):
    neighbors = {'n' + str(n + i - 1): 0, 'n' + str(i): 1, 'n' + str(n + i): 2}
    g.add_factor(neighbors, np.reshape(0.5 * np.eye(alphabet), (d, alphabet, d)))

# PBC
neighbors = {'n' + str(2 * n - 1): 0, 'n0': 1, 'n' + str(n): 2}
g.add_factor(neighbors, np.reshape(0.5 * np.eye(alphabet), (d, alphabet, d)))

# run BP
for t in range(1, t_max):
    g.sum_product(t, epsilon, 0.4)
    g.calc_node_belief()
    for i in range(n):
        node_marginals[:, i, t] = np.linalg.eigvals(g.node_belief['n' + str(i)])

plt.figure()
for i in range(n):
    plt.plot(list(range(t_max)), node_marginals[0, i, :], 'o')
    plt.plot(list(range(t_max)), node_marginals[1, i, :], 'v')
    plt.plot(list(range(t_max)), node_marginals[2, i, :], 's')
    plt.plot(list(range(t_max)), node_marginals[3, i, :], '.-')

plt.ylim([0, 1])
plt.grid()
plt.show()
'''

# ---------------------------------- 1D DEnFG Open BC-----------------------------------
# parameters
fac1 = np.array([[0.5, 0], [0, 0.5]])
fac2 = np.array([[0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5]])
n = 3
alphabet = 2
d = 2
t_max = 20
epsilon = 1e-5
dumping = 0.2

# saving data
node_marginals = np.zeros((alphabet, n, t_max))

# generate the graph
g = fg.Graph()

# add physical nodes
for i in range(n):
    g.add_node(alphabet, 'n' + str(i))

# add virtual nodes
for i in range(n):
    g.add_node(d, 'n' + str(g.node_count))

neighbors_left = {'n0': 0, 'n' + str(n): 1}
g.add_factor(neighbors_left, cp.copy(fac1))

# add factors
for i in range(1, n - 1):
    neighbors = {'n' + str(n + i - 1): 0, 'n' + str(i): 1, 'n' + str(n + i): 2}
    g.add_factor(neighbors, np.reshape(cp.copy(fac2), (d, alphabet, d)))

# OBC

neighbors_rigth = {'n' + str(n - 1): 1, 'n' + str(2 * n - 2): 0}
g.add_factor(neighbors_rigth, cp.copy(fac1))

# run BP
for t in range(1, t_max):
    g.sum_product(t, epsilon, dumping)
    g.calc_node_belief()
    for i in range(n):
        node_marginals[:, i, t] = np.linalg.eigvals(g.node_belief['n' + str(i)])

p, p_dic, p_order = g.exact_joint_probability()
n0_marginal = g.ni_ni_star_marginal(p, p_dic, p_order, 'n0', 'n0*')
n2_marginal = g.ni_ni_star_marginal(p, p_dic, p_order, 'n2', 'n2*')

plt.figure()

plt.plot(list(range(t_max)), node_marginals[0, 0, :], 'o')
plt.plot(list(range(t_max)), node_marginals[1, 0, :], 'v')
plt.plot(list(range(t_max)), node_marginals[0, 2, :], 'o')
plt.plot(list(range(t_max)), node_marginals[1, 2, :], 'v')

plt.ylim([0, 1])
plt.legend(['n0[0]', 'n0[1]', 'n2[0]', 'n2[1]'])
plt.grid()
plt.show()




'''
import numpy as np
import copy as cp
import BPupdate as su
from scipy import linalg
import matplotlib.pyplot as plt
import ncon_lists_generator as nlg
import ncon
import DEnFG as fg


#---------------------- Tensor Network paramas ------------------

N = 4 # number of spins
L = np.int(np.sqrt(N))

t_max = 40
epsilon = 1e-15
dumping = 0.8

d = 2  # virtual bond dimension
p = 2  # physical bond dimension
D_max = 2  # maximal virtual bond dimension



#------------- generating the finite PEPS structure matrix------------------
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


TT = []
for ii in range(n):
    TT.append(np.arange(np.float(p * (d ** 4))).reshape(p, d, d, d, d))
LL = []
for i in range(imat.shape[1]):
    LL.append(np.ones(d, dtype=float) / d)



BP_in_time = np.zeros((p, N, t_max))
for t in range(1, t_max):
    graph = fg.Graph()
    graph = su.PEPStoDEnFG_transform(graph, cp.deepcopy(TT), cp.deepcopy(LL), smat)
    graph.sum_product(t, epsilon, dumping)
    graph.calc_node_belief()
    for n in range(N):
        BP_in_time[:, n, t] = np.linalg.eigvals(graph.node_belief['n' + str(len(LL) + n)])

plt.figure()
for n in range(N):
    plt.plot(range(t_max), BP_in_time[0, n, :], '^')
    plt.plot(range(t_max), BP_in_time[1, n, :], 'o')
plt.grid()
plt.show()
'''