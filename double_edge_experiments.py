import numpy as np
import copy as cp
import BPupdate as su
from scipy import linalg
import matplotlib.pyplot as plt
import ncon_lists_generator as nlg
import ncon
import DEnFG as fg

'''
import numpy as np
import DEnFG as fg
import copy as cp
import matplotlib.pyplot as plt
#import pylustrator
#pylustrator.start()

# parameters
n = 10
alphabet = 2
d = 3
t_max = 40
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
    g.add_factor(neighbors, np.arange(np.float(d * d * alphabet)).reshape(d, alphabet, d))

# PBC
neighbors = {'n' + str(2 * n - 1): 0, 'n0': 1, 'n' + str(n): 2}
g.add_factor(neighbors, np.random.rand(d, alphabet, d))

# run BP
for t in range(1, t_max):
    g.sum_product(t, epsilon, 0.4)
    g.calc_node_belief()
    for i in range(n):
        node_marginals[:, i, t] = np.linalg.eigvals(g.node_belief['n' + str(i)])

plt.figure()
plt.plot(list(range(t_max)), node_marginals[0, 0, :], 'bo')
plt.plot(list(range(t_max)), node_marginals[1, 0, :], 'rv')
plt.ylim([0, 1])
plt.grid()
plt.show()
'''




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
