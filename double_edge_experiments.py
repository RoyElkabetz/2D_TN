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




