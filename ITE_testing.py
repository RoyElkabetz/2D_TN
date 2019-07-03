import numpy as np
import simple_update_algorithm2 as su
import copy as cp
import ncon

error = 0
R = np.random.rand(2, 4, 16)
L = np.random.rand(2, 4, 16)
lamda = np.random.rand(4)

A = np.einsum(R, [0, 1, 2], np.diag(lamda), [1, 3], [0, 3, 2])
A = np.einsum(A, [0, 1, 2], L, [3, 1, 4], [2, 0, 3, 4])

eye = np.reshape(np.eye(4), (2, 2, 2, 2))

B = np.einsum(A, [0, 1, 2, 3], eye, [1, 2, 4, 5], [0, 4, 5, 3])

tensor_list = [R, L, np.diag(lamda), eye]
indices = ([3, 1, -1], [4, 2, -3], [1, 2], [3, 4, -2, -4])
order = [1, 2, 3, 4]
forder = [-1, -2, -4, -3]
theta2 = ncon.ncon(tensor_list, indices, order, forder)


error = np.max(np.abs(A - B))
error2 = np.max(np.abs(A - theta2))

print(error)