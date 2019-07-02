import numpy as np
import simple_update_algorithm2 as su
import copy as cp
import ncon

error = 0
for i in range(100):
    R = np.random.rand(2, 4, 16)
    L = np.random.rand(2, 4, 16)
    lamda = np.random.rand(4)
    uij = np.reshape(np.eye(4), (2, 2, 2, 2))

    theta1 = su.imaginary_time_evolution(R, L, lamda, uij)


    tensor_list = [R, L, np.diag(lamda), uij]
    indices = ([3, 1, -1], [4, 2, -3], [1, 2], [3, -2, 4, -4])
    order = [1, 2, 3, 4]
    forder = [-2, -1, -4, -3]
    theta2 = ncon.ncon(tensor_list, indices, order, forder)


    Rt, lamdat, Lt = su.svd(theta2, [0, 1], [2, 3], keep_s='yes')
    Rt = np.reshape(Rt, (2, 32, 16))
    #Lt = np.transpose(Lt)
    Lt = np.reshape(Lt, (2, 32, 16))

    A = np.einsum(R, [0, 1, 2], np.diag(lamda), [1, 3], [0, 3, 2])
    A = np.einsum(A, [0, 1, 2], L, [3, 1, 4], [0, 2, 3, 4])

    B = np.einsum(Rt, [0, 1, 2], np.diag(lamdat), [1, 3], [0, 3, 2])
    B = np.einsum(B, [0, 1, 2], Lt, [3, 1, 4], [0, 2, 3, 4])

    error += np.max(np.abs(A - B))
error /= 100
print(error)