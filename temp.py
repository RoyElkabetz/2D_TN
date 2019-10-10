from numba import jit
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import copy as cp
import Tensor_Network_functions as tnf
import virtual_DEFG as defg
import BPupdate_PEPS_smart_trancation as BP

'''
Ek = 10
n, m = 3, 3
env_size = 1
smat, imat = tnf.PEPS_OBC_smat_imat(n, m)
emat = tnf.PEPS_OBC_edge_rect_env(Ek, smat, [n, m], env_size)
inside, outside = tnf.PEPS_OBC_divide_edge_regions(emat, smat)
omat = np.where(emat > -1, np.arange(n * m).reshape(n, m), -1)
tensors = omat[np.nonzero(emat > -1)]
sub_omat = tnf.PEPS_OBC_edge_environment_sub_order_matrix(emat)
n, m = sub_omat.shape
if n < m:
    a = np.transpose(sub_omat)
'''
p, d = 2, 3
n, m = 3, 2
smat, imat = tnf.PEPS_OBC_smat_imat(n, m)
TT, LL = tnf.PEPS_OBC_random_tn_gen(smat, p, d)
TT_new = BP.absorb_all_bond_vectors(TT, LL, smat)
TT_prime = tnf.PEPS_OBC_broadcast_to_Itai(TT_new, [n, m], p, d)
for t, T in enumerate(TT_prime):
    print(np.max(np.abs(TT_prime[t] - TT_prime_tilde[t])))

