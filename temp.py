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
Ek = 15
n, m = 4, 4
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
'''
p, d = 2, 3
n, m = 3, 2
smat, imat = tnf.PEPS_OBC_smat_imat(n, m)
TT, LL = tnf.PEPS_OBC_random_tn_gen(smat, p, d)
TT_new = BP.absorb_all_bond_vectors(TT, LL, smat)
TT_prime = tnf.PEPS_OBC_broadcast_to_Itai(TT_new, [n, m], p, d)
for t, T in enumerate(TT_prime):
    print(np.max(np.abs(TT_prime[t] - TT_prime_tilde[t])))
'''
def quicksort(arr, lo, hi):
    if lo < hi:
        pi = partition(arr, lo, hi)
        quicksort(arr, lo, pi - 1)
        quicksort(arr, pi + 1, hi)

def partition(arr, lo, hi):
    i = lo - 1
    p = arr[hi]
    for j in range(lo, hi):
        if arr[j] < p:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1

a = [6, 5, 77, 34, 2, 91, 8, 4, 1, 4, 0, 2, 45]
quicksort(a, 0, len(a) - 1)
print(a)























def quicksort2(arr, lo, hi):
    if lo < hi:
        pi = partition2(arr, lo, hi)
        quicksort2(arr, lo, pi - 1)
        quicksort2(arr, pi + 1, hi)

def partition2(arr, lo, hi):
    pivot = arr[hi]
    i = lo - 1
    for j in range(lo, hi):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1



