from numba import jit
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import copy as cp
import Tensor_Network_functions as tnf
import virtual_DEFG as defg
#import BPupdate_PEPS_smart_trancation as su

Ek = 102
n, m = 10, 10
env_size = 4
smat, imat = tnf.PEPS_OBC_smat_imat(n, m)
emat = tnf.PEPS_OBC_edge_rect_env(Ek, smat, [n, m], env_size)
inside, outside = tnf.PEPS_OBC_divide_edge_regions(emat, smat)
omat = np.where(emat > -1, np.arange(n * m).reshape(n, m), -1)
tensors = omat[np.nonzero(emat > -1)]
sub_omat = tnf.PEPS_OBC_edge_environment_sub_order_matrix(emat)
n, m = sub_omat.shape
if n < m:
    a = np.transpose(sub_omat)