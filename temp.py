from numba import jit
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import copy as cp
import Tensor_Network_functions as tnf
import virtual_DEFG as defg
import BPupdate_PEPS_smart_trancation as su

Ek = 1
n, m = 5, 5
smat, imat = tnf.PEPS_OBC_smat_imat(n, m)
emat = tnf.PEPS_OBC_edge_rect_env(Ek, smat, [n, m])
inside, outside = tnf.PEPS_OBC_divide_edge_regions(Ek, emat, smat)
omat = np.where(emat > -1, np.arange(n * m).reshape(n, m), -1)

#count = list(edges[0]).count(edges[0])
