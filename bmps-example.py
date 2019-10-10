#!/usr/bin/python3
#
# ======================================================================
#
#  bmps-example.py
# -----------------
# 
#  An example for using the bmpslib.py (boundary MPS) library. This
#  program creates a small random MxN PEPS, calculates its local RDM
#  exactly and then compares it to the calculation that is done by 
#  the bmpslib
#


import numpy as np
import scipy as sp

from numpy.linalg import norm
from numpy import zeros, ones, tensordot, sqrt, array, trace, \
    conj, identity

import ncon
import bmpslib
import Heisenberg_model_function as hmf
import BPupdate_PEPS_smart_trancation as BP
import Tensor_Network_functions as tnf





#------------------------------------------------------------------
#
#  Create a random MxN PEPS with bond dimension D.
#

def random_PEPS(M, N, D):
    
    p = bmpslib.peps(M,N)

    for i in range(M):
        for j in range(N):
            
            Dleft=D
            Dright=D
            Dup=D
            Ddown=D

            #
            # When we are at the boundaries, the bond-dimension should
            # be 1.
            #
            
            if j==0:
                Dleft = 1
            if j==N-1:
                Dright=1
            if i==0:
                Dup=1
            if i==M-1:
                Ddown=1
                
            A = np.random.normal(size=[d,Dleft,Dright,Dup,Ddown])
            
            p.set_site(A, i,j)
            
    return p
            



#------------------------------------------------------------------
#
#  Given a PEPS p, calculate its state as a big tensor 
#  psi[i_0,i_1,...,i_{MN-1}] - this is done for calculating the 
#  RDM by exact contraction.
#

def get_psi(p):
    
    M = p.M
    N = p.N

    # Create a list of tensors to be send to ncon.
    AA = []
    
    for i in range(M):
        for j in range(N):
            
            A = p.A[i][j]
            
            # A = d, left, right, up, down
            
            if i==0:
                
                A=A[:,:,:,0,:] # A=d, left,right, down
                
                if j==0:
                    A = A[:,0,:,:] # A=d,right, down
                elif j==N-1:
                    A = A[:,:,0,:] # A=d, left, down
                    
            elif i==M-1:
                A=A[:,:,:,:,0] # A=d, left,right, up
                
                if j==0:
                    A = A[:,0,:,:] # A=d,right, up
                elif j==N-1:
                    A = A[:,:,0,:] # A=d, left, up
                    
            else:
                if j==0:
                    A = A[:,0,:,:,:] # A=d,right, up, down
                elif j==N-1:
                    A = A[:,:,0,:,:] # A=d, left, up, down
                    
            AA.append(A)
            
            
    t=0
    v = [None]*M*N
    
    KR = 1
    KD = KR + M*N + 10
    
    for i in range(M):
        for j in range(N):
            
            ed = -t-1
            eR = KR + t
            eL = eR - 1
            eD = KD + t
            eU = eD - N
            
            v[t] = [ed,eL,eR,eU,eD]
            
            if i==0:
                
                v[t] = [ed, eL, eR, eD]
                
                
                if j==0:
                    v[t] = [ed, eR,eD]
                elif j==N-1:
                    v[t] = [ed,eL,eD]
                    
            elif i==M-1:
                v[t] = [ed, eL, eR, eU]
                
                if j==0:
                    v[t] = [ed,eR,eU]
                elif j==N-1:
                    v[t] = [ed,eL,eU]
                    
            else:
                if j==0:
                    v[t] = [ed,eR,eU,eD]
                elif j==N-1:
                    v[t] = [ed,eL,eU,eD]
                    
                    
            t += 1
            
    psi = ncon.ncon(AA,v)
 
            
    return psi
                
                
#-----------------------------------------------------------------------
#
# Returns the 2-body interaction of the anti-ferromagnetic Heizenberg
# model
#
     
def Heizenberg_AFM():
    
    
    X = array([[0,1.0],[1,0]])
    
    Y = array([[0,-1.0],[1,0]])  # we omit the i because it will not
                                 # appear in the end and we want to 
                                 # keep everything real (not complex)
    Z = array([[1,0],[0,-1]])
    
    h = 0.25*(tensordot(X,X,0) - tensordot(Y,Y,0) + tensordot(Z,Z,0))
    
    return h



#-----------------------------------------------------------------------
#
# Calculates the trace distance between two 2-body RDMs
# 
     
def trace_distance(rhoA, rhoB):
    
    d=rhoA.shape[0]
    
    rhoA = rhoA.transpose([0,2,1,3])
    rhoA = rhoA.reshape([d*d, d*d])

    rhoB = rhoB.transpose([0,2,1,3])
    rhoB = rhoB.reshape([d*d, d*d])
    
    Drho = 0.5*(rhoA-rhoB)
    eigs = np.linalg.eigvalsh(Drho)
    
    TD = sum(abs(eigs))
    
    
    return TD



#-----------------------------------------------------------------------
#
# Calculates the 2-body RDM from the state psi
# 
     
def direct_2RDM(psi, i, j):
    
    n = len(psi.shape)
    
    dims=[2**i, 2, 2**(j-i-1), 2, 2**(n-j-1)]
        
    psi1 = psi.reshape(dims)
    
    rho2 = tensordot(psi1, conj(psi1), axes=([0,2,4],[0,2,4]))
    
    rho2 = rho2.transpose([0,2,1,3])
    
    rho2 = rho2/trace(trace(rho2))
    
    return rho2
    
    
    
    
    
    












##########################       M A I N      ##########################






np.random.seed(3)

M=3    # how many rows
N=2    # how many columns
d=2    # physical dimension
D=2    # Bond dimension

Dp=200 # boundary MPS maximal dimension (usually Dp ~ 2*D^2 is good)

#
# creat the TN using Roy's BP_truncation
#

env_size = [0]
bc = 'open'
dE = 1e-5
t_max = 200
dumping = 0.2
epsilon = 1e-5
mu = -1
sigma = 0

pauli_z = np.array([[1, 0], [0, -1]])
pauli_y = np.array([[0, -1], [1, 0]])
pauli_x = np.array([[0, 1], [1, 0]])
sz = 0.5 * pauli_z
sy = 0.5 * pauli_y
sx = 0.5 * pauli_x
t_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]  # imaginary time evolution time steps list
iterations = 100
Opi = [sx, sy, sz]
Opj = [sx, sy, sz]
hij = np.zeros((d ** 2, d ** 2))

for i in range(3):
    hij += np.kron(Opi[i], Opj[i])
hij = hij.reshape(d, d, d, d)

Jk = np.random.normal(mu, sigma, np.int((M - 1) * N + (N - 1) * M)) # interaction constant list
data = hmf.Heisenberg_PEPS_BP(M, N, Jk, dE, D, t_max, epsilon, dumping, bc, env_size)
TT, LL, smat, TT_gpeps, LL_gpeps = data[4], data[5], data[6], data[7], data[8]
TT_prime_gpeps = BP.absorb_all_bond_vectors(TT_gpeps, LL_gpeps, smat)
TT_prime = BP.absorb_all_bond_vectors(TT, LL, smat)
TT_prime_gpeps = tnf.PEPS_OBC_broadcast_to_Itai(TT_prime_gpeps, [M, N], d, D)
TT_prime = tnf.PEPS_OBC_broadcast_to_Itai(TT_prime, [M, N], d, D)

p_roy = bmpslib.peps(M,N)
for t, T in enumerate(TT_prime):
    i, j = np.unravel_index(t, [M, N])
    p_roy.set_site(T, i, j)

p_roy_gpeps = bmpslib.peps(M,N)
for t, T in enumerate(TT_prime_gpeps):
    i, j = np.unravel_index(t, [M, N])
    p_roy_gpeps.set_site(T, i, j)
#
# Creating a random NxM PEPS with physical dimension d and bond
# dimension D
#

p = random_PEPS(M,N,D)


#
# Calculate the 2-local RDMs using the boundary-MPS method
#        


print("1. Calculate the 2-body RDMs using the boundary MPS method " \
      "with Dp={}\n".format(Dp))

rhoLA = bmpslib.calculate_PEPS_2RDM(p, Dp)


rhoLA_roy = bmpslib.calculate_PEPS_2RDM(p_roy, Dp)


rhoLA_roy_gpeps = bmpslib.calculate_PEPS_2RDM(p_roy_gpeps, Dp)

print("\n2. Calculate the 2-body RDMs directly by full contraction\n")

rhoLB = []
psi=get_psi(p)

for i in range(M):
    for j in range(N-1):
        t = i*N+j
        rhoLB.append(direct_2RDM(psi,t,t+1))

        
for j in range(N):
    for i in range(M-1):
        t = i*N+j
        rhoLB.append(direct_2RDM(psi,t,t+N))



#
# Calculate the average trace distance
#

s=0
s_roy = 0
s_roy_gpeps = 0
for i in range(len(rhoLA)):
    TD = trace_distance(rhoLA[i], rhoLB[i])
    TD_roy = trace_distance(rhoLA_roy[i], rhoLB[i])
    TD_roy_gpeps = trace_distance(rhoLA_roy_gpeps[i], rhoLB[i])
    s = s+TD
    s_roy += TD_roy
    s_roy_gpeps += TD_roy_gpeps
    
TD = s/len(rhoLA)
TD_roy = s_roy / len(rhoLA_roy)
TD_roy_gpeps = s_roy_gpeps / len(rhoLA_roy_gpeps)
print("Overall, calculated {} 2-local RDMS. Average trace distance: "\
    "{:}".format(len(rhoLA), TD))

print("Overall, calculated {} 2-local RDMS using roy's BP_truncation. Average trace distance: "\
    "{:}".format(len(rhoLA_roy), TD_roy))

print("Overall, calculated {} 2-local RDMS using roy's gPEPS. Average trace distance: "\
    "{:}".format(len(rhoLA_roy_gpeps), TD_roy_gpeps))


# comparing energies

ELB = 0
ELA = 0
ELA_roy = 0
for i in range(len(rhoLA)):
    ELA += np.einsum(rhoLA[i], [0, 1, 2, 3], hij,  [0, 1, 2, 3])
    ELA_roy += np.einsum(rhoLA_roy[i], [0, 1, 2, 3], hij, [0, 1, 2, 3])
    ELB += np.einsum(rhoLB[i], [0, 1, 2, 3], hij, [0, 1, 2, 3])
ELA /= M * N
ELA_roy /= M * N
ELB /= M * N

print('ELB = ', ELB)
print('ELA = ', ELA)
print('ELA roy = ', ELA_roy)






