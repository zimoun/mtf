#!/usr/bin/env python
# coding: utf8

import numpy as np

from assemb import MultiTrace, checker

from time import time
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt

meshname = "./geo/sphere-disjoint.msh"

kRef = 0.3 * np.pi

dd = [
    { 'name': '0',
      'phys': 1,
      'union': [-1, -2, -3],
  },
    { 'name': 'A',
      'phys': 2,
      'union': 1,
  },
    { 'name': 'B',
      'phys': 4,
      'union': 2,
  },
    { 'name': 'C',
      'phys': 16,
      'union': 3,
  }
]

def dir_data(x, normal, dom_ind, result):
    result[0] =  -np.exp( 1j * kRef * x[1])

def neu_data(x, normal, dom_ind, result):
    result[0] = -1j * normal[1] * kRef * np.exp( 1j * kRef * x[1])

mtf = MultiTrace(kRef, meshname, dd, J_is='CSC')
b = mtf.rhs(dir_data, neu_data)

At, X, J, iJ = mtf.tolinop()

shape = mtf.shape

A = 2.0 * At
A2 = A * iJ * A

Ce = 0.5 * J - At
Ci = 0.5 * J + At

Ce2 = Ce * iJ * Ce
Ci2 = Ci * iJ * Ci

print('')
print(mtf.shape, flush=True)
print('')


#################################################
#################################################
#################################################

from krylov import gmres, bicgstab

#################################################
#################################################
#################################################

tol = 1e-6
res = []
restart = None
if restart is None: scale = 1
else: scale = restart

maxiter = int((M.shape[0] / scale) * 0.05)
if maxiter < 50: maxiter = 50

norm_b = la.norm(b)

def rescaleRes(res, P, b):
    scale = 1.0 / la.norm(P(b))
    new_res = scale * res
    return new_res

#################################################

M = A
f = b

print('\nWO restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
tt = time()
xx, info = gmres(M, f,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oRes = np.array(res)
ResA = rescaleRes(oRes, lambda x: x, f)
print('#time: {}'.format(tt))

##############################################

M = iJ * A
f = iJ(b)

print('\nWO restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
tt = time()
xx, info = gmres(M, f,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oRes = np.array(res)
ResiJA = rescaleRes(oRes, lambda x: x, f)
print('#time: {}'.format(tt))

##############################################

iA = iJ * A * iJ

M = iA * A
f = iA(b)

print('\nWO restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
tt = time()
xx, info = gmres(M, f,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oRes = np.array(res)
ResiAA = rescaleRes(oRes, lambda x: x, f)
print('#time: {}'.format(tt))


def Res2Tuple(res):
    return np.arange(len(res)), res

its, res = Res2Tuple(ResA)
plt.semilogy(its, res, 'k-', linewidth=3,  label='A')

its, res = Res2Tuple(ResiJA)
plt.semilogy(its, res, 'g-', linewidth=3,  label='iJ A')

its, res = Res2Tuple(ResiAA)
plt.semilogy(its, res, 'b-', linewidth=3,  label='iA A')

plt.title('Convergence History', fontsize=20)
plt.xlabel('#iterations', fontsize=14)
plt.ylabel('normalized residual', fontsize=30)
plt.legend()

plt.grid(True)

plt.show()
