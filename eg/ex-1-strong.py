#!/usr/bin/env python
# coding: utf8

import numpy as np

from assemb import MultiTrace, checker

from time import time
import scipy.linalg as la

meshname = "./geo/sphere-disjoint.msh"

kRef = 0.5 * np.pi

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


mtf = MultiTrace(kRef, meshname, dd)

At, X, J, iJ = mtf.tolinop()

shape = mtf.shape

A = 2.0 * At
A2 = A * iJ * A

Ce = 0.5 * J - At
Ci = 0.5 * J + At

Ce2 = Ce * iJ * Ce
Ci2 = Ci * iJ * Ci

x = np.random.rand(shape[0])

checker('A2 = J', A2, J, x)
checker('exterior Proj.', Ce2, Ce, x)
checker('interior Proj.', Ci2, Ci, x)
checker('error-Calderon with random [no-sense]', A, J, x)

def dir_data(x, normal, dom_ind, result):
    result[0] =  -np.exp( 1j * kRef * x[1])

def neu_data(x, normal, dom_ind, result):
    result[0] = -1j * normal[1] * kRef * np.exp( 1j * kRef * x[1])

b = mtf.rhs(dir_data, neu_data)
M = A - X

print('')
print(mtf.shape, flush=True)
print('')

#################################################
#################################################
#################################################

Ats = iJ * At
As = iJ * A
Xs = iJ * X

Ms = As - Xs
bs = iJ(b)

#################################################
#################################################
#################################################

from krylov import gmres, bicgstab
from scipy.sparse.linalg import LinearOperator

#################################################
#################################################
#################################################

iA = iJ * A * iJ
iAs = As

#################################################

Pjac = iA
Pjacs = iAs

E = mtf.upper()
Es = iJ *  E

Pgs = iA + iA * E * iA
Pgss = iAs + iAs * Es * iAs

CCi = iJ * (0.5 * J + At)
CCe = (0.5 * J - At)

B = J - X
BD = B * CCi + CCe

I = LinearOperator(shape=shape, matvec=lambda x: x, dtype=complex)
CCis = (0.5 * I + Ats)
CCes = (0.5 * I - Ats)

Bs = I - Xs
BDs = Bs * CCis + CCes

#################################################
#################################################
#################################################

tol = 1e-6
res = []
restart = 20
if restart is None: scale = 1
else: scale = restart

maxiter = int((M.shape[0] / scale) * 0.05)
if maxiter < 50: maxiter = 50

norm_b = la.norm(b)

#################################################
#################################################
#################################################

def rescaleRes(res, P, b):
    scale = 1.0 / la.norm(P(b))
    new_res = scale * res
    return new_res

#################################################

print('\nWO restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
tt = time()
xx, info = gmres(J, b,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResWO = np.array(res)
ResWO_mass = rescaleRes(oResWO, lambda x: x, b)
print('#time: {}'.format(tt))

checker('Calderon WO', A, J, xx)

#################################################

print('\nWO restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
tt = time()
xwo, info = gmres(M, b,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResWO = np.array(res)
ResWO = rescaleRes(oResWO, lambda x: x, b)
print('#time: {}'.format(tt))


checker('Calderon WO', A, J, xwo)

#################################################

print('\nWO restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
tt = time()
xx, info = gmres(Ms, bs,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResWOs = np.array(res)
ResWOs = rescaleRes(oResWOs, lambda x: x, bs)
print('#time: {}'.format(tt))

checker('Calderon WO', A, J, xx)

#################################################

print('\nJac restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
tt = time()
xx, info = gmres(M, b,
                 M = Pjac,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResJac = np.array(res)
ResJac = rescaleRes(oResJac, Pjac, b)
print('#time: {}'.format(tt))

checker('Calderon Jac', A, J, xx)

#################################################

#################################################

print('\nJac restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
tt = time()
xx, info = gmres(Ms, bs,
                 M = Pjacs,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResJacs = np.array(res)
ResJacs = rescaleRes(oResJacs, Pjacs, bs)
print('#time: {}'.format(tt))

checker('Calderon Jac', A, J, xx)

#################################################
#################################################

print('\nGS restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
tt = time()
xx, info = gmres(M, b,
                 M = Pgs,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResGS = np.array(res)
ResGS = rescaleRes(oResGS, Pgs, b)
print('#time: {}'.format(tt))

checker('Calderon GS', A, J, xx)

#################################################

#################################################

print('\nGS restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
tt = time()
xx, info = gmres(Ms, bs,
                 M = Pgss,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResGSs = np.array(res)
ResGSs = rescaleRes(oResGSs, Pgss, bs)
print('#time: {}'.format(tt))

checker('Calderon GS', A, J, xx)

#################################################

print('\nBD WO restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
Mat, rhs = BD,  b
tt = time()
xx, info = gmres(Mat, rhs,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
xx = CCi(xx)
tt = time() - tt
print(info, len(res))
oResBDWO = np.array(res)
ResBDWO = rescaleRes(oResBDWO, lambda x: x, rhs)
print('#time: {}'.format(tt))

checker('BD WO', A, J, xx)

#################################################

print('\nBD WO restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
Mat, rhs = BDs,  bs
tt = time()
xx, info = gmres(Mat, rhs,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
xx = CCi(xx)
tt = time() - tt
print(info, len(res))
oResBDWO = np.array(res)
ResBDWOs = rescaleRes(oResBDWO, lambda x: x, rhs)
print('#time: {}'.format(tt))

checker('BD WO', A, J, xx)

####################################
####################################

import matplotlib.pyplot as plt

def Res2Tuple(res):
    return np.arange(len(res)), res

its, res = Res2Tuple(ResWO_mass)
plt.semilogy(its, res, 'ko', linewidth=3,  label='Mass')

its, res = Res2Tuple(ResWO)
plt.semilogy(its, res, 'k-', linewidth=3,  label='wo')

its, res = Res2Tuple(ResWOs)
plt.semilogy(its, res, 'k--', linewidth=3,  label='wo')

its, res = Res2Tuple(ResJac)
plt.semilogy(its, res, 'b-', linewidth=3,  label='Jacobi')

its, res = Res2Tuple(ResJacs)
plt.semilogy(its, res, 'b--', linewidth=3,  label='Jacobi')

its, res = Res2Tuple(ResGS)
plt.semilogy(its, res, 'r-', linewidth=3,  label='GS')

its, res = Res2Tuple(ResGSs)
plt.semilogy(its, res, 'r--', linewidth=3,  label='GS')

its, res = Res2Tuple(ResBDWO)
plt.semilogy(its, res, 'm-', linewidth=3,  label='Bott-Duffin')

its, res = Res2Tuple(ResBDWOs)
plt.semilogy(its, res, 'm--', linewidth=3,  label='Bott-Duffin')

plt.title('Convergence History', fontsize=20)
plt.xlabel('#iterations', fontsize=14)
plt.ylabel('normalized residual', fontsize=30)
plt.legend()

plt.grid(True)

plt.show()
