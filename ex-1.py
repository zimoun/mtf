#!/usr/bin/env python
# coding: utf8

import numpy as np

from assemb import MultiTrace, checker

from time import time
import scipy.linalg as la

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
      'phys': 3,
      'union': 2,
  },
    { 'name': 'C',
      'phys': 4,
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

x = np.random.rand(shape[0]) + 1j * np.random.rand(shape[0])
xr = np.random.rand(shape[0])

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

from krylov import gmres, bicgstab

#################################################
#################################################
#################################################

iA = iJ * A * iJ

#################################################

Pjac = iA

E = mtf.upper()
Pgs = iA + iA * E * iA

Msigma = lambda sigma: (A - J) + sigma * (J - X)

CCi = iJ * (0.5 * J + At)
CCe = (0.5 * J - At)

B = J - X
BD = B * CCi + CCe
F = B * CCi

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
xx, info = gmres(M, b,
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

checker('Calderon WO', A, J, xx)

#################################################

print('\nWO bicgstab maxiter={0}'.format(maxiter), flush=True)
del res
res = []
tt = time()
xx, info = bicgstab(M, b,
                 tol=tol,
                 residuals=res,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResWOcg = np.array(res)
ResWOcg = rescaleRes(oResWOcg, lambda x: x, b)
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

print('\nMass restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
tt = time()
xx, info = gmres(M, b,
                 M = iJ,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResiJ = np.array(res)
ResiJ = rescaleRes(oResiJ, iJ, b)
print('#time: {}'.format(tt))

checker('Calderon Mass', A, J, xx)

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

print('\nGS bicgstab maxiter={0}'.format(maxiter), flush=True)
del res
res = []
tt = time()
xx, info = bicgstab(M, b,
                 M = Pgs,
                 tol=tol,
                 residuals=res,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResGS = np.array(res)
ResGScg = rescaleRes(oResGS, lambda x: x, b)
print('#time: {}'.format(tt))

checker('Calderon GS', A, J, xx)

#################################################

sigma = -0.5
print('\nSigma={2} WO restart={0} maxiter={1}'.format(restart, maxiter, sigma), flush=True)
del res
res = []
Ms, bs = Msigma(sigma), sigma * b
tt = time()
xx, info = gmres(Ms, bs,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResSigWO = np.array(res)
ResSigJWO = rescaleRes(oResSigWO, lambda x: x, bs)
print('#time: {}'.format(tt))

checker('Sigma WO', A, J, xx)

#################################################

print('\nBD WO restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
del res
res = []
Ms, bs = BD,  b
tt = time()
xx, info = gmres(Ms, bs,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
xx = CCi(xx)
tt = time() - tt
print(info, len(res))
oResBDWO = np.array(res)
ResBDWO = rescaleRes(oResBDWO, lambda x: x, bs)
print('#time: {}'.format(tt))

checker('BD WO', A, J, xx)

#################################################

print('\nBD bicgstab WO maxiter={0}'.format(maxiter), flush=True)
del res
res = []
Ms, bs = BD,  b
tt = time()
xx, info = bicgstab(Ms, bs,
                 tol=tol,
                 residuals=res,
                 maxiter=maxiter)
xx = CCi(xx)
tt = time() - tt
print(info, len(res))
oResBDWO = np.array(res)
ResBDWOcg = rescaleRes(oResBDWO, lambda x: x, bs)
print('#time: {}'.format(tt))

checker('BD WO', A, J, xx)

####################################

print('\nF bicgstab WO maxiter={0}'.format(maxiter), flush=True)
del res
res = []
Ms, bs = F,  b
tt = time()
xx, info = bicgstab(Ms, bs,
                    tol=tol,
                    residuals=res,
                    maxiter=maxiter)
xx = CCi(xx)
tt = time() - tt
print(info, len(res))
oResBDWO = np.array(res)
ResFWOcg = rescaleRes(oResBDWO, lambda x: x, bs)
print('#time: {}'.format(tt))

checker('BD WO', A, J, xx)

####################################
####################################

import matplotlib.pyplot as plt


def Res2Tuple(res):
    return np.arange(len(res)), res

its, res = Res2Tuple(ResWO)
plt.semilogy(its, res, 'k-', linewidth=3,  label='wo')

its, res = Res2Tuple(ResWOcg)
plt.semilogy(its, res, 'k--', linewidth=3,  label='wo')

its, res = Res2Tuple(ResJac)
plt.semilogy(its, res, 'b-', linewidth=3,  label='Jacobi')

its, res = Res2Tuple(ResGS)
plt.semilogy(its, res, 'r-', linewidth=3,  label='Gauss-Siedel(1)')

its, res = Res2Tuple(ResGScg)
plt.semilogy(its, res, 'r--', linewidth=3,  label='Gauss-Siedel(1)')

its, res = Res2Tuple(ResSigJWO)
plt.semilogy(its, res, 'g-', linewidth=3,  label='Sigma')

its, res = Res2Tuple(ResBDWO)
plt.semilogy(its, res, 'm-', linewidth=3,  label='Bott-Duffin')

its, res = Res2Tuple(ResBDWOcg)
plt.semilogy(its, res, 'm--', linewidth=3,  label='Bott-Duffin')

its, res = Res2Tuple(ResiJ)
plt.semilogy(its, res, 'c-', linewidth=3,  label='Mass')

# its, res = Res2Tuple(ResFWOcg)
# plt.semilogy(its, res, 'c--', linewidth=3,  label='F')

plt.title('Convergence History', fontsize=20)
plt.xlabel('#iterations', fontsize=14)
plt.ylabel('normalized residual', fontsize=30)
plt.legend()

plt.grid(True)

plt.show()
