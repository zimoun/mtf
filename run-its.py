#!/usr/bin/env python
# coding: utf8

import numpy as np

from assemb import MultiTrace, checker

from time import time
import scipy.linalg as la

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

#################################################
#################################################
#################################################

from krylov import gmres, bicgstab

#################################################
#################################################
#################################################

tol = 1e-6
res = []
restart = 10
if restart is None: scale = 1
else: scale = restart

maxiter = int((M.shape[0] / scale) * 0.1)
if maxiter < 50: maxiter = 50
if maxiter > 1000: maxiter = 1000

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
checker('Transmission Mass', J, X, xx, b)

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
checker('Transmission Jac', J, X, xx, b)

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
checker('Transmission GS', J, X, xx, b)

sol = xx

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
oResBD = np.array(res)
ResBD = rescaleRes(oResBD, lambda x: x, bs)
print('#time: {}'.format(tt))

checker('Calderon BD', A, J, xx)
checker('Transmission WO', J, X, xx, b)

#################################################

# sigma = -0.5
# print('\nSigma={2} WO restart={0} maxiter={1}'.format(restart, maxiter, sigma), flush=True)
# del res
# res = []
# Ms, bs = Msigma(sigma), sigma * b
# tt = time()
# xx, info = gmres(Ms, bs,
#                  orthog='mgs',
#                  tol=tol,
#                  residuals=res,
#                  restrt=restart,
#                  maxiter=maxiter)
# tt = time() - tt
# print(info, len(res))
# oResSigWO = np.array(res)
# ResSigJWO = rescaleRes(oResSigWO, lambda x: x, bs)
# print('#time: {}'.format(tt))

# checker('Sigma WO', A, J, xx)

####################################
####################################

def Res2Tuple(res):
    return np.arange(len(res)), res

myfig = plt.figure()

its, res = Res2Tuple(ResWO)
plt.semilogy(its, res, 'k-', linewidth=3,  label='wo')

its, res = Res2Tuple(ResiJ)
plt.semilogy(its, res, 'c-', linewidth=3,  label='Mass')

its, res = Res2Tuple(ResJac)
plt.semilogy(its, res, 'b-', linewidth=3,  label='Jacobi')

its, res = Res2Tuple(ResGS)
plt.semilogy(its, res, 'r-', linewidth=3,  label='Gauss-Siedel(1)')

its, res = Res2Tuple(ResBD)
plt.semilogy(its, res, 'g--', linewidth=3,  label='Bott-Duffin')

# its, res = Res2Tuple(ResSigJWO)
# plt.semilogy(its, res, 'g-', linewidth=3,  label='Sigma')

plt.title('Convergence History', fontsize=20)
plt.xlabel('#iterations', fontsize=14)
plt.ylabel('normalized residual', fontsize=30)
plt.legend()

plt.grid(True)

# plt.show()
if restart is None:
    restrt = 'Inf'
else:
    restrt = restart
myfig.savefig('its_k{0}_r{1}.eps'.format(kRef, restrt))
