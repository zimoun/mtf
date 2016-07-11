#!/usr/bin/env python
# coding: utf8

import numpy as np
import scipy.linalg as la
from time import time

import bempp.api as bem

from assemb import MultiTrace, checker
from domains import *

lmbda = 0.7
kRef = 2 * np.pi / lmbda

#meshname = 'geo/mtf-logo.msh'
meshname = 'geo/rings.msh'

dd = [
    { 'name': 0,
      'union': [-1, -2, -3],
      'phys': 1.,
      },
    { 'name': 'M',
      'union': 1,
      'phys': 4.,
      },
    { 'name': 'T',
      'union': 2,
      'phys': 9.,
      },
    { 'name': 'F',
      'union': 3,
      'phys': 16.,
      },
]
doms = Domains(dd)
doms.write2dot()

mtf = MultiTrace(kRef, meshname, doms)

At, X, J, iJ = mtf.tolinop()

shape = mtf.shape

A = 2.0 * At
A2 = A * iJ * A

print('')
print(mtf.shape, flush=True)
print('')


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

#################################################
#################################################

def dir_data(x, normal, dom_ind, result):
    result[0] =  -np.exp( 1j * kRef * x[1])

def neu_data(x, normal, dom_ind, result):
    result[0] = -1j * normal[1] * kRef * np.exp( 1j * kRef * x[1])

#################################################
#################################################

b = mtf.rhs(dir_data, neu_data)
M = A - X

#################################################
#################################################
#################################################

from krylov import gmres, bicgstab

#################################################
#################################################

tol = 1e-6
res = []
restart = 50
if restart is None: scale = 1
else: scale = restart

maxiter = int((M.shape[0] / scale) * 0.05)
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

slices = mtf.getSlices()
s = slices['0']

sol = xx[s[0]:s[1]]
d = mtf.domains.getIndexDom('0')
(space, _) , (_, _) = mtf.spaces[d]

n, = sol.shape
n = int(n/2)
sold, soln = sol[:n], sol[n:]

gsold = bem.GridFunction(space, coefficients=sold)
gsoln = bem.GridFunction(space, coefficients=soln)

bem.export(grid_function=gsold,
           file_name="soldR.msh",
           transformation=np.real)
bem.export(grid_function=gsold,
           file_name="soldI.msh",
           transformation=np.imag)
