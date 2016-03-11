#!/usr/bin/env python
# coding: utf8

import numpy as np

from assemb import MultiTrace, checker

from time import time
import scipy.linalg as la

meshname = "./geo/sphere-disjoint.msh"

kRef = 0.1 * np.pi

dd = [
    { 'name': '0',
      'phys': 1,
      'union': [-1], #, -2, -3],
  },
    { 'name': 'A',
      'phys': 2,
      'union': 1,
  }
  # },
  #   { 'name': 'B',
  #     'phys': 1,
  #     'union': 2,
  #     }
  # },
  #   { 'name': 'C',
  #     'phys': 4,
  #     'union': 3,
  # }
]


mtf = MultiTrace(kRef, meshname, dd)

Aw = mtf.A_weak_form()

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
    ii = 0
    result[0] =  -np.exp( 1j * kRef * x[ii])

def neu_data(x, normal, dom_ind, result):
    ii = 0
    result[0] = -1j * normal[ii] * kRef * np.exp( 1j * kRef * x[ii])

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

tol = 1e-9
res = []
restart = None
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


slices = mtf.getSlices()

#for c in ['B', 'C', '0', 'A']:
# for c in ['B', '0', 'A']:
#     s = slices[c]
#     y = x[s[0]:s[1]]

#     d = mtf.domains.getIndexDom(c)
#     A = Aw[d, d]
#     z = A.dot(y)

s = slices['0']
sol = xx[s[0]:s[1]]

import bempp.api as bem
from miesphere import mie_D4grid, mie_N4grid

C = np.array([0, 0, 0])
k = kRef
kk = (0, k, 0)
R = 1
ce, ci = 1, np.sqrt(2)
jumpe, jumpi  = (1, 1), (1, 1)
N = 50
field = 'sca'
# field = 'int'
#field = 'inc'

def mieD(point, normal, dom_ind, result):
    val = mie_D4grid(field, kk, R, C, ce, ci, jumpe, jumpi, N, point)
    result[0] = val

def uinc(point, normal, dom_ind, result):
    result[0] = np.exp(1j * kRef * point[1])

def mieN(point, normal, dom_ind, result):
    val = mie_N4grid(field, kk, R, C, ce, ci, jumpe, jumpi, N, point)
    result[0] = val

def dnuinc(point, normal, dom_ind, result):
    result[0] = 1j * kRef * normal[1] * np.exp(1j * kRef * point[1])

# grid = bem.import_grid(meshname)
# space = bem.function_space(grid, "P", 1)
d = mtf.domains.getIndexDom('0')
(space, _) , (_, _) = mtf.spaces[d]


n, = sol.shape
n = int(n/2)
sold, soln = sol[:n], sol[n:]
gsold = bem.GridFunction(space, coefficients=sold)

gmie = bem.GridFunction(space, fun=mieD)
miecoeffs = gmie.coefficients

errd = sold - miecoeffs
aerrd = np.abs(errd)
gerrd = bem.GridFunction(space, coefficients=errd)
gaerrd = bem.GridFunction(space, coefficients=aerrd)
print(la.norm(errd), la.norm(aerrd), la.norm(errd)/la.norm(miecoeffs))
print(gerrd.l2_norm(), gaerrd.l2_norm(), gerrd.l2_norm()/gmie.l2_norm())


print(' ')

fmie = bem.GridFunction(space, fun=mieN)
dnmiecoeffs = fmie.coefficients

errn = soln - dnmiecoeffs
aerrn = np.abs(errn)
ferrn = bem.GridFunction(space, coefficients=errn)
faerrn = bem.GridFunction(space, coefficients=aerrn)
print(la.norm(errn), la.norm(aerrn), la.norm(errn)/la.norm(dnmiecoeffs))
print(ferrn.l2_norm(), faerrn.l2_norm(), ferrn.l2_norm()/fmie.l2_norm())


gui = bem.GridFunction(space, fun=uinc)
uicoeffs = gui.coefficients

fui = bem.GridFunction(space, fun=dnuinc)
dnuicoeffs = fui.coefficients
