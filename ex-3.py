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


#mtf = MultiTrace(kRef, meshname, dd)

#Aw = mtf.A_weak_form()

# # At, X, J, iJ = mtf.tolinop()

#shape = mtf.shape

# A = 2.0 * At
# A2 = A * iJ * A

# Ce = 0.5 * J - At
# Ci = 0.5 * J + At

# Ce2 = Ce * iJ * Ce
# Ci2 = Ci * iJ * Ci

#x = np.random.rand(shape[0]) + 1j * np.random.rand(shape[0])
# xr = np.random.rand(shape[0])

# checker('A2 = J', A2, J, x)
# checker('exterior Proj.', Ce2, Ce, x)
# checker('interior Proj.', Ci2, Ci, x)
# checker('error-Calderon with random [no-sense]', A, J, x)

# def dir_data(x, normal, dom_ind, result):
#     result[0] =  -np.exp( 1j * kRef * x[1])

# def neu_data(x, normal, dom_ind, result):
#     result[0] = -1j * normal[1] * kRef * np.exp( 1j * kRef * x[1])

# b = mtf.rhs(dir_data, neu_data)
# M = A - X

# print('')
# print(mtf.shape, flush=True)
# print('')

#slices = mtf.getSlices()

# for c in ['B', 'C', '0', 'A']:
#     s = slices[c]
#     y = x[s[0]:s[1]]

#     d = mtf.domains.getIndexDom(c)
#     A = Aw[d, d]
#     z = A.dot(y)


import bempp.api as bem
from miesphere import mie_D4grid, mie_N4grid

C = np.array([0, 0, 0])
k = kRef
kk = (0, k, 0)
R = 1
ce, ci = 1, np.sqrt(2)
N = 50
field = 'inc'

def mieD(point, normal, dom_ind, result):
    val = mie_D4grid(field, kk, R, C, ce, ci, N, point)
    result[0] = val

def uinc(point, normal, dom_ind, result):
    result[0] = np.exp(1j * kRef * point[1])

def mieN(point, normal, dom_ind, result):
    val = mie_N4grid(field, kk, R, C, ce, ci, N, point)
    result[0] = val

def dnuinc(point, normal, dom_ind, result):
    result[0] = 1j * kRef * normal[1] * np.exp(1j * kRef * point[1])

grid = bem.import_grid(meshname)
space = bem.function_space(grid, "P", 1)

gmie = bem.GridFunction(space, fun=mieD)
miecoeffs = gmie.coefficients

gui = bem.GridFunction(space, fun=uinc)
uicoeffs = gui.coefficients

fmie = bem.GridFunction(space, fun=mieN)
dnmiecoeffs = fmie.coefficients

fui = bem.GridFunction(space, fun=dnuinc)
dnuicoeffs = fui.coefficients
