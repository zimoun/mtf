#!/usr/bin/env python
# coding: utf8

import numpy as np

from assembMax import MultiTrace, checker

from time import time
import scipy.linalg as la

meshname = "./geo/sphere-disjoint.msh"

kRef = 0.1 * np.pi

dd = [
    { 'name': '0',
      'phys': 1,
      'union': [-1],
  },
    { 'name': 'A',
      'phys': 2,
      'union': 1,
  }
]
    # { 'name': 'B',
    #   'phys': 3,
    #   'union': 2,
    # }


mtf = MultiTrace(kRef, meshname, dd)
mtf.collecting()
At, X, J, iJ = mtf.tolinop()

shape = mtf.shape

A = 2.0 * At

Ce = -0.5 * J + At
Ci = 0.5 * J + At

Ce2 = Ce * iJ * Ce
Ci2 = Ci * iJ * Ci

CeCi = Ce * Ci
CiCe = Ci * Ce

Zero = 0.0 * J

x = np.random.rand(shape[0]) + 1j * np.random.rand(shape[0])
xr = np.random.rand(shape[0])

checker('Zero', Zero * A, A * Zero, x)
checker('exterior Proj.', Ce2, Ce, x)
checker('interior Proj.', Ci2, Ci, x)
checker('Ce Ci = 0', CeCi, Zero, x)
checker('Ce Ci = 0', CiCe, Zero, x)


def Einc(x):
    return np.array([np.exp(1j * kRef * x[2]), 0. * x[2], 0. * x[2]])

def Hinc(x):
    return np.array([0. * x[2], np.exp(1j * kRef * x[2]), 0. * x[2]])

def trace_Einc(x, n, domain_index, result):
    result[:] = np.cross(Einc(x), n, axis=0)

def trace_Hinc(x, n, domain_index, result):
    result[:] = np.cross(Hinc(x), n, axis=0)


b = mtf.rhs(trace_Einc, trace_Hinc)
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

B = J - X
BD = B * Ci + Ce
