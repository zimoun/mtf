#!/usr/bin/env python
# coding: utf8

import numpy as np

from assemb import MultiTrace, checker

from time import time
import scipy.linalg as la

meshname = "./geo/ellipse-disjoint.msh"

kRef = 3.3 * np.pi

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

print('')
print(mtf.shape, flush=True)
print('')

#################################################
#################################################
#################################################

iA = iJ * A * iJ

#################################################

Pjac = iA

E = mtf.upper()
Pgs = iA + iA * E * iA

Msigma = lambda sigma: (A - J) + sigma * (J - X)

