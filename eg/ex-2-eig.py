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


mtf = MultiTrace(kRef, meshname, dd, J_is='CSC')
#jmtf = MultiTrace(1j * kRef, meshname, dd)

At, X, J, iJ = mtf.tolinop()
#jAt, jX, jJ, jiJ = jmtf.tolinop()

shape = mtf.shape
#jshape = jmtf.shape

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


# jA = 2.0 * jAt
# jA2 = jA * iJ * jA

# jCe = 0.5 * jJ - jAt
# jCi = 0.5 * jJ + jAt

# jCe2 = jCe * jiJ * jCe
# jCi2 = jCi * jiJ * jCi

# x = np.random.rand(shape[0])

# checker('A2 = J', jA2, jJ, x)
# checker('exterior Proj.', jCe2, jCe, x)
# checker('interior Proj.', jCi2, jCi, x)
# checker('error-Calderon with random [no-sense]', jA, jJ, x)

# M = A - X
# jM = jA - jX

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

import sys, os
stdoutw = sys.stdout.write
stdoutf = sys.stdout.flush
N = shape[0]
e = np.zeros(N, dtype=np.complex)
M = np.zeros(shape=(N, 1), dtype=np.complex)
for i in range(N):
    stdoutw('\rdone: {0}%'.format(int(100*i/N)))
    e[i] = 1.0 + 1j * 0.0
    #v = Ci(e)
    #v = iJ * Ci(e)
    #v = (J - X) * iJ * Ci(e) + Ce(e)
    v = iJ * ( (J - X) * iJ * Ci(e) + Ce(e) )
    e[i] = 0.0 + 1j * 0.0
    v = np.reshape(v, (N, 1))
    M = np.concatenate((M, v), axis=1)
    stdoutf()
M = M[:, 1:]
print('done!', flush=True)

plt.spy(M, marker='.')
plt.show(block=False)

s = la.svd(M, compute_uv=False)
l = la.eig(M, right=False)
l.sort()

# N = int(Ce.shape[0]/2)
# v = spla.eigs(Ce, k=N, which='LM', return_eigenvectors=False)
