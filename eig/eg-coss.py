#!/usr/bin/env python
# coding: utf8

from time import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt

import bempp.api as bem
from krylov import gmres

from xtf import xTF, STF, MTF
from xtf import STF2, MTF2, iSTF
from algo import rhs, solve

its = False
MET = iSTF

N = 10
n = 4

k = np.linspace(2.5, 3.5, N+1, endpoint=True)

E = []
for kk in k:
    xtf = xTF(kk, n)
    Met = MET(xtf)
    met = Met.get()

    xtf.update(1j * kk)
    cMet = MET(xtf)
    cmet = cMet.get()

    print(' ')
    print('kk=', kk)
    print('shape(MET)=', met.shape)
    print(' ')

    print('converting...', end=' ', flush=True)
    A = bem.as_matrix(met)
    print('x', end=' ', flush=True)
    B = bem.as_matrix(cmet)
    print('eig...', end=' ', flush=True)
    w = la.eig(A, B, right=False)
    print('done.')
    E.append(1./np.min(np.abs(w)))
E = np.array(E)

plt.figure(1)
plt.plot(k, E)

plt.grid()
plt.show(block=False)
