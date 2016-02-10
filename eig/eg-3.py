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
from algo import solve

N = 50
n = 4

kRef = 1.

tol = 1e-6
res = []
restart = None
maxiter = 300

def dir_data(x, normal, dom_ind, result):
    result[0] =  -np.exp( 1j * kRef * x[1])

def neu_data(x, normal, dom_ind, result):
    result[0] = -1j * normal[1] * kRef * np.exp( 1j * kRef * x[1])

xtf = xTF(kRef, n)
xtf.setRHS(dir_data, neu_data)

space = xtf.space
shape = xtf.shape
fd, fn = xtf.getDir(), xtf.getNeu()
fdir, fneu = xtf.getGFdir(), xtf.getGFneu()

STF, MTF = STF(xtf), MTF(xtf)

stf, rhs_stf = STF.get(), STF.rhs()
mtf, rhs_mtf = MTF.get(), MTF.rhs()

x_stf = solve(stf, rhs_stf)
xd_stf, xn_stf = x_stf[0:shape], x_stf[shape:]

x_mtf = solve(mtf, rhs_mtf)
xd_mtf, xn_mtf = x_mtf[0:shape], x_mtf[shape:2*shape]
yd_mtf, yn_mtf = x_mtf[2*shape:3*shape], x_mtf[3*shape:4*shape]

print('')
print('l2 norm (relative)')
print(la.norm(xd_mtf - xd_stf), la.norm(xn_mtf - xn_stf))
print(la.norm(xd_mtf - yd_mtf - fd)/la.norm(xd_mtf),
      la.norm(-xn_mtf - yn_mtf -fn)/la.norm(xn_mtf))

gd_mtf = bem.GridFunction(space, coefficients=xd_mtf)
gn_mtf = bem.GridFunction(space, coefficients=xn_mtf)
ggd_mtf = bem.GridFunction(space, coefficients=yd_mtf)
ggn_mtf = bem.GridFunction(space, coefficients=yn_mtf)
gd_stf = bem.GridFunction(space, coefficients=xd_stf)
gn_stf = bem.GridFunction(space, coefficients=xn_stf)
print('L2 norm')
print((gd_mtf - gd_stf).l2_norm(), (gn_mtf - gn_stf).l2_norm())
print((gd_mtf - ggd_mtf - fdir).l2_norm(), (-gn_mtf - ggn_mtf - fneu).l2_norm())





# plt.figure(1)
# plt.plot(Psi.real, Psi.imag, 'b.-', label='contour')

# plt.xlabel('real')
# plt.ylabel('imag')
# plt.legend()
# plt.grid()
# plt.show()
