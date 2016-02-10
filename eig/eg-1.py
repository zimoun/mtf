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

N = 50
n = 4

tol = 1e-6
res = []
restart = None
maxiter = 300


def psi(t, mu=3, a=1, b=0.25):
    v = mu + a * np.cos(t) + 1j * b * np.sin(t)
    return v

t = 2*np.pi * np.linspace(0, 1, N)
Psi = psi(t)

kRef = Psi[0]
kRef = 1.

kExt, kInt = kRef, np.sqrt(n) * kRef

with open('in.geo', 'w') as fp:
    fp.write("k = {};".format(np.abs(kInt)))
    from subprocess import call
call(['gmsh', 'sphere-simple.script.geo', '-'])
meshname = 'sphere-simple.msh'

grid = bem.import_grid(meshname)
space = bem.function_space(grid, "P", 1)
shape = space.global_dof_count

K0 = bem.operators.boundary.helmholtz.double_layer(space, space, space, kExt)
V0 = bem.operators.boundary.helmholtz.single_layer(space, space, space, kExt)
W0 = bem.operators.boundary.helmholtz.hypersingular(space, space, space, kExt)
Q0 = bem.operators.boundary.helmholtz.adjoint_double_layer(space, space, space, kExt)

sK0, sQ0 = -K0, -Q0

K1 = bem.operators.boundary.helmholtz.double_layer(space, space, space, kInt)
V1 = bem.operators.boundary.helmholtz.single_layer(space, space, space, kInt)
W1 = bem.operators.boundary.helmholtz.hypersingular(space, space, space, kInt)
Q1 = bem.operators.boundary.helmholtz.adjoint_double_layer(space, space, space, kInt)

Id = bem.operators.boundary.sparse.identity(space, space, space)

STF = bem.BlockedOperator(2, 2)
STF[0, 0] = -sK0 + K1
STF[0, 1] = V0 + V1
STF[1, 0] = W0 + W1
STF[1, 1] = sQ0 - Q1

A0, A1 = bem.BlockedOperator(2, 2), bem.BlockedOperator(2, 2)
A0[0, 0], A0[0, 1] = -sK0, V0
A0[1, 0], A0[1, 1] = W0, sQ0
A1[0, 0], A1[0, 1] = -K1, V1
A1[1, 0], A1[1, 1] = W1, Q1

X01, X10 = bem.BlockedOperator(2, 2), bem.BlockedOperator(2, 2)
X01[0, 0], X01[1, 1] = Id, -Id
X10[0, 0], X10[1, 1] = Id, -Id

A, X = bem.BlockedOperator(2, 2), bem.BlockedOperator(2, 2)
A[0, 0], A[1, 1] = A0, A1
X[0, 1], X[1, 0] = X01, X10

MTF = A - 0.5 * X

print('Assembling...', end=' ', flush=True)
for op, who in zip([K0, V0, W0, Q0, K1, V1, W1, Q1],
                   ['K0', 'V0', 'W0', 'Q0', 'K1', 'V1', 'W1', 'Q1']):
    print(who, end=' ', flush=True)
    a = op.weak_form()
print('Assembling missing...', end='', flush=True)
mtf = MTF.weak_form()
stf = STF.weak_form()
print('done.')

def dir_data(x, normal, dom_ind, result):
    result[0] =  -np.exp( 1j * kRef * x[1])

def neu_data(x, normal, dom_ind, result):
    result[0] = -1j * normal[1] * kRef * np.exp( 1j * kRef * x[1])

fdir = bem.GridFunction(space, fun=dir_data)
fneu = bem.GridFunction(space, fun=neu_data)

a = (0.5 * Id + K1) * fdir - V1 * fneu
b = W1 * fdir + (-0.5 * Id + Q1) * fneu
a, b = a.projections(), b.projections()
rhs_stf = np.concatenate((a, b))

a, b = fdir.projections(), fneu.projections()
rhs_mtf = 0.5 * np.concatenate((a, -b, -a, -b))
fd, fn = a, b

rescaleRes = lambda res, P, rhs: res / la.norm(P(rhs))

print('\nSTF restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
Mat, b = stf, rhs_stf
print('size: ', stf.shape, 2*shape)
del res
res = []
tt = time()
x_stf, info = gmres(Mat, b,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oRes = np.array(res)
ResSTF = rescaleRes(oRes, lambda x: x, b)
print('#time: {}'.format(tt))

xd_stf, xn_stf = x_stf[0:shape], x_stf[shape:]

print('\nMTF restart={0} maxiter={1}'.format(restart, maxiter), flush=True)
Mat, b = mtf, rhs_mtf
print('size: ', mtf.shape, 4*shape)
del res
res = []
tt = time()
x_mtf, info = gmres(Mat, b,
                    orthog='mgs',
                    tol=tol,
                    residuals=res,
                    restrt=restart,
                    maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oRes = np.array(res)
ResMTF = rescaleRes(oRes, lambda x: x, b)
print('#time: {}'.format(tt))

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



J = bem.BlockedOperator(2, 2)
J[0, 0], J[1, 1] = Id, Id

C = 0.5 * J + A0
Cw = C.weak_form()
iJCw = C.strong_form()
CC = lambda x: Cw.matvec(iJCw.matvec(x))
x = np.random.rand(2*shape) + 1j * np.random.rand(2*shape)
y = Cw.matvec(x)
z = CC(x)




# plt.figure(1)
# plt.plot(Psi.real, Psi.imag, 'b.-', label='contour')

# plt.xlabel('real')
# plt.ylabel('imag')
# plt.legend()
# plt.grid()
# plt.show()
