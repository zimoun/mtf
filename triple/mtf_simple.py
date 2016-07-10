#!/usr/bin/env python
# coding: utf8

import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import bempp.api as bem

# ugly but simplify exchange config
# computations etc.
import config
import mtf_triple as mtf

s0 = mtf.s0

opA = bem.BlockedOperator(2, 2)

opA0 = mtf.opA0

k1 = config.k1
opA1 = bem.BlockedOperator(2, 2)

opK1 =  bem.operators.boundary.helmholtz.double_layer(s0, s0, s0, k1)
opV1 =  bem.operators.boundary.helmholtz.single_layer(s0, s0, s0, k1)
opW1 =  bem.operators.boundary.helmholtz.hypersingular(s0, s0, s0, k1)
opQ1 =  bem.operators.boundary.helmholtz.adjoint_double_layer(s0, s0, s0, k1)

opA1[0, 0], opA1[0, 1] = -1.0 * (-1.0 * opK1), opV1
opA1[1, 0], opA1[1, 1] = opW1, (-1.0 * opQ1)

opA[0, 0], opA[1, 1] = opA0, opA1


print('weak form (from simple)')
wA = opA.weak_form()

bX = bem.BlockedDiscreteOperator(2, 2)

XX = bem.BlockedDiscreteOperator(2, 2)

Id = mtf.Id0
wId = Id.weak_form()
sId = wId.sparse_operator

XX[0, 0], XX[1, 1] = sId, - sId

bX[0, 1], bX[1, 0] = XX, XX

bJ = bem.BlockedDiscreteOperator(2, 2)
JJ = bem.BlockedDiscreteOperator(2, 2)
JJ[0, 0], JJ[1, 1] = sId, sId
bJ[0, 0], bJ[1, 1] = JJ, JJ

iId = mtf.id0

ibJ = bem.BlockedDiscreteOperator(2, 2)
iJJ = bem.BlockedDiscreteOperator(2, 2)
iJJ[0, 0], iJJ[1, 1] = iId, iId
ibJ[0, 0], ibJ[1, 1] = iJJ, iJJ

shape = wA.shape
N, _ = shape
if N % 2 != 0:
    raise ValueError
else:
    n = int(N/2)
if n % 2 != 0:
    raise ValueError
else:
    m = int(n/2)

A = spla.LinearOperator(shape, matvec=wA.matvec)
X = spla.LinearOperator(shape, matvec=bX.matvec)
M = A - 0.5 * X
J = spla.LinearOperator(shape, matvec=bJ.matvec)
iJ = spla.LinearOperator(shape, matvec=ibJ.matvec)

x = np.random.rand(N)
y = M(x)

diri = bem.GridFunction(s0, fun=config.fdir)
neum = bem.GridFunction(s0, fun=config.fneu)

rhs = [diri, neum, -diri, neum]
b = np.array([], dtype=complex)
for r in rhs:
    b = np.concatenate((b, r.projections()))
b = 0.5 * b

print('solve')
x, info = spla.gmres(M, b)
print(info)

ged = bem.GridFunction(s0, coefficients=x[0:m])
gen = bem.GridFunction(s0, coefficients=x[m:2*m])

gid = bem.GridFunction(s0, coefficients=x[2*m:3*m])
gin = bem.GridFunction(s0, coefficients=x[3*m:4*m])

print(la.norm(2 * A(x) - J(x)))
print(la.norm(J(x) - X(x) - 2 * b))


#######################

print('mie')
from miesphere import mie_D4grid, mie_N4grid

eps = config.eps
iincident = config.iincident

k = config.k0
kk = [0, 0, 0]
for q in range(3):
    if q == iincident:
        kk[q] = k
kk = tuple(kk)
C = np.array([0, 0, 0])
R = 1
ce, ci = 1, np.sqrt(eps)
jumpe, jumpi  = (1, 1), (1, 1)
Nmodes = 50
field = 'sca'
# field = 'int'
# field = 'inc'

def mieD(point, normal, dom_ind, result):
    val = mie_D4grid(field, kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    result[0] = val

def uinc(point, normal, dom_ind, result):
    result[0] = np.exp(1j * kRef * point[iincident])

def mieN(point, normal, dom_ind, result):
    val = mie_N4grid(field, kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    result[0] = val

def dnuinc(point, normal, dom_ind, result):
    result[0] = 1j * kRef * normal[1] * np.exp(1j * kRef * point[iincident])

def mieD_int(point, normal, dom_ind, result):
    val = mie_D4grid('int', kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    result[0] = val

def mieN_int(point, normal, dom_ind, result):
    val = mie_N4grid('int', kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    result[0] = val

gmie = bem.GridFunction(s0, fun=mieD)
miecoeffs = gmie.coefficients

print('xref')
yref = np.array([], dtype=complex)

mie = bem.GridFunction(s0, fun=mieD)
yref = np.concatenate((yref, sId.dot(mie.coefficients)))
mie = bem.GridFunction(s0, fun=mieN)
yref = np.concatenate((yref, sId.dot(mie.coefficients)))
mie = bem.GridFunction(s0, fun=mieD_int)
yref = np.concatenate((yref, sId.dot(mie.coefficients)))
mie = bem.GridFunction(s0, fun=mieN_int)
yref = np.concatenate((yref, sId.dot(mie.coefficients)))

xref = iJ(yref)

print(la.norm(2 * A(xref) - J(xref)))
print(la.norm(J(xref) - X(xref) - 2 * b))
