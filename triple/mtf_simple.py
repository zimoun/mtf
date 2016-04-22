#!/usr/bin/env python
# coding: utf8

import numpy as np
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

opA1[0, 0], opA1[0, 1] = - (-opK1), opV1
opA1[1, 0], opA1[1, 1] = opW1, (-opQ1)

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


shape = wA.shape
N, _ = shape
if N % 2 != 0:
    raise ValueError
else:
    n = int(N/2)

A = spla.LinearOperator(shape, matvec=wA.matvec)
X = spla.LinearOperator(shape, matvec=bX.matvec)
M = A - 0.5 * X

x = np.random.rand(N)
y = M(x)

diri = bem.GridFunction(s0, fun=config.fdir)
neum = bem.GridFunction(s0, fun=config.fneu)

rhs = [diri, -neum, -diri, neum]
b = np.array([], dtype=complex)
for r in rhs:
    b = np.concatenate((b, r.projections()))


x, info = spla.gmres(M, b)

ged = bem.GridFunction(s0, coefficients=x[0:n])
