# conding: utf8

from time import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import  matplotlib.pyplot as plt

import bempp.api as bem

k = 0.2
lmbda = 2 * np.pi / k
nlmbda = 15

grid = bem.shapes.sphere(h=lmbda/nlmbda)

P1 = bem.function_space(grid, "P", 1)
#P0 = bem.function_space(grid, "DP", 0)
P0 = bem.function_space(grid, "P", 1)

N0, N1 = P0.global_dof_count, P1.global_dof_count

opV = bem.operators.boundary.helmholtz.single_layer(P0, P0, P0, k)
opI0 = bem.operators.boundary.sparse.identity(P0, P0, P0)

opK = bem.operators.boundary.helmholtz.double_layer(P1, P0, P0, k)
opI10 = bem.operators.boundary.sparse.identity(P1, P0, P0)

opQ = bem.operators.boundary.helmholtz.adjoint_double_layer(P0, P1, P1, k)
opI01 = bem.operators.boundary.sparse.identity(P0, P1, P1)

opW = bem.operators.boundary.helmholtz.hypersingular(P1, P1, P1, k)
opI1 = bem.operators.boundary.sparse.identity(P1, P1, P1)

print('V')
V = opV.weak_form()

print('K')
K = opK.weak_form()

print('W')
W = opW.weak_form()

print('Q')
Q = opQ.weak_form()

opB = bem.BlockedOperator(2, 2)
opB[0, 0], opB[0, 1] = opV, -opK
opB[1, 0], opB[1, 1] = opQ, opW

Bw = opB.weak_form()


II0 = opI0.weak_form()
II1 = opI1.weak_form()

I0 = II0.sparse_operator
I1 = II1.sparse_operator

opJ = bem.BlockedOperator(2, 2)
opJ[0, 0], opJ[1, 1] = opI0, opI1

Jw = opJ.weak_form()

iI0_lu = spla.splu(I0)
iI1_lu = spla.splu(I1)

iI0 = spla.LinearOperator(I0.shape, matvec=iI0_lu.solve)
iI1 = spla.LinearOperator(I1.shape, matvec=iI1_lu.solve)

iJb = bem.BlockedDiscreteOperator(2, 2)
iJb[0, 0], iJb[1, 1] = iI0, iI1

opJt = bem.BlockedOperator(2, 2)
opJt[0, 1], opJt[1, 0] = opI01, opI10

tJw = opJt.weak_form()

print('\n==Projection by Pseudo-Inverse')
II01 = opI01.weak_form()
II10 = opI10.weak_form()

I01 = II01.sparse_operator
I10 = II10.sparse_operator

tt = time()
dI01 = I01.todense()
dI10 = I10.todense()
tt = time() - tt
print('#time to convert Sparse to Dense:', tt)

tt = time()
pI01 = la.pinv(dI01)
pI10 = la.pinv(dI10)
tt = time() - tt
print('#time to evaluate Pseudo-Inverse (mean-square):', tt)
print(np.max(np.diag(pI01)), np.min(np.diag(pI01)))
print(np.max(np.diag(pI10)), np.min(np.diag(pI10)))
tt = time()
ppI01 = la.pinv2(dI01)
ppI10 = la.pinv2(dI10)
tt = time() - tt
print('#time to evaluate Pseudo-Inverse (svd):', tt)
print(np.max(np.diag(pI01)), np.min(np.diag(pI01)))
print(np.max(np.diag(pI10)), np.min(np.diag(pI10)))

fProj_f0_t1 = lambda x: pI10.dot(I0.dot(x))
fProj_f1_t0 = lambda x: pI01.dot(I1.dot(x))

# fProj_f0_t1 = lambda x: pI10.dot(x)
# fProj_f1_t0 = lambda x: pI01.dot(x)

prI01 = spla.LinearOperator((N1, N0), matvec=fProj_f0_t1)
prI10 = spla.LinearOperator((N0, N1), matvec=fProj_f1_t0)

ttJb = bem.BlockedDiscreteOperator(2, 2)
ttJb[0, 0], ttJb[1, 1] = prI10, prI01

print('==done.\n')

shape = Bw.shape

B = spla.LinearOperator(shape, matvec=Bw.matvec)
J = spla.LinearOperator(shape, matvec=Jw.matvec)
iJ = spla.LinearOperator(shape, matvec=iJb.matvec)
tJ = spla.LinearOperator(shape, matvec=tJw.matvec)
ttJ = spla.LinearOperator(shape, matvec=ttJb.matvec)


BB = B * iJ * B

C = 0.5 * tJ + B
CC = C * iJ * C

x = np.random.rand(shape[0]) + 1j * np.random.rand(shape[0])

y = 4. * BB(x) - J(x)
z = CC(x) - C(x)

nx = la.norm(x)
ny = la.norm(y)
nz = la.norm(z)
print(nx, ny, nz)

print('')

A = tJ * iJ * B

AA = A * iJ * A

C = 0.5 * J + A
CC = C * iJ * C

y = 4. * AA(x) - J(x)
z = CC(x) - C(x)

nx = la.norm(x)
ny = la.norm(y)
nz = la.norm(z)
print(nx, ny, nz)

print('')

A = ttJ * iJ * B

AA = A * iJ * A

C = 0.5 * J + A
CC = C * iJ * C

y = 4. * AA(x) - J(x)
z = CC(x) - C(x)

nx = la.norm(x)
ny = la.norm(y)
nz = la.norm(z)
print(nx, ny, nz)
