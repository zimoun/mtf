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
from algo import solve

its = False
MET = iSTF

N = 50
n = 4

l = 400

tolrank, tolres = 1e-9, 1e-7
mu, a, b = 3.8, 1, 0.25

t = 2*np.pi * np.linspace(0, 1, N+1, endpoint=True)
psi = lambda t: mu + a * np.cos(t) + 1j * b * np.sin(t)
phi = lambda t: - a * np.sin(t) + 1j * b * np.cos(t)
Psi, Phi = psi(t), phi(t)

kmax = np.max(np.abs(Psi))

tinit = time()
xtf = xTF(kmax, n)
Met = MET(xtf)
met = Met.get()

print('inital assemb time:', time() - tinit)
print('size MET: ', met.shape)
print(bem.global_parameters.assembly.boundary_operator_assembly_type)

m = met.shape[0]

# Vhat = np.random.rand(m, l) + 1j * np.random.rand(m, l)
Vhat = np.eye(m)
VVhat = Vhat[:, 0:l]
Vhat = np.zeros((m, 1), dtype=np.complex)
space = xtf.space
for v in VVhat.T:
    if m % 2 == 0:
        half = int(m/2)
    else:
        raise ValueError('Odd shape: WEIRD !! is possible ?')
    v1, v2 = v[:half], v[half:]
    gfun1 = bem.GridFunction(space, coefficients=v1)
    gfun2 = bem.GridFunction(space, coefficients=v2)
    u = np.concatenate((gfun1.projections(), gfun2.projections()), axis=0)
    Vhat = np.concatenate((Vhat, u.reshape((m, 1))), axis=1)
Vhat = Vhat[:, 1:]

AN = np.zeros((m, l), dtype=np.complex)
BN = np.zeros((m, l), dtype=np.complex)

An = np.zeros((m, l), dtype=np.complex)
Bn = np.zeros((m, l), dtype=np.complex)

count, tt = 1, time()
for tj in t[0:-1]:
    print('#', count, '/', N)
    count += 1
    print('updating...', end='\n  -> ', flush=True)
    tup = time()
    xtf.update(psi(tj))
    tup = time() - tup
    print('updated. time:', tup, end='\n', flush=True)
    Met = MET(xtf)
    met = Met.get()

    if bem.global_parameters.assembly.boundary_operator_assembly_type == 'dense':
        print('converting matrix...', end=' ', flush=True)
        tconv = time()
        mat = bem.as_matrix(met)
        tconv = time() - tconv
        print('done. time:', tconv, flush=True)
        print('solving...', end=' ', flush=True)
        tsolve = time()
        WWhat = la.solve(mat, Vhat)
        tsolve = time() - tsolve
        print('done. time:', tsolve, flush=True)
        An += phi(tj) * WWhat
        Bn += phi(tj) * psi(tj) * WWhat

    elif its == True:
        print('solving...', end=' ', flush=True)
        What = np.zeros((m, 1), dtype=np.complex)
        for ii in range(l):
            v = Vhat[:, ii]
            tsolve = time()
            WWhat = la.solve(met, v)
            What = np.concatenate((What, x.reshape(m, 1)), axis=1)
        What = What[:, 1:]
        tsolve = time() - tsolve
        print('done. time:', tsolve, flush=True)

        AN += phi(tj) * What
        BN += phi(tj) * psi(tj) * What

AN, BN = 1/(1j*N) * AN, 1/(1j*N) * BN
An, Bn = 1/(1j*N) * An, 1/(1j*N) * Bn

A, B = An, Bn

print(' ')
print('shape(MET)=', met.shape)
print('shape(AN)=', A.shape)
print('total time:', time() - tinit)
print(' ')


V, s, Wh = la.svd(A)
W = Wh.transpose().conjugate()

ii = [ i for i, val in enumerate(s) if val > tolrank ]

WARNING = False
if len(ii) == l:
    WARNING = True
    print('Warning')
k = len(ii)

V0, S0, W0 = V[:, ii], np.diag(s[ii]), W[0:l, ii]
V0h, iS0 = V0.transpose().conjugate(), la.inv(S0)
C = V0h.dot(B.dot(W0.dot(iS0)))
CC = V0h.dot(B.dot(W0))

w, M = la.eig(C)
M = V0.dot(M)
ww = la.eig(CC, b=S0, right=False)

reorder = lambda eigen: not ((eigen.real - mu)/a)**2 + (eigen.imag/b)**2 < 1.0
T, Z, sdim = la.schur(C, output='complex', sort=reorder)
www = np.diag(T)

plt.figure(1)
plt.plot(Psi.real, Psi.imag, 'b.-')
#plt.plot(C[0:5].real, C[0:5].imag, 'r.')
plt.plot(w.real, w.imag, 'ro', label='naive')
plt.plot(ww.real, ww.imag, 'g.', label='via generalized')
plt.plot(www.real, www.imag, 'c.', label='via Schur')

plt.plot(3.14159, 0, 'kx', label='theoretical')
plt.plot(3.69245, 0, 'kx')
plt.plot(4.26168, 0, 'kx')
plt.plot(4.83186, 0, 'kx')

plt.legend()
plt.xlabel('real')
plt.ylabel('imag')
plt.title('Eigenvalues in the complex plane')
plt.show(block=False)

goodeig = []
err = []
for j in range(len(w)):
    wj, vj = w[j], M[:, j]
    print('check#', j+1, '/', len(w), '[/', l, ']')
    print('updating...', end='\n  -> ', flush=True)
    xtf.update(wj)
    print('updated', end='.\n', flush=True)
    Met = MET(xtf)
    met = Met.get()

    er = la.norm(met.dot(vj))
    err.append(er)
    if er < tolres:
        goodeig.append(wj)
err = np.array(err)
goodeig = np.array(goodeig)

plt.figure(2)

plt.subplot(121)
plt.semilogy(err, 'bo')
plt.semilogy([ tolres for i in err ], 'k--')

plt.xlabel('j (of eigenvalue_j)')
plt.ylabel('error')
plt.title('Error = || EqInt(eigenvalue_j)eigenvector_j ||')

#plt.show(block=False)

#plt.figure(3)

plt.subplot(122)
plt.plot(Psi.real, Psi.imag, 'b.-')
plt.plot(goodeig.real, goodeig.imag, 'ro', label='computed')

plt.plot(3.14159, 0, 'kx', label='theoretical')
plt.plot(3.69245, 0, 'kx')
plt.plot(4.26168, 0, 'kx')
plt.plot(4.83186, 0, 'kx')

plt.legend()
plt.xlabel('real')
plt.ylabel('imag')
plt.title('WELL computed Eigenvalues in the complex plane')

plt.show()



if WARNING:
    print('bad rank approx.')
else:
    print('ok.')
