#/usr/bin/env python
# coding: utf8

from time import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt

from scipy.special import jv, yv
from scipy.special import sph_jn, sph_yn
from scipy.special import eval_legendre

from scipy.optimize import brentq

def fun_cart2sph(F):
    def f(n, z):
        return np.sqrt(np.pi/(2*z))* F(n+1/2.0, z)
        # return sp.sqrt(1/z)* F(n+1/2.0, z)
    return f

def fun_derivative(F):
    def f(n, z):
        return (n/z)*F(n, z) - F(n+1, z)
    return f

J = fun_cart2sph(jv)
Jp = fun_derivative(J)

Y = fun_cart2sph(yv)
Yp = fun_derivative(Y)

H1 = lambda n, z: J(n, z) + 1j*Y(n, z)
H1p = lambda n, z: Jp(n, z) + 1j*Yp(n, z)


n = 5
k = np.linspace(0.5, 8, 50)
ke, ki = k, np.sqrt(n) * k
R = 1
m = 7

det = lambda p: -J(p, ke*R)*ki*Jp(p, ki*R) + ke*Jp(p, ke*R)*J(p, ki*R)

# p = 0
# Det = []
# for kk in k:
#     kke, kki = kk, np.sqrt(n) * kk
#     val = -J(p, kke*R)*kki*Jp(p, kki*R) + Jp(p, kke*R)*J(p, kki*R)
#     Det.append(val)
# Det = np.array(Det)

# p = 0
# DDet = []
# for kk in k:
#     kke, kki = kk, np.sqrt(n) * kk
#     mat = np.array([[J(p, kke*R), J(p, kki*R)],
#                     [Jp(p, kke*R), -Jp(p, kki*R)]
#                 ])
#     val = la.det(mat)
#     DDet.append(val)
# DDet = np.array(DDet)



def fdet(k, p, n=4, R=1):
    ke, ki = k, np.sqrt(n) * k
    val =  -J(p, ke*R)*ki*Jp(p, ki*R) + ke*Jp(p, ke*R)*J(p, ki*R)
    return val

def locminmax(f):
    fm, I = [-1], [-1]
    fj =  0
    prec = 'pos'
    for i, fi in enumerate(f):
        if fi - fj > 0:
            if prec == 'neg':
                fm.append(fi)
                I.append(i)
                prec = 'pos'
        elif fi - fj < 0:
            if prec == 'pos':
                fm.append(fi)
                I.append(i)
                prec = 'neg'
        else:
            print('warning')
        fj = fi
    fm, I = fm[1:], I[1:]
    return fm, I

def changedsign(f):
    fm, I = list([0]), list([0])
    fj =  0
    prec = 'pos'
    for i, fi in enumerate(f):
        if fi - fj > 0:
            if prec == 'neg':
                if fm[-1] * fi <= 0.:
                    fm.append(fi)
                    I.append(i)
                prec = 'pos'
        elif fi - fj < 0:
            if prec == 'pos':
                if fm[-1] * fi <= 0.:
                    fm.append(fi)
                    I.append(i)
                prec = 'neg'
        else:
            print('warning')
        fj = fi
    fm, I = fm[1:], I[1:]
    return fm, I


Z = []
for ii in range(m):
    d, ind = changedsign(fdet(k, ii, n, R))
    for p in range(len(ind)-1):
        j, jj = ind[p], ind[p+1]
        a, b = k[j], k[jj]
        x, y = fdet(a, ii, n, R), fdet(b, ii, n, R)
        if x*y > 0:
            print('bad sign', x, y, a, b, ii)
        z = brentq(fdet, a, b, args=(ii, n, R))
        Z.append(z)
Z = np.array(Z)
Z.sort()

plt.figure()
plt.plot(k, 0*k, 'k--')
for ii in range(m):
    plt.plot(k, det(ii), label='n={}'.format(ii))
    plt.plot(Z, 0. * Z, 'gd')
    # d, ind = changedsign(fdet(k, ii, n, R))
    # plt.plot(k[ind], d, 's')


plt.plot(3.14159, 0, 'k.', label='from paper')
plt.plot(3.69245, 0, 'k.')
plt.plot(4.26168, 0, 'k.')
plt.plot(4.83186, 0, 'k.')

plt.title('determinant of Analytic by Harmonic (n)')
plt.xlabel('(real) wavenumber')
plt.ylabel('det')
plt.legend()
plt.show(block=False)

print(Z)
