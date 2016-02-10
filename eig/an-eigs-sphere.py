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

from scipy.optimize import brentq, root

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


n = 4
k = np.linspace(0.5, 8, 50)
R = 1
m = 7


def fdet(k, p, n=4, R=1):
    k_r, k_i = k[0], k[1]
    kk = k_r + 1j * k_i
    ke, ki = kk, np.sqrt(n) * kk
    val =  -H1(p, ke*R)*ki*Jp(p, ki*R) + ke*H1p(p, ke*R)*J(p, ki*R)
    return np.array([val.real, val.imag], dtype=np.float)



plt.figure()
plt.plot(k, 0*k, 'k--')
for ii in range(m):
    for kk in k:
        xo = [kk, 0.]
        sol = root(fdet, xo, args=(ii, n, R))
        if sol.success:
            x = sol.x
            plt.plot(x[0], x[1], '.') # label='n={}'.format(ii))
        else:
            print('not found', ii, xo)

plt.title('zeros of determinant of Analytic by Harmonic (n)')
plt.xlabel('real')
plt.ylabel('imag')
plt.legend()
plt.show(block=False)

# print(Z)
