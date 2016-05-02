
import numpy as np
from scipy.special import jv as besselj
from scipy.special import yv as bessely

import numpy.linalg as la

def fun_cart2sph(F):
    def f(n, z):
        y = np.sqrt(0.5 * np.pi / z) * F(n + 0.5, z)
        return y
    return f

def fun_derivative(F):
    def f(n, z):
        y = (n / z) * F(n, z) - F(n + 1, z)
        return y
    return f

J = fun_cart2sph(besselj)
Jp = fun_derivative(J)

Y = fun_cart2sph(bessely)
Yp = fun_derivative(Y)

H1 = lambda n, z: J(n, z) + 1j * Y(n, z)
H1p = lambda n, z: Jp(n, z) + 1j * Yp(n, z)


def get_size(Nmodes):
    size = 0
    for m in range(Nmodes):
        size += len(range(-m, m+1))
    return size, size

def self_interaction(Nmodes, k, rad, formula):
    size = get_size(Nmodes)
    N, _ = size
    Op = np.zeros(size, dtype=complex)
    for i in range(N):
        Op[i, i] = formula(i, k, rad)
    return Op

formula_single = lambda l, k, rad: 1j * k * (rad**4) * J(l, k * rad) * H1(l, k * rad)
formula_double = lambda l, k, rad: 1j * (k**2) * (rad**4) * Jp(l, k * rad) * H1(l, k * rad)
formula_adjoint = lambda l, k, rad: 1j * (k**2) * (rad**4) * J(l, k * rad) * H1p(l, k * rad)
formula_hyper = lambda l, k, rad: -1j * (k**3) * (rad**4) * Jp(l, k * rad) * H1p(l, k * rad)
formula_identity = lambda l, k, rad: rad**2


def single_layer(Nmodes, k, rad=1):
    op = self_interaction(Nmodes, k, rad, formula_single)
    return op
def double_layer(Nmodes, k, rad=1):
    """
    Return K
    Warning about sign of normal: K0 = -K ; K1 = K
    """
    op = self_interaction(Nmodes, k, rad, formula_double)
    return op
def adjoint_layer(Nmodes, k, rad=1):
    """
    Return Q
    Warning about sign of normal: Q0 = -Q ; Q1 = Q
    """
    op = self_interaction(Nmodes, k, rad, formula_adjoint)
    return op
def hyper_layer(Nmodes, k, rad=1):
    op = self_interaction(Nmodes, k, rad, formula_hyper)
    return op
def identity_layer(Nmodes, k, rad=1):
    op = self_interaction(Nmodes, k, rad, formula_identity)
    return op

k0 = 1.
k1 = 2.

rad = 1.

Nmodes = 8
shape = get_size(Nmodes)
N, _ = shape

V0 = single_layer(Nmodes, k0, rad)
K0 = double_layer(Nmodes, k0, rad)
Q0 = adjoint_layer(Nmodes, k0, rad)
W0 = hyper_layer(Nmodes, k0, rad)

V1 = single_layer(Nmodes, k1, rad)
K1 = double_layer(Nmodes, k1, rad)
Q1 = adjoint_layer(Nmodes, k1, rad)
W1 = hyper_layer(Nmodes, k1, rad)

Id = identity_layer(Nmodes, 1., rad)

A = np.zeros((4*N, 4*N), dtype=complex)
X = np.zeros((4*N, 4*N), dtype=complex)

def cald(K, V, W, Q):
    a = np.concatenate((K, V), axis=1)
    b = np.concatenate((W, Q), axis=1)
    c = np.concatenate((a, b), axis=0)
    return c

A0 = cald(K0, V0, W0, -Q0)
A1 = cald(-K1, V1, W1, Q1)

# AA = zeros(Complex, 4N ,4N)
# AA[1:2N, 1:2N] = A0
# AA[2N+1:end, 2N+1:end] = A1

# two = 1 + 0im
# D0, _ = eig(two*A0)
# D1, _ = eig(two*A1)
# D, _ = eig(two*A)
# DD, _ = eig(two*AA)
