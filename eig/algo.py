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


def solve(Mat, b, tol=1e-6, res=[], restart=None, maxiter=300):
    del res
    res = []
    tt = time()
    x, info = gmres(Mat, b,
                    orthog='mgs',
                    tol=tol,
                    residuals=res,
                    restrt=restart,
                    maxiter=maxiter)
    tt = time() - tt
    print(info, len(res))
    res = np.array(res)
    return x

def rhs(m, l, space, rand=True):
    print('Computing rhs', m, l)
    if rand:
        VVhat = np.random.rand(m, l) + 1j * np.random.rand(m, l)
    else:
        VVhat = np.eye(m)
        VVhat = Vhat[:, 0:l]
    Vhat = np.zeros((m, 1), dtype=np.complex)
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
    return Vhat
