#!/usr/bin/env python
# coding: utf8

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import bempp.api as bem

eps = 1.

k0 = 0.3 * np.pi
k1 = eps * k0
k2 = eps * k0

kRef = k0

iincident = 1

def fdir(x, normal, dom_ind, result):
    result[0] =  - np.exp( 1j * kRef * x[iincident])

def fneu(x, normal, dom_ind, result):
    result[0] = -1j * normal[iincident] * kRef * np.exp( 1j * kRef * x[iincident])

def fzero(point, normal, dom_ind, result):
    result[0] = 0. + 1j * 0.
