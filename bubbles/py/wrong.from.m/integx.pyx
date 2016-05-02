# cython: boundscheck=False, wraparound=False, cdivision=True, cdivision_warnings=False

import numpy as np
from scipy.special import lpmn as legendre, factorial

from funcs import Ylm

cimport numpy as np
from libc.math cimport M_PI
from libc.math cimport cos, sin
from libc.math cimport sqrt, exp

cdef extern from "complex.h":
    double complex cexp(double complex)

def integ(int l, int m, float a, float k0,
          np.ndarray[np.float64_t, ndim=1] thetas,
          np.ndarray[np.float64_t, ndim=1] phis):

    cdef int i1, j1, i2, j2
    cdef float th1, phi1, th2, phi2

    cdef float dd

    cdef complex dth, dphi
    cdef complex ik0, four_pi
    cdef complex a4, delta_square
    cdef complex dV0
    cdef int Ntheta, Nphi

    cdef float sTcP1, sTsP1, cT1
    cdef float sTcP2, sTsP2, cT2
    cdef complex Y1_delta
    cdef complex tmp1, tmp2

    cdef complex t

    Ntheta = thetas.shape[0]
    Nphi = phis.shape[0]

    dth, dphi = thetas[1] - thetas[0] , phis[1] - phis[0],

    ik0, four_pi = 1j * k0, 4 * M_PI
    delta_square = dth*dphi * dth*dphi
    a4 = a**4

    dV0 = 0 + 0j
    for i1 in range(Ntheta):
        th1 = thetas[i1]
        for j1 in range(Nphi):
            phi1 = phis[i1]
            sTcP1 = sin(th1) * cos(phi1) - 40
            sTsP1 = sin(th1) * sin(phi1)
            cT1 = cos(th1)
            Y1_delta = a4 * np.conj(Ylm(l, m, th1, phi1)) * sin(th1) * delta_square
            for i2 in range(Ntheta):
                th2 = thetas[i2]
                for j2 in range(Nphi):
                    phi2 = phis[j2]

                    tmp_sc = sTcP1 - sin(th2) * cos(phi2)
                    tmp_ss = sTsP1 - sin(th2) * sin(phi2)
                    tmp_c = cT1 - cos(th2)
                    dd = a * sqrt(tmp_sc**2 + tmp_ss**2 + tmp_c**2)
                    # t = cexp(ik0 * dd) * (four_pi * dd)
                    g0 = np.exp(ik0 * dd) / (four_pi * dd)
                    tmp1 = sin(th2) * g0 * Y1_delta
                    tmp2 = Ylm(l, m, th2, phi2)

                    dV0 += tmp1 * tmp2
    return dV0
