
from time import time

import numpy as np
from scipy.special import lpmn as legendre, factorial

from funcs import dsbesf1, dshaf11
from funcs import jn, hn
from funcs import Ylm
from funcs import cart2sph

from integ import integ

tt = time()

MM = 3

a = 0.005
freq = 650
N = 2
ww = 2 * np.pi * freq
rho0 = 1030; c0 = 1500; k0 = ww / c0;
rho1 = 1.18; c1 = 344.4475;  k1 = ww/c1;
a1 = rho1 / rho0;    a1inv = 1/a1;
PSM = 0;
Amp = 1;

dth, dphi = np.pi / 3, np.pi / 3

# thetas = np.arange(dth, np.pi - dth, dth, dtype=float)
# phis = np.arange(0, np.pi - dphi, dphi, dtype=float)

thetas = np.arange(dth, np.pi, dth, dtype=float)
phis = np.arange(0, 2*np.pi, dphi, dtype=float)

ik0, four_pi = 1j * k0, 4 * np.pi
delta_square = dth*dphi * dth*dphi
a4 = a**4

PSL = np.empty((MM+1, ), dtype=complex)
for l in range(0, MM+1):
    z0 = a  *k0
    K0 = - 1j * k0**2 *a4 * dsbesf1(l, z0) * hn(z0, l)
    W0 = - 1j * k0**3 *a4 * dsbesf1(l, z0) * dshaf11(l, z0)
    Ks0 = - 1j * k0**2 * a4 * jn(z0, l) * dshaf11(l, z0)
    V0 = 1j * k0 * a4 * hn(z0, l) * jn(z0, l)

    A0 = np.array([[-K0, V0], [W0, Ks0]], dtype=complex)

    z1 = a  *k1
    K1 = 1j * k1**2 * a**4 * jn(z1, l) * dshaf11(l, z1)
    Ks1 = 1j * k1**2 * a**4 * dsbesf1(l, z1) * hn(z1, l)
    W1 = -1j * k1**3 * a**4 * dshaf11(l, z1) * dsbesf1(l, z1)
    V1 = 1j * k1 * a**4 * jn(z1, l) * hn(z1, l)

    A1 = np.array([[-K1, V1], [W1, Ks1]], dtype=complex)

    for m in range(-l, l+1):
        dV0 = integ(l, m, a, k0, thetas, phis)
        print(dV0)
        V01 = dV0
        K01 =0
        W01 = 0
        Ks01 = 0
        A01 = np.array([[-K01, V01], [ W01, Ks01]], dtype=complex)

        ##
        Xx = a**2 * np.array([[1, 0], [0, -a1]], dtype=complex)
        Xinv = a**2 * np.array([[1, 0], [0, -a1inv]], dtype=complex)

        N_A0 = np.empty((2*N, 2*N), dtype=complex)
        N_A1 = np.empty((2*N, 2*N), dtype=complex)
        N_Xx = np.empty((2*N, 2*N), dtype=complex)
        N_Xinv = np.empty((2*N, 2*N), dtype=complex)
        for iN in range(1, N+1):
            N_A0[2 * iN - 1 -1 :2*iN, 2*iN - 1 -1 :2*iN] = A0
            N_A1[2 * iN - 1 -1 :2*iN, 2*iN - 1 -1 :2*iN] = A1
            N_Xx[2 * iN - 1 -1 :2*iN, 2*iN - 1 -1 :2*iN] = Xx
            N_Xinv[2 * iN - 1 -1 :2*iN, 2*iN - 1 -1 :2*iN] = Xinv

        N_A0[0:2, 2:4] = A01
        N_A0[2:4, 0:2] = A01

        Ma = np.concatenate((N_A0, -N_Xinv), axis=1)
        Mb = np.concatenate((-N_Xx, N_A1), axis=1)
        M = np.concatenate((Ma, Mb), axis=0)


        # rhs

        SrcX = -10;

        phs1 = np.exp(-1j * k0 * (SrcX+(20*a)) )
        phs2 = np.exp(-1j * k0 * (SrcX-(20*a)) )

        vec = np.array([[-jn(z0,l)*phs1],
                        [dsbesf1(l,z0)*k0*phs1],
                        [-jn(z0,l)*phs2],
                        [dsbesf1(l,z0)*k0*phs2],
                        [jn(z0,l)*phs1],
                        [a1*dsbesf1(l,z0)*k0*phs1],
                        [jn(z0,l)*phs2],
                        [a1*dsbesf1(l,z0)*k0*phs2]])
        b= 2 *np.sqrt((2*l+1) * np.pi) * Amp * (1j)**l * a**2 * vec

        pos = np.array([[-20*a, 0, 0], [20*a, 0, 0]])

        x = np.linalg.solve(M, b)

        D1 = dsbesf1(l, k0*a)
        J1 = jn(k0*a, l)
        R = np.array([-1, 0, 0])
        DRecN = np.empty((N, ))
        psb2 = np.empty((N, ), dtype=complex)
        for ii in range(1, N+1):
            th, phi, DRec = cart2sph(R[0]-pos[ii-1, 0], R[1]-pos[ii-1, 1], R[2]-pos[ii-1, 2])
            DRecN[ii-1] = DRec

            P, _ = legendre(l, l, np.cos(th))
            P = P[:, -1]
            Y = np.sqrt((2*l+1) * factorial(l - abs(m)) / (4 * np.pi*factorial(l+abs(m))))
            Y *= np.exp(1j * m * phi) * P[0]
            z2 = hn(k0 * DRec, l)

            tmp= 1j * k0 * a**2* Y * (k0*x[2*ii-1-1] * D1 * z2 + x[2*ii-1] * J1 * z2)
            tmp = tmp[0]
            psb2[ii-1] = np.abs(tmp) * np.exp(-1j*np.angle(tmp))

        ps = np.sum(psb2);
        PSM = PSM + ps;

    PSL[l] = PSM

conv = np.abs(PSL-PSL[-1]) / np.abs(PSL[-1])
print(conv)

tt = time() - tt
print('')
print('time', tt)
