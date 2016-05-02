
import numpy as np
from scipy.special import lpmn as legendre, factorial

from funcs import dsbesf1, dshaf11
from funcs import jn, hn
from funcs import Ylm
from funcs import cart2sph

def integ(l, m, a, k0, thetas, phis):
    dth, dphi = thetas[1] - thetas[0] , phis[1] - phis[0],

    ik0, four_pi = 1j * k0, 4 * np.pi
    delta_square = dth*dphi * dth*dphi
    a4 = a**4

    dV0 = 0 + 0j
    for th1 in thetas:
        for phi1 in phis:
            sTcP1 = np.sin(th1) * np.cos(phi1) - 40
            sTsP1 = np.sin(th1) * np.sin(phi1)
            cT1 = np.cos(th1)
            Y1_delta = a4 * np.conj(Ylm(l, m, th1, phi1)) * np.sin(th1) * delta_square
            for th2 in thetas:
                for phi2 in phis:

                    tmp_sc = sTcP1 - np.sin(th2) * np.cos(phi2)
                    tmp_ss = sTsP1 - np.sin(th2) * np.sin(phi2)
                    tmp_c = cT1 - np.cos(th2)
                    dd = a * np.sqrt(tmp_sc**2 + tmp_ss**2 + tmp_c**2)
                    g0 = np.exp(ik0 * dd) / (four_pi * dd)

                    dV0 += Ylm(l, m, th2, phi2) * np.sin(th2) * g0 * Y1_delta
    return dV0
