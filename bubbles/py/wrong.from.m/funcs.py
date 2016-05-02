
from numpy import sqrt, pi, cos, abs, exp, arctan2
from scipy.special import jv, hankel1, lpmn, factorial

def dsbesf1(m, z):
    """
    This function is based on information given in "Vibration and Sound",
    by Philip M. Morse, 2nd Edition., pp. 316-317.

    It uses recursion relations to compute the first derivative
    of spherical Bessel functions
    """

    if m == 0:
        y = - sqrt(0.5 * pi / z ) * jv(1.5, z);
    else:
        y = (m / (2*m+1)) * sqrt(0.5 * pi /z ) * jv(m - 0.5, z)
        t = ((m+1) / (2*m+1)) * sqrt(0.5 * pi /z) * jv(m + 1.5, z)
        y -= t
    return y

def dshaf11(m, z):
    """
    This function is based on information given in "Vibration and Sound",
    by Philip M. Morse, 2nd Edition., pp. 316-317.
    """

    if m == 0:
        y = - sqrt(0.5 * pi /z) * hankel1(1.5, z);
    else:
        y = (m / (2*m+1)) * sqrt(0.5 * pi /z) * hankel1(m - 0.5, z)
        t = ((m+1) / (2*m+1)) * sqrt(0.5 * pi /z) * hankel1(m + 1.5, z)
        y -= t
    return y

hn = lambda z, n: sqrt(0.5 * pi / z) * hankel1(n + 0.5, z)
jn = lambda z, n: sqrt(0.5 * pi / z) * jv(n + 0.5, z)

def Ylm(l, m, theta, phi):
    P, _ = lpmn(l, l, cos(theta))
    Y = sqrt((2*l+1) * factorial(l - abs(m)) / (4*pi*factorial((l+abs(m))))) * exp(1j * m * phi)
    Y *= P[abs(m),-1]
    return Y


def cart2sph(myx, myy, myz):
    XsqPlusYsq = myx**2 + myy**2
    r = sqrt(XsqPlusYsq + myz**2)
    elev = arctan2(myz, sqrt(XsqPlusYsq))
    az = arctan2(myy, myx)
    return az, elev, r
