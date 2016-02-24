#/usr/bin/env python

import scipy as sp

from scipy.special import jv, yv
from scipy.special import sph_jn, sph_yn
from scipy.special import eval_legendre

from mesh import Mesh, Cartesian
from mesh import myrange, stdoutw, stdoutf

def _print_progress(ii, N, what='done', mod=10):
    per = int(100*(ii+1)/N)
    if per%mod == 0:
        stdoutw('\rInfo: {0}% {3}. [{1}/{2}]'.format(
            per, ii+1, N, what))
        stdoutf()


def fun_cart2sph(F):
    def f(n, z):
        return sp.sqrt(sp.pi/(2*z))* F(n+1/2.0, z)
        #return sp.sqrt(1/z)* F(n+1/2.0, z)
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


def coeff_ext(n, params, k):
    R, ks, jumps = params
    ce, ci = ks
    jumpe, jumpi = jumps
    ke, ki = ce*k, ci*k
    ae, be = jumpe
    ai, bi = jumpi
    a = ke*J(n, ki*R)*Jp(n, ke*R) * ae*ai
    b = ki*Jp(n, ki*R)*J(n, ke*R) * ae*bi
    c = ki*H1(n, ke*R)*Jp(n, ki*R) * ae*bi
    d = ke*H1p(n, ke*R)*J(n, ki*R) * ai*be
    return (2*n+1)*( (1j)**n ) * (a - b) / (c - d)

def coeff_int(n, params, k):
    R, ks, jumps = params
    ce, ci = ks
    jumpe, jumpi = jumps
    ke, ki = ce*k, ci*k
    ae, be = jumpe
    ai, bi = jumpi
    a = ke*H1(n, ke*R)*Jp(n, ke*R) * be*ae
    b = ke*J(n, ke*R)*H1p(n, ke*R) * be*be
    c = ki*H1(n, ke*R)*Jp(n, ki*R) * ae*bi
    d = ke*H1p(n, ke*R)*J(n, ki*R) * ai*be
    return (2*n+1)*( (1j)**n ) * (a - b) / (c - d)


def ref_inc(mesh, Params, name='ref-inc.pos', ext=True):
    count = 0
    N, params, kk = Params
    R, (ce, ci) = params
    kk = Cartesian(kk)
    k = kk.norm()
    kk = kk.normalized()
    vals = [0+0j] * len(mesh.points)
    for ii, point in enumerate(mesh.points):
        _print_progress(ii, len(mesh.points), 'computed', mod=1)
        p = Cartesian(point)
        x, y, z = point
        count += 1
        if ins:
            val = sp.exp(1j * ci * k * kk.dot(p))
        else:
            val = sp.exp(1j * ce * k * kk.dot(p))
        vals[ii] = val
    print(' --> {0} computations.'.format(count))
    mesh.write(vals, name)

def mie_sphere(mesh, Params, name='mie-sphere.pos', field='sca'):
    count = 0
    N, params, kk = Params
    R, (ce, ci), jumps = params
    kk = Cartesian(kk)
    k = kk.norm()
    kk = kk.normalized()
    vals = [0+0j] * len(mesh.points)
    for ii, point in enumerate(mesh.points):
        _print_progress(ii, len(mesh.points), 'computed', mod=1)
        p = Cartesian(point)
        pnorm = p.norm()
        pn = p.normalized()
        costheta = pn.dot(kk)
        for n in myrange(N):
            if field == 'sca':
                cn = coeff_ext(n, params, k) * H1(n, k*pnorm)
            elif field == 'int':
                cn = coeff_int(n, params, k) * J(n, ci*k*pnorm)
            else:
                cn = ((1j)**n)*(2*n+1)*J(n, k*pnorm)
            c = eval_legendre(n, costheta)
            count += 1
            vals[ii] += cn * c
    print(' --> {1} computations i.e. N={0}.'.format(N, count))
    mesh.write(vals, name)
    return vals

def mie_D4grid(field, kk, R, C, ce, ci, jumpe, jumpi, N, point):
    """
    Requires:
     kk : numpy.array([kx, ky, kz])
     R  : radius of the sphere
     C  : center of the sphere
     ce, ci : contrast sqrt(epsExt), sqrt*espInt)
     jumpe: coeff jump exterior (alpha_Dir, beta_Neu)
     jumpi: coeff jump interior (alpha_Dir, beta_Neu)
     N  : Number of modes
    """
    pt = point[:]
    kk = Cartesian(kk)
    k = kk.norm()
    kk = kk.normalized()
    # be careful with this test !!
    if sp.linalg.norm(sp.linalg.norm(pt - C) - R) > 0.3:
        return 0. + 0j
    else:
        jumps = (jumpe, jumpi)
        params = (R, (ce, ci), jumps)
        p = Cartesian((pt[0], pt[1], pt[2]))
        pnorm = p.norm()
        pn = p.normalized()
        costheta = pn.dot(kk)
        val = 0
        for n in myrange(N):
            if field == 'sca':
                cn = coeff_ext(n, params, k) * H1(n, k*pnorm)
            elif field == 'int':
                cn = coeff_int(n, params, k) * J(n, ci*k*pnorm)
            else:
                cn = ((1j)**n)*(2*n+1)*J(n, k*pnorm)
            c = eval_legendre(n, costheta)
            val += cn * c
    return val

def mie_N4grid(field, kk, R, C, ce, ci, jumpe, jumpi, N, point):
    """
    Requires:
     kk : numpy.array([kx, ky, kz])
     R  : radius of the sphere
     C  : center of the sphere
     ce, ci : contrast sqrt(epsExt), sqrt*espInt)
     jumpe: coeff jump exterior (alpha_Dir, beta_Neu)
     jumpi: coeff jump interior (alpha_Dir, beta_Neu)
     N  : Number of modes
    """
    pt = point[:]
    kk = Cartesian(kk)
    k = kk.norm()
    kk = kk.normalized()
    # be careful with this test !!
    if sp.linalg.norm(sp.linalg.norm(pt - C) - R) > 0.3:
        return 0. + 0j
    else:
        jumps = (jumpe, jumpi)
        params = (R, (ce, ci), jumps)
        p = Cartesian((pt[0], pt[1], pt[2]))
        pnorm = p.norm()
        pn = p.normalized()
        costheta = pn.dot(kk)
        val = 0
        for n in myrange(N):
            if field == 'sca':
                cn = k * coeff_ext(n, params, k) * H1p(n, k*pnorm)
            elif field == 'int':
                cn = ci * k * coeff_int(n, params, k) * Jp(n, ci*k*pnorm)
            else:
                cn = k * ((1j)**n)*(2*n+1)*Jp(n, k*pnorm)
            c = eval_legendre(n, costheta)
            val += cn * c
    return val


if __name__ == "__main__":
    epsExt, epsInt = 1, 2
    ll = 9.5

    R = 1.0
    N = 10

    # k = 2*sp.pi / ll
    k = 0.3 * sp.pi # 3.1
    kk = (0., k, 0.)

    params = R, (sp.sqrt(epsExt), sp.sqrt(epsInt))
    Params = N, params, kk

    m = Mesh('geo/sphere-disjoint.msh', 1)
    sca = mie_sphere(m, Params, 'mie{}-sca.pos'.format(N), field='sca')
    ins = mie_sphere(m, Params, 'mie{}-int.pos'.format(N), field='int')
    inc = mie_sphere(m, Params, 'mie{}-inc.pos'.format(N), field='inc')

    ref_inc(m, Params)
    ref_inc(m, Params, name='ref-inc-int.pos', ext=False)

    ext = [ v  for ii, v in enumerate(sca) ]
    m.write(ext, 'mie{}-ext.pos'.format(N))
    diff = [ v - ins[ii] -inc[ii] for ii, v in enumerate(ext) ]
    m.write(diff, 'mie{}-diff.pos'.format(N))
