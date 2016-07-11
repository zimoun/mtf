#!/usr/bin/env python
# coding: utf8

from time import time
from os import system

import numpy as np
import scipy.linalg as la
import scipy.io as sio
import matplotlib.pyplot as plt

import bempp.api as bem

from assemb import MultiTrace, checker
from domains import sanitize_config, generate_disjoint_dict, write_params_geo

from miesphere import mie_D4grid, mie_N4grid
from krylov import gmres

#################################################

kRef_rc = 0.1 * np.pi
eps_rc = 2

N = 1
geoconf = sanitize_config(Nints=N,
                          names=[ str(i) for i in range(N+1) ],
                          phys=(eps_rc, 1, 1),
                          kRef=kRef_rc,
                          init_offset=True)
dd = generate_disjoint_dict(geoconf)
cmds = write_params_geo(geoconf)

nlambdas = [5, 10, 20, 40, 80, 160]

#################################################

kRef = geoconf['kRef']
eps, _, _ = geoconf['phys'][1]
Ndom = len(geoconf['names'])

#################################################

iincident = 1

def dir_data(x, normal, dom_ind, result):
    result[0] =  -np.exp( 1j * kRef * x[iincident])

def neu_data(x, normal, dom_ind, result):
    result[0] = -1j * normal[iincident] * kRef * np.exp( 1j * kRef * x[iincident])

#################################################

C = np.array([0, 0, 0])
k = kRef
kk = [0, 0, 0]
for q in range(3):
    if q == iincident:
        kk[q] = k
kk = tuple(kk)
R = 1
ce, ci = 1, np.sqrt(eps)
jumpe, jumpi  = (1, 1), (1, 1)
Nmodes = 50
field = 'sca'
# field = 'int'
# field = 'inc'

def mieD(point, normal, dom_ind, result):
    val = mie_D4grid(field, kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    result[0] = val

def uinc(point, normal, dom_ind, result):
    result[0] = np.exp(1j * kRef * point[iincident])

def mieN(point, normal, dom_ind, result):
    val = mie_N4grid(field, kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    result[0] = val

def dnuinc(point, normal, dom_ind, result):
    result[0] = 1j * kRef * normal[1] * np.exp(1j * kRef * point[iincident])

def mieD_int(point, normal, dom_ind, result):
    val = mie_D4grid('int', kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    result[0] = val

def mieN_int(point, normal, dom_ind, result):
    val = mie_N4grid('int', kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    result[0] = val

#################################################

def rescaleRes(res, P, b):
    scale = 1.0 / la.norm(P(b))
    new_res = scale * res
    return new_res

def my_gmres(mystr, M, b, tol, restart, maxiter):
    print(mystr.format(restart, maxiter), flush=True)
    res = []
    tt = time()
    xx, info = gmres(M, b,
                     orthog='mgs',
                     tol=tol,
                     residuals=res,
                     restrt=restart,
                     maxiter=maxiter)
    tt = time() - tt
    print(info, len(res))
    oRes = np.array(res)
    Res = rescaleRes(oRes, lambda x: x, b)
    print('#time: {}'.format(tt))
    return (xx, len(Res), Res, tt)

tol = 1e-9
restart = None
if restart is None: scale = 1
else: scale = restart
maxiter = 1000

#################################################

Size = []
H = []
ll = 2. * np.pi * kRef

Ecald, Etrans = [], []
Ecald_mie, Etrans_mie = [], []

dEl2, dEnl2 = [], []
dEL2, dEnL2 = [], []

nEl2, nEnl2 = [], []
nEL2, nEnL2 = [], []

for nlambda in nlambdas:

    print('\n')
    print('##################')
    print('\n')

    geoconf['nlambda'] = nlambda
    cmds = write_params_geo(geoconf)
    system("gmsh geo/sphere-disjoint.script.geo -")

    H.append(ll / nlambda)

    mtf = MultiTrace(geoconf['kRef'], 'geo/'+geoconf['meshname'], dd)

    Aw = mtf.A_weak_form()

    At, X, J, iJ = mtf.tolinop()

    shape = mtf.shape
    size, _ = shape
    Size.append(size)

    A = 2.0 * At
    A2 = A * iJ * A

    Ce = 0.5 * J - At
    Ci = 0.5 * J + At

    Ce2 = Ce * iJ * Ce
    Ci2 = Ci * iJ * Ci

    x = np.random.rand(shape[0]) + 1j * np.random.rand(shape[0])

    b = mtf.rhs(dir_data, neu_data)
    M = A - X

    print('')
    print(nlambda, mtf.shape, flush=True)
    print('')

    checker('A2 = J', A2, J, x)
    checker('exterior Proj.', Ce2, Ce, x)
    checker('interior Proj.', Ci2, Ci, x)
    checker('error-Calderon with random [no-sense]', A, J, x)

    xx, niter, res, tt = my_gmres("\nGmres restart={0} maxiter={1}",
                                   M, b, tol, restart, maxiter)

    ecald = checker('Calderon ', A, J, xx)
    etrans = checker('Transmission ', J, X, xx, b)

    Ecald.append(ecald)
    Etrans.append(etrans)

    slices = mtf.getSlices()

    s = slices['0']
    sol = xx[s[0]:s[1]]

    s = slices['1']
    soll = xx[s[0]:s[1]]

    d = mtf.domains.getIndexDom('0')
    (space, _) , (_, _) = mtf.spaces[d]

    n, = sol.shape
    n = int(n/2)
    sold, soln = sol[:n], sol[n:]

    print('\ndir. err, |err|, err_norm ; l2 <-| L2')

    gsold = bem.GridFunction(space, coefficients=sold)

    gmie = bem.GridFunction(space, fun=mieD)
    miecoeffs = gmie.coefficients
    ggmie = bem.GridFunction(space, coefficients=miecoeffs)

    errd = sold - miecoeffs
    aerrd = np.abs(errd)
    gerrd = bem.GridFunction(space, coefficients=errd)
    gaerrd = bem.GridFunction(space, coefficients=aerrd)
    el = la.norm(errd)
    enl = el / la.norm(miecoeffs)
    print(el, la.norm(aerrd), enl)
    eL = gerrd.l2_norm()
    enL = eL / gmie.l2_norm()
    print(eL, gaerrd.l2_norm(), enL)

    dEl2.append(el)
    dEnl2.append(enl)

    dEL2.append(eL)
    dEnL2.append(enL)

    print('\nneu. err, |err|, err_norm ; l2 <-| L2')

    fmie = bem.GridFunction(space, fun=mieN)
    dnmiecoeffs = fmie.coefficients

    errn = soln + dnmiecoeffs
    aerrn = np.abs(errn)
    ferrn = bem.GridFunction(space, coefficients=errn)
    faerrn = bem.GridFunction(space, coefficients=aerrn)
    el = la.norm(errn)
    enl = el / la.norm(dnmiecoeffs)
    print(el, la.norm(aerrn), enl)
    eL = ferrn.l2_norm()
    enL = eL /fmie.l2_norm()
    print(eL, faerrn.l2_norm(), enL)

    nEl2.append(el)
    nEnl2.append(enl)

    nEL2.append(eL)
    nEnL2.append(enL)

    print('\ninc.')

    gui = bem.GridFunction(space, fun=uinc)
    uicoeffs = gui.coefficients

    fui = bem.GridFunction(space, fun=dnuinc)
    dnuicoeffs = fui.coefficients

    print('interior')

    d = mtf.domains.getIndexDom('1')
    (space_int, _) , (_, _) = mtf.spaces[d]

    gmie_int = bem.GridFunction(space_int, fun=mieD_int)
    miecoeffs_int = gmie_int.coefficients
    fmie_int = bem.GridFunction(space_int, fun=mieN_int)
    dnmiecoeffs_int = fmie_int.coefficients

    ye = np.concatenate((miecoeffs, -dnmiecoeffs))
    yi = np.concatenate((miecoeffs_int, dnmiecoeffs_int))
    yy = np.concatenate((ye, yi))

    print('zeros')
    def zeros(point, normal, dom, res):
        res[0] = 0.0 + 1j * 0.0
    for ndom in range(2, Ndom):
        print(ndom)
        d = mtf.domains.getIndexDom(str(ndom))
        (space_b, _) , (_, _) = mtf.spaces[d]
        fmie_b = bem.GridFunction(space_b, fun=zeros)
        miecoeffs_b = fmie_b.coefficients
        yb = np.concatenate((miecoeffs_b, miecoeffs_b))
        yy = np.concatenate((yy, yb))

    ecald_mie = checker("Calderon Mie", A, J, yy)
    etrans_mie = checker('Transmission Mie', J, X, yy, b)

    Ecald_mie.append(ecald_mie)
    Etrans_mie.append(etrans_mie)


sio.savemat('dat/err.mat',
            {'kRef':kRef, 'eps':eps, 'Ndom':Ndom,
             'Size':Size,
             'H':H,
             'Nlambdas':nlambdas,
             'Ecald':Ecald, 'Etrans':Etrans,
             'Ecald_mie':Ecald_mie, 'Etrans_mie':Etrans_mie,
             'dEL2': dEL2, 'nEL2':nEL2,
             'dEl2': dEl2, 'nEl2':nEl2,
             'dEnL2':dEnL2, 'nEnL2':nEnL2,
             'dEnl2':dEnl2, 'nEnl2':nEnl2,
            })
