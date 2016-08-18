#!/usr/bin/env python
# coding: utf8

# ugly hack to do relative import
import sys
sys.path.append('..')

from time import time
from os import system

import numpy as np
import scipy.linalg as la
import scipy.io as sio
import matplotlib.pyplot as plt

import bempp.api as bem

from miesphere import mie_D4grid, mie_N4grid
from krylov import gmres

from assemb import checker

from my_relative_error import relative_error

from xtf import xTF, STF, MTF

bem.global_parameters.hmat.coarsening = False

kRef_rc = 0.1 * np.pi
eps_rc = 2
alpha_rc, beta_rc = 1-00j, 1-00j
# nlambdas = [5] #, 40, 80, 160]
# nlambdas = [40]
nlambdas = [5, 10, 20, 40, 80, 160]
# nlambdas = [5, 10, 20, 40, 80]#, 160]
#nlambdas = [50, 70, 80, 100, 160]

#################################################

kRef = kRef_rc
eps, alpha, beta = eps_rc, alpha_rc, beta_rc
Ndom = 2

#################################################

iincident = 1

def dir_data(x, normal, dom_ind, result):
    result[0] =  -np.exp(1j * kRef * x[iincident])

def neu_data(x, normal, dom_ind, result):
    result[0] = -1j * normal[iincident] * kRef * np.exp(1j * kRef * x[iincident])

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
jumpe, jumpi  = (1, 1), (alpha, beta)
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

##

def fmieD(point):
    val = mie_D4grid(field, kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    return val

def fuinc(point):
    return np.exp(1j * kRef * point[iincident])

def fmieN(point):
    """Return -neumann_solution"""
    val = mie_N4grid(field, kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    return -1.0 * val

def fdnuinc(point):
    return 1j * kRef * normal[1] * np.exp(1j * kRef * point[iincident])

def fmieD_int(point):
    val = mie_D4grid('int', kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    return val

def fmieN_int(point):
    val = mie_N4grid('int', kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    return val

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

##

Ecald_stf, Etrans_stf = [], []
Ecald_stf_mie_stf, Etrans_mie = [], []

dEl2_stf, dEnl2_stf = [], []
dEL2_stf, dEnL2_stf = [], []

nEl2_stf, nEnl2_stf = [], []
nEL2_stf, nEnL2_stf = [], []

for nlambda in nlambdas:

    print('\n')
    print('##################')
    print('\n')

    H.append(ll / nlambda)

    xtf = xTF(kRef, eps, nlambda)
    xtf.setRHS(dir_data, neu_data)
    # xtf.uncollect()
    # xtf.collect()
    # xtf.weak_form()

    mtf = MTF(xtf)

    Aw = mtf.get_A()

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

    print('')
    print('MTF classic', M.shape)
    print('')

    xx, niter, res, tt = my_gmres("\nGmres restart={0} maxiter={1}",
                                   M, b, tol, restart, maxiter)

    ecald = checker('Calderon ', A, J, xx)
    etrans = checker('Transmission ', J, X, xx, b)

    Ecald.append(ecald)
    Etrans.append(etrans)

    print('')
    print('STF classic', '[{}]'.format(int(M.shape[0] / 2)))
    print('')

    stf = STF(xtf)
    M_stf = stf.get()
    b_stf = stf.rhs()

    print(M_stf.shape)

    zz_stf, niter_stf, res_stf, tt_stf = my_gmres("\nGmres restart={0} maxiter={1}",
                                                  M_stf, b_stf, tol, restart, maxiter)

    n, = zz_stf.shape
    n = int(n/2)
    sold_stf, soln_stf = zz_stf[:n], zz_stf[n:]

    gui = bem.GridFunction(stf.space, fun=uinc)
    uicoeffs = gui.coefficients

    fui = bem.GridFunction(stf.space, fun=dnuinc)
    dnuicoeffs = fui.coefficients

    # There is a sign somewhere which is opposite.
    # in the Mie series ?
    soldi = sold_stf + uicoeffs
    solni = -soln_stf + dnuicoeffs
    sol0 = np.concatenate((sold_stf, soln_stf))
    sol1 = np.concatenate((soldi, solni))
    xx_stf = np.concatenate((sol0, sol1))

    ecald_stf = checker('Calderon ', A, J, xx_stf)
    etrans_stf = checker('Transmission ', J, X, xx_stf, b)

    Ecald_stf.append(ecald_stf)
    Etrans_stf.append(etrans_stf)

    print('')
    print('-----------')
    print('')

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

    sold_int, soln_int = soll[:n], soll[n:]

    print('\ndir. err, err_norm ; l2 <-| L2')

    gsold = bem.GridFunction(space, coefficients=sold)

    tt = time()
    gmie = bem.GridFunction(space, fun=mieD)
    miecoeffs = gmie.coefficients
    tt = time() - tt
    print('time gf:', tt)

    tt = time()

    errd = sold - miecoeffs
    el = la.norm(errd)
    enl = el / la.norm(miecoeffs)
    print(el, enl)
    enL = relative_error(gsold, fun=fmieD)
    eL = enL * gmie.l2_norm()
    print(eL, enL)

    tt = time() - tt
    print('time err:', tt)

    dEl2.append(el)
    dEnl2.append(enl)

    dEL2.append(eL)
    dEnL2.append(enL)

    ##
    print('')

    gsold_stf = bem.GridFunction(space, coefficients=sold_stf)

    tt = time()

    errd = sold_stf - miecoeffs
    el = la.norm(errd)
    enl = el / la.norm(miecoeffs)
    print(el, enl)
    enL = relative_error(gsold_stf, fun=fmieD)
    eL = enL * gmie.l2_norm()
    print(eL, enL)

    tt = time() - tt
    print('time err:', tt)

    dEl2_stf.append(el)
    dEnl2_stf.append(enl)

    dEL2_stf.append(eL)
    dEnL2_stf.append(enL)


    print('\nneu. err, err_norm ; l2 <-| L2')

    gsoln = bem.GridFunction(space, coefficients=soln)

    tt = time()
    fmie = bem.GridFunction(space, fun=mieN)
    dnmiecoeffs = fmie.coefficients
    tt = time() - tt
    print('time gf:', tt)

    tt = time()

    errn = soln + dnmiecoeffs
    el = la.norm(errn)
    enl = el / la.norm(dnmiecoeffs)
    print(el, enl)
    enL = relative_error(gsoln, fun=fmieN)
    eL = eL  * fmie.l2_norm()
    print(eL, enL)

    tt = time() - tt
    print('time err:', tt)

    nEl2.append(el)
    nEnl2.append(enl)

    nEL2.append(eL)
    nEnL2.append(enL)

    ##
    print('')

    gsoln_stf = bem.GridFunction(space, coefficients=soln_stf)

    tt = time()

    errn = soln_stf + dnmiecoeffs
    el = la.norm(errn)
    enl = el / la.norm(dnmiecoeffs)
    print(el, enl)
    enL = relative_error(gsoln_stf, fun=fmieN)
    eL = eL  * fmie.l2_norm()
    print(eL, enL)

    tt = time() - tt
    print('time err:', tt)

    nEl2_stf.append(el)
    nEnl2_stf.append(enl)

    nEL2_stf.append(eL)
    nEnL2_stf.append(enL)


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


sio.savemat('dat/err_stf.mat',
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
             'Ecald_stf':Ecald_stf, 'Etrans_stf':Etrans_stf,
             'dEL2_stf': dEL2_stf, 'nEL2_stf':nEL2_stf,
             'dEl2_stf': dEl2_stf, 'nEl2_stf':nEl2_stf,
             'dEnL2_stf':dEnL2_stf, 'nEnL2_stf':nEnL2_stf,
             'dEnl2_stf':dEnl2_stf, 'nEnl2_stf':nEnl2_stf,
            })
