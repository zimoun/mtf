# coding: utf8

from time import time

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import krylov
import krypy

import bempp.api as bem
from krypy.utils import ConvergenceError

spaces = [
    (("DP", 0), ("DP", 0)),
    (("DUAL", 0), ("B-P", 1)),
    (("P", 1), ("P", 1))
]

import warnings

def fdata(x, n, d, res):
    res[0] = x[0] + x[1] + x[2]

def solve_left(V, f, Ml=None):
    if Ml is not None:
        mv = lambda x: Ml(V(x))
        ff = Ml(f)
    else:
        mv = lambda x: V(x)
        ff = f
    res = []
    A, b = spla.LinearOperator(V.shape, matvec=mv), ff
    tt = time()
    x, info = krylov.cg(A, b, residuals=res)
    ts = time() - tt
    print(info, len(res))
    n = la.norm(b)
    if len(res) > 1:
        resn = [ v / n for v in res ]
        resn.append(la.norm(f - V(x)) / la.norm(f))
    else:
        resn = [-1]
    return resn[:], ts, x

def solve(V, f, Mr=None, Ml=None):
    if Mr is not None:
        mr = lambda x: Mr(x)
    else:
        mr = lambda x: x
    if Ml is not None:
        ml = lambda x: Ml(x)
    else:
        ml = lambda x: x
    Pr = spla.LinearOperator(V.shape, matvec=mr)
    Pl = spla.LinearOperator(V.shape, matvec=ml)
    A, b = spla.LinearOperator(V.shape, matvec=lambda x: V(x)), f
    linsys = krypy.linsys.LinearSystem(A, b,
                                       Mr=Pr, Ml=Pl,
                                       self_adjoint=True,
                                       positive_definite=True)
    try:
        tt = time()
        info = krypy.linsys.Cg(linsys)
    except ConvergenceError as e:
        info = e.solver
    finally:
        ts = time() - tt
    print(info.iter)

    if np.isnan(info.resnorms[-1]):
        resn, x = [-1], info.x0
    else:
        resn, x = info.resnorms, info.xk
    x = x.reshape((x.shape[0], ))
    resn.append(la.norm(f - V(x)) / la.norm(f))
    return resn, ts, x



grid = bem.shapes.sphere(h=0.5)

for spV, spW in spaces:

    if spV == spW:
        name, order = spV
        test_V = tria_V = bem.function_space(grid, name, order)
        test_W = tria_W = test_V
    else:
        name, order = spV
        test_V = tria_V = bem.function_space(grid, name, order)
        name, order = spW
        test_W = tria_W = bem.function_space(grid, name, order)

    opV = bem.operators.boundary.laplace.single_layer(tria_V, tria_W, test_V)
    opW = bem.operators.boundary.laplace.hypersingular(tria_W, tria_V, test_W)

    opJ = bem.operators.boundary.sparse.identity(opW.domain,
                                                 opV.dual_to_range,
                                                 opV.dual_to_range)
    opG = bem.operators.boundary.sparse.identity(opV.domain,
                                                 opW.dual_to_range,
                                                 opW.dual_to_range)

    print('Assembling V...')
    tt = time()
    V = opV.weak_form()
    tt_V = time() - tt

    print('Assembling W...')
    tt = time()
    W = opW.weak_form()
    tt_W = time() - tt

    Jw = opJ.weak_form()
    J = Jw.sparse_operator
    iJlu = spla.splu(J)
    Jt = J.transpose()
    iJtlu = spla.splu(Jt)
    iJ = spla.LinearOperator(iJlu.shape, matvec=iJlu.solve)
    iJt = spla.LinearOperator(iJtlu.shape, matvec=iJtlu.solve)

    Gw = opG.weak_form()
    G = Gw.sparse_operator
    iGlu = spla.splu(G)
    Gt = G.transpose()
    iGtlu = spla.splu(Gt)
    iG = spla.LinearOperator(iGlu.shape, matvec=iGlu.solve)
    iGt = spla.LinearOperator(iGtlu.shape, matvec=iGtlu.solve)

    gf = bem.GridFunction(test_V, fun=fdata)
    f = gf.projections()

    print('Solving V...')
    res_V, ts_V, x_V = solve(V, f)

    print('l Solving iJV...')
    lres_iJV, lts_iJV, lx_iJV = solve(V, f, Ml=lambda x: iJ(x))

    print('l Solving iJtV...')
    lres_iJtV, lts_iJtV, lx_iJtV = solve(V, f, Ml=lambda x: iJt(x))

    print('r Solving iJV...')
    rres_iJV, rts_iJV, rx_iJV = solve(V, f, Mr=lambda x: iJ(x))

    print('r Solving iJtV...')
    rres_iJtV, rts_iJtV, rx_iJtV = solve(V, f, Mr=lambda x: iJt(x))

    print('l Solving WiJtV...')
    lres_WiJtV, lts_WiJtV, lx_WiJtV = solve(V, f, Ml=lambda x: W(iJt(x)))

    print('l Solving iJWiJtV...')
    lres_iJWiJtV, lts_iJWiJtV, lx_iJWiJtV = solve(V, f,
                                                  Ml=lambda x: iJ(W(iJt(x))))

    print('lr Solving ViGtW...')
    rres_ViGtW, rts_ViGtW, rx_ViGtW = solve(V, f, Mr=lambda x: iGt(W(x)))

    print('lr Solving iGViGtW...')
    res_iGViGtW, ts_iGViGtW, x_iGViGtW = solve(V, f,
                                                  Ml=lambda x: iG(x),
                                                  Mr=lambda x: iGt(W(x)))



its = lambda ll: [ i for i, _ in enumerate(ll) ]
plt.figure()
lw, ms = 3, 10
plt.semilogy(its(res_V), res_V, 'k--', label='V',
             linewidth=lw, markersize=ms)
plt.semilogy(its(rres_iJV), rres_iJV, 'k-', label='iJV',
             linewidth=lw, markersize=ms)
plt.semilogy(its(lres_iJV), lres_iJV, 'k-.', label='ViJ',
             linewidth=lw, markersize=ms)
plt.semilogy(its(lres_WiJtV), lres_WiJtV, 'r--', label='WiJtV',
             linewidth=lw, markersize=ms)
plt.semilogy(its(lres_iJWiJtV), lres_iJWiJtV, 'r-', label='iJWiJtV',
             linewidth=lw, markersize=ms)
plt.semilogy(its(rres_ViGtW), rres_ViGtW, 'b--', label='ViGtW',
             linewidth=lw, markersize=ms)
plt.semilogy(its(res_iGViGtW), res_iGViGtW, 'b--', label='iGViGtW',
             linewidth=lw, markersize=ms)

plt.legend()
plt.xlabel('#its')
plt.ylabel('Normalized Residual')

plt.show(block=False)
