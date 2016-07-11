# coding: utf8

from time import time

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import krylov

import bempp.api as bem

bem.global_parameters.assembly.boundary_operator_assembly_type = 'hmat'
#bem.global_parameters.hmat.coarsening = False


def fdata(x, n, d, res):
    res[0] = x[0] + x[1] + x[2]

def solve(Mat, rhs, tol=1e-5, Ml=None, force_left=False, check=False):
    shape = Mat.shape
    if Ml is not None:
        if force_left:
            ml = lambda x: Ml(Mat(x))
            b = Ml(rhs)
        else:
            ml = lambda x: Ml(x)
            b = rhs
    else:
        ml = lambda x: x
        b = rhs
    if force_left:
        Pl = spla.LinearOperator(shape, matvec=lambda x: x)
        A = spla.LinearOperator(shape, matvec=ml)
    else:
        Pl = spla.LinearOperator(shape, matvec=ml)
        A = spla.LinearOperator(shape, matvec=lambda x: Mat(x))

    ok = True
    toll = tol
    x = np.zeros((shape[0], ))
    #solver = krylov.cg
    solver = krylov.gmres
    #solver = spla.cg
    while la.norm(rhs - Mat(x)) > tol * la.norm(rhs) and ok:
        x = np.zeros((shape[0], ))
        res = []
        tt = time()
        x, info = solver(A, b,
                         tol=toll,
                         M=Pl,
                         residuals=res,
                         restrt=None,
                         maxiter=shape[0],
                         orthog='mgs')
        ts = time() - tt
        print(info, len(res),
              la.norm(rhs - Mat(x)) / la.norm(rhs),
              la.norm(Pl(rhs) - Pl(Mat(x))) / la.norm(Pl(rhs)))
        toll = toll / 10
        if len(res) > shape[0]:
            print('converged ?')
            break
        if not check:
            break
    n = la.norm(b)
    if len(res) > 1:
        resn = [ v / n for v in res ]
        resn.append(la.norm(rhs - Mat(x)) / la.norm(rhs))
    else:
        resn = [-1]
    return resn, ts, x
#

H = [5, 10, 20, 30, 40, 50, 75, 100]

HH = np.zeros((len(H)))
Dof = np.zeros((len(H), 7))
Its = np.zeros((len(H), 7))
Tps = np.zeros((len(H), 7))
STps = np.zeros((len(H), 7))

for i, h in enumerate(H):
    print(h)

    k = 0.1 * np.pi
    ll = 2 * np.pi / k
    grid = bem.shapes.sphere(h=ll/h)
    HH[i] = ll/h

    P0 = bem.function_space(grid, "DP", 0)
    P1 = bem.function_space(grid, "P", 1)

    P0d = bem.function_space(grid, "DUAL", 0)
    P1b = bem.function_space(grid, "B-P", 1)

    opV00 = bem.operators.boundary.helmholtz.single_layer(P0, P1, P0, k)
    opW00 = bem.operators.boundary.helmholtz.hypersingular(P0, P1, P0, k,
                                                           use_slp=True)
    opJ00 = bem.operators.boundary.sparse.identity(P0, P1, P0)

    opV = bem.operators.boundary.helmholtz.single_layer(P0d, P1b, P0d, k)
    opW = bem.operators.boundary.helmholtz.hypersingular(P1b, P0d, P1b, k)

    opJ = bem.operators.boundary.sparse.identity(opV.domain,
                                                 opW.dual_to_range,
                                                 opW.dual_to_range)
    opG = bem.operators.boundary.sparse.identity(opW.domain,
                                                 opV.dual_to_range,
                                                 opV.dual_to_range)

    print('Assembling V00...')
    tt = time()
    V00 = opV00.weak_form()
    tt_V00 = time() - tt

    Dof[i, 0], Tps[i, 0] = V00.shape[0], tt_V00

    tt = time()
    J00w = opJ00.weak_form()
    J00 = J00w.sparse_operator
    J00 = J00.astype(np.complex)
    iJ00lu = spla.splu(J00)
    iJ00 = spla.LinearOperator(iJ00lu.shape, matvec=iJ00lu.solve)

    Dof[i, 1], Tps[i, 1] = V00.shape[0], time() - tt

    print('Assembling W00...')
    tt = time()
    W00 = opW00.weak_form()
    tt_W00 = time() - tt

    Dof[i, 6], Tps[i, 6] = W00.shape[0], tt_W00

    print('Assembling V...')
    tt = time()
    V = opV.weak_form()
    tt_V = time() - tt

    Dof[i, 2], Tps[i, 2] = V.shape[0], tt_V

    tt = time()
    Jw = opJ.weak_form()
    J = Jw.sparse_operator
    J = J.astype(np.complex)
    iJlu = spla.splu(J)
    Jt = J.transpose()
    iJtlu = spla.splu(Jt)
    iJ = spla.LinearOperator(iJlu.shape, matvec=iJlu.solve)
    iJt = spla.LinearOperator(iJtlu.shape, matvec=iJtlu.solve)

    Dof[i, 3], Tps[i, 3] =  V.shape[0], time() - tt

    Gw = opG.weak_form()
    G = Gw.sparse_operator
    iGlu = spla.splu(G)
    Gt = G.transpose()
    iGtlu = spla.splu(Gt)
    iG = spla.LinearOperator(iGlu.shape, matvec=iGlu.solve)
    iGt = spla.LinearOperator(iGtlu.shape, matvec=iGtlu.solve)

    print('Assembling W...')
    tt = time()
    W = opW.weak_form()
    tt_W = time() - tt

    Dof[i, 4], Tps[i, 4] = W.shape[0], tt_W
    Dof[i, 5], Tps[i, 5] = W.shape[0], tt_W


    gf0 = bem.GridFunction(P0, fun=fdata)
    f0 = gf0.projections()

    gf = bem.GridFunction(P0d, fun=fdata)
    f = gf.projections()

    print('Solving V00...')
    res_V0, ts_V0, x_V0 = solve(V00, f0)

    Its[i, 0], STps[i, 0] = len(res_V0), ts_V0

    print('Solving iJV0...')
    res_iJV0, ts_iJV0, x_iJV0 = solve(V00, f0,
                                      Ml=lambda x: iJ00(x))

    Its[i, 1], STps[i, 1] = len(res_iJV0), ts_iJV0

    print('Solving iJW0 iJV0...')
    res_iJWiJV0, ts_iJWiJV0, x_iJWiJV0 = solve(V00, f0,
                                               Ml=lambda x: iJ00(W00(iJ00(x))))

    Its[i, 6], STps[i, 6] = len(res_iJWiJV0), ts_iJWiJV0


    print('Solving V...')
    res_V, ts_V, x_V = solve(V, f)

    Its[i, 2], STps[i, 2] = len(res_V), ts_V

    print('Solving iJtV...')
    res_iJtV, ts_iJtV, x_iJtV = solve(V, f, Ml=lambda x: iJt(x),
                                      force_left=False)

    Its[i, 3], STps[i, 3] = len(res_iJtV), ts_iJtV

    # print('Solving WiJtV...')
    # res_WiJtV, ts_WiJtV, x_WiJtV = solve(V, f, Ml=lambda x: W(iJt(x)),
    #                                      force_left=False)

    # Its[i, 4], STps[i, 4] = len(res_WiJtV), ts_WiJtV

    print('Solving iJWiJtV...')
    res_iJWiJtV, ts_iJWiJtV, x_iJWiJtV = solve(V, f,
                                               Ml=lambda x: iJ(W(iJt(x))),
                                               force_left=False)

    Its[i, 5], STps[i, 5] = len(res_iJWiJtV), ts_iJWiJtV


    # its = lambda ll: [ i for i, _ in enumerate(ll) ]
    # plt.figure()
    # lw, ms = 3, 10
    # plt.semilogy(its(res_V0), res_V0, 'k-', label='V | P0',
    #              linewidth=lw, markersize=ms)
    # plt.semilogy(its(res_iJV0), res_iJV0, 'k--', label='J^-1 V | P0',
    #              linewidth=lw, markersize=ms)
    # plt.semilogy(its(res_V), res_V, 'b-', label='V | dP0',
    #              linewidth=lw, markersize=ms)
    # plt.semilogy(its(res_iJtV), res_iJtV, 'b--', label='J^-T V | dP0/bP1',
    #              linewidth=lw, markersize=ms)
    # plt.semilogy(its(res_WiJtV), res_WiJtV, 'r-', label='W J^-T V | dP0/bP1',
    #              linewidth=lw, markersize=ms)
    # plt.semilogy(its(res_iJWiJtV), res_iJWiJtV, 'r--',
    #              label='J^-1 W J^-T V | dP0/bP1',
    #              linewidth=lw, markersize=ms)

    # plt.legend()
    # plt.xlabel('#its')
    # plt.ylabel('Normalized Residual')
    # plt.title("h = {}".format(h))
    # plt.show(block=False)

# lw, ms = 3, 10
# f, axarr = plt.subplots(2, sharex=True)
# axarr[0].loglog(Dof[:, 0], Its[:, 0], 'ko-', label='V | P0',
#            linewidth=lw, markersize=ms)
# axarr[0].loglog(Dof[:, 1], Its[:, 1], 'kd--', label='J^-1 V | P0',
#            linewidth=lw, markersize=ms)
# axarr[0].loglog(Dof[:, 6], Its[:, 6], 'ks:', label='J^-1 W J^-1 V | P0',
#            linewidth=lw, markersize=ms)
# axarr[0].loglog(Dof[:, 2], Its[:, 2], 'bo-', label='V | dP0',
#            linewidth=lw, markersize=ms)
# axarr[0].loglog(Dof[:, 3], Its[:, 3], 'bd--', label='J^-T V | dP0/bP1',
#            linewidth=lw, markersize=ms)
# # axarr[0].loglog(Dof[:, 4], Its[:, 4], 'ro-', label='W J^-T V | dP0/bP1',
# #            linewidth=lw, markersize=ms)
# axarr[0].loglog(Dof[:, 5], Its[:, 5], 'rd--', label='J^-1 W J^-T V | dP0/bP1',
#            linewidth=lw, markersize=ms)

# axarr[0].grid(True, which="both")
# axarr[0].legend(loc=2)
# axarr[0].set_xlabel('Dof')
# axarr[0].set_ylabel('Number of Iterations')

# axarr[1].loglog(Dof[:, 0], Tps[:, 0], 'ko-', label='V | P0',
#            linewidth=lw, markersize=ms)
# axarr[1].loglog(Dof[:, 6], Tps[:, 6], 'ks:', label='W | P0',
#            linewidth=lw, markersize=ms)
# axarr[1].loglog(Dof[:, 2], Tps[:, 2], 'bo-', label='V | dP0',
#            linewidth=lw, markersize=ms)
# axarr[1].loglog(Dof[:, 5], Tps[:, 5], 'rd--', label='W | bP1',
#            linewidth=lw, markersize=ms)

# axarr[1].grid(True, which="both")
# axarr[1].legend(loc=2)
# axarr[1].set_ylabel('Assembling CPU time')
# axarr[1].set_xlabel('DoF')

# axarr[0].set_title("Helmholtz k = {}".format(k))

# plt.show()

plt.figure()
plt.loglog(HH, Tps[:, 0], 'ko-', label='V | P1',
           linewidth=lw, markersize=ms)
plt.loglog(HH, Tps[:, 6], 'ks:', label='W | P1',
           linewidth=lw, markersize=ms)
plt.loglog(HH, Tps[:, 2], 'bo-', label='V | dP0',
           linewidth=lw, markersize=ms)
plt.loglog(HH, Tps[:, 5], 'rd--', label='W | bP1',
           linewidth=lw, markersize=ms)

plt.grid(True, which="both")
plt.legend(loc=2)
plt.ylabel('Assembling CPU time')
plt.xlabel('h')

set_title("Sphere --  k = {}".format(k))

plt.show()
