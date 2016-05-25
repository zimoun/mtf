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
    solver = krylov.cg
    #solver = krylov.gmres
    #solver = spla.cg
    while la.norm(rhs - Mat(x)) > tol * la.norm(rhs) and ok:
        x = np.zeros((shape[0], ))
        res = []
        tt = time()
        x, info = solver(A, b,
                         tol=toll,
                         M=Pl,
                         residuals=res)
                         # restrt=None,
                         # maxiter=shape[0])
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


grid = bem.shapes.sphere(h=0.1)

P0 = bem.function_space(grid, "DP", 0)
P1 = bem.function_space(grid, "P", 1)

P0d = bem.function_space(grid, "DUAL", 0)
P1b = bem.function_space(grid, "B-P", 1)

opV00 = bem.operators.boundary.laplace.single_layer(P0, P1, P0)
opW00 = bem.operators.boundary.laplace.hypersingular(P0, P1, P0)
opJ00 = bem.operators.boundary.sparse.identity(P0, P1, P0)

opV = bem.operators.boundary.laplace.single_layer(P0d, P1b, P0d)
opW = bem.operators.boundary.laplace.hypersingular(P1b, P0d, P1b)

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

J00w = opJ00.weak_form()
J00 = J00w.sparse_operator
iJ00lu = spla.splu(J00)
iJ00 = spla.LinearOperator(iJ00lu.shape, matvec=iJ00lu.solve)

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

gf0 = bem.GridFunction(P0, fun=fdata)
f0 = gf0.projections()

gf = bem.GridFunction(P0d, fun=fdata)
f = gf.projections()

print('Solving V00...')
res_V0, ts_V0, x_V0 = solve(V00, f0)

print('Solving iJV...')
res_iJV0, ts_iJV0, x_iJV0 = solve(V00, f0, Ml=lambda x: iJ00(x))

print('Solving V...')
res_V, ts_V, x_V = solve(V, f)

print('Solving iJtV...')
res_iJtV, ts_iJtV, x_iJtV = solve(V, f, Ml=lambda x: iJt(x),
                                  force_left=False)

print('Solving WiJtV...')
res_WiJtV, ts_WiJtV, x_WiJtV = solve(V, f, Ml=lambda x: W(iJt(x)),
                                     force_left=False)

print('Solving iJWiJtV...')
res_iJWiJtV, ts_iJWiJtV, x_iJWiJtV = solve(V, f,
                                           Ml=lambda x: iJ(W(iJt(x))),
                                           force_left=False)



its = lambda ll: [ i for i, _ in enumerate(ll) ]
plt.figure()
lw, ms = 3, 10
plt.semilogy(its(res_V0), res_V0, 'k-', label='V | P0',
             linewidth=lw, markersize=ms)
plt.semilogy(its(res_iJV0), res_iJV0, 'k--', label='J^-1 V | P0',
             linewidth=lw, markersize=ms)
plt.semilogy(its(res_V), res_V, 'b-', label='V | dP0',
             linewidth=lw, markersize=ms)
plt.semilogy(its(res_iJtV), res_iJtV, 'b--', label='J^-T V | dP0/bP1',
             linewidth=lw, markersize=ms)
plt.semilogy(its(res_WiJtV), res_WiJtV, 'r-', label='W J^-T V | dP0/bP1',
             linewidth=lw, markersize=ms)
plt.semilogy(its(res_iJWiJtV), res_iJWiJtV, 'r--',
             label='J^-1 W J^-T V | dP0/bP1',
             linewidth=lw, markersize=ms)

plt.legend()
plt.xlabel('#its')
plt.ylabel('Normalized Residual')

plt.show(block=False)
