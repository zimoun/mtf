# coding: utf8

from time import time

import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import bempp.api as bem

from krylov import gmres

import check_Error_by_spaces as my

k = 0.5 * np.pi
eps = np.sqrt( 2 )
iincident = 0

bempp_shape = bem.shapes.cube

nlmbdas = [5, 10, 20, 30, 40, 50, 60, 70]

lmbda = 2 * np.pi / k
kk = [k, eps*k]

names = [ 'P0/P0', 'P1/P1', 'B-DP1/P0_d', 'B-P1/P0_d' ]
colors = [ 'k+--', 'bo--', 'rd--', 'gs--' ]
pls = zip(names, colors)

spaces = [
    ((("DP", 0), ("DP", 0)), (("DP", 0), ("DP", 0))),
    ((("P", 1), ("P", 1)), (("P", 1), ("P", 1))),
    ((("B-DP", 1), ("B-DP", 1)), (("DUAL", 0), ("DUAL", 0))),
    ((("B-P", 1), ("B-P", 1)), (("DUAL", 0), ("DUAL", 0))),

    ]

def fdir(x, n, d, res):
    res[0] =  -np.exp( 1j * k * x[iincident])
def fneu(x, n, d, res):
    res[0] =  -1j *  n[iincident] * k * np.exp( 1j * k * x[iincident])
funs = fdir, fneu

DofErr = [ np.zeros((len(nlmbdas), 4), dtype=float) for i in spaces ]
for i, nl in enumerate(nlmbdas):

    grid = bempp_shape(h=lmbda/(nl*eps))

    for j, (diri, neum) in enumerate(spaces):

        tt = time()
        space = my.create_spaces(grid, diri, neum)
        dof = 2*(space[0][0].global_dof_count + space[1][0].global_dof_count)
        print("nl:", nl, "dof:", dof)

        A, X = my.get_A(space, kk), my.get_X(space)
        b = my.rhs(space, funs)
        tt = time() - tt
        J = my.get_J(space)
        M = A - 0.5 * X
        MM = spla.LinearOperator(M.shape, matvec=M.matvec, dtype=complex)
        print('assembled.')

        res = []
        tol = 1e-5
        restart, maxiter = None, MM.shape[0]
        xx, info = gmres(MM, b,
                         orthog='mgs',
                         tol=tol,
                         residuals=res,
                         restrt=restart,
                         maxiter=maxiter)
        print(info, len(res), MM.shape)
        ea, et = my.check_sol(space, kk, xx, b, diri, neum,
                              A, X, J)

        DofErr[j][i, :] = dof, ea, et, tt


lw, ms = 3, 10
f, axarr = plt.subplots(2, sharex=True)
pls = zip(names, colors)
for i, pl in enumerate(pls):
    n, c = pl
    axarr[0].loglog(DofErr[i][:, 0], DofErr[i][:, 1], c, label=n,
                    linewidth=lw, markersize=ms)
    axarr[0].loglog(DofErr[i][:, 0], DofErr[i][:, 2], c[0]+c[2:],
                    linewidth=lw, markersize=ms)
    axarr[1].loglog(DofErr[i][:, 0], DofErr[i][:, 3], c, label=n,
                    linewidth=lw, markersize=ms)

axarr[0].grid(True, which="both")
axarr[0].legend()
axarr[0].set_ylabel('Relative Error l2')
axarr[0].set_xlabel('DoF')

axarr[1].grid(True, which="both")
axarr[1].legend(loc=2)
axarr[1].set_ylabel('Assembling CPU time')
axarr[1].set_xlabel('DoF')

plt.show(block=False)

lw, ms = 3, 10
plt.figure()
pls = zip(names, colors)
for i, pl in enumerate(pls):
    n, c = pl
    plt.loglog(DofErr[i][:, 1], DofErr[i][:, 3], c, label=n,
                    linewidth=lw, markersize=ms)

plt.grid(True, which="both")
plt.legend()
plt.xlabel('Relative Error l2')
plt.ylabel('Assembling CPU time')

plt.show(block=False)
