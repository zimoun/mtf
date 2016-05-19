# coding: utf8

from time import time

import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as spla

import bempp.api as bem


def create_spaces(diri, neum):

    trial, test = diri
    triad = bem.function_space(grid, trial[0], trial[1])
    testd = bem.function_space(grid, test[0], test[1])
    spaced = (triad, testd)

    trial, test = neum
    trian = bem.function_space(grid, trial[0], trial[1])
    testn = bem.function_space(grid, test[0], test[1])
    spacen = (trian, testn)

    return spaced, spacen

def get_opAi(spaces, k, sign=1):
    tt = time()

    spaced, spacen = spaces
    triad, testd = spaced
    trian, testn = spacen

    opK = bem.operators.boundary.helmholtz.double_layer(triad, triad, testd, k)
    opV = bem.operators.boundary.helmholtz.single_layer(trian, triad, testd, k)
    opW = bem.operators.boundary.helmholtz.hypersingular(triad, trian, testn, k)
    opQ = bem.operators.boundary.helmholtz.adjoint_double_layer(trian, trian, testn, k)

    opA = bem.BlockedOperator(2, 2)
    opA[0, 0], opA[0, 1] = - sign * opK, opV
    opA[1, 0], opA[1, 1] = opW, sign * opQ

    #print("collect A:", time()-tt)
    return opA


def get_opXij(spaces, k=None, sign=None):
    tt = time()

    spaced, spacen = spaces
    triad, testd = spaced
    trian, testn = spacen

    opId = bem.operators.boundary.sparse.identity(triad, triad, testd)
    opIn = bem.operators.boundary.sparse.identity(trian, trian, testn)

    opX = bem.BlockedOperator(2, 2)
    opX[0, 0], opX[1, 1] = opId, -opIn

    #print("collect X:", time()-tt)
    return opX


def get_opJi(spaces, k=None, sign=None):
    tt = time()

    spaced, spacen = spaces
    triad, testd = spaced
    trian, testn = spacen

    opId = bem.operators.boundary.sparse.identity(triad, triad, testd)
    opIn = bem.operators.boundary.sparse.identity(trian, trian, testn)

    opJ = bem.BlockedOperator(2, 2)
    opJ[0, 0], opJ[1, 1] = opId, opIn

    #print("collect J:", time()-tt)
    return opJ

def get_iJi(spaces, k=None, sign=None):
    tt = time()

    iJ = bem.BlockedDiscreteOperator(2, 2)

    opJ = get_opJi(spaces)
    opId, opIn = opJ[0, 0], opJ[1, 1]

    for ii, opI in enumerate([opId, opIn]):
        Iw = opI.weak_form()
        II = Iw.sparse_operator
        II = II.astype(complex)
        iII = spla.splu(II)
        iIl = spla.LinearOperator(II.shape, matvec=iII.solve, dtype=complex)
        iJ[ii, ii] = iIl

    #print("compute iJ:", time()-tt)
    return iJ


def check_Ci(spaces, k, sign=1):
    tt = time()

    opA = get_opAi(spaces, k, sign)
    Aw = opA.weak_form()

    opJ = get_opJi(spaces)
    Jw = opJ.weak_form()

    iJ = get_iJi(spaces)

    C = lambda x: 0.5*Jw(x) + Aw(x)
    CC = lambda x: C( iJ( C(x)))

    x = np.random.rand(Aw.shape[0]) + 1j * np.random.rand(Aw.shape[0])
    nx = la.norm(x)

    y = C(x)
    z = CC(x)

    print("D:", diri, "N:", neum,
          "errorX1e-2:", int(1e8 * la.norm(y - z)/nx) / 1e6,
          "cputime:", int(10 * (time() - tt)) / 10,
          "dof:", Aw.shape[0],
    )
#

def get_Diag(spaces, get_op,  k=None):
    if k is None:
        k = [None, None]

    D = bem.BlockedDiscreteOperator(2, 2)

    op0 = get_op(spaces, k[0], sign=1)
    op1 = get_op(spaces, k[1], sign=-1)

    for ii, op in enumerate([op0, op1]):
        try:
            Dw = op.weak_form()
        except:
            Dw = op
        D[ii, ii] = Dw

    return D

def get_A(spaces, k):
    A = get_Diag(spaces, get_opAi, k)
    return A

def get_J(spaces):
    J = get_Diag(spaces, get_opJi)
    return J

def get_iJ(spaces):
    iJ = get_Diag(spaces, get_iJi)
    return iJ

def get_X(spaces):
    X = bem.BlockedDiscreteOperator(2, 2)
    opX = get_opXij(spaces)
    Xw = opX.weak_form()
    X[0, 1], X[1, 0] = Xw, Xw
    return X

def check_C(spaces, k):
    tt = time()

    A = get_A(spaces, k)
    J = get_J(spaces)
    iJ = get_iJ(spaces)

    C = lambda x: 0.5*J(x) + A(x)
    CC = lambda x: C( iJ( C(x)))

    x = np.random.rand(A.shape[0]) + 1j * np.random.rand(A.shape[0])
    nx = la.norm(x)

    y = C(x)
    z = CC(x)

    print("D:", diri, "N:", neum,
          "errorX1e-2:", int(1e8 * la.norm(y - z)/nx) / 1e6,
          "cputime:", int(10 * (time() - tt)) / 10,
          "dof:", A.shape[0],
    )


def mtf(spaces, k):
    tt = time()
    A = get_A(spaces, k)
    X = get_X(spaces)
    M = A - 0.5 * X
    print("cputime:", time() - tt)
    return M

def rhs(spaces, funs):
    spaced, spacen = spaces
    triad, testd = spaced
    trian, testn = spacen

    fd, fn = funs

    gfd = bem.GridFunction(testd, fun=fd)
    gfn = bem.GridFunction(testn, fun=fn)

    pd, pn = gfd.projections(), gfn.projections()
    b = pd
    for p in [pn, pd, pn]:
        b = np.concatenate((b, -p))
    return b

def check_sol(spaces, k, x, b):

    A = get_A(spaces, k)
    X = get_X(spaces)
    J = get_J(spaces)

    nx = la.norm(x)

    ya, yt = A(x), J(x)
    za, zt = J(x), X(x)

    print("D:", diri, "N:", neum,
          "erAX1e-2:", int(1e8 * la.norm(ya - za)/nx) / 1e6,
          "erTX1e-2:", int(1e8 * la.norm(yt - zt - b)/nx) / 1e6,
    )


if __name__ == "__main__":

    k = 0.3 * np.pi
    eps = np.sqrt( 2 )
    nlmdba = 20
    iincident = 0

    lmbda = 2 * np.pi / k
    kk = [k, eps*k]

    #grid = bem.shapes.sphere(h=lmbda/(nlmdba*eps))
    grid = bem.shapes.cube(h=lmbda/(nlmdba*eps))

    # space
    # (diri, neum)
    # diri = (trial, test)

    spaces = [
        ((("DP", 0), ("DP", 0)), (("DP", 0), ("DP", 0))),
        ((("P", 1), ("P", 1)), (("P", 1), ("P", 1))),
        ((("B-DP", 1), ("B-DP", 1)), (("DUAL", 0), ("DUAL", 0))),
        ((("B-P", 1), ("B-P", 1)), (("DUAL", 0), ("DUAL", 0))),

        # ((("DP", 1), ("DP", 1)), (("DP", 1), ("DP", 1))),
        # ((("DP", 2), ("DP", 2)), (("DP", 2), ("DP", 2))),
        # ((("DP", 3), ("DP", 3)), (("DP", 3), ("DP", 3))),
        # ((("P", 2), ("P", 2)), (("P", 2), ("P", 2))),
        # ((("P", 3), ("P", 3)), (("P", 3), ("P", 3))),
        # ((("B-DP", 1), ("B-DP", 1)), (("B-DP", 1), ("B-DP", 1))),
        # ((("B-P", 1), ("B-P", 1)), (("B-P", 1), ("B-P", 1))),
        # ((("DUAL", 0), ("DUAL", 0)), (("B-P", 1), ("B-P", 1))),
        # ((("DUAL", 0), ("DUAL", 0)), (("B-DP", 1), ("B-DP", 1))),
    ]

    for diri, neum in spaces:
        space = create_spaces(diri, neum)
        check_Ci(space, k)

    print('')

    for diri, neum in spaces:
        space = create_spaces(diri, neum)
        check_C(space, kk)

    print('')

    def fdir(x, n, d, res):
        res[0] =  -np.exp( 1j * k * x[iincident])
    def fneu(x, n, d, res):
        res[0] =  -1j *  n[iincident] * k * np.exp( 1j * k * x[iincident])
    funs = fdir, fneu

    from krylov import gmres

    for diri, neum in spaces:
        space = create_spaces(diri, neum)
        M = mtf(space, kk)
        b = rhs(space, funs)
        MM = spla.LinearOperator(M.shape, matvec=M.matvec, dtype=complex)
        res = []
        tol = 1e-5
        restart, maxiter = None, MM.shape[0]
        xx, info = gmres(MM, b,
                         orthog='mgs',
                         tol=tol,
                         residuals=res,
                         restrt=restart,
                         maxiter=maxiter)
        print(info, len(res))
        check_sol(space, kk, xx, b)
