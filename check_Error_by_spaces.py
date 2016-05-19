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


def get_Xij(spaces):
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


def get_opJi(spaces):
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

def get_iJi(spaces):
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


def get_and_check_Ci(spaces, k, sign=1):
    tt = time()

    opA = get_opAi(spaces, k, sign)
    Aw = opA.weak_form()

    opJ = get_opJi(spaces)
    Jw = opJ.weak_form()

    iJ = get_iJi(spaces)

    C = lambda x: 0.5*Jw(x) + Aw(x)
    CC = lambda x: C( iJ( C(x)))

    x = np.random.rand(Aw.shape[0]) + 1j * np.random.rand(Aw.shape[0])

    y = C(x)
    z = CC(x)

    print("D:", diri, "N:", neum,
          "error:", int(1e4 * la.norm(y - z)) / 1e4,
          "cputime:", int(10 * (time() - tt)) / 10,
          "dof:", Aw.shape[0],
    )


if __name__ == "__main__":

    k = np.pi
    lmbda = 2 * np.pi / k

    grid = bem.shapes.sphere(h=lmbda/10)
    #grid = bem.shapes.cube(h=lmbda/10)

    # space
    # (diri, neum)
    # diri = (trial, test)

    spaces = [
        ((("DP", 0), ("DP", 0)), (("DP", 0), ("DP", 0))),
        ((("P", 1), ("P", 1)), (("P", 1), ("P", 1))),
        ((("B-P", 1), ("B-P", 1)), (("DUAL", 0), ("DUAL", 0))),

        ((("DP", 1), ("DP", 1)), (("DP", 1), ("DP", 1))),
        ((("DP", 2), ("DP", 2)), (("DP", 2), ("DP", 2))),
        ((("DP", 3), ("DP", 3)), (("DP", 3), ("DP", 3))),
        ((("P", 2), ("P", 2)), (("P", 2), ("P", 2))),
        ((("P", 3), ("P", 3)), (("P", 3), ("P", 3))),
        ((("B-DP", 1), ("B-DP", 1)), (("B-DP", 1), ("B-DP", 1))),
        ((("B-P", 1), ("B-P", 1)), (("B-P", 1), ("B-P", 1))),
        ((("DUAL", 0), ("DUAL", 0)), (("B-P", 1), ("B-P", 1))),
    ]

    for diri, neum in spaces:
        spaces = create_spaces(diri, neum)
        get_and_check_Ci(spaces, k)
