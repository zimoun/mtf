# coding: utf8

from time import time

import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as spla

import bempp.api as bem

def check_proj(diri, neum):
    tt = time()

    trial, test = diri
    triad = bem.function_space(grid, trial[0], trial[1])
    testd = bem.function_space(grid, test[0], test[1])

    trial, test = neum
    trian = bem.function_space(grid, trial[0], trial[1])
    testn = bem.function_space(grid, test[0], test[1])



    opK = bem.operators.boundary.helmholtz.double_layer(triad, triad, testd, k)
    opV = bem.operators.boundary.helmholtz.single_layer(trian, triad, testd, k)
    opW = bem.operators.boundary.helmholtz.hypersingular(triad, trian, testn, k)
    opQ = bem.operators.boundary.helmholtz.adjoint_double_layer(trian, trian, testn, k)

    opA = bem.BlockedOperator(2, 2)
    opA[0, 0], opA[0, 1] = -opK, opV
    opA[1, 0], opA[1, 1] = opW, opQ

    opId = bem.operators.boundary.sparse.identity(triad, triad, testd)
    opIn = bem.operators.boundary.sparse.identity(trian, trian, testn)
    opJ = bem.BlockedOperator(2, 2)
    opJ[0, 0], opJ[1, 1] = opId, opIn

    Idw = opId.weak_form()
    IId = Idw.sparse_operator
    IId = IId.astype(complex)
    iIId = spla.splu(IId)
    iIdl = spla.LinearOperator(IId.shape, matvec=iIId.solve, dtype=complex)

    Inw = opIn.weak_form()
    IIn = Idw.sparse_operator
    IIn = IId.astype(complex)
    iIIn = spla.splu(IIn)
    iInl = spla.LinearOperator(IIn.shape, matvec=iIIn.solve, dtype=complex)
    iJ = bem.BlockedDiscreteOperator(2, 2)
    iJ[0, 0], iJ[1, 1] = iIdl, iInl

    Aw = opA.weak_form()
    Jw = opJ.weak_form()

    C = lambda x: 0.5*Jw(x) + Aw(x)
    CC = lambda x: C( iJ( C(x)))

    x = np.random.rand(Aw.shape[0]) + 1j * np.random.rand(Aw.shape[0])

    y = C(x)
    z = CC(x)

    print("D:", diri, "N:", neum,
          "error:", la.norm(y - z),
          "cputime:", time() - tt,
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
        ((("DP", 1), ("DP", 1)), (("DP", 1), ("DP", 1))),
        ((("DP", 2), ("DP", 2)), (("DP", 2), ("DP", 2))),
        ((("DP", 3), ("DP", 3)), (("DP", 3), ("DP", 3))),
        ((("P", 1), ("P", 1)), (("P", 1), ("P", 1))),
        ((("P", 2), ("P", 2)), (("P", 2), ("P", 2))),
        ((("P", 3), ("P", 3)), (("P", 3), ("P", 3))),
        ((("B-DP", 1), ("B-DP", 1)), (("B-DP", 1), ("B-DP", 1))),
        ((("B-P", 1), ("B-P", 1)), (("B-P", 1), ("B-P", 1))),
    ]

    for diri, neum in spaces:
        check_proj(diri, neum)
