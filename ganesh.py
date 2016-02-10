#!/usr/bin/env python
# coding: utf8

from time import time

import numpy as np

import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import bempp.api as bem

from domains import *


meshname = "./geo/sphere-disjoint.msh"

kRef = 0.3 * np.pi

dd = [
    { 'name': '0',
      'phys': 1,
      'union': [-1, -2, -3],
  },
    { 'name': 'A',
      'phys': 2,
      'union': 1,
  },
    { 'name': 'B',
      'phys': 3,
      'union': 2,
  },
    { 'name': 'C',
      'phys': 4,
      'union': 3,
  }
]
domains = Domains(dd)


grid = bem.import_grid(meshname)

opA = bem.BlockedOperator(len(domains), len(domains))
opX = bem.BlockedOperator(len(domains), len(domains))
opI = bem.BlockedOperator(len(domains), len(domains))


kind_d = ("P", 1)
kind_n = ("P", 1)

funK = bem.operators.boundary.helmholtz.double_layer
funV = bem.operators.boundary.helmholtz.single_layer
funQ = bem.operators.boundary.helmholtz.adjoint_double_layer
funW = bem.operators.boundary.helmholtz.hypersingular

funI = bem.operators.boundary.sparse.identity

tinit = time()

print('\n=Collecting all the blocks')
for dom in domains:
    me = dom['name']
    ii = domains.getIndexDom(me)
    print('==Domain: {0}'.format(me))
    print('===Diag: Block #({0}, {0})'.format(ii))

    eps = dom['phys']
    k = kRef * np.sqrt(eps)

    opAA = bem.BlockedOperator(2, 2)
    opII = bem.BlockedOperator(2, 2)

    bem_space = bem.function_space

    Nints = len(dom['interfaces'])
    opK = bem.BlockedOperator(Nints, Nints)
    opV = bem.BlockedOperator(Nints, Nints)
    opW = bem.BlockedOperator(Nints, Nints)
    opQ = bem.BlockedOperator(Nints, Nints)
    opId = bem.BlockedOperator(Nints, Nints)
    opIn = bem.BlockedOperator(Nints, Nints)

    for facei, signi in zip(dom['interfaces'], dom['signs']):
        i = domains.getIndexInt(me, facei)

        space_test_d = bem_space(grid, kind_d[0], kind_d[1], domains=[facei])
        space_test_n = bem_space(grid, kind_n[0], kind_n[1], domains=[facei])

        space_range_d = bem_space(grid, kind_d[0], kind_d[1], domains=[facei])
        space_range_n = bem_space(grid, kind_n[0], kind_d[1], domains=[facei])

        for facej, signj in zip(dom['interfaces'], dom['signs']):
            j = domains.getIndexInt(me, facej)

            print('====Interface: ({0}, {1}) #({2}, {3})'.format(facei, facej,
                                                                 i, j))

            space_trial_d = bem_space(grid, kind_d[0], kind_d[1], domains=[facej])
            space_trial_n = bem_space(grid, kind_n[0], kind_n[1], domains=[facej])

            opKK = funK(space_trial_d, space_range_d, space_test_d, k)
            opVV = funV(space_trial_n, space_range_d, space_test_d, k)
            opWW = funW(space_trial_d, space_range_n, space_test_n, k)
            opQQ = funQ(space_trial_n, space_range_n, space_test_n, k)


            opK[i, j] = signj * opKK
            opV[i, j] = opVV
            opW[i, j] = opWW
            opQ[i, j] = signi * opQQ

            opIId = funI(space_trial_d, space_range_d, space_test_d)
            opIIn = funI(space_trial_n, space_range_n, space_test_n)

            opId[i, j] = opIId
            opIn[i, j] = opIIn

    opAA[0, 0] = - opK
    opAA[0, 1] = opV
    opAA[1, 0] = opW
    opAA[1, 1] = opQ

    opII[0, 0] = opId
    opII[1, 1] = opIn

    opA[ii, ii] = opAA
    opI[ii, ii] = opII


    for d in domains.getNeighborOf(dom['name']):
        other = d['name']
        jj = domains.getIndexDom(other)
        print('===Coupling {0} with {1}: Block #({2}, {3})'.format(other,
                                                                   domains.getName(jj),
                                                                   ii, jj))

        NintsMe, NintsOther = len(dom['interfaces']), len(d['interfaces'])
        opXXd = bem.BlockedOperator(NintsMe, NintsOther)
        opXXn = bem.BlockedOperator(NintsMe, NintsOther)

        for facei, signi in zip(dom['interfaces'], dom['signs']):

            i = domains.getIndexInt(me, facei)

            space_test_d = bem_space(grid, kind_d[0], kind_d[1], domains=[facei])
            space_test_n = bem_space(grid, kind_n[0], kind_n[1], domains=[facei])


            space_range_d = bem_space(grid, kind_d[0], kind_d[1], domains=[facei])
            space_range_n = bem_space(grid, kind_n[0], kind_d[1], domains=[facei])

            for facej, signj in zip(d['interfaces'], d['signs']):

                j = domains.getIndexInt(other, facej)

                print('====Interface: ({0}, {1}) #({2}, {3})'.format(facei, facej,
                                                                                 i, j))

                space_trial_d = bem_space(grid, kind_d[0], kind_d[1], domains=[facej])
                space_trial_n = bem_space(grid, kind_n[0], kind_n[1], domains=[facej])

                opXd = funI(space_trial_d, space_range_d, space_test_d)
                opXn = funI(space_trial_n, space_range_n, space_test_n)

                opXXd[i, j] = opXd
                opXXn[i, j] = -opXn

        opXX = bem.BlockedOperator(2, 2)
        opXX[0, 0] = opXXd
        opXX[1, 1] = opXXn

        opX[ii, jj] = opXX

#
##done


print('\n=RHS')

def evalIncDirichletTrace(x, normal, dom_ind, result):
    result[0] =  -np.exp( 1j * kRef * x[1])

def evalIncNeumannTrace(x, normal, dom_ind, result):
    result[0] = -1j * normal[1] * kRef * np.exp( 1j * kRef * x[1])

def evalZeros(point, normal, dom_ind, result):
    result[0] = 0. + 1j * 0.

inf = '0'

rhss = [] * len(domains)

rhs = [] * len(domains)
neighbors = domains.getNeighborOf(inf)
for ii in range(len(domains)):
    name = domains.getName(ii)
    dom = domains.getEntry(name)
    jj = domains.getIndexDom(dom['name'])
    print('==Domain: {0} #{1}'.format(dom['name'], ii))

    space_d = bem.function_space(grid, kind_d[0], kind_d[1], domains=dom['interfaces'])
    space_n = bem.function_space(grid, kind_n[0], kind_n[1], domains=dom['interfaces'])

    if dom['name'] == inf:
        IncDirichletTrace = bem.GridFunction(space_d, fun=evalIncDirichletTrace)
        IncNeumannTrace = bem.GridFunction(space_n, fun=evalIncNeumannTrace)
        idir, ineu = IncDirichletTrace, - IncNeumannTrace

    elif dom in neighbors:
        IncDirichletTrace = bem.GridFunction(space_d, fun=evalIncDirichletTrace)
        IncNeumannTrace = bem.GridFunction(space_n, fun=evalIncNeumannTrace)
        idir, ineu = - IncDirichletTrace, - IncNeumannTrace

    else:
        IncDirichletTrace = bem.GridFunction(space_d, fun=evalZeros)
        IncNeumannTrace = bem.GridFunction(space_n, fun=evalZeros)
        idir, ineu = IncDirichletTrace, IncNeumannTrace

    rhs.append(idir)
    rhs.append(ineu)

tt = time()
print('==Assembling RHS (projections)')
b = np.array([], dtype=complex)
for r in rhs:
    b = np.concatenate((b, r.projections()))
trhs = time() - tt
print('#time Assembling RHS: {}'.format(trhs))

##

tAssemb = time()
print('\n=Assembling all the matrices')
print('==BlockDiag assembling: A (be patient)')
for ii in range(len(domains)):
    tt = time()
    print('===Block: #({0}, {0})'.format(ii), end=' ')
    opp = opA[ii, ii]
    for i, j, who in zip([0, 0, 1, 1], [0, 1, 0, 1], ['K', 'V', 'W', 'Q']):
        print(who, end=' ', flush=True)
        op = opp[i, j]
        a = op.weak_form()
    print(' time: {}'.format(time() - tt))
# if something is missing... to be sure !
Aw = opA.weak_form()

print('==Coupling assembling: X')
Xw = opX.weak_form()

print('==MTF assembling: M ')
#########################
opMTF = opA - 0.5 * opX
########################
MTFw = opMTF.weak_form()
bmtf = 0.5 * b

print('==Identity assembling: J')
Jw = opI.weak_form()

tAssemb = time() - tAssemb
#
print('')
print('#total time Assembling: {}'.format(tAssemb))
print('')
#

tt = time()
Jcsc = sp.lil_matrix(Jw.shape, dtype=complex)
row_start, col_start = 0, 0
row_end, col_end = 0, 0
for ii in range(len(domains)):
    Jb = Jw[ii, ii]
    Jd, Jn = Jb[0, 0], Jb[1, 1]

    mat = bem.as_matrix(Jd)
    mat = sp.lil_matrix(mat)
    r, c = mat.shape

    row_end += r
    col_end += c
    Jcsc[row_start:row_end, col_start:col_end] = mat
    row_start, col_start = row_end, col_end

    mat = bem.as_matrix(Jn)
    mat = sp.lil_matrix(mat)
    r, c = mat.shape

    row_end += r
    col_end += c
    Jcsc[row_start:row_end, col_start:col_end] = mat
    row_start, col_start = row_end, col_end
Jcsc = Jcsc.tocsc()
print('##time convert Identity to CSC: {}'.format(time() - tt))
tt = time()
iJlu = spla.splu(Jcsc)
print('##time sparse J=LU: {}'.format(time() - tt))


tt = time()
Ecsc = sp.lil_matrix(Xw.shape, dtype=complex)
row_start, row_end = 0, 0
for r in range(len(domains)):
    row, col = 0, 0
    col_start, col_end = 0, 0
    first = True
    for c in range(len(domains)):
        if first:
            op = opI[r, r]
            row, _ = op.weak_form().shape
            row_end += row
            first = False
        op = opX[r, c]
        if not op is None:
            mat = bem.as_matrix(op.weak_form())
            mat = sp.lil_matrix(mat)
            _ , col = mat.shape
        else:
            opp =  opI[c, c]
            _ , col = opp.weak_form().shape
        col_end += col
        if c > r and (op is not None):
            Ecsc[row_start:row_end, col_start:col_end] = mat
        col_start = col_end
    row_start = row_end
Ecsc = Ecsc.tocsc()
print('##time to build E: {}'.format(time() - tt))

tpro = time() - tinit
#
print('')
print('#total time all processing: {}'.format(tpro))
print('')
#

print('=Size: {}'.format(MTFw.shape))

#
print('')
print(' ## All assembling done ! ##')
print("      == LET'S GO !! ==")
print('')
#

#################################################
#################################################
#################################################

norm = np.linalg.norm

shape = MTFw.shape

MTF = spla.LinearOperator(shape, matvec=MTFw.matvec, dtype=complex)

J = spla.LinearOperator(shape, matvec=Jw.matvec, dtype=complex)
iJ = spla.LinearOperator(shape, matvec=iJlu.solve, dtype=complex)

At = spla.LinearOperator(shape, matvec=Aw.matvec, dtype=complex)
X = spla.LinearOperator(shape, matvec=Xw.matvec, dtype=complex)

E = spla.LinearOperator(shape, matvec=Ecsc.dot, dtype=complex)

#####################################
#####################################
A = 2.0 * At
M = A - X
#####################################
#####################################


#################################################
#################################################
#################################################

from krylov import gmres

#################################################
#################################################
#################################################

iA = iJ * A * iJ

Pjac = iA
Pgs = iA + iA * E * iA

Msigma = lambda sigma: (A - J) + sigma * (J - X)

#################################################
#################################################
#################################################

tol = 1e-6
res = []
restart = 20
if restart is None: scale = 1
else: scale = restart

maxiter = int((M.shape[0] / scale) * 0.1)
if maxiter < 50: maxiter = 50

norm_b = la.norm(b)

#################################################
#################################################
#################################################

def rescaleRes(res, P, b):
    scale = 1.0 / la.norm(P(b))
    new_res = scale * res
    return new_res

#################################################

print('\nWO restart={0} maxiter={1}'.format(restart, maxiter))
del res
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
oResWO = np.array(res)
ResWO = rescaleRes(oResWO, lambda x: x, b)
print('#time: {}'.format(tt))

#sol = xx

print('=Error-Calderon WO')
y = A(xx)
z = J(xx)
e = la.norm(y - z)
ee = la.norm(xx)
print(e, ee, norm_b)

print('\nJac restart={0} maxiter={1}'.format(restart, maxiter))
del res
res = []
tt = time()
xx, info = gmres(M, b,
                 M = Pjac,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResJac = np.array(res)
ResJac = rescaleRes(oResJac, Pjac, b)
print('#time: {}'.format(tt))

print('=Error-Calderon Jacobi')
y = A(xx)
z = J(xx)
e = la.norm(y - z)
ee = la.norm(xx)
print(e, ee, norm_b)

print('\nGS restart={0} maxiter={1}'.format(restart, maxiter))
del res
res = []
tt = time()
xx, info = gmres(M, b,
                 M = Pgs,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResGS = np.array(res)
ResGS = rescaleRes(oResGS, Pgs, b)
print('#time: {}'.format(tt))

sol = xx

print('=Error-Calderon GS')
y = A(xx)
z = J(xx)
e = la.norm(y - z)
ee = la.norm(xx)
print(e, ee, norm_b)

sigma = -0.5
print('\nSigmaWO restart={0} maxiter={1}'.format(restart, maxiter))
del res
res = []
Ms, bs = Msigma(sigma), sigma * b
tt = time()
xx, info = gmres(Ms, bs,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResSigWO = np.array(res)
ResSigJWO = rescaleRes(oResSigWO, lambda x: x, bs)
print('#time: {}'.format(tt))

print('=Error-Calderon Sigma WO')
y = A(xx)
z = J(xx)
e = la.norm(y - z)
ee = la.norm(xx)
print(e, ee, norm_b)

####################################
####################################

import matplotlib.pyplot as plt


def Res2Tuple(res):
    return np.arange(len(res)), res

its, res = Res2Tuple(ResWO)
plt.semilogy(its, res, 'k-', linewidth=3,  label='wo')

its, res = Res2Tuple(ResJac)
plt.semilogy(its, res, 'b-', linewidth=3,  label='Jacobi')

its, res = Res2Tuple(ResGS)
plt.semilogy(its, res, 'r-', linewidth=3,  label='approx. Gauss-Siedel')

its, res = Res2Tuple(ResSigJWO)
plt.semilogy(its, res, 'g-', linewidth=3,  label='Sigma')

plt.title('Convergence History', fontsize=20)
plt.xlabel('#iterations', fontsize=14)
plt.ylabel('normalized residual', fontsize=30)
plt.legend()

plt.grid(True)

plt.show()
