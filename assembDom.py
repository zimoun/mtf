#!/usr/bin/env python
# coding: utf8

debug = True
if debug:
    print('Debug: {}'.format(debug))

from time import time

import numpy as np

import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import bempp.api as bem

from domains import *

debug = True

meshname = "./geo/sphere-disjoint.msh"
# meshname = "./geo/ellipse-concentric.msh"

# domains = Domains(generate_disjoint_Domains(2))

wl = 6.0
kRef = 2 * np.pi / wl

kRef = 0.5

dd = [
    { 'name': '0',
      'phys': 1,
      'union': [-10, -100],
  },
    { 'name': 'A',
      'phys': 2,
      'union': 10,
  },
    { 'name': 'B',
      'phys': 5,
      'union': 100,
  }
]
domains = Domains(dd)

# dd = [
#     { 'name': '0',
#       'phys': 1,
#       'union': [-1, -2, -3],
#   },
#     { 'name': 'A',
#       'phys': 2,
#       'union': 1,
#   },
#     { 'name': 'B',
#       'phys': 3,
#       'union': 2,
#   },
#     { 'name': 'C',
#       'phys': 4,
#       'union': 3,
#   }
# ]
# domains = Domains(dd)


# dd = [
#     { 'name': '0',
#       'phys': 1,
#       'union': [-1],
#   },
#     { 'name': 'A',
#       'phys': 2,
#       'union': [1, -2],
#   },
#     { 'name': 'B',
#       'phys': 1,
#       'union': [2, -3],
#   },
#     { 'name': 'C',
#       'phys': 1,
#       'union': 3,
#   }
# ]
# domains = Domains(dd)

if debug:
    print('Debug: {}'.format(debug))
    # ??
    # bem.enable_console_logging()
    # print('Dense')
    # bem.global_parameters.assembly.boundary_operator_assembly_type = 'dense'
    # bem.global_parameters.assembly.boundary_operator_assembly_type = 'hmat'
    #pass

# bem.global_parameters.assembly.boundary_operator_assembly_type = 'dense'

print(bem.global_parameters.assembly.boundary_operator_assembly_type)



grid = bem.import_grid(meshname)

# opM = bem.BlockedOperator(len(domains), len(domains))
opA = bem.BlockedOperator(len(domains), len(domains))
opX = bem.BlockedOperator(len(domains), len(domains))
opI = bem.BlockedOperator(len(domains), len(domains))

opY = bem.BlockedOperator(len(domains), len(domains))
opZ = bem.BlockedOperator(len(domains), len(domains))

# opR = bem.BlockedOperator(len(domains), len(domains))
opS = bem.BlockedOperator(len(domains), len(domains))


kind_d = ("P", 1)
kind_n = ("P", 1)

funK = bem.operators.boundary.helmholtz.double_layer
funV = bem.operators.boundary.helmholtz.single_layer
funQ = bem.operators.boundary.helmholtz.adjoint_double_layer
funW = bem.operators.boundary.helmholtz.hypersingular

funI = bem.operators.boundary.sparse.identity

# kind_d = ("RT", 0)
# kind_n = ("RT", 0)

# funK = bem.operators.boundary.maxwell.magnetic_field
# funV = bem.operators.boundary.maxwell.electric_field
# funQ = bem.operators.boundary.maxwell.magnetic_field
# funW = bem.operators.boundary.maxwell.electric_field

# funI = bem.operators.boundary.sparse.maxwell_identity



tinit = time()

print('\n=Collecting all the blocks')
for dom in domains:
    ii = domains.getIndexDom(dom['name'])
    print('==Domain: {0}'.format(dom['name']))
    print('===Diag: Block #({0}, {0})'.format(ii))

    eps = dom['phys']
    k = kRef * np.sqrt(eps)

    sig = dom['sign']

    opAA = bem.BlockedOperator(2, 2)
    opII = bem.BlockedOperator(2, 2)

    opSS = bem.BlockedOperator(2, 2)

    bem_space = bem.function_space

    space_trial_d = bem_space(grid, kind_d[0], kind_d[1], domains=dom['interfaces'])
    space_trial_n = bem_space(grid, kind_n[0], kind_n[1], domains=dom['interfaces'])

    space_range_d = bem_space(grid, kind_d[0], kind_d[1], domains=dom['interfaces'])
    space_range_n = bem_space(grid, kind_n[0], kind_d[1], domains=dom['interfaces'])

    space_test_d = bem_space(grid, kind_d[0], kind_d[1], domains=dom['interfaces'])
    space_test_n = bem_space(grid, kind_n[0], kind_n[1], domains=dom['interfaces'])

    opK = funK(space_trial_d, space_range_d, space_test_d, k)
    opV = funV(space_trial_n, space_range_d, space_test_d, k)
    opW = funW(space_trial_d, space_range_n, space_test_n, k)
    opQ = funQ(space_trial_n, space_range_n, space_test_n, k)

    opId = funI(space_trial_d, space_range_d, space_test_d)
    opIn = funI(space_trial_n, space_range_n, space_test_n)

    opAA[0, 0] = - sig * opK
    opAA[0, 1] = opV
    opAA[1, 0] = opW
    opAA[1, 1] = sig * opQ

    opII[0, 0] = opId
    opII[1, 1] = opIn

    opA[ii, ii] = opAA
    opI[ii, ii] = opII

    opSS[0, 0] =  sig * opK
    opSS[0, 1] = - opV
    opSS[1, 0] = opW
    opSS[1, 1] = sig * opQ
    opS[ii, ii] = opSS


    for d in domains.getNeighborOf(dom['name']):
        jj = domains.getIndexDom(d['name'])
        print('===Coupling {0} with {1}: Block #({2}, {3})'.format(dom['name'],
                                                                   domains.getName(jj),
                                                                   ii, jj))

        opXX = bem.BlockedOperator(2, 2)

        opYY = bem.BlockedOperator(2, 2)
        opZZ = bem.BlockedOperator(2, 2)


        space_trial_d = bem_space(grid, kind_d[0], kind_d[1], domains=d['interfaces'])
        space_trial_n = bem_space(grid, kind_n[0], kind_n[1], domains=d['interfaces'])

        space_range_d = bem_space(grid, kind_d[0], kind_d[1], domains=d['interfaces'])
        space_range_n = bem_space(grid, kind_n[0], kind_n[1], domains=d['interfaces'])

        opXd = funI(space_trial_d, space_range_d, space_test_d)
        opXn = funI(space_trial_n, space_range_n, space_test_n)

        opXX[0, 0] = opXd
        opXX[1, 1] = - opXn

        opX[ii, jj] = opXX

        eps = d['phys']
        kb = kRef * np.sqrt(eps)


        opIdd = funI(space_trial_d, space_range_d, space_test_d)
        opInd = funI(space_trial_n, space_range_d, space_test_d)
        opIdn = funI(space_trial_d, space_range_n, space_test_n)
        opInn = funI(space_trial_n, space_range_n, space_test_n)

        a, b = 1.0, 1.0
        c, d = 1.0, 1.0

        ntd = 1.0 / (1j * kb)
        dtn = 1j * k

        opZZ[0, 0] = a * opIdd
        opZZ[0, 1] = b * ntd * opInd
        opZZ[1, 0] = c * dtn * opIdn
        opZZ[1, 1] = d * opInn

        opZ[ii, jj] = opZZ

        opYY[0, 0] = a * opIdd
        opYY[0, 1] = - b * ntd * opInd
        opYY[1, 0] = c * dtn * opIdn
        opYY[1, 1] = - d * opInn

        opY[ii, jj] = opYY

        # opRR[0, 0] = opXd
        # opRR[1, 1] = opXn
        # opR[ii, jj] = opRR

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

###############################
opA2 = opA * opA
opC = 0.5 * opI + opA
opC2 = opC * opC

print('assembling and inverting in BEM++ (remove this part) [checking]')
A2w = opA2.weak_form()
Cw = opC.weak_form()
C2w = opC2.weak_form()
#################################


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


A2 = A * iJ * A

Ce = 0.5 * J - At
Ci = 0.5 * J + At

Ce2 = Ce * iJ * Ce
Ci2 = Ci * iJ * Ci

x = np.random.rand(shape[0])

print('\n=Checking')
print('==check: M = A - X')
t0 = time()
y = M(x)
t1 = time() - t0

t0 = time()
z = A(x) - X(x)
t2 = time() - t0
print('#time: {0} {1} {2}'.format(t1, t2, t1 - t2))
e = la.norm(y - z)
print(e)

print('==check: A2 = J')
t0 = time()
y2 = A2(x)
t1 = time() - t0

t0 = time()
z2 = J(x)
t2 = time() - t0
print('#time: {0} {1} {2}'.format(t1, t2, t1 - t2))
e2 = la.norm(y2 - z2)
print(e2)

#######################
y = A2w.matvec(x)
z = 0.25 * Jw.matvec(x)
e = norm(y - z)
print(e)
#######################

print('==check exterior Proj.')
t0 = time()
yy = Ce2(x)
t1 = time() - t0

t0 = time()
zz = Ce(x)
t2 = time() - t0
print('#time: {0} {1} {2}'.format(t1, t2, t1 - t2))
ee = la.norm(yy - zz)
print(ee)

#######################
y = C2w.matvec(x)
z = Cw.matvec(x)
e = norm(y - z)
print(e)
#######################

print('==check interior Proj.')
t0 = time()
yy = Ci2(x)
t1 = time() - t0

t0 = time()
zz = Ci(x)
t2 = time() - t0
print('#time: {0} {1} {2}'.format(t1, t2, t1 - t2))
ee = la.norm(yy - zz)
print(ee)

print('==Error-Calderon with random [no-sense]')
t0 = time()
y = A(x)
t1 = time() - t0

t0 = time()
z = J(x)
t2 = time() - t0
print('#time: {0} {1} {2}'.format(t1, t2, t1 - t2))
e = la.norm(y - z)
print(e)

#################################################
#################################################
#################################################

from krylov import gmres

#################################################
#################################################
#################################################

iA = iJ * A * iJ

#################################################
print('==Error-inv A')
t0 = time()
y = iA(A(x))
t1 = time() - t0

t0 = time()
z = x
t2 = time() - t0
print('#time: {0} {1} {2}'.format(t1, t2, t1 - t2))
e = la.norm(y - z)
print(e)

t0 = time()
y = A(iA(x))
t1 = time() - t0

t0 = time()
z = x
t2 = time() - t0
print('#time: {0} {1} {2}'.format(t1, t2, t1 - t2))
e = la.norm(y - z)
print(e)
#################################################

Pjac = iA
Pgs = iA + iA * E * iA

Malpha = lambda alpha: (1.0 - alpha) * (A - J) + alpha * (J - X)

# Msigma = lambda sigma: A - J + sigma - sigma * iJ * X
# Mtau = lambda tau: tau * iJ * A - tau + J - X

Msigma = lambda sigma: (A - J) + sigma * (J - X)
Mtau = lambda tau: tau * (A - J) + J - X

Sw = opS.weak_form()
Sa = spla.LinearOperator(shape, matvec=Sw.matvec, dtype=complex)
S = 0.5 * J + Sa

# Rw = opR.weak_form()
# R = spla.LinearOperator(shape, matvec=Rw.matvec, dtype=complex)

# Yw = opY.weak_form()
# Y = spla.LinearOperator(shape, matvec=Yw.matvec, dtype=complex)

Mtheta = lambda theta: (A - J) + theta * (S - S * X)

#####################################
#####################################

Yw = opY.weak_form()
Zw = opZ.weak_form()
Y = spla.LinearOperator(shape, matvec=Yw.matvec, dtype=complex)
Z = spla.LinearOperator(shape, matvec=Zw.matvec, dtype=complex)

Mrho = lambda rho: (A - J) + rho * (Z - Y)

#####################################
#####################################

print('\n=Approx DtN/NtD')

class MyBlock:
    def __init__(self, shape, matvec, rhs, maxiter=10):
        self._matvec = spla.LinearOperator(shape, matvec=matvec, dtype=complex)
        self._rhs = spla.LinearOperator(shape, matvec=rhs, dtype=complex)
        self.shape = shape
        self.maxiter = maxiter
        self.dtype = np.dtype(np.complex)
        self.dot = self.matvec

    def matvec(self, x):
        tol = 1e-6
        maxiter = self.maxiter
        A = self._matvec
        b = self._rhs(x)
        y, info = gmres(A, b,
                        orthog='mgs',
                        tol=tol,
                        maxiter=maxiter)
        return y

from MatBlock import MatBlock

SP = MatBlock(dtype=complex)

for dom in domains:
        ii = domains.getIndexDom(dom['name'])
        print('==Domain: {0}'.format(dom['name']))
        print('===Diag: Block #({0}, {0})'.format(ii))

        opAA = bem.BlockedOperator(2, 2)
        opII = bem.BlockedOperator(2, 2)

        opAA = opA[ii, ii]
        # opmK already contains sign minus from def A
        opmK = opAA[0, 0]
        opV = opAA[0, 1]
        opW = opAA[1, 0]
        opQ = opAA[1, 1]

        opII = opI[ii, ii]
        opId = opII[0, 0]
        opIn = opII[1, 1]

        mKw = opmK.weak_form()
        Vw = opV.weak_form()
        Ww = opW.weak_form()
        Qw = opQ.weak_form()
        Jdw = opId.weak_form()
        Jnw = opIn.weak_form()

        mK = spla.LinearOperator(mKw.shape, matvec=mKw.matvec, dtype=complex)
        V = spla.LinearOperator(Vw.shape, matvec=Vw.matvec, dtype=complex)
        W = spla.LinearOperator(Ww.shape, matvec=Ww.matvec, dtype=complex)
        Q = spla.LinearOperator(Qw.shape, matvec=Qw.matvec, dtype=complex)
        Jd = spla.LinearOperator(Jdw.shape, matvec=Jdw.matvec, dtype=complex)
        Jn = spla.LinearOperator(Jnw.shape, matvec=Jnw.matvec, dtype=complex)

        IK = 0.5 * Jd - mK
        IQ = 0.5 * Jn + Q
        mV = -1.0 * V

        mNtD = MyBlock(IK.shape, IK.matvec, mV.matvec, 100)
        DtN = MyBlock(IK.shape, IQ.matvec, W.matvec, 100)

        spp = MatBlock(dtype=complex)

        spp.addBlock((0, 0), Jd)
        spp.addBlock((0, 1), mNtD)
        spp.addBlock((1, 1), Jn)
        spp.addBlock((1, 0), DtN)

        SP.addBlock((ii, ii), spp)


T = spla.LinearOperator(shape, matvec=SP.matvec, dtype=complex)

Mphi = lambda phi: (A - J) + phi * (T - T *X)

#################################################
#################################################



# F = 2.0 * At + 0.5 * X - X * At

CCi = iJ * (0.5 * J + At)
CCe = (0.5 * J - At)

B = J - X
F = B * CCi + CCe

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
print('-')
y = Ce(xx)
e = la.norm(y)
y = Ci(xx)
z = J(xx)
ee = la.norm(y - z)
y = CCi(xx)
z = xx
eee = la.norm(y - z)
print(e, eee)


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

print('\nSigmaJac restart={0} maxiter={1}'.format(restart, maxiter))
del res
res = []
Ms, bs = Msigma(sigma), sigma * b
tt = time()
xx, info = gmres(Ms, bs,
                 M = Pjac,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResSigJac = np.array(res)
ResSigJJac = rescaleRes(oResSigJac, Pjac, bs)
print('#time: {}'.format(tt))

print('=Error-Calderon Sigma Jacobi')
y = A(xx)
z = J(xx)
e = la.norm(y - z)
print(e)

print('\nSigmaGS restart={0} maxiter={1}'.format(restart, maxiter))
del res
res = []
Ms, bs = Msigma(sigma), sigma * b
tt = time()
xx, info = gmres(Ms, bs,
                 M = Pgs,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResSigGS = np.array(res)
ResSigJGS = rescaleRes(oResSigGS, Pgs, bs)
print('#time: {}'.format(tt))

print('=Error-Calderon Sigma GS')
y = A(xx)
z = J(xx)
e = la.norm(y - z)
print(e)

# ####################################
# ####################################


#tau = -X * iJ * (J + A)
tau = -0.5
print('\nTauWO restart={0} maxiter={1}'.format(restart, maxiter))
del res
res = []
Ms, bs = Mtau(tau), b
tt = time()
xx, info = gmres(Ms, bs,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
tt = time() - tt
print(info, len(res))
oResTauWO = np.array(res)
ResTauWO = rescaleRes(oResTauWO, lambda x: x, bs)
print('#time: {}'.format(tt))

print('=Error-Calderon Tau WO')
y = A(xx)
z = J(xx)
e = la.norm(y - z)
ee = la.norm(xx)
print(e, ee, norm_b)

# ####################################
# ####################################

print('\nFWO restart={0} maxiter={1}'.format(restart, maxiter))
del res
res = []
Ms, bs = F,  b
tt = time()
xx, info = gmres(Ms, bs,
                 orthog='mgs',
                 tol=tol,
                 residuals=res,
                 restrt=restart,
                 maxiter=maxiter)
xx = CCi(xx)
tt = time() - tt
print(info, len(res))
oResFWO = np.array(res)
ResFWO = rescaleRes(oResFWO, lambda x: x, bs)
print('#time: {}'.format(tt))

u = xx

print('=Error-Calderon F WO')
y = A(xx)
z = J(xx)
e = la.norm(y - z)
ee = la.norm(xx)
print(e, ee, norm_b)
print('-')
y = At(xx)
z = 0.5 * J(xx)
e = la.norm(y - z)
y = Ce(xx)
ee = la.norm(y)
y = Ci(xx)
z = J(xx)
eee = la.norm(y - z)
y = CCi(xx)
z = xx
eeee = la.norm(y - z)
eeeee = la.norm(B(xx) - b)
print(e, ee, eee, eeee, eeeee)


# ####################################
# ####################################

# phi = 0.5
# print('\nPhiWO restart={0} maxiter={1}'.format(restart, maxiter))
# del res
# res = []
# Ms, bs = Mphi(phi), phi * T(b)
# tt = time()
# xx, info = gmres(Ms, bs,
#                  orthog='mgs',
#                  tol=tol,
#                  residuals=res,
#                  restrt=restart,
#                  maxiter=maxiter)
# tt = time() - tt
# print(info, len(res))
# oResPhiWO = np.array(res)
# ResPhiWO = rescaleRes(oResPhiWO, lambda x: x, bs)
# print('#time: {}'.format(tt))

# print('=Error-Calderon Phi WO')
# y = A(xx)
# z = J(xx)
# e = la.norm(y - z)
# ee = la.norm(xx)
# print(e, ee, norm_b)


# theta = 1.0
# print('\nThetaWO restart={0} maxiter={1}'.format(restart, maxiter))
# del res
# res = []
# Ms, bs = Mtheta(theta), theta * S(b)
# tt = time()
# xx, info = gmres(Ms, bs,
#                  orthog='mgs',
#                  tol=tol,
#                  residuals=res,
#                  restrt=restart,
#                  maxiter=maxiter)
# tt = time() - tt
# print(info, len(res))
# oResThetaWO = np.array(res)
# ResThetaWO = rescaleRes(oResThetaWO, lambda x: x, bs)
# print('#time: {}'.format(tt))

# print('=Error-Calderon Theta WO')
# y = A(xx)
# z = J(xx)
# e = la.norm(y - z)
# ee = la.norm(xx)
# print(e, ee, norm_b)


# rho = 1.0
# print('\nRhoWO restart={0} maxiter={1}'.format(restart, maxiter))
# del res
# res = []
# Ms, bs = Mrho(rho), rho * Z(b)
# tt = time()
# xx, info = gmres(Ms, bs,
#                  orthog='mgs',
#                  tol=tol,
#                  residuals=res,
#                  restrt=restart,
#                  maxiter=maxiter)
# tt = time() - tt
# print(info, len(res))
# oResRhoWO = np.array(res)
# ResRhoWO = rescaleRes(oResRhoWO, lambda x: x, bs)
# print('#time: {}'.format(tt))

# print('=Error-Calderon Rho WO')
# y = A(xx)
# z = J(xx)
# e = la.norm(y - z)
# ee = la.norm(xx)
# print(e, ee, norm_b)


# ####################################
# ####################################

# print('')

# from krypy.linsys import LinearSystem
# from krypy.deflation import DeflatedGmres as GmresD

# #####################################

# linear_system = LinearSystem(M, b)

# tt = time()
# solverd = GmresD(linear_system,
#                    ortho='mgs',
#                    tol=tol,
#                    maxiter=maxiter*restart)
# tt = time() - tt
# print('#time: {}'.format(tt))

# #####################################

# PM = Pjac * M
# Pb = Pjac(b)
# linear_system = LinearSystem(PM, Pb)

# tt = time()
# solverdp = GmresD(linear_system,
#                    ortho='mgs',
#                    tol=tol,
#                    maxiter=maxiter*restart)
# tt = time() - tt
# print('#time: {}'.format(tt))

# #####################################

# sigma = -1.0
# Ms, bs = Msigma(sigma), sigma * b

# linear_system = LinearSystem(Ms, bs)

# tt = time()
# solverds = GmresD(linear_system,
#                  ortho='mgs',
#                  tol=tol,
#                  maxiter=maxiter*restart)
# tt = time() - tt
# print('#time: {}'.format(tt))

# #####################################

# tau = -1.0
# Mt, bt = Mtau(tau), b

# linear_system = LinearSystem(Mt, bt)

# tt = time()
# solverdt = GmresD(linear_system,
#                   ortho='mgs',
#                   tol=tol,
#                   maxiter=maxiter*restart)
# tt = time() - tt
# print('#time: {}'.format(tt))

# print('')

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

its, res = Res2Tuple(ResSigJJac)
plt.semilogy(its, res, 'b--', linewidth=3,  label='Sigma Jacobi')

its, res = Res2Tuple(ResSigJGS)
plt.semilogy(its, res, 'r--', linewidth=3,  label='Sigma Gauss-Siedel')

its, res = Res2Tuple(ResTauWO)
plt.semilogy(its, res, 'm-', linewidth=3,  label='Tau')

its, res = Res2Tuple(ResFWO)
plt.semilogy(its, res, 'g--', linewidth=3,  label='F')


# its, res = Res2Tuple(ResPhiWO)
# plt.semilogy(its, res, 'y-', linewidth=3,  label='Phi')

# its, res = Res2Tuple(ResThetaWO)
# plt.semilogy(its, res, 'c-', linewidth=3,  label='Theta')

# its, res = Res2Tuple(ResRhoWO)
# plt.semilogy(its, res, 'c-', linewidth=3,  label='Rho')

# its, res = Res2Tuple(ResSigSJac)
# plt.semilogy(its, res, 'b-.', linewidth=3,  label='Sigma S / Jac')

# its, res = Res2Tuple(solverd.resnorms)
# plt.semilogy(its, res, 'k--', linewidth=3,  label='Deflated')

# its, res = Res2Tuple(solverdp.resnorms)
# plt.semilogy(its, res, 'b--', linewidth=3,  label='Deflated + Jacobi')

# its, res = Res2Tuple(solverds.resnorms)
# plt.semilogy(its, res, 'g--', linewidth=3,  label='Sigma Deflated')

# its, res = Res2Tuple(solverdt.resnorms)
# plt.semilogy(its, res, 'm--', linewidth=3,  label='Tau Deflated')


plt.title('Convergence History', fontsize=20)
plt.xlabel('#iterations', fontsize=14)
plt.ylabel('normalized residual', fontsize=30)
plt.legend()

plt.grid(True)

plt.show()
