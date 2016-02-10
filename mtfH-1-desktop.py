#!/usr/bin/env python

# This script calculates the field generated by a z-polarised, x-propagating
# plane wave incident on a dielectric sphere.

# Help Python find the bempp module
import sys
sys.path.append("..")

from bempp.lib import *
import numpy as np

from matplotlib import pylab as pl

import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import time

# Physical Parameters

epsInt = 1.5**2
epsExt = 1
muInt = 1
muExt = 1
wavelengthVacuum = 10.
kVacuum = 2 * np.pi / wavelengthVacuum
kExt = kVacuum * np.sqrt(epsExt * muExt)
kInt = kVacuum * np.sqrt(epsInt * muInt)
rho = (kInt * muExt) / (kExt * muInt)

# Boundary conditions

def evalIncDirichletTrace(point, normal):
    x, y, z = point
    return np.exp( 1j * kExt * (x))

def evalIncNeumannTrace(point, normal):
    x, y, z = point
    nx, ny, nz = normal
#    return -1j * (nx+ny) * kExt * np.exp( 1j * kExt * (x+y))
#    return 1j * nx * kExt * np.exp( 1j * kExt * x)
    return -1j * nx * kExt * np.exp( 1j * kExt * x)


# 
print('==== Create quadrature strategy')

accuracyOptions = createAccuracyOptions()
accuracyOptions.doubleRegular.setRelativeQuadratureOrder(2)
accuracyOptions.doubleSingular.setRelativeQuadratureOrder(1)
quadStrategy = createNumericalQuadratureStrategy(
    "float64", "complex128", accuracyOptions)

# 
print('==== Create assembly context')

assemblyOptions = createAssemblyOptions()
#assemblyOptions.switchToAcaMode(createAcaOptions())
context = createContext(quadStrategy, assemblyOptions)

# 
print('==== Load mesh')
#meshname = "./sphere-simple.msh"
#meshname = "./sphere.msh"
meshname = "./cube.msh"
grid = createGridFactory().importGmshGrid(
    "triangular", meshname)

# 
print('==== Initialize functional spaces')

#space = createPiecewiseConstantScalarSpace(context, grid)
#space = createPiecewiseConstantDualGridScalarSpace(context, grid)

space = createPiecewiseLinearContinuousScalarSpace(context, grid)

#space = createPiecewisePolynomialContinuousScalarSpace(context, grid, 
                                                       # polynomialOrder=1)
#space = createPiecewiseLinearDiscontinuousScalarSpace(context, grid)

# 
print('==== Define the grid functions for the traces of incident field')

incDirichletTrace = createGridFunction(
    context, space, space, evalIncDirichletTrace,
    surfaceNormalDependent=True)
incNeumannTrace = createGridFunction(
    context, space, space, evalIncNeumannTrace,
    surfaceNormalDependent=True)

# Extract plane wave

exportToGmsh(incDirichletTrace, 'inc-d', 'Inc_D.pos')
exportToGmsh(incNeumannTrace, 'inc-n', 'Inc_N.pos')

def RevalIncNeumannTrace(point, normal):
    x, y, z = point
    nx, ny, nz = normal
    return np.real(-1j * nx * kExt * np.exp( 1j * kExt * x))

def IevalIncNeumannTrace(point, normal):
    x, y, z = point
    nx, ny, nz = normal
    return np.imag(-1j * nx * kExt * np.exp( 1j * kExt * x))

RincNeumannTrace = createGridFunction(
    context, space, space, RevalIncNeumannTrace,
    surfaceNormalDependent=True)
IincNeumannTrace = createGridFunction(
    context, space, space, IevalIncNeumannTrace,
    surfaceNormalDependent=True)
CincNeumannTrace = RincNeumannTrace + 1j * IincNeumannTrace

exportToGmsh(CincNeumannTrace, 'inc-n-c', 'Inc_Nc.pos')
exportToGmsh(RincNeumannTrace, 'inc-n-r', 'Inc_Nr.pos')
exportToGmsh(IincNeumannTrace, 'inc-n-i', 'Inc_Ni.pos')


#sys.exit('Yahoga!')

#
print('==== Construct elementary operators')


V0 = createHelmholtz3dSingleLayerBoundaryOperator(
    context, space, space, space, kExt, "V0")
K0 = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, space, space, space, kExt, "K0")
K0a = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, space, space, space, kExt, "Q0")
W0 = createHelmholtz3dHypersingularBoundaryOperator(
    context, space, space, space, kExt, "W")
V1 = createHelmholtz3dSingleLayerBoundaryOperator(
    context, space, space, space, kInt, "V1")
K1 = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, space, space, space, kInt, "K1")
K1a = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, space, space, space, kInt, "Q1")
W1 = createHelmholtz3dHypersingularBoundaryOperator(
    context, space, space, space, kInt, "W1")

Id = createIdentityOperator(
    context, space, space, space, "Id")
Zd = createNullOperator(
    context, space, space, space, "Zd")

# reorientation normal
ak, bq = -1. , -1.
K0, K0a = ak * K0, bq * K0a

# 
print('==== Form the STF matrix')

STF00 = -K0+K1
STF01 = V0+V1
STF10 = W0+W1
STF11 = K0a-K1a

STF_list = [[STF00, STF01], [STF10, STF11]]

STF = createBlockedBoundaryOperator(
    context, STF_list)

for obj in STF_list:
    for o in obj:
        a = o.weakForm()
a = Id.weakForm()

rhsP = [Id * incNeumannTrace, Id * incDirichletTrace]
rhsSTF = [ (0.5*Id + K1) * incDirichletTrace - V1 * incNeumannTrace,
           +W1 * incDirichletTrace + (-0.5*Id+K1a) * incNeumannTrace]


# 
print('==== Form the MTF matrix')

Half= -0.5 * Id
ZZ =  Zd

MTF_list = [ 
    [-K0, V0, Half, ZZ], 
    [W0, K0a, ZZ, -Half],
    [Half, ZZ, -K1, V1], 
    [ZZ, -Half, W1, K1a] 
    ]

MTF = createBlockedBoundaryOperator(
    context, MTF_list)

rhsMTF = [0.5*Id * incDirichletTrace,
          -0.5*Id * incNeumannTrace, 
          -0.5*Id * incDirichletTrace,
          -0.5*Id * incNeumannTrace]

rhsMTF_wo = [0.5 * incDirichletTrace,
          -0.5 * incNeumannTrace, 
          -0.5 * incDirichletTrace,
          -0.5 * incNeumannTrace]


for os in MTF_list:
    for o in os:
        a = o.weakForm()
for o in rhsMTF:
    a = o.projections()

# 
print('==== Initialize the solver')

STFsolver = createDefaultDirectSolver(STF)
MTFsolver = createDefaultDirectSolver(MTF)

# 
print('==== Solve STF \t'),
print(STF.weakForm().shape)
sys.stdout.flush()

t0 = time.time()
STFsolution = STFsolver.solve(rhsSTF)
t1 = time.time()
print(t1-t0)
print STFsolution.solverMessage()

print('==== Solve MTF \t'),
print(MTF.weakForm().shape)
sys.stdout.flush()

t0 = time.time()
MTFsolution = MTFsolver.solve(rhsMTF)
t1 = time.time()
print(t1-t0)
print MTFsolution.solverMessage()

print('==== end solved... extraction...')

########################################################
########################################################

# STF extraction
# Extract the solution components in the form of grid functions

STFextDirichletTrace = STFsolution.gridFunction(0)
STFextNeumannTrace = STFsolution.gridFunction(1)

exportToGmsh(STFextDirichletTrace, 'STF-dext', 'STF_Dext.pos')
exportToGmsh(STFextNeumannTrace, 'STF-next', 'STF_Next.pos')

STFscattDirichletTrace = STFextDirichletTrace - incDirichletTrace
STFscattNeumannTrace = STFextNeumannTrace - incNeumannTrace

exportToGmsh(STFscattDirichletTrace, 'STF-dsca', 'STF_Dsca.pos')

STFintDirichletTrace = STFextDirichletTrace
STFintNeumannTrace = STFextNeumannTrace 

########################################################
########################################################

# MTF extraction
# Extract the solution components in the form of grid functions

MTFextDirichletTrace = MTFsolution.gridFunction(0)
MTFextNeumannTrace = MTFsolution.gridFunction(1)

exportToGmsh(MTFextDirichletTrace, 'MTF-dext', 'MTF_Dext.pos')
exportToGmsh(MTFextNeumannTrace, 'MTF-next', 'MTF_Next.pos')

MTFscattDirichletTrace = MTFextDirichletTrace - incDirichletTrace
MTFscattNeumannTrace = MTFextNeumannTrace + incNeumannTrace

exportToGmsh(MTFscattDirichletTrace, 'MTF-dsca', 'MTF_Dsca.pos')
exportToGmsh(-MTFscattNeumannTrace, 'MTF-nsca', 'MTF_Nsca.pos')

MTFintDirichletTrace = MTFsolution.gridFunction(2)
MTFintNeumannTrace = MTFsolution.gridFunction(3)

exportToGmsh(MTFintDirichletTrace, 'MTF-dint', 'MTF_Dint.pos')
exportToGmsh(MTFintNeumannTrace, 'MTF-nint', 'MTF_Nint.pos')

exportToGmsh(-0.5j*MTFintNeumannTrace, 'MTF-nint2', 'MTF_Nint2.pos')

# #
# MTFextDirichletTraceMass = Id *MTFsolution.gridFunction(0)
# MTFextNeumannTraceMass = Id * MTFsolution.gridFunction(1)

# exportToGmsh(MTFextDirichletTraceMass, 'MTF-dextM', 'MTF_Dmext.pos')
# exportToGmsh(MTFextNeumannTraceMass, 'MTF-nextM', 'MTF_Nmext.pos')

MTFintDirichletTraceMass = Id * MTFsolution.gridFunction(2)
MTFintNeumannTraceMass = Id * MTFsolution.gridFunction(3)

exportToGmsh(MTFintDirichletTraceMass, 'MTF-dintM', 'MTF_Dmint.pos')
exportToGmsh(MTFintNeumannTraceMass, 'MTF-nintM', 'MTF_Nmint.pos')

########################################################
########################################################

print('\n post\n')

Half= 0.0 * Id
ZZ =  Zd

a = [ 
    [-K0, V0, Half, ZZ], 
    [W0, K0a, ZZ, -Half],
    [Half, ZZ, -K1, V1], 
    [ZZ, -Half, W1, K1a] 
    ]
t = time.time()
AAblock = createBlockedBoundaryOperator(context, a)
tt = time.time()
print(tt-t)
t = time.time()
a = [ 
    [-K0, V0], 
    [W0, K0a]
    ]
A0block = createBlockedBoundaryOperator(context, a)
tt = time.time()
print(tt-t)
t = time.time()
a = [
    [-K1, V1], 
    [W1, K1a]
    ]
A1block = createBlockedBoundaryOperator(context, a)
tt = time.time()
print(tt-t)
t = time.time()
Ismallblock = createBlockedBoundaryOperator(
    context, 
    [ [Id, Zd], 
      [Zd, Id] ]
    )
tt = time.time()
print(tt-t)
t = time.time()
IIblock = createBlockedBoundaryOperator(
    context, 
    [ [Id, ZZ, Half, ZZ], 
      [ZZ, Id, ZZ, Half],
      [Half, ZZ, Id, ZZ], 
      [ZZ, Half, ZZ, Id] ]
    )
tt = time.time()
print(tt-t)

print('are times to instanciate all the missing matrices.\n')

t = time.time()
print('building M...'),
sys.stdout.flush()
M = MTF.weakForm() #.asMatrix()
tt = time.time()
print(tt-t),

print('building b...'),
sys.stdout.flush()

b = np.array([])
for v in rhsMTF:
    b = np.concatenate((b, v.projections()))
tt = time.time()
print(tt-t),
print('done.\n')

# print('solve MTF with Numpy... be patient [take time]'),
# print(M.shape)
# sys.stdout.flush()
# t0 = time.time()
# x = np.linalg.solve(M, b)
# t1 = time.time()
# print(t1-t0)

print('')

t = time.time()
print('building A...'),
sys.stdout.flush()
A = AAblock.weakForm() #.asMatrix()
tt = time.time()
print(tt-t),
t = time.time()
print('building II...'),
sys.stdout.flush()
II = IIblock.weakForm() #.asMatrix()
tt = time.time()
print(tt-t),
t = time.time()

A0 = A0block.weakForm() #.asMatrix()
A1 = A1block.weakForm() #.asMatrix()
IJ = Ismallblock.weakForm() #.asMatrix()

bb = np.array([])
for v in rhsMTF_wo:
    bb = np.concatenate((bb, v.projections()))
tt = time.time()
# print(tt-t),

print('done.\n')

gD = incDirichletTrace.coefficients()
gN = incNeumannTrace.coefficients()

y = np.array([])
for ii in xrange(len(rhsMTF)):
    gf = MTFsolution.gridFunction(ii)
    y = np.concatenate((y, gf.coefficients()))

yext = np.array([])
yint = np.array([])

yDext = MTFsolution.gridFunction(0).coefficients()
yNext = MTFsolution.gridFunction(1).coefficients()
yext = np.concatenate((yext, yDext))
yext = np.concatenate((yext, yNext))

yDint = MTFsolution.gridFunction(2).coefficients()
yNint = MTFsolution.gridFunction(3).coefficients()
yint = np.concatenate((yint, yDint))
yint = np.concatenate((yint, yNint))

yint_sign = np.array([])
yint_sign = np.concatenate((yint_sign, MTFsolution.gridFunction(2).coefficients()))
yint_sign = np.concatenate((yint_sign, -MTFsolution.gridFunction(3).coefficients()))

ysca = np.array([])
yDsca = MTFscattDirichletTrace.coefficients()
yNsca = MTFscattNeumannTrace.coefficients()
ysca = np.concatenate((ysca, yDsca))
ysca = np.concatenate((ysca, yNsca))


yyext = np.array([])
yyint = np.array([])
yy = np.array([])

yyDext = STFsolution.gridFunction(0).coefficients()
yyNext = STFsolution.gridFunction(1).coefficients()
yyext = np.concatenate((yyext, yyDext))
yyext = np.concatenate((yyext, yyNext))

yyDint, yyNint = yyDext-gD, -yyNext-gN
yyint = np.concatenate((yyint, yyDint))
yyint = np.concatenate((yyint, yyNint))

yy = np.concatenate((yy, yyext))
yy = np.concatenate((yy, yyint))


x = y

print('\n\nCalderon error: absolute ; relative | rhs')
e = 2*A.matvec(x) - II.matvec(x)
print(np.linalg.norm(e)),
print('\t ; \t'),
print(np.linalg.norm(e) / np.linalg.norm(x)),
print('\t | \t'),
print(np.linalg.norm(2*A.matvec(b)-II.matvec(b)))


E_mtf = np.linalg.norm( 2*A.matvec(y) - II.matvec(y) )
E_mtf_ext = np.linalg.norm( 2*A0.matvec(yext) - IJ.matvec(yext) )
E_mtf_int = np.linalg.norm( 2*A1.matvec(yint) - IJ.matvec(yint) )
E_mtf_diff = np.linalg.norm( ysca-yint_sign )

E_stf = np.linalg.norm( 2*A.matvec(yy) - II.matvec(yy) )
E_stf_ext = np.linalg.norm( 2*A0.matvec(yyext) - IJ.matvec(yyext) )
E_stf_int = np.linalg.norm( 2*A1.matvec(yyint) - IJ.matvec(yyint) )
E_stf_diff = -1


def mynorm(vec):
    gf = createGridFunction(context, space, coefficients=vec)
    return gf.L2Norm()

normDext, normDint =  mynorm(yDext), mynorm(yDint),
normNext, normNint =  mynorm(yNext), mynorm(yNint),

normDDext, normDDint =  mynorm(yyDext), mynorm(yyDint),
normNNext, normNNint =  mynorm(yyNext), mynorm(yyNint),

normEDext, normEDint =  mynorm(yDext-yyDext), mynorm(yDint-yyDint),
normENext, normENint =  mynorm(yNext-yyNext), mynorm(yNint-yyNint),

e_mtf_diffD, e_mtf_diffN = mynorm(yDext-gD-yDint), mynorm(yNext+gN+yNint),


print('\n\n')

print('# ell^2')
print("# norm  {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15}".format(
        'Dext', 'Dint', 
        'Next', 'Nint',
        'sol', 'ext', 'int'
        ))
print("# MTF   {:^15E} {:^15E} {:^15E} {:^15E} {:^15E} {:^15E} {:^15E}".format(
        np.linalg.norm(yDext), np.linalg.norm(yDint),
        np.linalg.norm(yNext), np.linalg.norm(yNint),
        np.linalg.norm(y), np.linalg.norm(yext), np.linalg.norm(yint)
        ))
print("# STF   {:^15E} {:^15E} {:^15E} {:^15E} {:^15E} {:^15E} {:^15E}".format(
        np.linalg.norm(yyDext), np.linalg.norm(yyDint),
        np.linalg.norm(yyNext), np.linalg.norm(yyNint),
        np.linalg.norm(yy), np.linalg.norm(yyext), np.linalg.norm(yyint)
        ))

print("# M-S|TF{:^15E} {:^15E} {:^15E} {:^15E} {:^15E} {:^15E} {:^15E}".format(
        np.linalg.norm(yDext-yyDext), np.linalg.norm(yDint-yyDint),
        np.linalg.norm(yNext-yyNext), np.linalg.norm(yNint-yyNint),
        np.linalg.norm(y-yy), np.linalg.norm(yext-yyext), np.linalg.norm(yint-yyint)
        ))


print('')

print("# error {:^15} {:^15} {:^15} {:^15} {:^15} {:^15}".format(
        'jumpD', 'jumpN', 'jump',
        'tot', 'ext', 'int'
        ))
print("# MTF   {:^15E} {:^15E} {:^15E} {:^15E} {:^15E} {:^15E}".format(
        np.linalg.norm(yDext-gD-yDint), np.linalg.norm(yNext+gN+yNint),  E_mtf_diff,
        E_mtf, E_mtf_ext, E_mtf_int
        ))
print("# STF   {:^15E} {:^15E} {:^15E} {:^15E} {:^15E} {:^15E}".format(
        -1, -1,  E_stf_diff,
        E_stf, E_stf_ext, E_stf_int
        ))

print('\n')


print('# L^2')
print("# norm  {:^15} {:^15} {:^15} {:^15}".format(
        'Dext', 'Dint', 
        'Next', 'Nint'
        ))
print("# MTF   {:^15E} {:^15E} {:^15E} {:^15E}".format(
        normDext, normDint,
        normNext, normNint
        ))
print("# STF   {:^15E} {:^15E} {:^15E} {:^15E}".format(
        normDDext, normDDint,
        normNNext, normNNint
        ))

print("# M-S|TF{:^15E} {:^15E} {:^15E} {:^15E}".format(
        normEDext, normEDint,
        normENext, normENint
        ))

print('')

print("# error {:^15} {:^15} {:^15} {:^15} {:^15} {:^15}".format(
        'jumpD', 'jumpN', 'jump',
        'tot', 'ext', 'int'
        ))
print("# MTF   {:^15E} {:^15E} {:^15E} {:^15E} {:^15E} {:^15E}".format(
        e_mtf_diffD, e_mtf_diffN,
        -1, -1, -1, -1
        ))
