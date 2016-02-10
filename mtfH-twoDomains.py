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


eps = [1, 2, 3]

epsInt = 1
epsExt = 1
muInt = 1
muExt = 1
wavelengthVacuum = 9.5
kVacuum = 2 * np.pi / wavelengthVacuum

kExt = kVacuum * np.sqrt(eps[0] * muExt)
kInt1 = kVacuum * np.sqrt(eps[1] * muInt)
kInt2 = kVacuum * np.sqrt(eps[2] * muInt)

# Boundary conditions

def evalIncDirichletTrace(point, normal):
    x, y, z = point
    return -np.exp( 1j * kExt * (x))

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
meshname = "./sphere-sphere.msh"
#meshname = "./sphere.msh"
#meshname = "./cube.msh"
#meshname = "./cylinder-simple.msh"
grid = createGridFactory().importGmshGrid(
    "triangular", meshname)

interfaceA = (GridSegment.closedDomain(grid, 10))
interfaceB = (GridSegment.closedDomain(grid, 100))


#
print('==== Initialize functional spaces')

space = createPiecewiseLinearContinuousScalarSpace(context, grid)

P1_A = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceA,
                                                  strictlyOnSegment=True)
P1_B = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceB,
                                                      strictlyOnSegment=True)


#
print('==== Define the grid functions for the traces of incident field')

incDirichletTraceA = createGridFunction(
    context, P1_A, P1_A, evalIncDirichletTrace,
    surfaceNormalDependent=True)
incNeumannTraceA = createGridFunction(
    context, P1_A, P1_A, evalIncNeumannTrace,
    surfaceNormalDependent=True)

incDirichletTraceB = createGridFunction(
    context, P1_B, P1_B, evalIncDirichletTrace,
    surfaceNormalDependent=True)
incNeumannTraceB = createGridFunction(
    context, P1_B, P1_B, evalIncNeumannTrace,
    surfaceNormalDependent=True)

# Extract plane wave

exportToGmsh(incDirichletTraceA, 'inc-dext', 'Inc_D_A.pos')
exportToGmsh(incDirichletTraceB, 'inc-dext', 'Inc_D_B.pos')


#sys.exit('Yahoga!')


print('==== Construct elementary operators')


V0aa = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P1_A, P1_A, P1_A, kExt, "V0")
K0aa = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_A, P1_A, P1_A, kExt, "K0")
Q0aa = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P1_A, P1_A, P1_A, kExt, "Q0")
W0aa = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_A, P1_A, P1_A, kExt, "W0")

V1 = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P1_A, P1_A, P1_A, kInt1, "V1")
K1 = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_A, P1_A, P1_A, kInt1, "K1")
Q1 = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P1_A, P1_A, P1_A, kInt1, "Q1")
W1 = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_A, P1_A, P1_A, kInt1, "W1")

Idaa = createIdentityOperator(
    context, P1_A, P1_A, P1_A, "Id")
Zdaa = createNullOperator(
    context, P1_A, P1_A, P1_A, "Zd")

V0bb = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P1_B, P1_B, P1_B, kExt, "V0")
K0bb = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_B, P1_B, P1_B, kExt, "K0")
Q0bb = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P1_B, P1_B, P1_B, kExt, "Q0")
W0bb = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_B, P1_B, P1_B, kExt, "W0")

V2 = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P1_B, P1_B, P1_B, kInt2, "V2")
K2 = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_B, P1_B, P1_B, kInt2, "K2")
Q2 = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P1_B, P1_B, P1_B, kInt2, "Q2")
W2 = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_B, P1_B, P1_B, kInt2, "W2")

Idbb = createIdentityOperator(
    context, P1_B, P1_B, P1_B, "Id")
Zdbb = createNullOperator(
    context, P1_B, P1_B, P1_B, "Zd")

## mixed

V0ab = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P1_A, P1_A, P1_B, kExt, "V0")
K0ab = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_A, P1_A, P1_B, kExt, "K0")
Q0ab = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P1_A, P1_A, P1_B, kExt, "Q0")
W0ab = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_A, P1_A, P1_B, kExt, "W0")

Idab = createIdentityOperator(
    context, P1_A, P1_A, P1_B, "Id")
Zdab = createNullOperator(
    context, P1_A, P1_A, P1_B, "Zd")

V0ba = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P1_B, P1_B, P1_A, kExt, "V0")
K0ba = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_B, P1_B, P1_A, kExt, "K0")
Q0ba = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P1_B, P1_B, P1_A, kExt, "Q0")
W0ba = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_B, P1_B, P1_A, kExt, "W0")

Idba = createIdentityOperator(
    context, P1_B, P1_B, P1_A, "Id")
Zdba = createNullOperator(
    context, P1_B, P1_B, P1_A, "Zd")


# reorientation normal
K0aa, K0ab = -1.0* K0aa, -1.0* K0ab
K0ba, K0bb = -1.0* K0ba, -1.0* K0bb

Q0aa, Q0ab = -1.0* Q0aa, -1.0* Q0ab
Q0ba, Q0bb = -1.0* Q0ba, -1.0* Q0bb

#
print('==== Form the MTF matrix')

HalfAA= -0.5 * Idaa
HalfBB= -0.5 * Idbb
#
HalfAB= -0.5 * Idab
HalfBA= -0.5 * Idba

zAA= 0. * Zdaa
zBB= 0. * Zdbb
#
zAB= 0. * Zdab
zBA= 0. * Zdba


MTF_list = [

    [-K0aa, -K0ba, V0aa, V0ba, HalfAA, zAA, zBA, zBA],
    [-K0ab, -K0bb, V0ab, V0bb, zAB, zAB, HalfBB, zBB],

    [W0aa, W0ba, Q0aa, Q0ba, zAA, -HalfAA, zBA, zBA],
    [W0ab, W0bb, Q0ab, Q0bb, zAB, zAB, zBB, -HalfBB],

    [HalfAA, zBA, zAA, zBA, -K1, V1, zBA, zBA],
    [zAA, zBA, -HalfAA, zBA, W1, Q1, zBA, zBA],

    [zAB, HalfBB, zAB, zBB, zAB, zAB, -K2, V2],
    [zAB, zBB, zAB, -HalfBB, zAB, zAB, W2, Q2],

    ]

MTF = createBlockedBoundaryOperator(
    context, MTF_list)

rhsMTF = [
    0.5* incDirichletTraceA,
    0.5* incDirichletTraceB,
    -0.5* incNeumannTraceA,
    -0.5* incNeumannTraceB,
    -0.5* incDirichletTraceA,
    -0.5* incNeumannTraceA,
    -0.5* incDirichletTraceB,
    -0.5* incNeumannTraceB
]



for os in MTF_list:
    for o in os:
        a = o.weakForm()
for o in rhsMTF:
    a = o.projections()

#
print('==== Initialize the solver')

# MTFsolver = createDefaultDirectSolver(MTF)

MTFsolver = createDefaultIterativeSolver(MTF)
MTFsolver.initializeSolver(defaultGmresParameterList(1e-8))

#
print('==== Solve MTF \t'),
print(MTF.weakForm().shape)
sys.stdout.flush()

t0 = time.time()
MTFsolution = MTFsolver.solve(rhsMTF)
t1 = time.time()
print(t1-t0)
print(MTFsolution.solverMessage())
print(MTFsolution.iterationCount())


print('==== end solved... extraction...')

########################################################
########################################################

# MTF extraction
# Extract the solution components in the form of grid functions

MTFextDirichletTraceA = MTFsolution.gridFunction(0)
MTFextDirichletTraceB = MTFsolution.gridFunction(1)
MTFextNeumannTraceA = MTFsolution.gridFunction(2)
MTFextNeumannTraceB = MTFsolution.gridFunction(3)

# exportToGmsh(MTFextDirichletTrace, 'MTF-dext', 'MTF_Dext.pos')
# exportToGmsh(MTFextNeumannTrace, 'MTF-next', 'MTF_Next.pos')

# MTFscattDirichletTrace = MTFextDirichletTrace - incDirichletTrace
# MTFscattNeumannTrace = MTFextNeumannTrace + incNeumannTrace

# exportToGmsh(MTFscattDirichletTrace, 'MTF-dsca', 'MTF_Dsca.pos')
# exportToGmsh(-MTFscattNeumannTrace, 'MTF-nsca', 'MTF_Nsca.pos')

# imagMTFscattDirichletTrace = - 1j * MTFscattDirichletTrace
# exportToGmsh(imagMTFscattDirichletTrace, 'MTF-dsca-i', 'MTF_DscaI.pos')
# exportToGmsh(MTFscattDirichletTrace, 'MTF-dsca-r', 'MTF_DscaR.pos')

MTFintDirichletTraceA = MTFsolution.gridFunction(4)
MTFintNeumannTraceA = MTFsolution.gridFunction(5)

MTFintDirichletTraceB = MTFsolution.gridFunction(6)
MTFintNeumannTraceB = MTFsolution.gridFunction(7)


# exportToGmsh(MTFintDirichletTrace, 'MTF-dint', 'MTF_Dint.pos')
# exportToGmsh(MTFintNeumannTrace, 'MTF-nint', 'MTF_Nint.pos')

# exportToGmsh(-0.5j*MTFintNeumannTrace, 'MTF-nint2', 'MTF_Nint2.pos')

# #
# MTFextDirichletTraceMass = Id *MTFsolution.gridFunction(0)
# MTFextNeumannTraceMass = Id * MTFsolution.gridFunction(1)

# exportToGmsh(MTFextDirichletTraceMass, 'MTF-dextM', 'MTF_Dmext.pos')
# exportToGmsh(MTFextNeumannTraceMass, 'MTF-nextM', 'MTF_Nmext.pos')

# MTFintDirichletTraceMass = Id * MTFsolution.gridFunction(2)
# MTFintNeumannTraceMass = Id * MTFsolution.gridFunction(3)

# exportToGmsh(MTFintDirichletTraceMass, 'MTF-dintM', 'MTF_Dmint.pos')
# exportToGmsh(MTFintNeumannTraceMass, 'MTF-nintM', 'MTF_Nmint.pos')

t =  MTFextDirichletTraceA
exportToGmsh(t, 'MTF-dext-r', 'MTF_DextR-A.pos')
tI = -1j * t
exportToGmsh(tI, 'MTF-dext-i', 'MTF_DextI-A.pos')

t =  MTFintDirichletTraceA
exportToGmsh(t, 'MTF-dint-r', 'MTF_DintR-A.pos')
tI = -1j * t
exportToGmsh(tI, 'MTF-dint-i', 'MTF_DintI-A.pos')

t =  MTFintDirichletTraceB
exportToGmsh(t, 'MTF-dint-r', 'MTF_DintR-B.pos')
tI = -1j * t
exportToGmsh(tI, 'MTF-dint-i', 'MTF_DintI-B.pos')


# t =  MTFscattDirichletTrace
# exportToGmsh(t, 'MTF-dsca-r', 'MTF_DscaR.pos')
# tI = -1j * t
# exportToGmsh(tI, 'MTF-dsca-i', 'MTF_DscaI.pos')

# t =  MTFscattNeumannTrace
# exportToGmsh(t, 'MTF-nsca-r', 'MTF_NscaR.pos')
# tI = -1j * t
# exportToGmsh(tI, 'MTF-nsca-i', 'MTF_NscaI.pos')

########################################################
########################################################

print('\n post\n')

normDextA = MTFextDirichletTraceA.L2Norm()
normDextB = MTFextDirichletTraceB.L2Norm()
normNextA = MTFextNeumannTraceA.L2Norm()
normNextB = MTFextNeumannTraceB.L2Norm()
normDintA = MTFintDirichletTraceA.L2Norm()
normDintB = MTFintDirichletTraceB.L2Norm()
normNintA = MTFintNeumannTraceA.L2Norm()
normNintB = MTFintNeumannTraceB.L2Norm()

normDincA = incDirichletTraceA.L2Norm()
normNincA = incNeumannTraceA.L2Norm()

normDincB = incDirichletTraceB.L2Norm()
normNincB = incNeumannTraceB.L2Norm()


jDA = (MTFextDirichletTraceA - MTFintDirichletTraceA -incDirichletTraceA).L2Norm()
jNA = (-MTFextNeumannTraceA - MTFintNeumannTraceA -incNeumannTraceA).L2Norm()

jDB = (MTFextDirichletTraceB - MTFintDirichletTraceB -incDirichletTraceB).L2Norm()
jNB = (-MTFextNeumannTraceB - MTFintNeumannTraceB -incNeumannTraceB).L2Norm()


print('')
print('# L^2')
# print("# norm  {:^15} {:^15} {:^15} {:^15} {:^15} {:^15}".format(
#     'DextA', 'DintA',
#     'DincA',
#     'NextA', 'NintA',
#     'NincA'
# ))
print("# norm  {:^15} {:^15} {:^15} {:^15} {:^15} {:^15}".format(
    'DextA/B', 'DintA/B',
    'DincA/B',
    'NextA/B', 'NintA/B',
    'NincA/B'
))
print("# MTF   {:^15E} {:^15E} {:^15E} {:^15E} {:^15E} {:^15E}".format(
    normDextA, normDintA,
    normDincA,
    normNextA, normNintA,
    normNincA
))
#
# print("# norm  {:^15} {:^15} {:^15} {:^15} {:^15} {:^15}".format(
#     'DextB', 'DintB',
#     'DincB',
#     'NextB', 'NintB',
#     'NincB'
# ))
print("# MTF   {:^15E} {:^15E} {:^15E} {:^15E} {:^15E} {:^15E}".format(
    normDextB, normDintB,
    normDincB,
    normNextB, normNintB,
    normNincB
))
##
print('')
print("# error  {:^15} {:^15} {:^15} {:^15}".format(
    'jumpDA', 'jumpNA',
    'jumpDB', 'jumpNB'
    ))
print("# MTF    {:^15E} {:^15E} {:^15E} {:^15E}".format(
    jDA, jNA, jDB, jNB
))
print('')


sys.exit('bye.')

M = MTF.weakForm().asMatrix()

pdA = incDirichletTraceA.projections()
pdB = incDirichletTraceB.projections()
pnA = incNeumannTraceA.projections()
pnB = incNeumannTraceB.projections()

cdA = incDirichletTraceA.coefficients()
cdB = incDirichletTraceB.coefficients()
cnA = incNeumannTraceA.coefficients()
cnB = incNeumannTraceB.coefficients()

v = np.array([], dtype=complex)
v = np.concatenate((v, 0*cdA))
v = np.concatenate((v, 0*cdB))
v = np.concatenate((v, 0*cnA))
v = np.concatenate((v, 0*cnB))

v = np.concatenate((v, -cdA))
v = np.concatenate((v, -cnA))
v = np.concatenate((v, -cdB))
v = np.concatenate((v, -cnB))

b = M.dot(v)

beg, end = 0, len(pdA)
ppdA, beg, end = 2.0 * b[beg:end], end, end+len(pdB)
ppdB, beg, end = 2.0 * b[beg:end], end, end+len(pnA)
ppnA, beg, end = -2.0 * b[beg:end], end, end+len(pnB)
ppnB, beg, end = -2.0 * b[beg:end], end, end+len(pnB)

ppdA_1, beg, end = -2.0 * b[beg:end], end, end+len(pdA)
ppnA_1, beg, end = -2.0 * b[beg:end], end, end+len(pnA)

ppdB_2, beg, end = -2.0 * b[beg:end], end, end+len(pdB)
ppnB_2, beg, end = -2.0 * b[beg:end], end, end+len(pnB)
