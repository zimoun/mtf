#!/usr/bin/env python

# This script calculates the field generated by a z-polarised, x-propagating
# plane wave incident on a dielectric sphere.

# Help Python find the bempp module
import sys
sys.path.append("..")

from bempp.lib import *
import numpy as np
import scipy.linalg as sp
from matplotlib import pylab as pl

# Physical Parameters

epsInt = 1.2**2
epsExt = 1
muInt = 1
muExt = 1
wavelengthVacuum = 1.5
kVacuum = 2 * np.pi / wavelengthVacuum
kExt = kVacuum * np.sqrt(epsExt * muExt)
kInt = kVacuum * np.sqrt(epsInt * muInt)
rho = (kInt * muExt) / (kExt * muInt)

# 
print('==== Load mesh')

grid = createGridFactory().importGmshGrid(
    "triangular", "./sphere-simple.msh")

# Boundary conditions

def evalIncDirichletTrace(point, normal):
    field = evalIncField(point)
    result = np.cross(field, normal, axis=0)
    return result

def evalIncNeumannTrace(point, normal):
    x, y, z = point
    curl = np.array([x * 0., -1j * kExt * np.exp(1j * kExt * x), x * 0.])
    result = np.cross(curl / (1j * kExt), normal, axis=0)
    return result

def evalIncField(point):
    x, y, z = point
    field = np.array([x * 0., y * 0., np.exp(1j * kExt * x)])
    return field


# 
print('==== Create quadrature strategy')

accuracyOptions = createAccuracyOptions()
# Increase by 2 the order of quadrature rule used to approximate
# integrals of regular functions on pairs on elements
accuracyOptions.doubleRegular.setRelativeQuadratureOrder(2)
# Increase by 2 the order of quadrature rule used to approximate
# integrals of regular functions on single elements
accuracyOptions.singleRegular.setRelativeQuadratureOrder(2)
quadStrategy = createNumericalQuadratureStrategy(
    "float64", "complex128", accuracyOptions)

# 
print('==== Create assembly context')

assemblyOptions = createAssemblyOptions()
#assemblyOptions.switchToAcaMode(createAcaOptions())
context = createContext(quadStrategy, assemblyOptions)

# 
print('==== Initialize functional spaces')

space = createRaviartThomas0VectorSpace(context, grid)

# 
print('==== Define the grid functions for the traces of incident field')

incDirichletTrace = createGridFunction(
    context, space, space, evalIncDirichletTrace,
    surfaceNormalDependent=True)
incNeumannTrace = createGridFunction(
    context, space, space, evalIncNeumannTrace,
    surfaceNormalDependent=True)

#
print('==== Construct elementary operators')

S0 = createMaxwell3dSingleLayerBoundaryOperator(
    context, space, space, space, kExt, "SLP_ext")
C0 = createMaxwell3dDoubleLayerBoundaryOperator(
    context, space, space, space, kExt, "DLP_ext")
S1 = createMaxwell3dSingleLayerBoundaryOperator(
    context, space, space, space, kInt, "SLP_int")
C1 = createMaxwell3dDoubleLayerBoundaryOperator(
    context, space, space, space, kInt, "DLP_int")
Id = createMaxwell3dIdentityOperator(
    context, space, space, space, "Id")

# 
print('==== Form the STF matrix')

# STF00 = -(S0 + rho * S1)
# STF01 = STF10 = C0 + C1
# STF11 = S0 + (1. / rho) * S1

# STF = createBlockedBoundaryOperator(
#     context, [[STF00, STF01], 
#               [STF10, STF11]])

# rhsP = [Id * incNeumannTrace, Id * incDirichletTrace]



a = C0 + C1
b = S0 + (1/rho)*S1
c = S0 + rho * S1

STF0 = createBlockedBoundaryOperator(
    context, [[a, b], 
              [-c, a]])

rhsSTF0 = [  ( (-0.5*Id+C1) * incDirichletTrace + (1/rho)*S1 * incNeumannTrace ),
              ( (-0.5*Id+C1) * incNeumannTrace - rho*S1 * incDirichletTrace ) ]


d = S0+S1
ar, dr = rho*a, rho*d

STF1 = createBlockedBoundaryOperator(
    context, [[-a, -dr], 
              [d, -ar]])

rhsSTF1 = [ - ( (0.5*Id+C0) * incDirichletTrace + S0 * incNeumannTrace ),
             - ( (0.5*Id+C0) * incNeumannTrace - S0 * incDirichletTrace ) ]


# 
print('==== Form the MTF matrix')

Half= -0.5 * Id
Rho, iRho = -0.5*rho *Id, -0.5*(1/rho) * Id
ZZ =  0.0*Id

MTF = createBlockedBoundaryOperator(
    context, 
    [ [C0, S0, Half, ZZ], 
      [-S0, C0, ZZ, Rho],
      [-Half, ZZ, C1, S1], 
      [ZZ, -iRho, -S1, C1] ]
    )

rhsMTF = [-0.5*Id * incDirichletTrace,
          -0.5*Id * incNeumannTrace, 
          0.5*Id * incDirichletTrace,
          0.5*(1/rho)*Id * incNeumannTrace]


# 
print('==== Initialize the solver')

# precTol = 1e-2
# invLhsOp00 = acaOperatorApproximateLuInverse(
#     lhsOp00.weakForm().asDiscreteAcaBoundaryOperator(), precTol)
# invLhsOp11 = acaOperatorApproximateLuInverse(
#     lhsOp11.weakForm().asDiscreteAcaBoundaryOperator(), precTol)
# prec = discreteBlockDiagonalPreconditioner([invLhsOp00, invLhsOp11])

# solver = createDefaultIterativeSolver(lhsOp)
# solver.initializeSolver(defaultGmresParameterList(1e-8), prec)

STF0solver = createDefaultDirectSolver(STF0)
STF1solver = createDefaultDirectSolver(STF1)
MTFsolver = createDefaultDirectSolver(MTF)

# 
print('==== Solve STF')

#Psolution = STFsolver.solve(rhsP)
STF0solution = STF0solver.solve(rhsSTF0)
print STF0solution.solverMessage()
print('----------------------')
STF1solution = STF1solver.solve(rhsSTF1)
print STF1solution.solverMessage()

print('==== Solve MTF')

MTFsolution = MTFsolver.solve(rhsMTF)
print MTFsolution.solverMessage()

print('==== end solved... extraction...')


# STF extraction
# Extract the solution components in the form of grid functions


STF0extDirichletTrace = STF0solution.gridFunction(0)
STF0extNeumannTrace = STF0solution.gridFunction(1)

exportToGmsh(STF0extDirichletTrace, 'STF0-dext', 'STF0_Dext.pos')
exportToGmsh(STF0extNeumannTrace, 'STF0-next', 'STF0_Next.pos')

STF0intDirichletTrace = -STF0extDirichletTrace + incDirichletTrace
STF0intNeumannTrace = -(1/rho) * STF0extNeumannTrace + (1/rho) * incNeumannTrace

exportToGmsh(STF0intDirichletTrace, 'STF0-dint', 'STF0_Dint.pos')
exportToGmsh(STF0intNeumannTrace, 'STF0-nint', 'STF0_Nint.pos')

###

STF1intDirichletTrace = STF1solution.gridFunction(0)
STF1intNeumannTrace = STF1solution.gridFunction(1)

exportToGmsh(STF1intDirichletTrace, 'STF1-dint', 'STF1_Dint.pos')
exportToGmsh(STF1intNeumannTrace, 'STF1-nint', 'STF1_Nint.pos')

STF1extDirichletTrace = -STF1intDirichletTrace + incDirichletTrace
STF1extNeumannTrace = -rho * STF1intNeumannTrace + incNeumannTrace

exportToGmsh(STF1extDirichletTrace, 'STF1-dext', 'STF1_Dext.pos')
exportToGmsh(STF1extNeumannTrace, 'STF1-next', 'STF1_Next.pos')


########################################################
########################################################

# MTF extraction
# Extract the solution components in the form of grid functions

MTFextDirichletTrace = MTFsolution.gridFunction(0)
MTFextNeumannTrace = MTFsolution.gridFunction(1)

exportToGmsh(MTFextDirichletTrace, 'MTF-dext', 'MTF_Dext.pos')
exportToGmsh(MTFextNeumannTrace, 'MTF-next', 'MTF_Next.pos')

MTFscattDirichletTrace = -(MTFextDirichletTrace - incDirichletTrace)
MTFscattNeumannTrace = -(1/rho)*(MTFextNeumannTrace - incNeumannTrace)

exportToGmsh(MTFscattDirichletTrace, 'MTF-dsca', 'MTF_Dsca.pos')
exportToGmsh(MTFscattNeumannTrace, 'MTF-nsca', 'MTF_Nsca.pos')

MTFintDirichletTrace = MTFsolution.gridFunction(2)
MTFintNeumannTrace = MTFsolution.gridFunction(3)

exportToGmsh(MTFintDirichletTrace, 'MTF-dint', 'MTF_Dint.pos')
exportToGmsh(MTFintNeumannTrace, 'MTF-nint', 'MTF_Nint.pos')

########################################################
########################################################

print('\n post\n')

Half= -0.0 * Id
Rho, iRho = -0.0*rho *Id, -0.0*(1/rho) * Id
ZZ =  0.0*Id

AAblock = createBlockedBoundaryOperator(
    context, 
    [ [C0, S0, Half, ZZ], 
      [-S0, C0, ZZ, Rho],
      [-Half, ZZ, C1, S1], 
      [ZZ, -iRho, -S1, C1] ]
    )
IIblock = createBlockedBoundaryOperator(
    context, 
    [ [Id, ZZ, Half, ZZ], 
      [ZZ, Id, ZZ, Rho],
      [-Half, ZZ, Id, ZZ], 
      [ZZ, -iRho, ZZ, Id] ]
    )


M = MTF.weakForm().asMatrix()
A = AAblock.weakForm().asMatrix()
II = IIblock.weakForm().asMatrix()
b = np.array([])
for v in rhsMTF:
    b = np.concatenate((b, v.projections()))
y = np.array([])
for ii in xrange(len(rhsMTF)):
    gf = MTFsolution.gridFunction(ii)
    y = np.concatenate((y, gf.coefficients()))

print('solve MTF with Numpy... be patient [take time]')
x = np.linalg.solve(M, b)

print(np.allclose(x, y)),
print(np.allclose(M.dot(x), b)),
print(np.allclose(M.dot(y), b)),

print('\nCalderon error: absolute ; relative')
e = 2*A.dot(x) - II.dot(x)
print(np.linalg.norm(e)),
print('\t ; \t'),
print(np.linalg.norm(e) / np.linalg.norm(x)),
