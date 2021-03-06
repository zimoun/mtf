#!/usr/bin/env python

# This script calculates the field generated by a z-polarised, x-propagating
# plane wave incident on a dielectric sphere.

# Help Python find the bempp module
import sys
sys.path.append("..")

from bempp.lib import *
import numpy as np

# Physical Parameters

epsInt = 1.5**2
epsExt = 1
muInt = 1
muExt = 1
wavelengthVacuum = 1.8
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
    x, y, z = point
    return np.exp( 1j * kExt * x)

def evalIncNeumannTrace(point, normal):
    x, y, z = point
    nx, ny, nz = normal
    return 1j * nx * kExt * np.exp( 1j * kExt * x)


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
print('==== Initialize functional spaces')


space = createPiecewiseLinearContinuousScalarSpace(context, grid)


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


V0 = createHelmholtz3dSingleLayerBoundaryOperator(
    context, space, space, space, kExt, "V")
K0 = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, space, space, space, kExt, "K")
K0a = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, space, space, space, kExt, "K")
W0 = createHelmholtz3dHypersingularBoundaryOperator(
    context, space, space, space, kExt, "W")
V1 = createHelmholtz3dSingleLayerBoundaryOperator(
    context, space, space, space, kInt, "V1")
K1 = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, space, space, space, kInt, "K1")
K1a = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, space, space, space, kInt, "K1")
W1 = createHelmholtz3dHypersingularBoundaryOperator(
    context, space, space, space, kInt, "W1")

Id = createIdentityOperator(
    context, space, space, space, "Id")

# 
print('==== Form the STF matrix')

STF00 = -K0+K1
STF01 = V0+V1
STF10 = W0+W1
STF11 = K0a-K1a

STF = createBlockedBoundaryOperator(
    context, [[STF00, STF01], [STF10, STF11]])

rhsP = [Id * incNeumannTrace, Id * incDirichletTrace]
rhsSTF = [ (0.5*Id + K1) * incDirichletTrace - V1 * incNeumannTrace,
           +W1 * incDirichletTrace + (-0.5*Id+K1a) * incNeumannTrace]


# 
print('==== Form the MTF matrix')

Half= 0.5 * Id
ZZ =  0.0*Id

MTF = createBlockedBoundaryOperator(
    context, 
    [ [-K0, V0, -Half, ZZ], 
      [W0, K0a, ZZ, Half],
      [-Half, ZZ, -K1, V1], 
      [ZZ, Half, W1, K1a] ]
    )

rhsMTF = [0.5*Id * incDirichletTrace,
          -0.5*Id * incNeumannTrace, 
          -0.5*Id * incDirichletTrace,
          -0.5*Id * incNeumannTrace]


# 
print('==== Initialize the solver')

STFsolver = createDefaultDirectSolver(STF)
MTFsolver = createDefaultDirectSolver(MTF)

# 
print('==== Solve STF')

STFsolution = STFsolver.solve(rhsSTF)
print STFsolution.solverMessage()

print('==== Solve MTF')

MTFsolution = MTFsolver.solve(rhsMTF)
print MTFsolution.solverMessage()

print('==== end solved... extraction...')


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

# # Prepare to evaluate the solution on an annulus outside the sphere

# # Create the potential operators entering the Green's representation formula

# slPotInt = createMaxwell3dSingleLayerPotentialOperator(context, kInt)
# dlPotInt = createMaxwell3dDoubleLayerPotentialOperator(context, kInt)
# slPotExt = createMaxwell3dSingleLayerPotentialOperator(context, kExt)
# dlPotExt = createMaxwell3dDoubleLayerPotentialOperator(context, kExt)

# # Create a grid of points

# nPointsX = 201
# nPointsY = nPointsX
# x, y, z = np.mgrid[-3:3:nPointsX*1j, -3:3:nPointsY*1j, 0:0:1j]
# points = np.vstack((x.ravel(), y.ravel(), z.ravel()))

# # Split the points into those located inside and outside the scatterer

# inside = areInside(grid, points)
# outside = np.logical_not(inside)

# # Use appropriate Green's representation formulas to evaluate the total field
# # inside and outside the scatterer

# evalOptions = createEvaluationOptions()
# # Evaluate field (all three components) at exterior points
# valsExt = (- slPotExt.evaluateAtPoints(STFscattNeumannTrace,
#                                        points[:,outside], evalOptions)
#            - dlPotExt.evaluateAtPoints(STFscattDirichletTrace,
#                                        points[:,outside], evalOptions)
#            + evalIncField(points[:,outside]))

# # Evaluate field (all three components) at interior points
# valsInt = (  slPotInt.evaluateAtPoints(STFintNeumannTrace,
#                                        points[:,inside], evalOptions)
#            + dlPotInt.evaluateAtPoints(STFintDirichletTrace,
#                                        points[:,inside], evalOptions))

# # Combine the results obtained for points inside and outside the scatterer
# # in a single array

# vals_STF = np.empty((3, nPointsX * nPointsY), dtype=complex)
# for dim in range(3): # iterate over coordinates
#     np.place(vals_STF[dim], outside, valsExt[dim].ravel())
#     np.place(vals_STF[dim], inside, valsInt[dim].ravel())

# # Display the field plot

# from bempp import visualization2 as vis
# tvtkVals = vis.tvtkStructuredGridData(points, vals_STF, (nPointsX, nPointsY))
# tvtkGrid = vis.tvtkGrid(grid)
# #vis.plotVectorData(tvtkGrids=tvtkGrid, tvtkStructuredGridData=tvtkVals)

# # Export the results into a VTK file

# from tvtk.api import write_data
# write_data(tvtkVals, "stf.vts")

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


# Prepare to evaluate the solution on an annulus outside the sphere

# # Create the potential operators entering the Green's representation formula

# slPotInt = createMaxwell3dSingleLayerPotentialOperator(context, kInt)
# dlPotInt = createMaxwell3dDoubleLayerPotentialOperator(context, kInt)
# slPotExt = createMaxwell3dSingleLayerPotentialOperator(context, kExt)
# dlPotExt = createMaxwell3dDoubleLayerPotentialOperator(context, kExt)

# # Create a grid of points

# nPointsX = 201
# nPointsY = nPointsX
# x, y, z = np.mgrid[-3:3:nPointsX*1j, -3:3:nPointsY*1j, 0:0:1j]
# points = np.vstack((x.ravel(), y.ravel(), z.ravel()))

# # Split the points into those located inside and outside the scatterer

# inside = areInside(grid, points)
# outside = np.logical_not(inside)

# # Use appropriate Green's representation formulas to evaluate the total field
# # inside and outside the scatterer

# evalOptions = createEvaluationOptions()
# # Evaluate field (all three components) at exterior points
# valsExt = (- slPotExt.evaluateAtPoints(MTFscattNeumannTrace,
#                                        points[:,outside], evalOptions)
#            - dlPotExt.evaluateAtPoints(MTFscattDirichletTrace,
#                                        points[:,outside], evalOptions)
#            + evalIncField(points[:,outside]))

# # Evaluate field (all three components) at interior points
# valsInt = (  slPotInt.evaluateAtPoints(MTFintNeumannTrace,
#                                        points[:,inside], evalOptions)
#            + dlPotInt.evaluateAtPoints(MTFintDirichletTrace,
#                                        points[:,inside], evalOptions))

# # Combine the results obtained for points inside and outside the scatterer
# # in a single array

# vals_MTF = np.empty((3, nPointsX * nPointsY), dtype=complex)
# for dim in range(3): # iterate over coordinates
#     np.place(vals_MTF[dim], outside, valsExt[dim].ravel())
#     np.place(vals_MTF[dim], inside, valsInt[dim].ravel())

# # Display the field plot

# from bempp import visualization2 as vis
# tvtkVals = vis.tvtkStructuredGridData(points, vals_MTF, (nPointsX, nPointsY))
# tvtkGrid = vis.tvtkGrid(grid)
# #vis.plotVectorData(tvtkGrids=tvtkGrid, tvtkStructuredGridData=tvtkVals)

# # Export the results into a VTK file

# from tvtk.api import write_data
# write_data(tvtkVals, "mtf.vts")


A0op = createBlockedBoundaryOperator(
    context, [[-K0 , V0], [W0, K0a]])

Xop = createBlockedBoundaryOperator(
    context, [[Id , ZZ], [ZZ, -Id]])

A0 = A0op.weakForm().asMatrix()
lod = STFsolution.gridFunction(0).coefficients()
lon = STFsolution.gridFunction(1).coefficients()
lo = np.concatenate((lod, lon))

STFextDirichletVec = createGridFunction(context, space, coefficients=lod )
exportToGmsh(STFextDirichletVec, 'STF-dextV', 'STF_DextV.pos')


loo = 2*A0.dot(lo)
N=lod.shape[0]
myI = np.eye(N)

A0STFextDirichletVec = createGridFunction(context, space, coefficients=loo[0:N])
exportToGmsh(A0STFextDirichletVec, 'A0STF-dextV', 'A0STF_DextV.pos')
