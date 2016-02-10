#!/usr/bin/env python

# Help Python find the bempp module
import sys
sys.path.append("..")

from bempp.lib import *
import numpy as np

# Physical Parameters

discretization = 'P1'

wavelengthVacuum = 10.5
kVacuum = 2 * np.pi / wavelengthVacuum

kVacuum = 1

eps = [1, 1, 1]
mu = [1, 1, 1]
k = []
for e, m in zip(eps, mu):
    k.append( kVacuum * np.sqrt( e*m ) )

# source: incident plane wave
def evalIncDirichletTrace(point, normal):
    x, y, z = point
    return -np.exp( 1j * k[0] * x)
#
def evalIncNeumannTrace(point, normal):
    x, y, z = point
    nx, ny, nz = normal
    return -1j * nx * k[0] * np.exp( 1j * k[0] * x)
#

##
print('==== Create context')
#
accuracyOptions = createAccuracyOptions()
accuracyOptions.doubleRegular.setRelativeQuadratureOrder(2)
accuracyOptions.doubleSingular.setRelativeQuadratureOrder(1)
quadStrategy = createNumericalQuadratureStrategy(
    "float64", "complex128", accuracyOptions)
#
assemblyOptions = createAssemblyOptions()
context = createContext(quadStrategy, assemblyOptions)
#
##


####

## 
print('==== Load mesh and create closed boundary domain')
#
grid = createGridFactory().importGmshGrid(
    "triangular", "./full-interfaces.msh")
#
interfaceA = (GridSegment.closedDomain(grid, 110))
interfaceB = (GridSegment.closedDomain(grid, 21))
interfaceC = (GridSegment.closedDomain(grid, 210))
#
# bnd0 = interface110.union_(interface210) # wrong orientation
# bnd1 = interface110.union_(interface21)  # one bad 
# bnd2 = interface210.union_(interface21)  # all right
#
##

##
print('==== Initialize functional spaces')
#
P0 = createPiecewiseConstantScalarSpace(context, grid)
P1 = createPiecewiseLinearContinuousScalarSpace(context, grid)
##

if discretization == 'P0P1':
    P0_A = createPiecewiseConstantScalarSpace(context, grid, interfaceA)
    P0_B = createPiecewiseConstantScalarSpace(context, grid, interfaceB)
    P0_C = createPiecewiseConstantScalarSpace(context, grid, interfaceC)
#                                            
    P1_A = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceA,
                                                      strictlyOnSegment=True)
    P1_B = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceB, 
                                                      strictlyOnSegment=True)
    P1_C = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceC, 
                                                      strictlyOnSegment=True)

elif discretization == 'P0':                               
    P0_A = createPiecewiseConstantScalarSpace(context, grid, interfaceA)
    P0_B = createPiecewiseConstantScalarSpace(context, grid, interfaceB)
    P0_C = createPiecewiseConstantScalarSpace(context, grid, interfaceC)
#
    P1_A = createPiecewiseConstantScalarSpace(context, grid, interfaceA)
    P1_B = createPiecewiseConstantScalarSpace(context, grid, interfaceB)
    P1_C = createPiecewiseConstantScalarSpace(context, grid, interfaceC)

elif discretization == 'P1':
    P0_A = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceA,
                                                      strictlyOnSegment=True)
    P0_B = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceB, 
                                                      strictlyOnSegment=True)
    P0_C = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceC, 
                                                      strictlyOnSegment=True)
#
    P1_A = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceA,
                                                      strictlyOnSegment=True)
    P1_B = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceB, 
                                                      strictlyOnSegment=True)
    P1_C = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceC, 
                                                      strictlyOnSegment=True)
else:
    discretization = 'P0P1'
    P0_A = createPiecewiseConstantScalarSpace(context, grid, interfaceA)
    P0_B = createPiecewiseConstantScalarSpace(context, grid, interfaceB)
    P0_C = createPiecewiseConstantScalarSpace(context, grid, interfaceC)
#                                            
    P1_A = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceA,
                                                      strictlyOnSegment=True)
    P1_B = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceB, 
                                                      strictlyOnSegment=True)
    P1_C = createPiecewiseLinearContinuousScalarSpace(context, grid, interfaceC, 
                                                      strictlyOnSegment=True)

##

##
print('==== Project the source onto the exterior subgrid')

# incDirichletTrace0 = createGridFunction(
#     context, P1_0, P1_0, evalIncDirichletTrace,
#     surfaceNormalDependent=True)
# incNeumannTrace0 = createGridFunction(
#     context, P0_0, P0_0, evalIncNeumannTrace,
#     surfaceNormalDependent=True)
#
#
print('==== Declare all the useful operators : per interface')
#

############################################################
## create A0: bnd = -A, -C
############################################################
tK0_A_A = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_A, P1_A, P0_A, k[0], "K0 aa")
tK0_A_C = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_C, P1_A, P0_A, k[0], "K0 ac")
tK0_C_A = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_A, P1_C, P0_C, k[0], "K0 ca")
tK0_C_C = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_C, P1_C, P0_C, k[0], "K0 cc")
# correction bad normal orientation
K0_A_A, K0_A_C = -tK0_A_A, -tK0_A_C
K0_C_A, K0_C_C = -tK0_C_A, -tK0_C_C

tV0_A_A = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P0_A, P1_A, P0_A, k[0], "V0 aa")
tV0_A_C = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P0_C, P1_A, P0_A, k[0], "V0 ac")
tV0_C_A = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P0_A, P1_C, P0_C, k[0], "V0 ca")
tV0_C_C = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P0_C, P1_C, P0_C, k[0], "V0 cc")
# do not depend on normal
V0_A_A, V0_A_C = tV0_A_A, tV0_A_C
V0_C_A, V0_C_C = tV0_C_A, tV0_C_C

tW0_A_A = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_A, P0_A, P1_A, k[0], "W0 aa")
tW0_A_C = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_C, P0_A, P1_A, k[0], "W0 ac")
tW0_C_A = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_A, P0_C, P1_C, k[0], "W0 ca")
tW0_C_C = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_C, P0_C, P1_C, k[0], "W0 cc")
# double wrong normal orientation => compensated
W0_A_A, W0_A_C = tW0_A_A, tW0_A_C
W0_C_A, W0_C_C = tW0_C_A, tW0_C_C

tQ0_A_A = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P0_A, P0_A, P1_A, k[0], "Q0 aa")
tQ0_A_C = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P0_C, P0_A, P1_A, k[0], "Q0 ac")
tQ0_C_A = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P0_A, P0_C, P1_C, k[0], "Q0 ca")
tQ0_C_C = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P0_C, P0_C, P1_C, k[0], "Q0 cc")
# correction bad normal orientation
Q0_A_A, Q0_A_C = -tQ0_A_A, -tQ0_A_C
Q0_C_A, Q0_C_C = -tQ0_C_A, -tQ0_C_C


############################################################
## create X01 
############################################################
kX01a = createIdentityOperator(
    context, P1_A, P1_A, P0_A, "X01 aa")
kX01b = createNullOperator(
    context, P1_B, P1_A, P0_A, "X01 ab")
kX01c = createNullOperator(
    context, P1_A, P1_C, P0_C, "X01 ca")
kX01d = createNullOperator(
    context, P1_B, P1_C, P0_C, "X01 cb")
# do not depend on normal
kX01_A_A, kX01_A_B = kX01a, kX01b
kX01_C_A, kX01_C_B = kX01c, kX01d

vX01a = createNullOperator(
    context, P0_A, P1_A, P0_A, "X01 aa")
vX01b = createNullOperator(
    context, P0_B, P1_A, P0_A, "X01 ab")
vX01c = createNullOperator(
    context, P0_A, P1_C, P0_C, "X01 ca")
vX01d = createNullOperator(
    context, P0_B, P1_C, P0_C, "X01 cb")
# do not depend on normal
vX01_A_A, vX01_A_B = vX01a, vX01b
vX01_C_A, vX01_C_B = vX01c, vX01d

wX01a = createNullOperator(
    context, P1_A, P0_A, P1_A, "X01 aa")
wX01b = createNullOperator(
    context, P1_B, P0_A, P1_A, "X01 ab")
wX01c = createNullOperator(
    context, P1_A, P0_C, P1_C, "X01 ca")
wX01d = createNullOperator(
    context, P1_B, P0_C, P1_C, "X01 cb")
# do not depend on normal
wX01_A_A, wX01_A_B = wX01a, wX01b
wX01_C_A, wX01_C_B = wX01c, wX01d

qX01a = createIdentityOperator(
    context, P0_A, P0_A, P1_A, "X01 aa")
qX01b = createNullOperator(
    context, P0_B, P0_A, P1_A, "X01 ab")
qX01c = createNullOperator(
    context, P0_A, P0_C, P1_C, "X01 ca")
qX01d = createNullOperator(
    context, P0_B, P0_C, P1_C, "X01 cb")
# do not depend on normal
qX01_A_A, qX01_A_B = -qX01a, -qX01b
qX01_C_A, qX01_C_B = -qX01c, -qX01d
# minus comes from Neumann sign X=[Id,0;0,-Id]

############################################################
## create X02 
############################################################
kX02a = createNullOperator(
    context, P1_C, P1_A, P0_A, "X02 ca")
kX02b = createNullOperator(
    context, P1_B, P1_A, P0_A, "X02 ab")
kX02c = createIdentityOperator(
    context, P1_C, P1_C, P0_C, "X02 cc")
kX02d = createNullOperator(
    context, P1_B, P1_C, P0_C, "X02 cb")
# do not depend on normal
kX02_A_C, kX02_A_B = kX02a, kX02b
kX02_C_C, kX02_C_B = kX02c, kX02d

vX02a = createNullOperator(
    context, P0_C, P1_A, P0_A, "X02 ca")
vX02b = createNullOperator(           
    context, P0_B, P1_A, P0_A, "X02 ab")
vX02c = createNullOperator(           
    context, P0_C, P1_C, P0_C, "X02 cc")
vX02d = createNullOperator(           
    context, P0_B, P1_C, P0_C, "X02 cb")
# do not depend on normal
vX02_A_C, vX02_A_B = vX02a, vX02b
vX02_C_C, vX02_C_B = vX02c, vX02d

wX02a = createNullOperator(
    context, P1_C, P0_A, P1_A, "X02 ca")
wX02b = createNullOperator(          
    context, P1_B, P0_A, P1_A, "X02 ab")
wX02c = createNullOperator(          
    context, P1_C, P0_C, P1_C, "X02 cc")
wX02d = createNullOperator(          
    context, P1_B, P0_C, P1_C, "X02 cb")
# do not depend on normal
wX02_A_C, wX02_A_B = wX02a, wX02b
wX02_C_C, wX02_C_B = wX02c, wX02d

qX02a = createNullOperator(
    context, P0_C, P0_A, P1_A, "X02 ca")
qX02b = createNullOperator(          
    context, P0_B, P0_A, P1_A, "X02 ab")
qX02c = createIdentityOperator(          
    context, P0_C, P0_C, P1_C, "X02 cc")
qX02d = createNullOperator(          
    context, P0_B, P0_C, P1_C, "X02 cb")
# do not depend on normal
qX02_A_C, qX02_A_B = -qX02a, -qX02b
qX02_C_C, qX02_C_B = -qX02c, -qX02d

############################################################
# create missing useful operators bnd0: -A, -C
############################################################
tK0_A_A = createIdentityOperator(
    context, P1_A, P1_A, P0_A, "iK0 aa")
tK0_A_C = createNullOperator(
    context, P1_C, P1_A, P0_A, "oK0 ac")
tK0_C_A = createNullOperator(
    context, P1_A, P1_C, P0_C, "oK0 ca")
tK0_C_C = createIdentityOperator(
    context, P1_C, P1_C, P0_C, "iK0 cc")
# correction bad normal orientation
iK0_A_A, oK0_A_C = tK0_A_A, tK0_A_C
oK0_C_A, iK0_C_C = tK0_C_A, tK0_C_C

tV0_A_A = createNullOperator(
    context, P0_A, P1_A, P0_A, "oV0 aa")
tV0_A_C = createNullOperator(
    context, P0_C, P1_A, P0_A, "oV0 ac")
tV0_C_A = createNullOperator(
    context, P0_A, P1_C, P0_C, "oV0 ca")
tV0_C_C = createNullOperator(
    context, P0_C, P1_C, P0_C, "oV0 cc")
# do not depend on normal
oV0_A_A, oV0_A_C = tV0_A_A, tV0_A_C
oV0_C_A, oV0_C_C = tV0_C_A, tV0_C_C

tW0_A_A = createNullOperator(
    context, P1_A, P0_A, P1_A, "oW0 aa")
tW0_A_C = createNullOperator(
    context, P1_C, P0_A, P1_A, "oW0 ac")
tW0_C_A = createNullOperator(
    context, P1_A, P0_C, P1_C, "oW0 ca")
tW0_C_C = createNullOperator(
    context, P1_C, P0_C, P1_C, "oW0 cc")
# double wrong normal orientation => compensated
oW0_A_A, oW0_A_C = tW0_A_A, tW0_A_C
oW0_C_A, oW0_C_C = tW0_C_A, tW0_C_C

tQ0_A_A = createIdentityOperator(
    context, P0_A, P0_A, P1_A, "iQ0 aa")
tQ0_A_C = createNullOperator(
    context, P0_C, P0_A, P1_A, "oQ0 ac")
tQ0_C_A = createNullOperator(
    context, P0_A, P0_C, P1_C, "oQ0 ca")
tQ0_C_C = createIdentityOperator(
    context, P0_C, P0_C, P1_C, "iQ0 cc")
# correction bad normal orientation
iQ0_A_A, oQ0_A_C = tQ0_A_A, tQ0_A_C
oQ0_C_A, iQ0_C_C = tQ0_C_A, tQ0_C_C
############################################################
############################################################



############################################################
## create A1: bnd = A, -B
############################################################
tK1_A_A = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_A, P1_A, P0_A, k[1], "K0 aa")
tK1_A_B = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_B, P1_A, P0_A, k[1], "K0 ab")
tK1_B_A = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_A, P1_B, P0_B, k[1], "K0 ba")
tK1_B_B = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_B, P1_B, P0_B, k[1], "K0 bb")
# correction bad normal orientation
K1_A_A, K1_A_B = tK1_A_A, -tK1_A_B
K1_B_A, K1_B_B = tK1_B_A, -tK1_B_B

tV1_A_A = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P0_A, P1_A, P0_A, k[1], "V0 aa")
tV1_A_B = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P0_B, P1_A, P0_A, k[1], "V0 ab")
tV1_B_A = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P0_A, P1_B, P0_B, k[1], "V0 ba")
tV1_B_B = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P0_B, P1_B, P0_B, k[1], "V0 bb")
# do not depend on normal
V1_A_A, V1_A_B = tV1_A_A, tV1_A_B
V1_B_A, V1_B_B = tV1_B_A, tV1_B_B

tW1_A_A = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_A, P0_A, P1_A, k[1], "W0 aa")
tW1_A_B = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_B, P0_A, P1_A, k[1], "W0 ab")
tW1_B_A = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_A, P0_B, P1_B, k[1], "W0 ba")
tW1_B_B = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_B, P0_B, P1_B, k[1], "W0 bb")
# double wrong normal orientation => compensated
W1_A_A, W1_A_B = tW1_A_A, tW1_A_B
W1_B_A, W1_B_B = -tW1_B_A, -tW1_B_B

tQ1_A_A = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P0_A, P0_A, P1_A, k[1], "Q0 aa")
tQ1_A_B = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P0_B, P0_A, P1_A, k[1], "Q0 ab")
tQ1_B_A = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P0_A, P0_B, P1_B, k[1], "Q0 ba")
tQ1_B_B = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P0_B, P0_B, P1_B, k[1], "Q0 bb")
# correction bad normal orientation
Q1_A_A, Q1_A_B = tQ1_A_A, tQ1_A_B
Q1_B_A, Q1_B_B = -tQ1_B_A, -tQ1_B_B


############################################################
## create X10
############################################################
kX10a = createIdentityOperator(
    context, P1_A, P1_A, P0_A, "X10 aa")
kX10b = createNullOperator(
    context, P1_C, P1_A, P0_A, "X10 ac")
kX10c = createNullOperator(
    context, P1_A, P1_B, P0_B, "X10 ba")
kX10d = createNullOperator(
    context, P1_C, P1_B, P0_B, "X10 bc")
# do not depend on normal
kX10_A_A, kX10_A_C = kX10a, kX10b
kX10_B_A, kX10_B_C = kX10c, kX10d

vX10a = createNullOperator(
    context, P0_A, P1_A, P0_A, "X10 aa")
vX10b = createNullOperator(
    context, P0_C, P1_A, P0_A, "X10 ab")
vX10c = createNullOperator(
    context, P0_A, P1_B, P0_B, "X10 ca")
vX10d = createNullOperator(
    context, P0_C, P1_B, P0_B, "X10 cb")
# do not depend on normal
vX10_A_A, vX10_A_C = vX10a, vX10b
vX10_B_A, vX10_B_C = vX10c, vX10d

wX10a = createNullOperator(
    context, P1_A, P0_A, P1_A, "X10 aa")
wX10b = createNullOperator(
    context, P1_C, P0_A, P1_A, "X10 ab")
wX10c = createNullOperator(
    context, P1_A, P0_B, P1_B, "X10 ca")
wX10d = createNullOperator(
    context, P1_C, P0_B, P1_B, "X10 cb")
# do not depend on normal
wX10_A_A, wX10_A_C = wX10a, wX10b
wX10_B_A, wX10_B_C = wX10c, wX10d

qX10a = createIdentityOperator(
    context, P0_A, P0_A, P1_A, "X10 aa")
qX10b = createNullOperator(
    context, P0_C, P0_A, P1_A, "X10 ab")
qX10c = createNullOperator(
    context, P0_A, P0_B, P1_B, "X10 ca")
qX10d = createNullOperator(
    context, P0_C, P0_B, P1_B, "X10 cb")
# do not depend on normal
qX10_A_A, qX10_A_C = -qX10a, -qX10b
qX10_B_A, qX10_B_C = -qX10c, -qX10d

############################################################
## create X12 
############################################################
kX12a = createNullOperator(
    context, P1_C, P1_A, P0_A, "X12 ca")
kX12b = createNullOperator(
    context, P1_B, P1_A, P0_A, "X12 ab")
kX12c = createNullOperator(
    context, P1_C, P1_B, P0_B, "X12 cc")
kX12d = createIdentityOperator(
    context, P1_B, P1_B, P0_B, "X12 cb")
# do not depend on normal
kX12_A_C, kX12_A_B = kX12a, kX12b
kX12_B_C, kX12_B_B = kX12c, kX12d

vX12a = createNullOperator(
    context, P0_C, P1_A, P0_A, "X12 ca")
vX12b = createNullOperator(           
    context, P0_B, P1_A, P0_A, "X12 ab")
vX12c = createNullOperator(           
    context, P0_C, P1_B, P0_B, "X12 cc")
vX12d = createNullOperator(           
    context, P0_B, P1_B, P0_B, "X12 cb")
# do not depend on normal
vX12_A_C, vX12_A_B = vX12a, vX12b
vX12_B_C, vX12_B_B = vX12c, vX12d

wX12a = createNullOperator(
    context, P1_C, P0_A, P1_A, "X12 ca")
wX12b = createNullOperator(          
    context, P1_B, P0_A, P1_A, "X12 ab")
wX12c = createNullOperator(          
    context, P1_C, P0_B, P1_B, "X12 cc")
wX12d = createNullOperator(          
    context, P1_B, P0_B, P1_B, "X12 cb")
# do not depend on normal
wX12_A_C, wX12_A_B = wX12a, wX12b
wX12_B_C, wX12_B_B = wX12c, wX12d

qX12a = createNullOperator(
    context, P0_C, P0_A, P1_A, "X12 ca")
qX12b = createNullOperator(          
    context, P0_B, P0_A, P1_A, "X12 ab")
qX12c = createNullOperator(          
    context, P0_C, P0_B, P1_B, "X12 cc")
qX12d = createIdentityOperator(          
    context, P0_B, P0_B, P1_B, "X12 cb")
# do not depend on normal
qX12_A_C, qX12_A_B = -qX12a, -qX12b
qX12_B_C, qX12_B_B = -qX12c, -qX12d
############################################################

############################################################
# create missing useful operators bnd1: A, -B
############################################################
tK1_A_A = createIdentityOperator(
    context, P1_A, P1_A, P0_A, "iK1 aa")
tK1_A_B = createNullOperator(
    context, P1_B, P1_A, P0_A, "oK1 ac")
tK1_B_A = createNullOperator(
    context, P1_A, P1_B, P0_B, "oK1 ca")
tK1_B_B = createIdentityOperator(
    context, P1_B, P1_B, P0_B, "iK1 cc")
# correction bad normal orientation
iK1_A_A, oK1_A_B = tK1_A_A, tK1_A_B
oK1_B_A, iK1_B_B = tK1_B_A, tK1_B_B

tV1_A_A = createNullOperator(
    context, P0_A, P1_A, P0_A, "oV1 aa")
tV1_A_B = createNullOperator(
    context, P0_B, P1_A, P0_A, "oV1 ac")
tV1_B_A = createNullOperator(
    context, P0_A, P1_B, P0_B, "oV1 ca")
tV1_B_B = createNullOperator(
    context, P0_B, P1_B, P0_B, "oV1 cc")
# do not depend on normal
oV1_A_A, oV1_A_B = tV1_A_A, tV1_A_B
oV1_B_A, oV1_B_B = tV1_B_A, tV1_B_B

tW1_A_A = createNullOperator(
    context, P1_A, P0_A, P1_A, "oW1 aa")
tW1_A_B = createNullOperator(
    context, P1_B, P0_A, P1_A, "oW1 ab")
tW1_B_A = createNullOperator(
    context, P1_A, P0_B, P1_B, "oW1 ba")
tW1_B_B = createNullOperator(
    context, P1_B, P0_B, P1_B, "oW1 bb")
# double wrong normal orientation => bompensated
oW1_A_A, oW1_A_B = tW1_A_A, tW1_A_B
oW1_B_A, oW1_B_B = tW1_B_A, tW1_B_B

tQ1_A_A = createIdentityOperator(
    context, P0_A, P0_A, P1_A, "iQ1 aa")
tQ1_A_B = createNullOperator(
    context, P0_B, P0_A, P1_A, "oQ1 ab")
tQ1_B_A = createNullOperator(
    context, P0_A, P0_B, P1_B, "oQ1 ba")
tQ1_B_B = createIdentityOperator(
    context, P0_B, P0_B, P1_B, "iQ1 bb")
# borrebtion bad normal orientation
iQ1_A_A, oQ1_A_B = tQ1_A_A, tQ1_A_B
oQ1_B_A, iQ1_B_B = tQ1_B_A, tQ1_B_B
############################################################
############################################################


############################################################
## create A2: bnd = C, B
############################################################
tK2_C_C = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_C, P1_C, P0_C, k[2], "K2 cc")
tK2_C_B = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_B, P1_C, P0_C, k[2], "K2 cb")
tK2_B_C = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_C, P1_B, P0_B, k[2], "K2 bc")
tK2_B_B = createHelmholtz3dDoubleLayerBoundaryOperator(
    context, P1_B, P1_B, P0_B, k[2], "K2 bb")
# good normal
K2_C_C, K2_C_B = tK2_C_C, tK2_C_B
K2_B_C, K2_B_B = tK2_B_C, tK2_B_B

tV2_C_C = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P0_C, P1_C, P0_C, k[2], "V2 cc")
tV2_C_B = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P0_B, P1_C, P0_C, k[2], "V2 cb")
tV2_B_C = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P0_C, P1_B, P0_B, k[2], "V2 bc")
tV2_B_B = createHelmholtz3dSingleLayerBoundaryOperator(
    context, P0_B, P1_B, P0_B, k[2], "V2 bb")
# good normal
V2_C_C, V2_C_B = tV2_C_C, tV2_C_B
V2_B_C, V2_B_B = tV2_B_C, tV2_B_B

tW2_C_C = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_C, P0_C, P1_C, k[2], "W2 cc")
tW2_C_B = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_B, P0_C, P1_C, k[2], "W2 cb")
tW2_B_C = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_C, P0_B, P1_B, k[2], "W2 bc")
tW2_B_B = createHelmholtz3dHypersingularBoundaryOperator(
    context, P1_B, P0_B, P1_B, k[2], "W2 bb")
# good normal
W2_C_C, W2_C_B = tW2_C_C, tW2_C_B
W2_B_C, W2_B_B = tW2_B_C, tW2_B_B

tQ2_C_C = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P0_C, P0_C, P1_C, k[2], "Q0 cc")
tQ2_C_B = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P0_B, P0_C, P1_C, k[2], "Q0 cb")
tQ2_B_C = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P0_C, P0_B, P1_B, k[2], "Q0 bc")
tQ2_B_B = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, P0_B, P0_B, P1_B, k[2], "Q0 bb")
# good normal
Q2_C_C, Q2_C_B = tQ2_C_C, tQ2_C_B
Q2_B_C, Q2_B_B = tQ2_B_C, tQ2_B_B


############################################################
## create X20
############################################################
kX20a = createNullOperator(
    context, P1_A, P1_C, P0_C, "X20 ca")
kX20b = createIdentityOperator(
    context, P1_C, P1_C, P0_C, "X20 cc")
kX20c = createNullOperator(
    context, P1_A, P1_B, P0_B, "X20 ba")
kX20d = createNullOperator(
    context, P1_C, P1_B, P0_B, "X20 bc")
# do not depend on normal and the normal is the right one
kX20_C_A, kX20_C_C = kX20a, kX20b
kX20_B_A, kX20_B_C = kX20c, kX20d

vX20a = createNullOperator(
    context, P0_A, P1_C, P0_C, "X20 ca")
vX20b = createNullOperator(
    context, P0_C, P1_C, P0_C, "X20 cc")
vX20c = createNullOperator(
    context, P0_A, P1_B, P0_B, "X20 ba")
vX20d = createNullOperator(
    context, P0_C, P1_B, P0_B, "X20 bc")
# do not depend on normal and the normal is the right one
vX20_C_A, vX20_C_C = vX20a, vX20b
vX20_B_A, vX20_B_C = vX20c, vX20d

wX20a = createNullOperator(
    context, P1_A, P0_C, P1_C, "X20 ca")
wX20b = createNullOperator(
    context, P1_C, P0_C, P1_C, "X20 cc")
wX20c = createNullOperator(
    context, P1_A, P0_B, P1_B, "X20 ba")
wX20d = createNullOperator(
    context, P1_C, P0_B, P1_B, "X20 bc")
# do not depend on normal and the normal is the right one
wX20_C_A, wX20_C_C = wX20a, wX20b
wX20_B_A, wX20_B_C = wX20c, wX20d

qX20a = createNullOperator(
    context, P0_A, P0_C, P1_C, "X20 ca")
qX20b = createIdentityOperator(
    context, P0_C, P0_C, P1_C, "X20 cc")
qX20c = createNullOperator(
    context, P0_A, P0_B, P1_B, "X20 ba")
qX20d = createNullOperator(
    context, P0_C, P0_B, P1_B, "X20 bc")
# do not depend on normal and the normal is the right one
qX20_C_A, qX20_C_C = -qX20a, -qX20b
qX20_B_A, qX20_B_C = -qX20c, -qX20d

############################################################
## create X21 
############################################################
kX21a = createNullOperator(
    context, P1_A, P1_C, P0_C, "X21 ca")
kX21b = createNullOperator(
    context, P1_B, P1_C, P0_C, "X21 cb")
kX21c = createNullOperator(
    context, P1_A, P1_B, P0_B, "X21 ba")
kX21d = createIdentityOperator(
    context, P1_B, P1_B, P0_B, "X21 bb")
# do not depend on normal and the normal is the right one
kX21_C_A, kX21_C_B = kX21a, kX21b
kX21_B_A, kX21_B_B = kX21c, kX21d

vX21a = createNullOperator(
    context, P0_A, P1_C, P0_C, "X21 ca")
vX21b = createNullOperator(           
    context, P0_B, P1_C, P0_C, "X21 cb")
vX21c = createNullOperator(           
    context, P0_A, P1_B, P0_B, "X21 ba")
vX21d = createNullOperator(           
    context, P0_B, P1_B, P0_B, "X21 bb")
# do not depend on normal and the normal is the right one
vX21_C_A, vX21_C_B = vX21a, vX21b
vX21_B_A, vX21_B_B = vX21c, vX21d

wX21a = createNullOperator(
    context, P1_A, P0_C, P1_C, "X21 ca")
wX21b = createNullOperator(          
    context, P1_B, P0_C, P1_C, "X21 cb")
wX21c = createNullOperator(          
    context, P1_A, P0_B, P1_B, "X21 ba")
wX21d = createNullOperator(          
    context, P1_B, P0_B, P1_B, "X21 bb")
# do not depend on normal and the normal is the right one
wX21_C_A, wX21_C_B = wX21a, wX21b
wX21_B_A, wX21_B_B = wX21c, wX21d

qX21a = createNullOperator(
    context, P0_A, P0_C, P1_C, "X21 ca")
qX21b = createNullOperator(          
    context, P0_B, P0_C, P1_C, "X21 cb")
qX21c = createNullOperator(          
    context, P0_A, P0_B, P1_B, "X21 ba")
qX21d = createIdentityOperator(          
    context, P0_B, P0_B, P1_B, "X21 bb")
# do not depend on normal and the normal is the right one
qX21_C_A, qX21_C_B = -qX21a, -qX21b
qX21_B_A, qX21_B_B = -qX21c, -qX21d
############################################################

############################################################
# create missing useful operators bnd2: C, B
############################################################
tK2_C_C = createIdentityOperator(
    context, P1_C, P1_C, P0_C, "iK2 cc")
tK2_C_B = createNullOperator(
    context, P1_B, P1_C, P0_C, "oK2 cc")
tK2_B_C = createNullOperator(
    context, P1_C, P1_B, P0_B, "oK2 cc")
tK2_B_B = createIdentityOperator(
    context, P1_B, P1_B, P0_B, "iK2 cc")
# correction bcd normcl orientction
iK2_C_C, oK2_C_B = tK2_C_C, tK2_C_B
oK2_B_C, iK2_B_B = tK2_B_C, tK2_B_B

tV2_C_C = createNullOperator(
    context, P0_C, P1_C, P0_C, "oV2 cc")
tV2_C_B = createNullOperator(
    context, P0_B, P1_C, P0_C, "oV2 cc")
tV2_B_C = createNullOperator(
    context, P0_C, P1_B, P0_B, "oV2 cc")
tV2_B_B = createNullOperator(
    context, P0_B, P1_B, P0_B, "oV2 cc")
# do not depend on normcl
oV2_C_C, oV2_C_B = tV2_C_C, tV2_C_B
oV2_B_C, oV2_B_B = tV2_B_C, tV2_B_B

tW2_C_C = createNullOperator(
    context, P1_C, P0_C, P1_C, "oW2 cc")
tW2_C_B = createNullOperator(
    context, P1_B, P0_C, P1_C, "oW2 cb")
tW2_B_C = createNullOperator(
    context, P1_C, P0_B, P1_B, "oW2 bc")
tW2_B_B = createNullOperator(
    context, P1_B, P0_B, P1_B, "oW2 bb")
# double wrong normcl orientction => bompenscted
oW2_C_C, oW2_C_B = tW2_C_C, tW2_C_B
oW2_B_C, oW2_B_B = tW2_B_C, tW2_B_B

tQ2_C_C = createIdentityOperator(
    context, P0_C, P0_C, P1_C, "iQ2 cc")
tQ2_C_B = createNullOperator(
    context, P0_B, P0_C, P1_C, "oQ2 cb")
tQ2_B_C = createNullOperator(
    context, P0_C, P0_B, P1_B, "oQ2 bc")
tQ2_B_B = createIdentityOperator(
    context, P0_B, P0_B, P1_B, "iQ2 bb")
# borrebtion bcd normcl orientction
iQ2_C_C, oQ2_C_B = tQ2_C_C, tQ2_C_B
oQ2_B_C, iQ2_B_B = tQ2_B_C, tQ2_B_B
############################################################
############################################################

h = -0.5
Ms = [
    [ 
        -K0_A_A, -K0_A_C, V0_A_A, V0_A_C, 
         
         h*kX01_A_A, h*kX01_A_B, h*vX01_A_A, h*vX01_A_B,
         
         h*kX02_A_C, h*kX02_A_B, h*vX02_A_C, h*vX02_A_B,
         ],    
    [
        -K0_C_A, -K0_C_C, V0_C_A, V0_C_C,

         h*kX01_C_A, h*kX01_C_B, h*vX01_C_A, h*vX01_C_B,
         
         h*kX02_C_C, h*kX02_C_B, h*vX02_C_C, h*vX02_C_B,
         ],
    [
        W0_A_A, W0_A_C, Q0_A_A, Q0_A_C, 
        
        h*wX01_A_A, h*wX01_A_B, h*qX01_A_A, h*qX01_A_B,
        
        h*wX02_A_C, h*wX02_A_B, h*qX02_A_C, h*qX02_A_B,
        ],     
    [
        W0_C_A, W0_C_C, Q0_C_A, Q0_C_C,
         
        h*wX01_C_A, h*wX01_C_B, h*qX01_C_A, h*qX01_C_B,
        
        h*wX02_C_C, h*wX02_C_B, h*qX02_C_C, h*qX02_C_B,
        ],
      

    [ 
         h*kX10_A_A, h*kX10_A_C, h*vX10_A_A, h*vX10_A_C,

        -K1_A_A, -K1_A_B, V1_A_A, V1_A_B,          
         
         h*kX12_A_C, h*kX12_A_B, h*vX12_A_C, h*vX12_A_B,
         ],    
    [
         h*kX10_B_A, h*kX10_B_C, h*vX10_B_A, h*vX10_B_C,

        -K1_B_A, -K1_B_B, V1_B_A, V1_B_B,
         
         h*kX12_B_C, h*kX12_B_B, h*vX12_B_C, h*vX12_B_B,
         ],
    [        
        h*wX10_A_A, h*wX10_A_C, h*qX10_A_A, h*qX10_A_C,

        W1_A_A, W1_A_B, Q1_A_A, Q1_A_B, 
        
        h*wX12_A_C, h*wX12_A_B, h*qX12_A_C, h*qX12_A_B,
        ],     
    [         
        h*wX10_B_A, h*wX10_B_C, h*qX10_B_A, h*qX10_B_C,

        W1_B_A, W1_B_B, Q1_B_A, Q1_B_B,
        
        h*wX12_B_C, h*wX12_B_B, h*qX12_B_C, h*qX12_B_B,
        ],


    [          
         h*kX20_C_A, h*kX20_C_C, h*vX20_C_A, h*vX20_C_C,
         
         h*kX21_C_A, h*kX21_C_B, h*vX21_C_A, h*vX21_C_B,

        -K2_C_C, -K2_C_B, V2_C_C, V2_C_B, 
         ],    
    [
         h*kX20_B_A, h*kX20_B_C, h*vX20_B_A, h*vX20_B_C,
         
         h*kX21_B_A, h*kX21_B_B, h*vX21_B_A, h*vX21_B_B,

        -K2_B_C, -K2_B_B, V2_B_C, V2_B_B,
         ],
    [        
        h*wX20_C_A, h*wX20_C_C, h*qX20_C_A, h*qX20_C_C,
        
        h*wX21_C_A, h*wX21_C_B, h*qX21_C_A, h*qX21_C_B,

        W2_C_C, W2_C_B, Q2_C_C, Q2_C_B,
        ],     
    [         
        h*wX20_B_A, h*wX20_B_C, h*qX20_B_A, h*qX20_B_C,
        
        h*wX21_B_A, h*wX21_B_B, h*qX21_B_A, h*qX21_B_B,

        W2_B_C, W2_B_B, Q2_B_C, Q2_B_B,
        ],
    ]


#
print('==== Compute [weakForm] all the Operators (blocks) / Dense=Ouch!'),
print('----> could be time consuming...')
#
#Mlist = []
for obj in Ms:
#    l = []
    for o in obj:
        a = o.weakForm()
#        l.append(a.asMatrix())
#    Mlist.append(l)

# for obj in [
#     iK0_A_A, oK0_A_C, oK0_C_A, iK0_C_C, 
#     oV0_A_A, oV0_A_C, oV0_C_A, oV0_C_C, 
#     oW0_A_A, oW0_A_C, oW0_C_A, oW0_C_C,
#     iQ0_A_A, oQ0_A_C, oQ0_C_A, iQ0_C_C, 
    
#     iK1_A_A, oK1_A_B, oK1_B_A, iK1_B_B, 
#     oV1_A_A, oV1_A_B, oV1_B_A, oV1_B_B, 
#     oW1_A_A, oW1_A_B, oW1_B_A, oW1_B_B,
#     iQ1_A_A, oQ1_A_B, oQ1_B_A, iQ1_B_B,

#     iK2_C_C, oK2_C_B, oK2_B_C, iK2_B_B, 
#     oV2_C_C, oV2_C_B, oV2_B_C, oV2_B_B, 
#     oW2_C_C, oW2_C_B, oW2_B_C, oW2_B_B,
#     iQ2_C_C, oQ2_C_B, oQ2_B_C, iQ2_B_B,
#     ]:
#     a = obj.weakForm()
    

print('==== Assembly RHS [NotImplementedYet]')

# rhs = [
#     I0_k * incDirichletTrace0, I0_q * incNeumannTrace0,
#     I10_k * incDirichletTrace0, I10_q * incNeumannTrace0,
#     I20_k * incDirichletTrace0, I20_q * incNeumannTrace0
#     ]
# for v in rhs: v.projections()

#func = np.random.rand
func = np.ones

rhs = []
for trial, test in zip([P1_A, P1_C, P0_A, P0_C,
                        P1_A, P1_B, P0_A, P0_B,
                        P1_C, P1_B, P0_C, P0_B],
                       [P0_A, P0_C, P1_A, P1_C,
                        P0_A, P0_B, P1_A, P1_B,
                        P0_C, P0_B, P1_C, P1_B]):
    gf = createGridFunction(
            context, trial, test,
            coefficients = func(trial.globalDofCount()))
    gf.projections()
    rhs.append(gf)

print('==== Create BlockedBoundaryOperator')
MTF = createBlockedBoundaryOperator(context, Ms)
#
print(' note: if the operators were previously wearForm-ed'),
print('then it is only a checking')
print('\t(of the shapes of the different blocks)')
mtf = MTF.weakForm() 

#
print('==== Initialize the solver')
#
solver = createDefaultDirectSolver(MTF)

#
print("\nWell done. Now let compute, assembly and solve...\n")
#

#
print('==== Solve')
#
solution = solver.solve(rhs)
print(solution.solverMessage())

#
print("\nWell done. Now let extract...\n")
#
gfD0_A = solution.gridFunction(0)
gfD0_C = solution.gridFunction(1)
gfN0_A = solution.gridFunction(2)
gfN0_C = solution.gridFunction(3)

gfD1_A = solution.gridFunction(4)
gfD1_B = solution.gridFunction(5)
gfN1_A = solution.gridFunction(6)
gfN1_B = solution.gridFunction(7)

gfD2_C = solution.gridFunction(8)
gfD2_B = solution.gridFunction(9)
gfN2_C = solution.gridFunction(10)
gfN2_B = solution.gridFunction(11)

# exportToGmsh(gfD0_A, 'dir 0', 'dir0-a.pos')
# exportToGmsh(gfD0_C, 'dir 0', 'dir0-c.pos')

# lD0_A = gfD0_A.coefficients()
# lD0_C = gfD0_C.coefficients()
# lN0_A = gfN0_A.coefficients()
# lN0_C = gfN0_C.coefficients()

# lD1_A = gfD1_A.coefficients()
# lD1_B = gfD1_B.coefficients()
# lN1_A = gfN1_A.coefficients()
# lN1_B = gfN1_B.coefficients()

# lD2_C = gfD2_C.coefficients()
# lD2_B = gfD2_B.coefficients()
# lN2_C = gfN2_C.coefficients()
# lN2_B = gfN2_B.coefficients()

############################################################
############################################################
############################################################

print("\nWell done. Now let check...\n")

# ###

# M = mtf.asMatrix()
# b = np.array([])
# for v in rhs:
#     b = np.concatenate((b, v.projections()))

# print('numpy solver\n')
# x = np.linalg.solve(M, b)

# y = np.array([])
# y = np.concatenate((y, lD0_A))
# y = np.concatenate((y, lD0_C))
# y = np.concatenate((y, lN0_A))
# y = np.concatenate((y, lN0_C))
# y = np.concatenate((y, lD1_A))
# y = np.concatenate((y, lD1_B))
# y = np.concatenate((y, lN1_A))
# y = np.concatenate((y, lN1_B))
# y = np.concatenate((y, lD2_C))
# y = np.concatenate((y, lD2_B))
# y = np.concatenate((y, lN2_C))
# y = np.concatenate((y, lN2_B))

# print(np.allclose(M.dot(x), b)),
# print(np.allclose(mtf.matvec(x), b)),

# print(np.allclose(M.dot(y), b)),
# print(np.allclose(mtf.matvec(y), b))

# print('\nsolution by hand VS bem++ : '),
# print(np.allclose(y, x))

# ##

# print('\ncheck Calderon...\n')

# h = 0.0
# As = [
#     [ 
#         -K0_A_A, -K0_A_C, V0_A_A, V0_A_C, 
         
#          h*kX01_A_A, h*kX01_A_B, h*vX01_A_A, h*vX01_A_B,
         
#          h*kX02_A_C, h*kX02_A_B, h*vX02_A_C, h*vX02_A_B,
#          ],    
#     [
#         -K0_C_A, -K0_C_C, V0_C_A, V0_C_C,

#          h*kX01_C_A, h*kX01_C_B, h*vX01_C_A, h*vX01_C_B,
         
#          h*kX02_C_C, h*kX02_C_B, h*vX02_C_C, h*vX02_C_B,
#          ],
#     [
#         W0_A_A, W0_A_C, Q0_A_A, Q0_A_C, 
        
#         h*wX01_A_A, h*wX01_A_B, h*qX01_A_A, h*qX01_A_B,
        
#         h*wX02_A_C, h*wX02_A_B, h*qX02_A_C, h*qX02_A_B,
#         ],     
#     [
#         W0_C_A, W0_C_C, Q0_C_A, Q0_C_C,
         
#         h*wX01_C_A, h*wX01_C_B, h*qX01_C_A, h*qX01_C_B,
        
#         h*wX02_C_C, h*wX02_C_B, h*qX02_C_C, h*qX02_C_B,
#         ],
      

#     [ 
#          h*kX10_A_A, h*kX10_A_C, h*vX10_A_A, h*vX10_A_C,

#         -K1_A_A, -K1_A_B, V1_A_A, V1_A_B,          
         
#          h*kX12_A_C, h*kX12_A_B, h*vX12_A_C, h*vX12_A_B,
#          ],    
#     [
#          h*kX10_B_A, h*kX10_B_C, h*vX10_B_A, h*vX10_B_C,

#         -K1_B_A, -K1_B_B, V1_B_A, V1_B_B,
         
#          h*kX12_B_C, h*kX12_B_B, h*vX12_B_C, h*vX12_B_B,
#          ],
#     [        
#         h*wX10_A_A, h*wX10_A_C, h*qX10_A_A, h*qX10_A_C,

#         W1_A_A, W1_A_B, Q1_A_A, Q1_A_B, 
        
#         h*wX12_A_C, h*wX12_A_B, h*qX12_A_C, h*qX12_A_B,
#         ],     
#     [         
#         h*wX10_B_A, h*wX10_B_C, h*qX10_B_A, h*qX10_B_C,

#         W1_B_A, W1_B_B, Q1_B_A, Q1_B_B,
        
#         h*wX12_B_C, h*wX12_B_B, h*qX12_B_C, h*qX12_B_B,
#         ],


#     [          
#          h*kX20_C_A, h*kX20_C_C, h*vX20_C_A, h*vX20_C_C,
         
#          h*kX21_C_A, h*kX21_C_B, h*vX21_C_A, h*vX21_C_B,

#         -K2_C_C, -K2_C_B, V2_C_C, V2_C_B, 
#          ],    
#     [
#          h*kX20_B_A, h*kX20_B_C, h*vX20_B_A, h*vX20_B_C,
         
#          h*kX21_B_A, h*kX21_B_B, h*vX21_B_A, h*vX21_B_B,

#         -K2_B_C, -K2_B_B, V2_B_C, V2_B_B,
#          ],
#     [        
#         h*wX20_C_A, h*wX20_C_C, h*qX20_C_A, h*qX20_C_C,
        
#         h*wX21_C_A, h*wX21_C_B, h*qX21_C_A, h*qX21_C_B,

#         W2_C_C, W2_C_B, Q2_C_C, Q2_C_B,
#         ],     
#     [         
#         h*wX20_B_A, h*wX20_B_C, h*qX20_B_A, h*qX20_B_C,
        
#         h*wX21_B_A, h*wX21_B_B, h*qX21_B_A, h*qX21_B_B,

#         W2_B_C, W2_B_B, Q2_B_C, Q2_B_B,
#         ],
#     ]
# DiagCald = createBlockedBoundaryOperator(context, As)
# diagCald = DiagCald.weakForm()
# A = diagCald.asMatrix()

# h = 0.0
# Is = [
#     [ 
#         iK0_A_A, oK0_A_C, oV0_A_A, oV0_A_C, 
         
#         h*kX01_A_A, h*kX01_A_B, h*vX01_A_A, h*vX01_A_B,
        
#         h*kX02_A_C, h*kX02_A_B, h*vX02_A_C, h*vX02_A_B,
#          ],    
#     [
#         oK0_C_A, iK0_C_C, oV0_C_A, oV0_C_C,

#         h*kX01_C_A, h*kX01_C_B, h*vX01_C_A, h*vX01_C_B,
        
#         h*kX02_C_C, h*kX02_C_B, h*vX02_C_C, h*vX02_C_B,
#          ],
#     [
#         oW0_A_A, oW0_A_C, iQ0_A_A, oQ0_A_C, 
        
#         h*wX01_A_A, h*wX01_A_B, h*qX01_A_A, h*qX01_A_B,
        
#         h*wX02_A_C, h*wX02_A_B, h*qX02_A_C, h*qX02_A_B,
#         ],     
#     [
#         oW0_C_A, oW0_C_C, oQ0_C_A, iQ0_C_C,
         
#         h*wX01_C_A, h*wX01_C_B, h*qX01_C_A, h*qX01_C_B,
        
#         h*wX02_C_C, h*wX02_C_B, h*qX02_C_C, h*qX02_C_B,
#         ],
      

#     [ 
#          h*kX10_A_A, h*kX10_A_C, h*vX10_A_A, h*vX10_A_C,

#          iK1_A_A, oK1_A_B, oV1_A_A, oV1_A_B,          
         
#          h*kX12_A_C, h*kX12_A_B, h*vX12_A_C, h*vX12_A_B,
#          ],    
#     [
#          h*kX10_B_A, h*kX10_B_C, h*vX10_B_A, h*vX10_B_C,

#          oK1_B_A, iK1_B_B, oV1_B_A, oV1_B_B,
         
#          h*kX12_B_C, h*kX12_B_B, h*vX12_B_C, h*vX12_B_B,
#          ],
#     [        
#         h*wX10_A_A, h*wX10_A_C, h*qX10_A_A, h*qX10_A_C,

#         oW1_A_A, oW1_A_B, iQ1_A_A, oQ1_A_B, 
        
#         h*wX12_A_C, h*wX12_A_B, h*qX12_A_C, h*qX12_A_B,
#         ],     
#     [         
#         h*wX10_B_A, h*wX10_B_C, h*qX10_B_A, h*qX10_B_C,

#         oW1_B_A, oW1_B_B, oQ1_B_A, iQ1_B_B,
        
#         h*wX12_B_C, h*wX12_B_B, h*qX12_B_C, h*qX12_B_B,
#         ],


#     [          
#          h*kX20_C_A, h*kX20_C_C, h*vX20_C_A, h*vX20_C_C,
         
#          h*kX21_C_A, h*kX21_C_B, h*vX21_C_A, h*vX21_C_B,

#          iK2_C_C, oK2_C_B, oV2_C_C, oV2_C_B, 
#          ],    
#     [
#          h*kX20_B_A, h*kX20_B_C, h*vX20_B_A, h*vX20_B_C,
         
#          h*kX21_B_A, h*kX21_B_B, h*vX21_B_A, h*vX21_B_B,

#          oK2_B_C, iK2_B_B, oV2_B_C, oV2_B_B,
#          ],
#     [        
#         h*wX20_C_A, h*wX20_C_C, h*qX20_C_A, h*qX20_C_C,
        
#         h*wX21_C_A, h*wX21_C_B, h*qX21_C_A, h*qX21_C_B,

#         oW2_C_C, oW2_C_B, iQ2_C_C, oQ2_C_B,
#         ],     
#     [         
#         h*wX20_B_A, h*wX20_B_C, h*qX20_B_A, h*qX20_B_C,
        
#         h*wX21_B_A, h*wX21_B_B, h*qX21_B_A, h*qX21_B_B,

#         oW2_B_C, oW2_B_B, oQ2_B_C, iQ2_B_B,
#         ],
#     ]
# Identity = createBlockedBoundaryOperator(context, Is)
# identity = Identity.weakForm()
# II = identity.asMatrix()

# # AA = np.linalg.solve(II, A)
# # r = AA.dot(x)
# # print(np.allclose(r, x))
# # e = np.linalg.norm(r-x)
# # print(e)

# a = 0.5

# try:
#     r = np.linalg.solve(II, A.dot(x))
#     print(np.allclose(a*r, x))
#     e = np.linalg.norm(a*r - x)
#     print(e)

#     r = 4*A.dot(r)
#     e = np.linalg.norm(r - II.dot(x))
#     print(e)
# except:
#     pass

# print(np.allclose(a*II.dot(x), A.dot(x)))
# e = np.linalg.norm(a*II.dot(x) - A.dot(x))
# print(e)




# # MM = np.linalg.solve(II, M)
# # r = MM.dot(x)
# # print(np.allclose(r, x))
# # e = np.linalg.norm(r-x)
# # print(e)


# print('\n\ndiscretization scheme : ' + discretization)
