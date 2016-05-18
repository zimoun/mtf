#!/usr/bin/env python
# coding: utf8

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import bempp.api as bem

import my_grid
import config

k0 = config.k0
k1 = config.k1
k2 = config.k2

g0 = bem.import_grid('0.msh')
g1 = bem.import_grid('1.msh')
g2 = bem.import_grid('2.msh')

def function_space(grid, **kwargs):
    return bem.function_space(grid, "P", 1, **kwargs)

s0 = function_space(g0)
s1 = function_space(g1)
s2 = function_space(g2)

N0, N1, N2 = s0.global_dof_count, s1.global_dof_count, s2.global_dof_count

###
# Block Diagonal Part
###

opA = bem.BlockedOperator(3, 3)

opA0 = bem.BlockedOperator(2, 2)
opA1 = bem.BlockedOperator(2, 2)
opA2 = bem.BlockedOperator(2, 2)

opK0 = bem.operators.boundary.helmholtz.double_layer(s0, s0, s0, k0)
opV0 = bem.operators.boundary.helmholtz.single_layer(s0, s0, s0, k0)
opW0 = bem.operators.boundary.helmholtz.hypersingular(s0, s0, s0, k0)
opQ0 = bem.operators.boundary.helmholtz.adjoint_double_layer(s0, s0, s0, k0)

opA0[0, 0], opA0[0, 1] = -opK0, opV0
opA0[1, 0], opA0[1, 1] = opW0, opQ0


opK1 = bem.operators.boundary.helmholtz.double_layer(s1, s1, s1, k1)
opV1 = bem.operators.boundary.helmholtz.single_layer(s1, s1, s1, k1)
opW1 = bem.operators.boundary.helmholtz.hypersingular(s1, s1, s1, k1)
opQ1 = bem.operators.boundary.helmholtz.adjoint_double_layer(s1, s1, s1, k1)

opA1[0, 0], opA1[0, 1] = -opK1, opV1
opA1[1, 0], opA1[1, 1] = opW1, opQ1


opK2 = bem.operators.boundary.helmholtz.double_layer(s2, s2, s2, k2)
opV2 = bem.operators.boundary.helmholtz.single_layer(s2, s2, s2, k2)
opW2 = bem.operators.boundary.helmholtz.hypersingular(s2, s2, s2, k2)
opQ2 = bem.operators.boundary.helmholtz.adjoint_double_layer(s2, s2, s2, k2)

opA2[0, 0], opA2[0, 1] = -opK2, opV2
opA2[1, 0], opA2[1, 1] = opW2, opQ2


opA[0, 0], opA[1, 1], opA[2, 2] = opA0, opA1, opA2

print('weak form (from triple)')

wA = opA.weak_form()

shape = wA.shape
N, _ = shape

print('consuming done.')

##############################
#
# The Issue : coupling terms
#
##############################

bX = bem.BlockedDiscreteOperator(3, 3)

X01 = bem.BlockedDiscreteOperator(2, 2)
X02 = bem.BlockedDiscreteOperator(2, 2)
X12 = bem.BlockedDiscreteOperator(2, 2)

X10 = bem.BlockedDiscreteOperator(2, 2)
X20 = bem.BlockedDiscreteOperator(2, 2)
X21 = bem.BlockedDiscreteOperator(2, 2)

###
# First collect all the geometrical info
###

print('collect my_grid')

nodes0 = my_grid.get_vertices(s0)
nodes1 = my_grid.get_vertices(s1)
nodes2 = my_grid.get_vertices(s2)

nds_0_1, nds_1_0 = my_grid.common_vertices(nodes0, nodes1)
nds_0_2, nds_2_0 = my_grid.common_vertices(nodes0, nodes2)
nds_1_2, nds_2_1 = my_grid.common_vertices(nodes1, nodes2)

el2nd0 = my_grid.vertices_by_elements(s0)
el2nd1 = my_grid.vertices_by_elements(s1)
el2nd2 = my_grid.vertices_by_elements(s2)

faces0 = my_grid.get_elements(s0)
faces1 = my_grid.get_elements(s1)
faces2 = my_grid.get_elements(s2)

boule = True
els_0_1, els_1_0 = my_grid.common_elements(faces0, faces1, only=boule)
els_0_2, els_2_0 = my_grid.common_elements(faces0, faces2, only=boule)
els_1_2, els_2_1 = my_grid.common_elements(faces1, faces2, only=boule)

dof_to_n0 = my_grid.dofs_to_nodes(s0)
dof_to_n1 = my_grid.dofs_to_nodes(s1)
dof_to_n2 = my_grid.dofs_to_nodes(s2)

print('done.')

###
# HACK !! still an issue for general considerations
###

def dof_to_dof(arguments):
    """
    This function depends on the `functional space`.

    Question: What is lacking to obtain a general procedure ?
    Answer: The link between geometrical information and DoF numbering.

    This is hard coded somewhere, and this information should help a lot.
    Instead of doing some reverse engineering.

     - If "P0", then it is the most obvious since the DoF are attached to elements.
     - If "P1", then it is obvious since the DoF are attached to nodes.
     - Otherwise, welcome to the mess !!

    Easy to get:
    where space{1, 2} attached to grid{1, 2}

     - Nodes_list(space1) <---> Nodes_list(space2)
     - Elements_list(space1) <---> Elements_list(space2)

    (see my_grid.py)
    """
    pass

###
# Second build the closed domains identity
###

Id0 = bem.operators.boundary.sparse.identity(s0, s0, s0)
Id1 = bem.operators.boundary.sparse.identity(s1, s1, s1)
Id2 = bem.operators.boundary.sparse.identity(s2, s2, s2)

wId0 = Id0.weak_form()
sId0 = wId0.sparse_operator

wId1 = Id1.weak_form()
sId1 = wId1.sparse_operator

wId2 = Id2.weak_form()
sId2 = wId2.sparse_operator

###
# Third plug in the right order
###

# by-pass the issue
# since considering P1, the obvious mechanism is:
#
#  dof(elements_space_1) -> list_nodes1
#  nodes_from_space1_to_space2(list_nodes) -> list_nodes2
#  inverse_dof(list_nodes) -> dof_space2
#

i01 = sp.lil_matrix((N0, N1), dtype=complex)
i10 = sp.lil_matrix((N1, N0), dtype=complex)

for e0, e1 in els_0_1:
    dof0 = s0.get_global_dofs(faces0[e0])
    n0 = el2nd0[e0]

    n0_to_1 = [ nds_0_1[i] for i in n0 ]
    dof0_in_1 = [ dof_to_n1[i] for i in n0_to_1 ]

    for r, c in zip(dof0, dof0_in_1):
        for i, j in zip(dof0, dof0_in_1):
            i01[r, j] = sId0[r, i]

    dof1 = s1.get_global_dofs(faces1[e1])
    n1 = el2nd1[e1]

    n1_to_0 = [ nds_1_0[i] for i in n1 ]
    dof1_in_0 = [ dof_to_n0[i] for i in n1_to_0 ]

    for r, c in zip(dof1, dof1_in_0):
        for i, j in zip(dof1, dof1_in_0):
            i10[r, j] = sId1[r, i]

X01[0, 0], X01[1, 1] = i01, - i01
X10[0, 0], X10[1, 1] = i10, - i10

i02 = sp.lil_matrix((N0, N2), dtype=complex)
i20 = sp.lil_matrix((N2, N0), dtype=complex)

for e0, e2 in els_0_2:
    dof0 = s0.get_global_dofs(faces0[e0])
    n0 = el2nd0[e0]

    n0_to_2 = [ nds_0_2[i] for i in n0 ]
    dof0_in_2 = [ dof_to_n2[i] for i in n0_to_2 ]

    for r, c in zip(dof0, dof0_in_2):
        for i, j in zip(dof0, dof0_in_2):
            i02[r, j] = sId0[r, i]

    dof2 = s2.get_global_dofs(faces2[e2])
    n2 = el2nd2[e2]

    n2_to_0 = [ nds_2_0[i] for i in n2 ]
    dof2_in_0 = [ dof_to_n0[i] for i in n2_to_0 ]

    for r, c in zip(dof2, dof2_in_0):
        for i, j in zip(dof2, dof2_in_0):
            i20[r, j] = sId2[r, i]

X02[0, 0], X02[1, 1] = i02, - i02
X20[0, 0], X20[1, 1] = i20, - i20

i21 = sp.lil_matrix((N2, N1), dtype=complex)
i12 = sp.lil_matrix((N1, N2), dtype=complex)

for e1, e2 in els_1_2:
    dof1 = s1.get_global_dofs(faces1[e1])
    n1 = el2nd1[e1]

    n1_to_2 = [ nds_1_2[i] for i in n1 ]
    dof1_in_2 = [ dof_to_n2[i] for i in n1_to_2 ]

    for r, c in zip(dof1, dof1_in_2):
        for i, j in zip(dof1, dof1_in_2):
            i12[r, j] = sId1[r, i]

    dof2 = s2.get_global_dofs(faces2[e2])
    n2 = el2nd2[e2]

    n2_to_1 = [ nds_2_1[i] for i in n2 ]
    dof2_in_1 = [ dof_to_n1[i] for i in n2_to_1 ]

    for r, c in zip(dof2, dof2_in_1):
        for i, j in zip(dof2, dof2_in_1):
            i21[r, j] = sId2[r, i]

X12[0, 0], X12[1, 1] = i12, - i12
X21[0, 0], X21[1, 1] = i21, - i21

bX[0, 1], bX[0, 2] = X01, X02
bX[1, 0], bX[1, 2] = X10, X12
bX[2, 0], bX[2, 1] = X20, X21


####
#
# END
#
###

A = spla.LinearOperator(shape, matvec=wA.matvec, dtype=complex)
X = spla.LinearOperator(shape, matvec=bX.matvec, dtype=complex)
M = A - 0.5 * X

x = np.random.rand(N)
y = M(x)


b = np.array([], dtype=complex)
get_null = lambda lst: [ i for i, j in enumerate(lst) if j >= 0 ]

diri = bem.GridFunction(s0, fun=config.fdir)
neum = bem.GridFunction(s0, fun=config.fneu)

rhs = [diri, -neum]
b = np.concatenate((b, rhs[0].projections()))
b = np.concatenate((b, rhs[1].projections()))


c = diri.coefficients
b = np.concatenate((b, -i10.dot(c)))
c = neum.coefficients
b = np.concatenate((b, i10.dot(c)))

c = diri.coefficients
b = np.concatenate((b, -i20.dot(c)))
c = neum.coefficients
b = np.concatenate((b, i20.dot(c)))

b = 0.5 * b

print('solve')
x, info = spla.gmres(M, b)

ged = bem.GridFunction(s0, coefficients=x[0:N0])
#ged.plot()

gid1 = bem.GridFunction(s1, coefficients=x[2*N0:2*N0+N1])
gid2 = bem.GridFunction(s2, coefficients=x[2*N0+2*N1:2*N0+2*N1+N2])

from miesphere import mie_D4grid, mie_N4grid

eps = config.eps
iincident = config.iincident

k = k0
kk = [0, 0, 0]
for q in range(3):
    if q == iincident:
        kk[q] = k
kk = tuple(kk)
C = np.array([0, 0, 0])
R = 1
ce, ci = 1, np.sqrt(eps)
jumpe, jumpi  = (1, 1), (1, 1)
Nmodes = 50
field = 'sca'
# field = 'int'
# field = 'inc'

def mieD(point, normal, dom_ind, result):
    val = mie_D4grid(field, kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    result[0] = val

def uinc(point, normal, dom_ind, result):
    result[0] = np.exp(1j * kRef * point[iincident])

def mieN(point, normal, dom_ind, result):
    val = mie_N4grid(field, kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    result[0] = val

def dnuinc(point, normal, dom_ind, result):
    result[0] = 1j * kRef * normal[1] * np.exp(1j * kRef * point[iincident])

def mieD_int(point, normal, dom_ind, result):
    val = mie_D4grid('int', kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    result[0] = val

def mieN_int(point, normal, dom_ind, result):
    val = mie_N4grid('int', kk, R, C, ce, ci, jumpe, jumpi, Nmodes, point)
    result[0] = val

gmie = bem.GridFunction(s0, fun=mieD)
miecoeffs = gmie.coefficients
