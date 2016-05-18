#!/usr/bin/env python
# coding: utf8

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt

import bempp.api as bem

import my_grid

g0 = bem.import_grid('0.msh')
g1 = bem.import_grid('1.msh')
g2 = bem.import_grid('2.msh')

def function_space(grid, **kwargs):
    return bem.function_space(grid, "P", 1, **kwargs)

s0 = function_space(g0)
s1 = function_space(g1)
s2 = function_space(g2)

N0, N1 = s0.global_dof_count, s1.global_dof_count
N2 = s2.global_dof_count

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

Id0 = bem.operators.boundary.sparse.identity(s0, s0, s0)
Id1 = bem.operators.boundary.sparse.identity(s1, s1, s1)
Id2 = bem.operators.boundary.sparse.identity(s2, s2, s2)

wId0 = Id0.weak_form()
sId0 = wId0.sparse_operator

wId1 = Id1.weak_form()
sId1 = wId1.sparse_operator

wId2 = Id2.weak_form()
sId2 = wId2.sparse_operator

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
#

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
#

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
#


def val_at(point, normal, dom_ind, result):
    # if point[0] < 0:
    #     result[0] = 1 + point[0]
    # else:
    #     result[0] = 1 - point[0]
    #result[0] = 2 + point[0] #+ point[1] + point[2]
    result[0] = np.exp(1j * 0.5 * np.pi * point[1])
gf0 = bem.GridFunction(s0, fun=val_at)
gf1 = bem.GridFunction(s1, fun=val_at)
gf2 = bem.GridFunction(s2, fun=val_at)

proj0 = gf0.projections()
proj1 = gf1.projections()
proj2 = gf2.projections()

coef0 = gf0.coefficients
coef1 = gf1.coefficients
coef2 = gf2.coefficients

v1_onto0 = i01.dot(coef1)
gf1_onto0 = bem.GridFunction(s0, projections=v1_onto0)
gff1_onto0 = bem.GridFunction(s0, coefficients=v1_onto0)

gf_err10 = bem.GridFunction(s0, projections=(v1_onto0 - proj0))

v2_onto0 = i02.dot(coef2)
gf2_onto0 = bem.GridFunction(s0, projections=v2_onto0)

gf_err20 = bem.GridFunction(s0, projections=(v2_onto0 - proj0))


gf12_onto0 = bem.GridFunction(s0, projections=(0.5*v1_onto0 + 0.5*v2_onto0))

gf_err0 = bem.GridFunction(s0, projections=(0.5*(v1_onto0 + v2_onto0) - proj0))

#

v0_onto1 = i10.dot(coef0)
gf0_onto1 = bem.GridFunction(s1, projections=v0_onto1)

gf_err01 = bem.GridFunction(s1, projections=(v0_onto1 - proj1))
