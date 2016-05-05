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

def function_space(grid, **kwargs):
    return bem.function_space(grid, "P", 1, **kwargs)

s0 = function_space(g0)
s1 = function_space(g1)

N0, N1 = s0.global_dof_count, s1.global_dof_count

nodes0 = my_grid.get_vertices(s0)
nodes1 = my_grid.get_vertices(s1)

nds_0_1, nds_1_0 = my_grid.common_vertices(nodes0, nodes1)

el2nd0 = my_grid.vertices_by_elements(s0)
el2nd1 = my_grid.vertices_by_elements(s1)

faces0 = my_grid.get_elements(s0)
faces1 = my_grid.get_elements(s1)

boule = True
els_0_1, els_1_0 = my_grid.common_elements(faces0, faces1, only=boule)

dof_to_n0 = my_grid.dofs_to_nodes(s0)
dof_to_n1 = my_grid.dofs_to_nodes(s1)

Id0 = bem.operators.boundary.sparse.identity(s0, s0, s0)
Id1 = bem.operators.boundary.sparse.identity(s1, s1, s1)

wId0 = Id0.weak_form()
sId0 = wId0.sparse_operator

wId1 = Id1.weak_form()
sId1 = wId1.sparse_operator

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


def val_at(point, normal, dom_ind, result):
    result[0] = point[0] * point[1] * point[2]
gf0 = bem.GridFunction(s0, fun=val_at)
gf1 = bem.GridFunction(s1, fun=val_at)

proj0 = gf0.projections()
proj1 = gf1.projections()

coef0 = gf0.coefficients
coef1 = gf1.coefficients
