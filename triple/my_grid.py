#!/usr/bin/env python
# coding: utf8

"""
Hack around grid and space

Aim: be able to work with different grids

Issue: be able to establish the correspondance
       between the numbering of both `functional space`
       defined by two different `grid`.

                         ?
DoF_numbering(space_1) <---> DoF_numbering(space_2)
"""

import numpy as np
import bempp.api as bem


def get_entities(space, kind):
    """ kind is one of values:
                 0 element
                 1 edge
                 2 vertex
    """

    if not kind in [0, 1, 2]:
        raise ValueError
    lst = []
    for e in space.grid.leaf_view.entity_iterator(kind):
        lst.append(e)
    return lst


def vertices_by_elements(space):
    """
    reorganized grid.leaf_view.elements in list
    """

    nodes = space.grid.leaf_view.elements.T
    lst = []
    for n in nodes:
        p, q, r = n
        lst.append((p, q, r))
    return lst

def common_vertices(nodes1, nodes2, only=False):
    """
    Assume that dim_world is 3

    Correspondance based on coordinate vertices comparison.

    WARNING: bad practise !!
       -> returning data depends on `only` variable
    """
    # should not be used
    norm = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2)

    common = []
    first2second = [-1] * len(nodes1)
    second2first = [-1] * len(nodes2)
    for i, n in enumerate(nodes1):
        x, y, z = n.geometry.corners
        for j, m in enumerate(nodes2):
            a, b, c = m.geometry.corners

            if x == a and y == b and z == c:
                common.append((i, j))
                first2second[i] = j
                second2first[j] = i
                break

            elif norm(x-a, y-b, z-c) < 1e-6:
                # should never happens !!
                pass
                break

    # print('common nodes done', flush=True)
    if only:
        return [common, -1]
    else:
        return [first2second, second2first]


def common_elements(faces1, faces2, only=False):
    """
    Assume that faces are triangles, defined by 3 points

    Based on coordinated vertices comparison
    (purely geometric correspondance, not considering the normal)


    WARNING: bad practise !!
       -> returning data depends on `only` variable
    """

    common = []
    first2second = [-1] * len(faces1)
    second2first = [-1] * len(faces2)
    for i, f in enumerate(faces1):
        p0, p1, p2 = f.geometry.corners.transpose()
        for j, g in enumerate(faces2):
            q0, q1, q2 = g.geometry.corners.transpose()

            check = 0
            for x, y, z in [p0, p1, p2]:
                for a, b, c in [q0, q1, q2]:
                    if x == a and y == b and z == c:
                        check += 1
            if check == 3:
                common.append((i, j))
                first2second[i] = j
                second2first[j] = i
                break

    # print('common faces done', flush=True)
    if only:
        return [common, -1]
    else:
        return [first2second, second2first]

get_vertices = lambda space: get_entities(space, 2)
get_elements = lambda space: get_entities(space, 0)

def dofs_to_nodes(space):
    """
    This is the function which is difficult to generalize
    for any `functional space`.
    It is obvious for 'P1' since DoF are attached to nodes.
    To be precise, return identity: #Dof = #Node if closed surface
    """

    dof2nod = [-1] * space.global_dof_count
    els = get_elements(space)  # consuming, but proof-of-concept
    nodes_per_elem = vertices_by_elements(space) # again consuming
    for i, e in enumerate(els):
        dofs = space.get_global_dofs(e)

        # because P1
        d1, d2, d3 = dofs
        n1, n2, n3 = nodes_per_elem[i]
        for d, n in zip([d1, d2, d3],
                        [n1, n2, n3]):
            dof2nod[d] = n
    return dof2nod



if __name__ == "__main__":

    gints = bem.import_grid('full-interfaces.msh')

    g0 = bem.import_grid('0.msh')
    g1 = bem.import_grid('1.msh')
    g2 = bem.import_grid('2.msh')


    def function_space(grid, **kwargs):
        return bem.function_space(grid, "P", 1, **kwargs)

    sints = function_space(gints)
    s110 = function_space(gints, domains=[110])
    s210 = function_space(gints, domains=[210])
    s12 = function_space(gints, domains=[12])

    s0 = function_space(g0)
    s1 = function_space(g1)
    s2 = function_space(g2)


    n0 = get_entities(s0, 2)
    f0 = get_entities(s0, 0)

    n1 = get_entities(s1, 2)
    f1 = get_entities(s1, 0)

    # for f in f0:
    #     print(s1.get_global_dofs(f))

    nodes_0_to_1, nodes_1_to_0 = common_vertices(n0, n1)
    print(nodes_0_to_1)

    vals, _  = common_vertices(n0, n1, only=True)
    print(vals)

    faces_0_1, faces_1_0 = common_elements(f0, f1)
    print(faces_0_1)

    vals, _ = common_elements(f0, f1, only=True)
    print(vals)

    nodes = vertices_by_elements(s0)


    dd = dofs_to_nodes(s0)
    print(dd)

    # els_from_0_to_1 = sorted(els_0_1, key=lambda tpl: tpl[0])
    # els_from_1_to_0 = sorted(els_0_1, key=lambda tpl: tpl[1])

    # els_from_0_to_2 = sorted(els_0_2, key=lambda tpl: tpl[0])
    # els_from_2_to_0 = sorted(els_0_2, key=lambda tpl: tpl[1])

    # els_from_1_to_2 = sorted(els_1_2, key=lambda tpl: tpl[0])
    # els_from_2_to_1 = sorted(els_1_2, key=lambda tpl: tpl[1])
