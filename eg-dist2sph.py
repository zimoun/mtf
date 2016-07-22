#!/usr/bin/env python
# coding: utf8

import numpy as np

from assemb import MultiTrace, checker

from time import time
import scipy.linalg as la

kRef = 0.6 * np.pi

L = 5.1


##################################

import numpy as np
import numpy.linalg as la

from subprocess import call

from domains import sanitize_config
from domains import generate_concentric_dict
from domains import write_params_geo
from domains import merge_msh_bubbles
from domains import Domains


rads = [1, 1]
phys = [(2, 1, 1), (2, 1, 1)]

centers = [ (0, 0, 0), (L, 0, 0)]

x, y, z = centers[0]
xx, yy, zz = centers[1]
d = np.sqrt( (x-xx)**2 + (y-yy)**2 + (z-zz)**2 )
if 0.99*d < rads[0]+rads[1]:
    raise ValueError('The spheres are not separated!')

if len(rads) != len(centers):
    raise ValueError('different length')
elif len(rads) != len(phys):
    raise ValueError('different length')
else:
    N = len(rads)


dgens = []
for i in range(N):
    if i == 0:
        truefalse = True
    else:
        truefalse = False
    conf = sanitize_config(init_offset=truefalse,
                           tag=i+1,
                           kRef=kRef,
                           rad=rads[i],
                           phys=phys[i],
                           center=centers[i],
    )
    dgen = generate_concentric_dict(conf)
    cmds = write_params_geo(conf)
    call(cmds)

    dgens.append(dgen)

doms = merge_msh_bubbles(dgens)

myd = Domains(doms)
myd.write2dot('graph.dot')
call(['dot', '-Teps', 'graph.dot'], stdout=open('graph.eps', 'wb'))

dd = myd

print(N)

##################################

meshname = "./geo/all.msh"


mtf = MultiTrace(kRef, meshname, dd)

At, X, J, iJ = mtf.tolinop()

shape = mtf.shape

A = 2.0 * At
A2 = A * iJ * A

Ce = 0.5 * J - At
Ci = 0.5 * J + At

Ce2 = Ce * iJ * Ce
Ci2 = Ci * iJ * Ci

x = np.random.rand(shape[0]) + 1j * np.random.rand(shape[0])
xr = np.random.rand(shape[0])

checker('A2 = J', A2, J, x)
checker('exterior Proj.', Ce2, Ce, x)
checker('interior Proj.', Ci2, Ci, x)
checker('error-Calderon with random [no-sense]', A, J, x)

#################################################
#################################################

def dir_data(x, normal, dom_ind, result):
    result[0] =  -np.exp( 1j * kRef * x[1])

def neu_data(x, normal, dom_ind, result):
    result[0] = -1j * normal[1] * kRef * np.exp( 1j * kRef * x[1])

#################################################
#################################################

b = mtf.rhs(dir_data, neu_data)
M = A - X

print('')
print(mtf.shape, flush=True)
print('')

#################################################
#################################################
#################################################

iA = iJ * A * iJ

#################################################

Pjac = iA

E = mtf.upper()
Pgs = iA + iA * E * iA


CCi = iJ * (0.5 * J + At)
CCe = (0.5 * J - At)

B = J - X
BD = B * CCi + CCe
F = B * CCi
