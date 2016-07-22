
import numpy as np
import numpy.linalg as la

from subprocess import call

from domains import sanitize_config
from domains import generate_concentric_dict
from domains import write_params_geo
from domains import merge_msh_bubbles
from domains import Domains

def gen_new_rad():
    r = 0.2
    return r

def gen_rads(N):
    rads = [ gen_new_rad() for i in range(N) ]
    return rads

def gen_new_phy():
    p = (2, 1, 1)
    return p

def gen_phys(N):
    phys = [ gen_new_phy() for i in range(N) ]
    return phys

def gen_centers(Nx, Ny=None, Nz=None, Lx=1., Ly=1., Lz=1.):
    if Ny is None:
        Ny = Nx
    if Nz is None:
        Nz = Nx

    r = gen_new_rad()

    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

    if 0.99*dx < 2*r:
        raise ValueError('too dense in x-direction')
    if 0.99*dy < 2*r:
        raise ValueError('too dense in y-direction')
    if 0.99*dz < 2*r:
        raise ValueError('too dense in z-direction')

    n = 0
    centers = []
    for ix in range(Nx):
        x = ix * dx
        for iy in range(Ny):
            y = iy * dy
            for iz in range(Nz):
                z = iz * dz
                centers.append((x, y, z))
                n += 1
    print('#centers:', len(centers), Nx*Ny*Nz)
    return centers



N = 7
kRef = 0.01 * np.pi

Lx, Ly, Lz = 50, 50, 50


centers = gen_centers(N, N, N, Lx, Ly, Lz)

N = len(centers)

rads = gen_rads(N)
phys = gen_phys(N)

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
