
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

def check_centers(p, r, centers, rads):
    x, y, z = p
    for cc, rr in zip(centers, rads):
        xx, yy, zz = cc
        d = np.sqrt( (x-xx)**2 + (y-yy)**2 + (z-zz)**2 )
        #print(d, r+rr)
        if 0.99*d < r+rr:
            # print('false')
            return False
    # print('true')
    return True

def gen_centers(rads, Lx=1., Ly=1., Lz=1.):
    N = len(rads)
    M = 5 * N

    n = 0
    centers = []
    for r in rads:
        for ntry in range(M):
            X, Y, Z = np.random.rand(N), np.random.rand(N), np.random.rand(N)
            X, Y, Z = Lx * X, Ly * Y, Lz * Z
            for x, y, z in zip(X, Y, Z):
                tf = check_centers((x, y, z), r, centers, rads)
                if tf is True:
                    centers.append((x, y, z))
                    n += 1
                    if n == N:
                        return centers
    print('#centers:', len(centers))
    return centers

##################################################
##################################################

N = 5
kRef = 0.01 * np.pi

Lx, Ly, Lz = 5, 5, 5

rads = gen_rads(N)
phys = gen_phys(N)

centers = gen_centers(rads, Lx, Ly, Lz)

N = len(centers)
rads = rads[:N]
phys = phys[:N]


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
