

import numpy as np
import matplotlib.pyplot as plt

from mesh import Mesh
from miesphere import mie_sphere

m = Mesh('geo/sphere-disjoint.msh', 1)

R = 1.
epsExt, epsInt = 1., 4.
ks = (np.sqrt(epsExt), np.sqrt(epsInt))
jumps = ((1., 1.), (1., 1.))

params = R, ks, jumps

lmbda = 0.3
k = 2 * np.pi / lmbda
kk_high = (k, 0., 0.)

lmbda = 1.
k = 2 * np.pi / lmbda
kk_medium = (k, 0., 0.)

lmbda = 50.
k = 2 * np.pi / lmbda
kk_low = (k, 0., 0.)

# Nref = 20
# Params_high = Nref, params, kk_high
# Params_medium = Nref, params, kk_medium
# Params_low = Nref, params, kk_low

# scaRef_high = mie_sphere(m, Params_high, 'mie{}-sca_high.pos'.format(Nref), field='sca')
# scaRef_medium = mie_sphere(m, Params_medium, 'mie{}-sca_medium.pos'.format(Nref), field='sca')
# scaRef_low = mie_sphere(m, Params_low, 'mie{}-sca_low.pos'.format(Nref), field='sca')


sca_high = np.zeros(len(m.points))
sca_medium = np.zeros(len(m.points))
sca_low = np.zeros(len(m.points))

ns = []
rh, rm, rl = [], [] ,[]
for n in range(1, 30):

    sh = sca_high
    sm = sca_medium
    sl = sca_low

    Params_high = n, params, kk_high
    Params_medium = n, params, kk_medium
    Params_low = n, params, kk_low
    N = n - 1

    sca_high = mie_sphere(m, Params_high, 'mie{}-sca_high.pos'.format(N), field='sca')
    sca_medium = mie_sphere(m, Params_medium, 'mie{}-sca_medium.pos'.format(N), field='sca')
    sca_low = mie_sphere(m, Params_low, 'mie{}-sca_low.pos'.format(N), field='sca')

    dh = np.linalg.norm(np.array(sca_high) - sh)
    dm = np.linalg.norm(np.array(sca_medium) - sm)
    dl = np.linalg.norm(np.array(sca_low) - sl)


    ns.append(N)
    rh.append(dh)
    rm.append(dm)
    rl.append(dl)

ns = np.array(ns)
rh = np.array(rh)
rm = np.array(rm)
rl = np.array(rl)

plt.semilogy(ns, rh, 'ko-', label='High Freq.')
plt.semilogy(ns, rm, 'bs-', label='Medium Freq.')
plt.semilogy(ns, rl, 'gd-', label='Low Freq.')

plt.legend()
plt.title('R=1, k1=2k0')
plt.xlabel('#mode')
plt.ylabel('norm(un)')
plt.show()
