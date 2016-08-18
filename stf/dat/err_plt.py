#!/usr/bin/env python
# coding: utf8

import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

mat = sio.loadmat('err_stf.mat')

k = mat['kRef'][0, 0]
eps = mat['eps'][0, 0]
Ndom = mat['Ndom'][0, 0]

Size = mat['Size']
H = mat['H']
Ecald, Etrans = mat['Ecald'], mat['Etrans']
Ecald_mie, Etrans_mie = mat['Ecald_mie'], mat['Etrans_mie']
dEL2, dEl2 = mat['dEL2'], mat['dEl2']
nEL2, nEl2 = mat['nEL2'], mat['nEl2']
dEnL2, dEnl2 = mat['dEnL2'], mat['dEnl2']
nEnL2, nEnl2 = mat['nEnL2'], mat['nEnl2']

Ecald_stf, Etrans_stf = mat['Ecald_stf'], mat['Etrans_stf']
dEL2_stf, dEl2_stf = mat['dEL2_stf'], mat['dEl2_stf']
nEL2_stf, nEl2_stf = mat['nEL2_stf'], mat['nEl2_stf']
dEnL2_stf, dEnl2_stf = mat['dEnL2_stf'], mat['dEnl2_stf']
nEnL2_stf, nEnl2_stf = mat['nEnL2_stf'], mat['nEnl2_stf']

shaper = lambda ar: ar.reshape((ar.shape[1],))

Size = shaper(Size)
H = shaper(H)
Ecald, Etrans = shaper(Ecald), shaper(Etrans)
Ecald_mie, Etrans_mie = shaper(Ecald_mie), shaper(Etrans_mie)
dEL2, dEl2 = shaper(dEL2), shaper(dEl2)
nEL2, nEl2 = shaper(nEL2), shaper(nEl2)
dEnL2, dEnl2 = shaper(dEnL2), shaper(dEnl2)
nEnL2, nEnl2 = shaper(nEnL2), shaper(nEnl2)

Ecald_stf, Etrans_stf = shaper(Ecald_stf), shaper(Etrans_stf)
dEL2_stf, dEl2_stf = shaper(dEL2_stf), shaper(dEl2_stf)
nEL2_stf, nEl2_stf = shaper(nEL2_stf), shaper(nEl2_stf)
dEnL2_stf, dEnl2_stf = shaper(dEnL2_stf), shaper(dEnl2_stf)
nEnL2_stf, nEnl2_stf = shaper(nEnL2_stf), shaper(nEnl2_stf)

iSize = 1. / Size
iH = 1. / H

Nlambdas = shaper(mat['Nlambdas'])

myfig = plt.figure(1)
lw, ms = 3, 10

ax = myfig.add_subplot(111)

point = np.exp(np.log(10) * (np.log10(dEnl2[-1]) - 2 * np.log10(H[-1])))
ax.loglog(iH, H**2 * point, 'k--', linewidth=0.5, markersize=ms)

point = np.exp(np.log(10) * (np.log10(dEnl2[0]) - np.log10(H[0])))
ax.loglog(iH, H * point, 'k-.', linewidth=1.5, markersize=ms)

##

point = np.exp(np.log(10) * (np.log10(dEnl2_stf[-1]) - 2 * np.log10(H[-1])))
ax.loglog(iH, H**2 * point, 'k--', linewidth=0.5, markersize=ms)

point = np.exp(np.log(10) * (np.log10(dEnl2_stf[0]) - np.log10(H[0])))
ax.loglog(iH, H * point, 'k-.', linewidth=1.5, markersize=ms)

##

if dEnl2[0] < 1e6:
    ax.loglog(iH, dEnl2, 'rd', linewidth=lw, markersize=ms, label='MTF norm. l2 Dir.')
    ax.loglog(iH, dEnL2, 'r-', linewidth=lw, markersize=ms, label='MTF norm. L2 Dir.')

if dEnl2_stf[0] < 1e6:
    ax.loglog(iH, dEnl2_stf, 'mv', linewidth=lw, markersize=ms, label='STF norm. l2 Dir.')
    ax.loglog(iH, dEnL2_stf, 'm-', linewidth=lw, markersize=ms, label='STF norm. L2 Dir.')


if nEnl2[0] < 1e6:
    ax.loglog(iH, nEnl2, 'bd', linewidth=lw, markersize=ms, label='MTF norm. l2 Neu.')
    ax.loglog(iH, nEnL2, 'b-', linewidth=lw, markersize=ms, label='MTF norm. L2 Neu.')

if nEnl2_stf[0] < 1e6:
    ax.loglog(iH, nEnl2_stf, 'cv', linewidth=lw, markersize=ms, label='STF norm. l2 Neu.')
    ax.loglog(iH, nEnL2_stf, 'c-', linewidth=lw, markersize=ms, label='STF norm. L2 Neu.')

# ax.loglog(iH, Ecald, 'm-', linewidth=lw, markersize=ms, label='Calderon l2 Sol.')
# ax.loglog(iH, Etrans, 'mv', linewidth=lw, markersize=ms, label='Transmission l2 Sol.')

# ax.loglog(iH, Ecald_mie, 'c-', linewidth=lw, markersize=ms, label='Calderon l2 Mie')
# ax.loglog(iH, Etrans_mie, 'c*',linewidth=lw, markersize=ms, label='Transmission l2 Mie')

ax.set_xlabel('1/h')
ax.set_ylabel('Error')

ax.grid(True, which="both")

#ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.legend(loc=3)

# axx = ax.twiny()
# axx.set_xticks(Nlambdas)
# axx.set_xticklabels(Nlambdas)
# axx.set_xlabel(r"Mesh Density: $\frac{\lambda}{n_\lambda}$")

#myfig.show()
myfig.savefig('err_eps{0}_k{1}_Ndom{2}.eps'.format(eps, k, Ndom))
myfig.savefig('err_conv.eps')

# sio.savemat('err_eps{0}_k{1}_Ndom{2}.mat'.format(eps, k, Ndom),
#             {'kRef':k, 'eps':eps, 'Ndom':Ndom,
#              'Size':Size,
#              'Nlambdas':Nlambdas,
#              'Ecald':Ecald, 'Etrans':Etrans,
#              'Ecald_mie':Ecald_mie, 'Etrans_mie':Etrans_mie,
#              'dEL2': dEL2, 'nEL2':nEL2,
#              'dEl2': dEl2, 'nEl2':nEl2,
#              'dEnL2':dEnL2, 'nEnL2':nEnL2,
#              'dEnl2':dEnl2, 'nEnl2':nEnl2,
#             })
