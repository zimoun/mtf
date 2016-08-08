#!/usr/bin/env python
# coding: utf8

import numpy as np
from matplotlib import pylab as pl

x = np.linspace(-5, 5, 200)


fp = lambda x: -1.0 + x + np.sqrt(1.0 + x**2)
fm = lambda x: -1.0 + x - np.sqrt(1.0 + x**2)

def cond(xx, fp, fm):
    p = lambda x: np.abs(fp(x))
    m = lambda x: np.abs(fm(x))
    c = []
    for x in xx:
        if m(x) == 0.0 or p(x) == 0.0:
            c.append(0)
        if p(x) >= m(x):
            c.append(p(x) / m(x))
        else:
            c.append(m(x) / p(x))
    return np.array(c)


pl.figure(1)

pl.plot(x, fp(x), 'b-', label='+')
pl.plot(x, fm(x), 'r-', label='-')

pl.legend()
pl.grid()
pl.title(r'$f_{\pm}: IR \rightarrow IR$')
#pl.show()

####
pl.figure(2)

z = x
xx, y = np.abs(z), cond(z, fp, fm)
pl.plot(x, y, 'b-', label='I=real')

z = 1j * x
xx, y =  np.abs(z), cond(z, fp, fm)
pl.plot(xx, y, 'r-', label='I=imag')

z = (0.2 + 1j) * x
xx, y = np.abs(z), cond(z, fp, fm)
pl.plot(xx, y, 'k-', label='I=cplx')

pl.legend()
pl.grid()
pl.title(r'conditionning number $\kappa: I \rightarrow IR$')


###
pl.figure(3)

theta = np.linspace(-1.0 * np.pi, 1.0 * np.pi, 2000, endpoint=True)

eio = 1.0 * np.exp(1j * theta)

x = np.real(eio)
y = np.imag(eio)
# pl.plot(x, y, 'b-')

v = fp(eio)
x = np.real(v)
y = np.imag(v)
pl.plot(x, y, 'b-', label='+')

v = fm(eio)
x = np.real(v)
y = np.imag(v)
pl.plot(x, y, 'r-', label='-')


v = cond(eio, fp, fm)
x = v
y = v
pl.plot(x, y, 'kx', label='c')

pl.grid()
pl.legend()
pl.title(r'$f_{\pm},\kappa: e^{i\theta} \rightarrow IC$ with $\theta\in[-\pi/2,\pi/2]$')

###
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

fig = pl.figure(30)
ax = fig.gca(projection='3d')

theta = np.linspace(-1.0 * np.pi, 1.0 * np.pi, 2000, endpoint=True)
R = np.linspace(0.25, 2, 10, endpoint=True)

for r in R:
    eio = r * np.exp(1j * theta)
    v = cond(eio, fp, fm)
    x = np.real(eio)
    y = np.imag(eio)
    ax.plot(x, y, v, label='{}'.format(r))
ax.grid()
ax.legend()


###
pl.figure(4)

theta = np.linspace(-1.0 * np.pi, 1.0 * np.pi, 2000, endpoint=True)

eio = 1. * np.exp(1j * theta)

v = cond(eio, fp, fm)
x = np.real(eio)
y = v
pl.plot(x, y, 'bo-', label='x-real')

pl.plot(x, x, 'b--', label='x-reall')


v = cond(eio, fp, fm)
x = np.imag(eio)
y = v
pl.plot(x, y, 'r-', label='x-imag')
pl.plot(x, x, 'r--', label='x-imagg')


pl.legend()
pl.grid()
pl.show()
