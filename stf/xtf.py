#!/usr/bin/env python
# coding: utf8

from time import time
from subprocess import call

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt

import bempp.api as bem
from krylov import gmres

#bem.global_parameters.assembly.boundary_operator_assembly_type = 'dense'
#bem.global_parameters.assembly.boundary_operator_assembly_type = 'hmat'
#bem.global_parameters.hmat.coarsening = False

#bem.global_parameters.assembly.enable_interpolation_for_oscillatory_kernels = True
#bem.api.global_parameters.quadrature.double_singular = 320

class Bridge:
    def __init__(self, shape, dot, dtype):
        self.shape = shape
        self.pot = dot
        # hack because the type-system of bem++ is weird
        self.dtype = np.dtype(dtype)

    def dot(self, x):
        return self.pot(x)

class xTF:
    def __init__(self, kcomplex, eps, nlambda,
                 geofile='sphere-simple.script.geo', meshname='sphere.msh'):
        self.ks = [kcomplex]
        self.kExt = kcomplex
        self.eps = eps
        self.nlambda = nlambda
        self.kInt = kcomplex * np.sqrt(eps)

        self.geofile = geofile
        self.meshname = meshname

        with open('in.geo', 'w') as fp:
            fp.write("k = {};".format(np.abs(self.kExt)))
            fp.write("eps = {};".format(np.abs(self.eps)))
            fp.write("nlambda = {};".format(np.abs(self.nlambda)))
        with open('out.geo', 'w') as fp:
            fp.write('\nSave "{}";\n'.format(meshname))
        call(['gmsh', geofile, '-'])

        self.grid = grid = bem.import_grid(meshname)
        self.space = space = bem.function_space(grid, "P", 1)
        self.shape = self.space.global_dof_count

        self.collect()
        self.weak_form()

    def collect(self):
        print('collecting...', end=' ', flush=True)
        kExt, kInt = self.kExt, self.kInt
        space = self.space

        K0 = bem.operators.boundary.helmholtz.double_layer(
            space, space, space, kExt)
        self.V0 = bem.operators.boundary.helmholtz.single_layer(
            space, space, space, kExt)
        self.W0 = bem.operators.boundary.helmholtz.hypersingular(
            space, space, space, kExt)
        Q0 = bem.operators.boundary.helmholtz.adjoint_double_layer(
            space, space, space, kExt)

        # # normal orientation
        self.K0, self.Q0 = -K0, -Q0
        #self.K0, self.Q0 = K0, Q0

        self.K1 = bem.operators.boundary.helmholtz.double_layer(
            space, space, space, kInt)
        self.V1 = bem.operators.boundary.helmholtz.single_layer(
            space, space, space, kInt)
        self.W1 = bem.operators.boundary.helmholtz.hypersingular(
            space, space, space, kInt)
        self.Q1 = bem.operators.boundary.helmholtz.adjoint_double_layer(
            space, space, space, kInt)

        self.Id = bem.operators.boundary.sparse.identity(
            space, space, space)
        print('done.', flush=True)


    def uncollect(self):
        print('uncollecting...', end=' ', flush=True)
        for op, who in zip([self.K0, self.V0, self.W0, self.Q0,
                            self.K1, self.V1, self.W1, self.Q1,
                            self.Id],
                           ['K0', 'V0', 'W0', 'Q0',
                            'K1', 'V1', 'W1', 'Q1',
                            'Id']):
            print(who, end=' ', flush=True)
            op = None
        print('done.', flush=True)

    def weak_form(self):
        print('Assembling...', end=' ', flush=True)
        for op, who in zip([self.K0, self.V0, self.W0, self.Q0,
                            self.K1, self.V1, self.W1, self.Q1,
                            self.Id],
                           ['K0', 'V0', 'W0', 'Q0',
                            'K1', 'V1', 'W1', 'Q1',
                            'Id']):
            print(who, end=' ', flush=True)
            a = op.weak_form()
        print('done.', flush=True)

    def update(self, kcomplex):
        self.ks.append(kcomplex)
        self.kExt = kcomplex
        self.kInt = kcomplex * np.sqrt(self.n)

        self.uncollect()
        self.collect()
        self.weak_form()

    def setRHS(self, dir_data, neu_data):
        self.fdir = bem.GridFunction(self.space, fun=dir_data)
        self.fneu = bem.GridFunction(self.space, fun=neu_data)

    def getDir(self):
        return self.fdir.projections()
    def getNeu(self):
        return self.fneu.projections()
    def getGFdir(self):
        return self.fdir
    def getGFneu(self):
        return self.fneu

    def get_A0(self):
        K0, V0 = self.K0, self.V0
        W0, Q0 = self.W0, self.Q0
        A0 = bem.BlockedOperator(2, 2)
        A0[0, 0], A0[0, 1] = -K0, V0
        A0[1, 0], A0[1, 1] = W0, Q0
        return A0

    def get_A1(self):
        K1, V1 = self.K1, self.V1
        W1, Q1 = self.W1, self.Q1
        A1 = bem.BlockedOperator(2, 2)
        A1[0, 0], A1[0, 1] = -K1, V1
        A1[1, 0], A1[1, 1] = W1, Q1
        return A1

    def get_IId(self):
        Id = self.Id
        IId = bem.BlockedOperator(2, 2)
        IId[0, 0], IId[1, 1] = Id, Id
        return IId

    def get_iJlu(self):
        Id = self.Id
        Idw = Id.weak_form()
        mId = Idw.sparse_operator
        mId = mId.astype(complex)
        iId = spla.splu(mId)
        iJlu = spla.LinearOperator(iId.shape, matvec=iId.solve, dtype=complex)
        return iJlu



class iSTF:
    def __init__(self, xtf):
        self.xtf = xtf

    def get(self):
        K0, V0 = self.xtf.K0, self.xtf.V0
        W0, Q0 = self.xtf.W0, self.xtf.Q0
        K1, V1 = self.xtf.K1, self.xtf.V1
        W1, Q1 = self.xtf.W1, self.xtf.Q1
        W0, W1 = -W0, -W1
        iSTF = bem.BlockedOperator(2, 2)
        iSTF[0, 0] = V1 - V0
        iSTF[0, 1] = -K1 + K0
        iSTF[1, 0] = Q1 - Q0
        iSTF[1, 1] = -W1 + W0
        istf = iSTF.weak_form()
        return istf


class STF:
    def __init__(self, xtf):
        self.xtf = xtf
        self.space = xtf.space

    def get(self):
        K0, V0 = self.xtf.K0, self.xtf.V0
        W0, Q0 = self.xtf.W0, self.xtf.Q0
        K1, V1 = self.xtf.K1, self.xtf.V1
        W1, Q1 = self.xtf.W1, self.xtf.Q1
        STF = bem.BlockedOperator(2, 2)
        STF[0, 0] = -K0 + K1
        STF[0, 1] = V0 + V1
        STF[1, 0] = W0 + W1
        STF[1, 1] = Q0 - Q1
        stf = STF.weak_form()
        return stf

    def rhs(self):
        Id = self.xtf.Id
        K1, V1 = self.xtf.K1, self.xtf.V1
        W1, Q1 = self.xtf.W1, self.xtf.Q1

        fdir, fneu = self.xtf.fdir, self.xtf.fneu

        a = (0.5 * Id + K1) * fdir - V1 * fneu
        b = W1 * fdir + (-0.5 * Id + Q1) * fneu
        a, b = a.projections(), b.projections()
        rhs = np.concatenate((a, b))
        return rhs

class Domains:
    """ hack for compatibility """
    def __init__(self):
        self.tmp = 0
    def getIndexDom(self, ind):
        return self.tmp

class MTF:
    def __init__(self, xtf):
        self.xtf = xtf
        self.domains = Domains()
        self.spaces = [
            ((xtf.space, xtf.space), (xtf.space, xtf.space)),
            ((xtf.space, xtf.space), (xtf.space, xtf.space))
        ]

    def getSlices(self):
        d = { '0': (0, 2*self.xtf.shape), '1': (2*self.xtf.shape, 4*self.xtf.shape) }
        return d

    def get_J(self):
        IId = self.xtf.get_IId()
        J = bem.BlockedOperator(2, 2)
        J[0, 0], J[1, 1] = IId, IId
        Jw = J.weak_form()
        return Jw

    def get_iJ(self):
        iJlu = self.xtf.get_iJlu()
        iJ0 = bem.BlockedDiscreteOperator(2, 2)
        iJ0[0, 0], iJ0[1, 1] = iJlu, iJlu
        iJ1 = bem.BlockedDiscreteOperator(2, 2)
        iJ1[0, 0], iJ1[1, 1] = iJlu, iJlu
        iJ = bem.BlockedDiscreteOperator(2, 2)
        iJ[0, 0], iJ[1, 1] = iJ0, iJ1
        return iJ

    def get_A(self):
        A0 = self.xtf.get_A0()
        A1 = self.xtf.get_A1()
        A = bem.BlockedOperator(2, 2)
        A[0, 0], A[1, 1] = A0, A1
        Aw = A.weak_form()
        return Aw

    def get_X(self):
        Id = self.xtf.Id
        X01, X10 = bem.BlockedOperator(2, 2), bem.BlockedOperator(2, 2)
        X01[0, 0], X01[1, 1] = Id, -Id
        X10[0, 0], X10[1, 1] = Id, -Id
        X = bem.BlockedOperator(2, 2)
        X[0, 1], X[1, 0] = X01, X10
        Xw = X.weak_form()
        return Xw

    def tolinop(self):
        Aw = self.get_A()
        At = spla.LinearOperator(Aw.shape, matvec=Aw.matvec, dtype=complex)

        Xw = self.get_X()
        X = spla.LinearOperator(Xw.shape, matvec=Xw.matvec, dtype=complex)

        Jw = self.get_J()
        J = spla.LinearOperator(Jw.shape, matvec=Jw.matvec, dtype=complex)

        iJw = self.get_iJ()
        iJ = spla.LinearOperator(iJw.shape, matvec=iJw.matvec, dtype=complex)

        self.shape = Aw.shape

        return At, X, J, iJ

        MTF = A - 0.5 * X
        mtf = MTF.weak_form()
        return mtf

    def rhs(self, dir_data, neu_data):
        fdir, fneu = self.xtf.fdir, self.xtf.fneu
        a, b = fdir.projections(), fneu.projections()
        rhs = np.concatenate((a, -b, -a, -b))
        return rhs
