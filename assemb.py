# coding: utf8

from time import time
import warnings

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import bempp.api as bem

from domains import Domains

class Bridge:
    def __init__(self, shape, dot, dtype):
        self.shape = shape
        self.pot = dot
        # hack because the type-system of bem++ is weird
        self.dtype = np.dtype(dtype)

    def dot(self, x):
        return self.pot(x)

class MultiTrace:

    def __init__(self, kRef, meshname, doms,
                 J_is='BlockedDiscrete',
                 X_is='BlockedDiscrete',
                 use_slp=True):

        if isinstance(doms, list):
            domains = Domains(doms)
        elif isinstance(doms, Domains):
            domains = doms
        else:
            raise TypeError('configuration needs to be a list or a Domains class')

        self._collected = False

        self._assembled = False
        self._A_assembled = False
        self._X_assembled = False
        self._J_assembled = False
        self._iJ_assembled = False

        if not J_is in ['BlockedDiscrete', 'CSC', 'Blocked']:
            warnings.warn('{0} is not supported. Default used {1}'.format(J_is,
                                                                         'BlockedDiscrete'))
            J_is = 'BlockedDiscrete'
        self._J_is = J_is

        if not X_is in ['BlockedDiscrete', 'Blocked']:
            warnings.warn('{0} is not supported. Default used {1}'.format(J_is,
                                                                         'BlockedDiscrete'))
            X_is = 'BlockedDiscrete'
        self._X_is = X_is

        self.use_slp = use_slp

        print('==J_is: {0} , X_is: {1} , use_slp={2}'.format(J_is, X_is,
                                                             use_slp))


        self.domains = domains
        self.N = len(domains)

        print('==Importing Grid/Mesh {}'.format(meshname), end=' ', flush=True)
        self.grid = grid = bem.import_grid(meshname)
        print('done.', flush=True)

        N = self.N
        self.opA = bem.BlockedOperator(N, N)
        self.opX = bem.BlockedOperator(N, N)
        self.opI = bem.BlockedOperator(N, N)

        if np.abs(kRef) == 0.0:
            kernel = 'lapl'
            self.kRef = 0.0
            dtype = np.float
        else:
            kernel = 'helm'
            self.kRef = kRef #float(kRef)
            dtype = np.complex
        self.kernel = kernel

        if not dtype is None:
            if kernel == 'helm':
                if dtype != np.complex:
                    warnings.warn('Helmholtz is complex. dtype={}'.format(np.complex))
                    dtype = np.complex
            else:
                if dtype != np.float:
                    warnings.warn('Unsupported dtype. Converted to {}'.format(np.float))
                    dtype = np.float
        self.dtype = dtype

        if kernel == 'helm':
            funK = lambda trial, ran, test, k: bem.operators.boundary.helmholtz.double_layer(trial, ran, test, k)
            funV = lambda trial, ran, test, k: bem.operators.boundary.helmholtz.single_layer(trial, ran, test, k)
            funQ = lambda trial, ran, test, k: bem.operators.boundary.helmholtz.adjoint_double_layer(trial, ran, test, k)
            funW = lambda trial, ran, test, k: bem.operators.boundary.helmholtz.hypersingular(trial, ran, test, k, use_slp=use_slp)
        else:
            funK = lambda trial, ran, test, k: bem.operators.boundary.laplace.double_layer(trial, ran, test)
            funV = lambda trial, ran, test, k: bem.operators.boundary.laplace.single_layer(trial, ran, test)
            funQ = lambda trial, ran, test, k: bem.operators.boundary.laplace.adjoint_double_layer(trial, ran, test)
            funW = lambda trial, ran, test, k: bem.operators.boundary.laplace.hypersingular(trial, ran, test, use_slp=use_slp)

        self._funK, self._funV = funK, funV
        self._funW, self._funQ = funW, funQ

        self._funI = bem.operators.boundary.sparse.identity

        self.spaces = [ (
                ('test_d', 'trial_d'),
                ('test_n', 'trial_n')
                ) for d in domains ]


    def collecting(self):
        if self._collected:
            print('Already collected')
            return self.tcollect

        tinit = time()

        space = bem.function_space

        dtype = self.dtype

        grid = self.grid
        kRef = self.kRef
        domains = self.domains

        opA = self.opA
        opX = self.opX
        opI = self.opI

        funK = self._funK
        funV = self._funV
        funW = self._funW
        funQ = self._funQ
        funI = self._funI

        nrow, ncol = 0, 0
        print('\n=Collecting all the blocks')
        for dom in domains:
            ii = domains.getIndexDom(dom['name'])

            eps, alpha, beta = dom['phys']
            k = kRef * np.sqrt(eps)

            sig = dom['sign']

            print('==Domain: {0}'.format(dom['name']))
            print('====info {0}: #{1}, eps={2}, (alpha={3}, beta={4}), sig={5}'.format(
                    dom['name'], ii, eps, alpha, beta, sig))
            print('===Diag: Block #({0}, {0})'.format(ii))

            opAA = bem.BlockedOperator(2, 2)
            opII = bem.BlockedOperator(2, 2)

            space_trial_d = space(grid, "P", 1, domains=dom['interfaces'])

            if self.use_slp:
                space_trial_n = space_trial_d
                space_range_d, space_range_n = space_trial_d, space_trial_n
                space_test_d, space_test_n = space_trial_d, space_trial_n

            else:
                space_trial_n = space(grid, "P", 1, domains=dom['interfaces'])

                space_range_d = space(grid, "P", 1, domains=dom['interfaces'])
                space_range_n = space(grid, "P", 1, domains=dom['interfaces'])

                space_test_d = space(grid, "P", 1, domains=dom['interfaces'])
                space_test_n = space(grid, "P", 1, domains=dom['interfaces'])

            space_d = (space_test_d, space_trial_d)
            space_n = (space_test_n, space_trial_n)
            spaces = (space_d, space_n)
            self.spaces[ii] = spaces

            ncol += space_test_d.global_dof_count + space_test_n.global_dof_count
            nrow += space_trial_d.global_dof_count + space_trial_n.global_dof_count

            # the kernel type is managed in __init__
            opK = funK(space_trial_d, space_range_d, space_test_d, k)
            opV = funV(space_trial_n, space_range_d, space_test_d, k)
            opW = funW(space_trial_d, space_range_n, space_test_n, k)
            opQ = funQ(space_trial_n, space_range_n, space_test_n, k)

            opId = funI(space_trial_d, space_range_d, space_test_d)
            opIn = funI(space_trial_n, space_range_n, space_test_n)

            opAA[0, 0] = - sig * opK
            opAA[0, 1] = opV
            opAA[1, 0] = opW
            opAA[1, 1] = sig * opQ

            opII[0, 0] = opId
            opII[1, 1] = opIn

            opA[ii, ii] = opAA
            opI[ii, ii] = opII

            for d in domains.getNeighborOf(dom['name']):
                jj = domains.getIndexDom(d['name'])
                print('===Coupling {0} with {1}: Block #({2}, {3})'.format(dom['name'],
                                                                           domains.getName(jj),
                                                                           ii, jj))

                _, alph, bet = d['phys']

                opXX = bem.BlockedOperator(2, 2)

                space_trial_d = space(grid, "P", 1, domains=d['interfaces'])
                space_trial_n = space(grid, "P", 1, domains=d['interfaces'])

                space_range_d = space(grid, "P", 1, domains=d['interfaces'])
                space_range_n = space(grid, "P", 1, domains=d['interfaces'])

                opXd = funI(space_trial_d, space_range_d, space_test_d)
                opXn = funI(space_trial_n, space_range_n, space_test_n)

                print('====coeffs jumps: alpha_j/i={0:e}  beta_j/i={1:e}'.format(alph/alpha,
                                                                             -bet/beta))

                opXX[0, 0] = (alph/alpha) * opXd
                opXX[1, 1] = - (bet/beta) * opXn

                opX[ii, jj] = opXX


        self.opA = opA
        self.opX = opX
        self.opI = opI

        self.shape = (nrow, ncol)

        self._collected = True
        self.tcollect = time() - tinit
        return self.tcollect


    def _check_shape(self, shape):
        if hasattr(self, 'shape'):
            if shape != self.shape:
                raise ValueError('Inconsistent shape')
        else:
            self.shape = shape

    def _collect(self):
        if not self._collected:
            self.tcollect = self.collecting()


    def A_weak_form(self):
        self._collect()

        if self._A_assembled:
            return self.Aw

        tinit = time()
        opA = self.opA
        N = self.N
        print('==BlockDiag assembling: A (be patient)')
        for ii in range(N):
            tt = time()
            print('===Block: #({0}, {0})'.format(ii), end=' ')
            opp = opA[ii, ii]
            for i, j, who in zip([0, 0, 1, 1],
                                 [0, 1, 0, 1],
                                 ['K', 'V', 'W', 'Q']):
                print(who, end=' ', flush=True)
                op = opp[i, j]
                a = op.weak_form()
            print(' time: {}'.format(time() - tt))
        # if something is missing... to be sure !
        self.Aw = opA.weak_form()
        self._check_shape(self.Aw.shape)
        self._A_assembled = True
        self.tassembA = time() - tinit
        return self.Aw

    def X_weak_form(self, X_is='Blocked'):
        self._collect()

        if self._X_assembled:
            return self.Xw

        tinit = time()

        domains = self.domains
        dtype = self.dtype

        print('==Coupling assembling: X', end=' ')
        Xw = self.opX.weak_form()
        self._check_shape(Xw.shape)

        if X_is == 'Blocked':
            self._X_is = 'Blocked'
            self.Xw = Xw
            self._X_assembled = True
            self.tassembX = time() - tinit
            print('Blocked time: {0}'.format(time() - tinit))
            return self.Xw

        N = len(domains)
        for dom in domains:
            ii = domains.getIndexDom(dom['name'])
            for d in domains.getNeighborOf(dom['name']):
                jj = domains.getIndexDom(d['name'])
                Xb = Xw[ii, jj]
                Xd, Xn = Xb[0, 0], Xb[1, 1]

                xs = Xd.sparse_operator
                xs = xs.astype(dtype)
                Xd = Bridge(xs.shape, dot=xs.dot, dtype=dtype)

                xs = Xn.sparse_operator
                xs = xs.astype(dtype)
                Xn = Bridge(xs.shape, dot=xs.dot, dtype=dtype)

                Xloc = bem.BlockedDiscreteOperator(2, 2)
                Xloc[0, 0], Xloc[1, 1] = Xd, Xn

                Xw[ii, jj] = Xloc
        self.Xw = Xw
        self._X_assembled = True
        self._X_is = 'BlockedDiscrete'
        self.tassembX = time() - tinit
        print('BlockedDiscrete time: {0}'.format(time() - tinit))
        return self.Xw


    def J_weak_form(self, J_is='CSC'):
        self._collect()

        if self._J_assembled:
            return self.Jw

        tinit = time()

        domains = self.domains
        dtype = self.dtype

        print('==Identity assembling: J', end=' ', flush=True)
        Jw = self.opI.weak_form()
        self._check_shape(Jw.shape)

        if J_is == 'Blocked':
            self._J_is = 'Blocked'
            self.Jw = Jw
            self._J_assembled = True
            self.tassembJ = time() - tinit
            print('Blocked time: {0}'.format(time() - tinit))
            return self.Jw


        if J_is == 'BlockedDiscrete':
            self._J_is = 'BlockedDiscrete'
            N = len(domains)
            for ii in range(N):
                Jb = Jw[ii, ii]
                Jd, Jn = Jb[0, 0], Jb[1, 1]

                js = Jd.sparse_operator
                js = js.astype(dtype)
                Jd = Bridge(js.shape, dot=js.dot, dtype=dtype)

                js = Jn.sparse_operator
                js = js.astype(dtype)
                Jn = Bridge(js.shape, dot=js.dot, dtype=dtype)

                Jloc = bem.BlockedDiscreteOperator(2, 2)
                Jloc[0, 0], Jloc[1, 1] = Jd, Jn

                Jw[ii, ii] = Jloc
            self.Jw = Jw
            self._J_assembled = True
            self.tassembJ = time() - tinit
            print('BlockedDiscrete time: {0}'.format(time()-tinit))
            return self.Jw

        tt = time()
        Jsp = sp.lil_matrix(Jw.shape, dtype=np.float)
        row_start, col_start = 0, 0
        row_end, col_end = 0, 0
        for ii in range(len(domains)):
            Jb = Jw[ii, ii]
            Jd, Jn = Jb[0, 0], Jb[1, 1]

            #mat = bem.as_matrix(Jd)
            mat = Jd.sparse_operator.toarray()
            #mat = sp.lil_matrix(mat, dtype=dtype)
            r, c = mat.shape

            row_end += r
            col_end += c
            Jsp[row_start:row_end, col_start:col_end] = mat
            row_start, col_start = row_end, col_end

            #mat = bem.as_matrix(Jn)
            mat = Jn.sparse_operator.toarray()
            #mat = sp.lil_matrix(mat, dtype=dtype)
            r, c = mat.shape

            row_end += r
            col_end += c
            Jsp[row_start:row_end, col_start:col_end] = mat
            row_start, col_start = row_end, col_end
        Jsp = Jsp.astype(dtype)
        print('CSC time: {0}'.format(time()-tinit))
        tt = time()
        self.Jw = Jsp.tocsc()
        self._J_is = 'CSC'
        self._J_assembled = True
        self.tassembJ = time() - tinit
        print('##time convert Identity to {0} CSC: {1}'.format(dtype,
                                                               time()-tt))
        return self.Jw

    def iJ_weak_form(self):
        self._collect()

        if self._iJ_assembled:
            return self.iJlu

        tinit = time()

        print('==Factorization LU: J')

        if self._J_is == 'CSC':
            self.iJlu = spla.splu(self.Jw)
            self._iJ_assembled = True
            self.tassembiJ = time() - tinit
            print('##time CSC J=LU: {}'.format(self.tassembiJ))
            return self.iJlu

        Jw = self.opI.weak_form()

        N = len(self.domains)
        dtype = self.dtype

        iJlu = bem.BlockedDiscreteOperator(N, N)
        for ii in range(N):
            Jb = Jw[ii, ii]
            Jd, Jn = Jb[0, 0], Jb[1, 1]

            js = Jd.sparse_operator
            js = js.astype(dtype)
            js = spla.splu(js)
            iJd = Bridge(js.shape, dot=js.solve, dtype=dtype)

            js = Jn.sparse_operator
            js = js.astype(dtype)
            js = spla.splu(js)
            iJn = Bridge(js.shape, dot=js.solve, dtype=dtype)

            iJloc = bem.BlockedDiscreteOperator(2, 2)
            iJloc[0, 0], iJloc[1, 1] = iJd, iJn

            iJlu[ii, ii] = iJloc
        self.iJlu = iJlu
        self._iJ_assembled = True
        self.tassembiJ = time() - tinit

        print('##time blocked-discrete J=LU: {}'.format(self.tassembiJ))
        return self.iJlu


    def weak_form(self):
        self._collect()

        tinit = time()

        print('\n=Assembling all the matrices')
        self.A_weak_form()
        self.X_weak_form(X_is=self._X_is)
        self.J_weak_form(J_is=self._J_is)
        self.iJ_weak_form()
        tA, tX  = self.tassembA, self.tassembX
        tJ, tiJ = self.tassembJ, self.tassembiJ
        tassemb = time() - tinit
        if tassemb < tA + tX + tJ + tiJ:
            tassemb = tA + tX + tJ + tiJ
        self.tassemb = tassemb
        #
        print('')
        print('#total time Assembling: {0}'.format(tassemb))
        print('')
        #
        return tassemb


    def J_tolinop(self):
        self.J_weak_form(self._J_is)
        J_is = self._J_is
        if J_is == 'CSC':
            mv = self.Jw.dot
        else:
            mv = self.Jw.matvec
        J = spla.LinearOperator(self.shape,
                                matvec=mv,
                                dtype=self.dtype)
        return J

    def iJ_tolinop(self):
        self.iJ_weak_form()
        J_is = self._J_is
        if J_is == 'CSC':
            mv = self.iJlu.solve
        else:
            mv = self.iJlu.matvec
        iJ = spla.LinearOperator(self.shape,
                                matvec=mv,
                                dtype=self.dtype)
        return iJ

    def X_tolinop(self):
        self.X_weak_form(self._X_is)
        X = spla.LinearOperator(self.shape,
                                matvec=self.Xw.matvec,
                                dtype=self.dtype)
        return X

    def A_tolinop(self):
        self.A_weak_form()
        A = spla.LinearOperator(self.shape,
                                    matvec=self.Aw.matvec,
                                    dtype=self.dtype)
        return A



    def tolinop(self):
        self.weak_form()
        A, X = self.A_tolinop(), self.X_tolinop()
        J, iJ = self.J_tolinop(), self.iJ_tolinop()
        return A, X, J, iJ


    def upper(self):
        print('==building Upper: E', flush=True)
        domains = self.domains
        dtype = self.dtype
        N = len(domains)

        if self._J_is == 'CSC':
            Jw = self.opI.weak_form()
        else:
            Jw = self.J_weak_form(self._J_is)
        # Xw = self.X_weak_form(self._X_is)
        Xw = self.opX.weak_form()
        ## not nice, because the type conversion is done 2 times
        ## need to fix with the Bridge

        tt = time()
        E = bem.BlockedDiscreteOperator(N, N)
        for dom in domains:
            ii = domains.getIndexDom(dom['name'])
            Jb = Jw[ii, ii]
            es = Bridge(Jb.shape, dot=lambda x: 0.0 * x, dtype=dtype)
            E[ii, ii] = es

            for d in domains.getNeighborOf(dom['name']):
                jj = domains.getIndexDom(d['name'])
                if jj > ii:
                    Xij = Xw[ii, jj]
                    Xd, Xn = Xij[0, 0], Xij[1, 1]

                    ed = Xd.sparse_operator
                    ed = ed.astype(dtype)
                    en = Xn.sparse_operator
                    en = en.astype(dtype)

                    es = bem.BlockedDiscreteOperator(2, 2)
                    es[0, 0], es[1, 1] = ed, en

                    E[ii, jj] = es
        print('##time to build E: {}'.format(time() - tt), flush=True)
        E = spla.LinearOperator(self.shape,
                                matvec=E.matvec,
                                dtype=self.dtype)

        return E

        # tt = time()
        # Esp = sp.lil_matrix(self.Xw.shape, dtype=np.float)
        # row_start, row_end = 0, 0
        # for r in range(N):
        #     row, col = 0, 0
        #     col_start, col_end = 0, 0
        #     first = True
        #     for c in range(N):
        #         if first:
        #             op = opI[r, r]
        #             row, _ = op.weak_form().shape
        #             row_end += row
        #             first = False
        #         op = opX[r, c]
        #         if not op is None:
        #             mat = bem.as_matrix(op.weak_form())
        #             # mat = sp.lil_matrix(mat)
        #             # mat = op.weak_form().sparse_operator.toarray()
        #             _ , col = mat.shape
        #         else:
        #             opp =  opI[c, c]
        #             _ , col = opp.weak_form().shape
        #         col_end += col
        #         if c > r and (op is not None):
        #             Esp[row_start:row_end, col_start:col_end] = mat
        #         col_start = col_end
        #     row_start = row_end
        # print('===converting Upper: LIL to CSC', flush=True)
        # Esp = Esp.astype(dtype)
        # Esp = Esp.tocsc()
        # print('##time to build E: {}'.format(time() - tt), flush=True)
        # E = spla.LinearOperator(self.shape,
        #                         matvec=Esp.dot,
        #                         dtype=self.dtype)

        # return E


    ##################################
    def rhs(self, fdir, fneu, inf='0'):
        print('\n=RHS')

        def fzero(point, normal, dom_ind, result):
            result[0] = 0. + 1j * 0.

        dtype = self.dtype
        grid = self.grid
        N = self.N
        domains = self.domains

        space = bem.function_space
        grid_fun = bem.GridFunction

        for dom in domains:
            if dom['name'] == inf:
                _, alpha, beta = dom['phys']

        rhs = [] * N
        neighbors = domains.getNeighborOf(inf)
        for ii in range(N):
            name = domains.getName(ii)
            dom = domains.getEntry(name)
            jj = domains.getIndexDom(dom['name'])
            _, alph, bet = dom['phys']
            print('==Domain: {0} #{1}  \t (alpha={2}, beta={3})'.format(
                    dom['name'], ii, alph, bet))

            space_d = space(grid, "P", 1, domains=dom['interfaces'])
            space_n = space(grid, "P", 1, domains=dom['interfaces'])

            if dom['name'] == inf:
                diri = grid_fun(space_d, fun=fdir)
                neum = grid_fun(space_n, fun=fneu)
                idir, ineu = diri, - neum

            elif dom in neighbors:
                diri = grid_fun(space_d, fun=fdir)
                neum = grid_fun(space_n, fun=fneu)
                a, b = alpha / alph, beta / bet
                idir, ineu = - a * diri, - b * neum

            else:
                diri = grid_fun(space_d, fun=fzero)
                neum = grid_fun(space_n, fun=fzero)
                idir, ineu = diri, neum

            rhs.append(idir)
            rhs.append(ineu)

        tt = time()
        print('==Assembling RHS (projections)')
        b = np.array([], dtype=dtype)
        for r in rhs:
            b = np.concatenate((b, r.projections()))
        trhs = time() - tt
        print('#time Assembling RHS: {}'.format(trhs))
        return b


    def getSlices(self):
        if self._J_is == 'CSC':
            Jw = self.opI.weak_form()
        else:
            Jw = self.J_weak_form(self._J_is)

        domains = self.domains

        slices = {}
        start, end = 0, 0
        for ii in range(len(domains)):
            name = domains.getName(ii)
            s = Jw[ii, ii].shape
            if s[0] != s[1]:
                print('Warning: block #{0} = ({1}, {2}) rectangular'.format(ii, s[0], s[1]))
            end = start + s[1]
            slices[name] = (start, end)
            start = end
        self.slices = slices
        return slices



###########################
def checker(string, A, B, x, b=None):
    if not isinstance(A, spla.LinearOperator):
        raise TypeError('A has to be a LinearOperator')
    if not isinstance(B, spla.LinearOperator):
        raise TypeError('B has to be a LinearOperator')
    if A.shape != B.shape:
        raise ValueError('Inconsistent shape')

    print('==check ' + string)
    t0 = time()
    y = A(x)
    t1 = time() - t0

    t0 = time()
    z = B(x)
    t2 = time() - t0
    if b is None:
        e = la.norm(y - z)
    else:
        e = la.norm(y - z - b)
    print(e)
    print('#time: {0} [{1} {2} {3}]'.format(t1 + t2, t1, t2, t1 - t2))
    return e
###########################
