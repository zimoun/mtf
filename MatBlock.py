#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

__docformat__ = "restructuredtext en"

#__all__ = ['lil_matrix','isspmatrix_lil']

from bisect import bisect_left

import numpy as np
from scipy.sparse.sputils import getdtype
from scipy.sparse import lil_matrix

# support for the last version of LinearOperator
# from interface import *

def isshape(x):
    try:
        (m, n) = x
    except:
        return False
    else:
        if isinstance(m, int) and isinstance(n, int):
            return True
        else:
            return False

class Block:
    """
    A block is an object with at least two attributes
    -- shape : (M,N) tuple of integer
    -- matvec or dot : a function
    that returns a vector of length M
    when applies a vector of length N

    the initial object with all its attributes is referenced in .orig

    Block owns attributes:
    -- shape
    -- matvec
    -- type
    -- orig
    """
    def __init__(self, block):
        if  hasattr(block, 'shape'):
            self.shape = block.shape
        else:
            raise  TypeError('a block needs a shape.')
        if hasattr(block, 'dot'):
            self.matvec = block.dot
        elif hasattr(block, 'matvec'):
            self.matvec = block.matvec
        else:
            raise  TypeError('a block needs a matvec/dot.')
        self.type = type(block)
        self.orig = block
        if hasattr(block, 'dtype'):
            self.dtype= block.dtype
        else:
            self.dtype = float

class MatBlock:
    """ class which manages a linear operator by blocks, e.g.,
    \t[ A B ]
    \t[ C D ]
    with A, B, C and D are:
    numpy.ndarray OR scipy.sparse.linalg.LinearOperator

    It is compatible with scipy.sparse.linalg.LinearOperator,
    i.e., all the linear solvers from scipy are usable.

    Moreover, the classical operations scalar*, + and - are available.
    \n
    Row-based linked list sparse matrix
    """

    def __init__(self,  shape=(0, 0), dtype=int):
        if not isshape(shape):
            raise ValueError('not an acceptable tuple.')
        try:
            self.dtype = dtype
            r, c = shape
            self.Shape = (r, c)
            if c>0:
                # see if array+concatenate is not better?
                ##
                # this part slows down addBlock()
                # because the current update of the offsets needs more loops
                # FIXME: remove unnecessary loops (those of empty value)
                self.cols = [ list([]) for i in range(c) ]
                self.data = [ list([]) for i in range(c) ]
            else:
                self.cols = [[]]
                self.data = [[]]
            self.shape = (0, 0)
            self._shapes_row = np.array([ -1 for i in range(r) ], dtype=int)
            self._shapes_col = np.array([ -1 for i in range(c) ], dtype=int)

            self._offsets_row = np.array([ 0 for i in range(r) ], dtype=int)
            self._offsets_col = np.array([ 0 for i in range(c) ], dtype=int)
        except:
            raise TypeError('MatBlock constructor fails. Hum?')

    def addBlock(self, pos, b, expert=False):
        if not isshape(pos):
            raise ValueError('not an acceptable tuple.')
        block = Block(b)

        if block.dtype == float and self.dtype == int:
            self.dtype = float
        elif block.dtype == complex and self.dtype == float:
            self.dtype = complex
        elif block.dtype == complex and self.dtype == int:
            self.dtype = complex

        R, C = self.shape
        rr, cc = self.Shape

        r, c = pos
        if r+1>=rr:
            rr = r+1
        if c+1>=cc:
            cc = c+1
        self.Shape = (rr, cc)

        le = len(self._shapes_row)
        if le < rr:
            t = np.array([ 0 for i in range(le, rr) ], dtype=int)
            s = self._shapes_row
            self._shapes_row = np.concatenate((s,t))
            s = self._offsets_row
            self._offsets_row = np.concatenate((s,t))

        le = len(self._shapes_col)
        if le < cc:
            t = np.array([ 0 for i in range(le, cc) ], dtype=int)
            s = self._shapes_col
            self._shapes_col = np.concatenate((s,t))
            s = self._offsets_col
            self._offsets_col = np.concatenate((s,t))

        le = len(self.cols)
        if le < cc:
            t = [ [] for i in range(le, cc) ]
            self.cols.extend(t)
            # warning: reference, pointer and memory
            t = [ [] for i in range(le, cc) ]
            self.data.extend(t)

        if self._shapes_row[r] == 0:
            self._shapes_row[r] = block.shape[0]
            R += block.shape[0]
        elif self._shapes_row[r] != block.shape[0]:
            raise ValueError('invalid block size.')
        if self._shapes_col[c] == 0:
            self._shapes_col[c] = block.shape[1]
            C += block.shape[1]
        elif self._shapes_col[c] != block.shape[1]:
            raise ValueError('invalid block size.')

        self.shape = (R, C)

        ###
        # the blocks are stored row-sorted
        ###
        # if the blocks are added in a natural order
        # ie (i,j) added before (I,J) ; i<I j<J
        # then the insertion is replaced by an append
        # because it is faster
        # (more or less, memory allocation/reallocation)
        #
        # the sort is time consuming
        #  FIXME(for large matrix, expert=True+_expert())
        ##
        # it is assumed that:
        # add an already existing block is a rare case
        ##
        N, append = len(self.cols[c]), False
        if N == 0:
            append = True
        else:
            if r > self.cols[c][N-1]:
                append = True
            else:
                # this is a naive insertion-sort
                ## make time if not almost sorted
                if expert:
                    for ii in range(N-1, -2, -1):
                        if r > self.cols[c][ii]:
                            self.cols[c].insert(ii+1, r)
                            self.data[c].insert(ii+1, block)
                            break
                        elif r == self.cols[c][ii]:
                            print('Warning: a block already exists here. Now, overwritten.')
                            self.data[c][ii] = block
                            ii = N # ugly hack!
                            break
                    if ii == -1:
                        self.cols[c].insert(0, r)
                        self.data[c].insert(0, block)
                else:
                    append = True
        if append:
            self.cols[c].append(r)
            self.data[c].append(block)

        if not expert:
            # these should make time
            self._expert()

    def _expert(self):
        tot = 0
        for ii, s in enumerate(self._shapes_col[0:-1]):
            tot += s
            self._offsets_col[ii+1] = tot
        tot = 0
        for ii, s in enumerate(self._shapes_row[0:-1]):
            tot += s
            self._offsets_row[ii+1] = tot


    def getBlock(self, pos):
        if not isshape(pos):
            raise ValueError('not an acceptable tuple.')

        r, c = pos
        R, C = self.Shape
        if r > R-1:
            raise ValueError('inexisting block. [row]')
        if c > C-1:
            raise ValueError('inexisting block. [col]')

        try:
            i = self.cols[c].index(r)
        except:
            i = -1
        if i >= 0:
            return self.data[c][i].orig
        else:
            return 0

    def diag(self):
        M = MatBlock()
        R, C = self.Shape
        I = max(R, C)
        for i in range(I):
            b = self.getBlock((i, i))
            if not isinstance(b, int):
                M.addBlock((i, i), b)
        return M

    def rmBlock(self, pos):
        if isinstance(pos, tuple):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def matvec(self, x):
        x = np.asanyarray(x)

        M, N = self.shape
        if x.shape != (N,) and x.shape != (N,1):
            raise ValueError('dimension mismatch.')

        y = np.zeros(x.shape, dtype=self.dtype)

        for c, blocks in enumerate(self.data):
            beg = self._offsets_col[c]
            end = beg + self._shapes_col[c]
            xi = x[beg:end]
            for i, block in enumerate(blocks):
                r = self.cols[c][i]
                begg = self._offsets_row[r]
                endd = begg + self._shapes_row[r]
                y[begg:endd] += block.matvec(xi)
        return y


    def concatenate(self, M, overwrite=False):
        pass

    def toLinOp(self):
        return LinearOperator(shape=self.shape, matvec=self.matvec)

    def tolil(self):
        lil = lil_matrix(self.shape, dtype=self.dtype)
        for c, blocks in enumerate(self.data):
            beg = self._offsets_col[c]
            end = beg + self._shapes_col[c]
            for i, block in enumerate(blocks):
                r = self.cols[c][i]
                begg = self._offsets_row[r]
                endd = begg + self._shapes_row[r]
                if isinstance(block.orig, np.ndarray) \
                        or isinstance(block.orig, np.matrix):
                    b = block.orig
                else:
                    try:
                        b = block.orig.todense()
                    except:
                        n, m = block.shape
                        b = np.empty((n, m), dtype=self.dtype)
                        for i in range(m):
                            ei = np.zeros((m,))
                            ei[i] = 1.0
                            b[:,i] = block.matvec(ei)
                lil[begg:endd,beg:end] = b
        return lil

    def todense(self):
        return self.tolil().todense()

    def spy(self):
        pass

    def view(self):
        print(self.todense().view())

def isBlock(x):
    return isinstance(x, Block)
def isMatBlock(x):
    return isinstance(x, MatBlock)

class Myeye:
    def __init__(self,shape=(0,0)):
        self.shape=shape
        self.dtype=float
    def matvec(self,x):
        return 2.2*x


if __name__ == "__main__":
    print("test")

    M1 = MatBlock(dtype=float)
#     M2 = MatBlock((2,3))

# #    M1.addBlock((1,1),2)



    from scipy.sparse import eye

    f = Myeye((4,4))

    M1.addBlock((0,0),np.ones((2,2)))
    M1.addBlock((1,1),2*eye(3,3))
    M1.addBlock((2,0),2*np.ones((4,2)))
    M1.addBlock((0,2),3*np.ones((2,4)))
    M1.addBlock((2,2),f)
    u = np.ones(M1.shape[0])
    for i in range(M1.shape[0]):
        u[i] = i+1
    v = M1.matvec(u)

    print('----New')

    a = M1.toLinOp()
    M2 = MatBlock()
    M2.addBlock((0,0),M1)
    M2.addBlock((1,1),a)
    uu = np.ones(M2.shape[0])
    # for i in range(M2.shape[0]):
    #     uu[i] = i+1
    vv = M2.matvec(uu)

#     a = np.eye(3)
#     b = np.eye(4)

#     M = MatBlock()

#     M.addBlock((7,0), 7*a)
#     M.addBlock((4,0), 4*a)
#     M.addBlock((2,0), 2*a)
#     M.addBlock((6,0), 6*a)
# #    M.addBlock((1,0), a)
#     M.addBlock((5,0), 5*a)
#     M.addBlock((3,0), 3*a)
#     M.addBlock((0,0), 0.5*a)

#     M.addBlock((1,1), a)

#     u = np.ones(M.shape[1])
#     v = M.matvec(u)

#     a = np.eye(3)
#     b = np.eye(4)

#     Mb = MatBlock((2,8))

#     Mb.addBlock((0,7), 7*a)
#     Mb.addBlock((0,4), 4*a)
#     Mb.addBlock((0,2), 2*a)
#     Mb.addBlock((0,6), 6*a)
# #    Mb.addBlock((0,1), a)
#     Mb.addBlock((0,5), 5*a)
#     Mb.addBlock((0,3), 3*a)
#     Mb.addBlock((0,0), 0.5*a)

# #    Mb.addBlock((1,0), 2*a)
#     Mb.addBlock((1,1), a)

#     u = np.ones(Mb.shape[1])
#     v = Mb.matvec(u)

#     N, coef = 5, 3
#     a = np.eye(N)
# #    M = MatBlock()
#     M = MatBlock((N,N**coef))
#     # for j in range(N):
#     #     for i in range(N**coef):
#     #         M.addBlock((i,j), j*i*a, expert=True)


#     from numpy.random import randint
#     for j in randint(N, size=(N,)):
#         for i in randint(N**coef, size=(N**coef,)):
#             M.addBlock((i,j), j*i*a, expert=True)



# def test1(N,coef):
#     a = np.eye(N)
#     M = MatBlock()
#     for j in xrange(N**coef):
#         for i in xrange(N**coef):
#             M.addBlock((j,i), j*i*a)

# def test2(N,coef):
#     a = np.eye(N)
#     M = MatBlock()
#     for j in xrange(N**coef,-1,-1):
#         for i in xrange(N**coef,-1,-1):
#             M.addBlock((j,i), j*i*a)

# def test3(N,coef):
#     a = np.eye(N)
#     M = MatBlock( (N**coef, N**coef) )
#     for j in xrange(N**coef):
#         for i in xrange(N**coef):
#             M.addBlock((j,i), j*i*a)

# def test(M,r,c,N=2):
#     a = np.eye(N)
#     for j in xrange(r):
#         for i in xrange(c):
#             M.addBlock((j,i), j*i*a)
#     return M
