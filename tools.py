#!/usr/bin/env python
# coding: utf8

import sys
sys.path.append("..")

from bempp.lib import createNullOperator
from numpy import abs
from copy import deepcopy


def createList(Context, Shape=(0, 0)):

    context, spaces, doms = Context

    Mat = []
    for ii in range(Shape[0]):
        Mat.append([ 0 for jj in range(Shape[1])])

    MyZeros_symbol = deepcopy(Mat)

    row = 0
    MyZeros = deepcopy(Mat) ## could be better with append instead of assign
    for dom in doms:
        myname = dom['name']
        for my in dom['union']:
            Ni = len(dom['union'])
            itest = abs(my)
            test = spaces[itest]
            col = 0
            for odom in doms:
                oname = odom['name']
                for other in odom['union']:
                    ni = len(odom['union'])
                    itrial = abs(other)
                    trial = spaces[itrial]

                    sz = 'Z_' + myname+oname + '_{}{}'.format(itest, itrial)
                    Zd = createNullOperator(
                        context, trial, trial, test, sz)

                    # print(row, col, 'd-'+sz, ni)
                    MyZeros_symbol[row][col] = sz
                    MyZeros[row][col] = Zd

                    # print(row, col+ni, 'n-'+sz)
                    MyZeros_symbol[row][col+ni] = sz
                    MyZeros[row][col+ni] = Zd

                    # print(row+Ni, col, 'n-'+sz, ni)
                    MyZeros_symbol[row+Ni][col] = sz
                    MyZeros[row+Ni][col] = Zd

                    # print(row+Ni, col+ni, 'd-'+sz)
                    MyZeros_symbol[row+Ni][col+ni] = sz
                    MyZeros[row+Ni][col+ni] = Zd

                    col += 1
                col += ni
            row += 1
        row += Ni
    return MyZeros_symbol, MyZeros
