#!/usr/bin/env python
#coding: utf8



def write(k, eps, rad, L,
         alpha=10,
         name='my.msh',
         filename='geo/params.geo'):

    if len(eps) != len(rad):
        if len(eps) != len(L):
            raise ValueError("Incoherent parameters: check eps, k, L")
    if L[0] != 0:
        raise ValueError("L must start with 0, not {}".format(L[0]))

    with open(filename, 'w') as fp:
        def fp_write(mystr, myvals):
            fp.write("{} = {{".format(mystr))
            for v in myvals[:-1]:
                fp.write(" {},".format(v))
            fp.write(" {} }};\n".format(myvals[-1]))
        fp.write("alpha = {};\n\n".format(alpha))
        fp.write("k = {};\n\n".format(k))
        fp_write("eps", eps)
        fp_write("rad", rad)
        fp_write("L", L)
        fp.write("\nname = '{}';\n".format(name))


if __name__ == "__main__":

    k = 0.1
    eps = [2, 3, 4]
    rad = [1, 1, 0.5]
    L = [0, 0.5, 1]

    write(k, eps, rad, L)
