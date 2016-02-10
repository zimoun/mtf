#/usr/bin/env python

import sys, os
import scipy as sp

stdoutw = sys.stdout.write
stdoutf = sys.stdout.flush


# historical: when PY2 and PY3 was supported
def myrange(n=0, start=-1, stop=False):
    i = start+1
    if not stop:
        stop = n + start+1
    while i < stop:
        yield i
        i += 1

class Cartesian:
    def __init__(self, coord=(0, 0, 0)):
        if not isinstance(coord, tuple):
            raise IOError('need a tuple, not {}'.format(type(coord)))
        if len(coord) != 3:
            raise IOError('works only in IR^3 [{}]'.format(len(coord)))
        self.coord = coord

    def tosph(self):
        x, y, z = self.coord
        rho = self.norm()
        if sp.absolute(x) < 1e-8:
            if y >= 0:
                theta = sp.pi/2
            else:
                theta = 3*sp.pi/2
        else:
            theta = sp.arctan(y/x)
        phi = sp.arccos(z/rho)
        return rho, theta, phi

    def dot(self, other):
        if not isinstance(other, self.__class__):
            raise IOError('operation impossible')
        x, y, z = self.coord
        a, b, c = other.coord
        return x*a + y*b + z*c

    def norm(self):
        return sp.sqrt(self.dot(self))

    def normalized(self):
        n = self.norm()
        C = [ v/n for v in self.coord ]
        return Cartesian(coord=(C[0], C[1], C[2]))


class Mesh:
    def __init__(self, name, tag):
        if not os.path.isfile(name):
            print('Error: {0} does not exist.'.format(name))
            raise IOError('check if {} exists'.format(name))
        self.name = name
        if not isinstance(tag, list):
            tag = [tag]
        self.tag = tag
        self.points = []
        self.elems = []
        self.read()

    def read(self):
        with open(self.name, 'r') as fp:
            self._move(fp, '$Nodes')
            if not self._read_nodes(fp):
                raise IOError('check the nodes [{0}]'.format(self.name))
            self._move(fp, '$Elements')
            if not self._read_elements(fp):
                raise IOError('check the elements [{0}]'.format(self.name))

    def _move(self, fp, where):
        line = fp.readline()
        while not line.startswith(where):
            line = fp.readline().strip()
            if line == '':
                fp.seek(0, 0)

    def _that(self, fp, what):
        if fp.readline().strip().startswith(what):
            print(' --> {0} read.'.format(what[4:]))
            return True
        else:
            print('Error: corrupted file: {0} \n[{1}]'.format(self.name, fp))
            return False

    def _print_progress(self, ii, N, what='done', mod=10):
        per = int(100*(ii+1)/N)
        if per%mod == 0:
            stdoutw('\rInfo: {0}% {3}. [{1}/{2}]'.format(
                per, ii+1, N, what))


    def _read_nodes(self, fp):
        N = int(fp.readline().strip())
        self.points = [(None, None, None)] * N
        for ii in myrange(N):
            ll = fp.readline().strip().split()
            p, xyz = int(ll[0]), [ float(coord) for coord in ll[1:] ]
            self.points[p-1] = tuple(xyz)
            self._print_progress(ii, N, 'read')

        return self._that(fp, '$EndNodes')

    def _read_elements(self, fp):
        N = int(fp.readline().strip())
        self.elems = [] # [(None, None, None)] * N
        for ii in myrange(N):
            ll = fp.readline().strip().split()
            p, t, n = int(ll[0]), int(ll[1]), int(ll[2])
            if t != 2:
                print('Warning: elem #{0} is not a triangle. [{1}]'.format(p, t))
            tags = ll[3:3+n]
            if int(tags[0]) in self.tag:
                self.elems.append(tuple([ int(e) for e in ll[3+n:] ]))
                #self.elems[p-1] = tuple([ int(e) for e in ll[3+n:] ])
            self._print_progress(ii, N, 'read')
        return self._that(fp, '$EndElements')


    def write(self, vals, name, label='Mie'):
        with open(name, 'w') as fp:
            N = len(self.elems)
            fp.write("View \"{0}\" {{\n".format(label))
            for ii, elem in enumerate(self.elems):
                fp.write('ST( ')
                for jj, n in enumerate(elem):
                    x, y, z = self.points[n-1]
                    fp.write('{0}, {1}, {2}'.format(x, y, z))
                    if jj == len(elem)-1:
                        fp.write(')')
                    else:
                        fp.write(', ')
                fp.write('{')
                for jj, n in enumerate(elem):
                    rv = sp.real(vals[n-1])
                    fp.write('{0}, '.format(rv))
                for jj, n in enumerate(elem):
                    iv = sp.imag(vals[n-1])
                    fp.write('{0}'.format(iv))
                    if jj == len(elem)-1:
                        fp.write('};\n')
                    else:
                        fp.write(', ')
                self._print_progress(ii, N, 'written')
            fp.write("TIME{0, 1};\n};\n\n")
            print(' --> {0} written.'.format(name))




if __name__ == "__main__":
    name = 'geo/sphere-disjoint.msh'
    m = Mesh(name, 1)

    v = sp.ones(len(m.points)) + 10j*sp.ones(len(m.points))
    m.write(v, 'test.pos')

    # name = 'sphere-simple.pos'
    # mm = Mesh(name)
