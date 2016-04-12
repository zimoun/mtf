#!/usr/bin/env python
# coding: utf8

debug = False
if debug:
    print('Debug: {}'.format(debug))

from copy import deepcopy
import warnings as warning

class Domains:

    def __init__(self, doms):

        mydoms = deepcopy(doms)

        self._check(mydoms)
        self.doms = mydoms
        self.numbering_doms = self._numbering_doms()
        self.neighs = self._neighbor()

        self.numbering_ints = self._numbering_ints()
        # list of all the interfaces: self.interfaces

    def _check(self, doms):
        if not isinstance(doms, list):
            raise TypeError("Domain has to be a list, not {}".format(type(doms)))

        unsigned = lambda l: [ x if x > 0 else -x for x in l ]
        names = []
        interfaces = []

        for d in doms:
            if not isinstance(d, dict):
                td = type(d)
                raise TypeError("Each domain has to be a dict, not {}".format(td))
            try:
                name = d['name']
                if not isinstance(name, str):
                    try:
                        d['name'] = str(name)
                    except:
                        raise ValueError("Name of domain has to be 'str' convertible")
                v = d['phys']
                if isinstance(v, list):
                    k, alpha, beta = v
                else:
                    k, alpha, beta = v, 1., 1.
                k, alpha, beta = complex(k), complex(alpha), complex(beta)
                d['phys'] = [k, alpha, beta]

                v = d['union']
                if not isinstance(v, list):
                    # if debug:
                    #     tv = type(v)
                    #     tl = type([])
                    #     print("Trying to convert {0} into {1}".format(tv, tl))
                    try:
                        d['union'] = list(v)
                    except:
                        d['union'] = [v]
                p, m = 0, 0
                for v in d['union']:
                    if v > 0:
                        p += 1
                    elif v < 0:
                        m -= 1
                    else:
                        raise ValueError('0 is not a Physical Tag of Gmsh')
                if len(d['union']) != p and  len(d['union']) != -m:
                    if debug:
                        warning.warn('all the interfaces must have the same orientation', UserWarning)

                d['interfaces'] = unsigned(d['union'])
                sig = int((p + m) / len(d['union']))
                d['sign'] = sig
                d['signs'] = [ complex(ii / jj)
                              for ii, jj in zip(d['interfaces'], d['union']) ]

                for ii in d['interfaces']:
                    if not ii in interfaces:
                        interfaces.append(ii)
            except:
                raise SyntaxError("Check 'name', 'phys' or 'union'.")

            if not name in names:
                names.append(name)
            else:
                raise ValueError("Domain {} already exists.".format(name))

            self.ints = interfaces


    def _neighbor(self):
        if debug:
            print("Computing the neighbor from 'union'", end="")
            print(" ; dict: name -> tuple(name, interface)")
        neigh = {}
        for dom in self.doms:
            name = dom['name']
            surfs = dom['interfaces']

            temp = self.doms[:]
            temp.remove(dom)

            for d in temp:
                n = d['name']
                ss = d['interfaces']
                for s in ss:
                    if s in surfs:
                        t = (n, s)
                        try:
                            neigh[name].append(t)
                        except:
                            neigh[name] = [t]
        return neigh

    def _numbering_doms(self):
        if debug:
            print("Numbering domains ; dict: name -> int")
        numbering = {}
        for ii, d in enumerate(self.doms):
            numbering[d['name']] = ii
        return numbering

    def _numbering_ints(self):
        if debug:
            print("Numbering interface ; dict: name -> int")
        numbering = {}
        for d in self:
            name = d['name']
            local = {}
            for ii, face in enumerate(d['interfaces']):
                local[face] = ii
            numbering[name] = local
        return numbering


    ### end constructor

    def getEntry(self, name):
        for ii, vals in enumerate(self.doms):
            if vals['name'] == name:
                return self.doms[ii]
        return {}

    def getIndexDom(self, name):
        return self.numbering_doms[name]

    def getIndexInt(self, name, face):
        local = self.numbering_ints[name]
        return local[face]

    def getName(self, index):
        for name, ii in self.numbering_doms.items():
            if ii == index:
                return name
        raise ValueError("NotFound")

    def getNeighborOf(self, name):
        doms = []
        for n, vals in self.neighs.items():
            if n == name:
                for v in vals:
                    doms.append(self.getEntry(v[0]))
        return doms

    def write2dot(self, filename="graph.dot", label=False):
        if debug:
            print("Writing dotfile.")
        with open(filename, 'w') as f:
            f.write('graph {\n')
            f.write('\n')
            for name, vals in self.neighs.items():
                for n, i in vals:
                    name = str(name)
                    n = str(n)
                    if label:
                        f.write(name + ' -- ' + n
                                + ' [label={}]'.format(i) + ';\n')
                    else:
                        f.write(name + ' -- ' + n + ' ;\n')
                    f.write('\n')
            f.write('}\n')
        if debug:
            print("Try: $ dot -Teps " + filename)


    def add(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Operation not supported.')
        this, that = self.doms[:], other.doms[:]
        for ii, d in enumerate(self):
            for jj, o in enumerate(other):
                if d['name'] == o['name']:
                    del this[ii]
                    del that[jj]
                    break
        this.extend(that)
        ddd = self.__class__(this)
        return ddd

    def __add__(self, other):
        return self.add(other)

    def __len__(self):
        return len(self.doms)

    def __iter__(self):
        for d in self.doms:
            yield d

    def __repr__(self):
        mystr = '['
        orders = ['name', 'phys', 'union', 'interfaces', 'sign', 'signs']
        for d in self:
            mystr += "\n{\n"
            for key in orders:
                mystr += "{0}\t: {1}".format(key[0:6], d[key])
                mystr += ",\n"
            mystr += "},\n"
        mystr += "]\n"
        return mystr


#####

def generate_disjoint_dict(N, phys=2, infty='0', offset=1):
    if str(infty) == str(offset):
        raise ValueError('infty and offset MUST be different.')
    if not isinstance(phys, list):
        v = phys
        phys = [ v for i in range(N) ]
    else:
        if len(phys) != N:
            raise ValueError('Incompatible length')
    surfs = [ i+offset for i in range(N) ]
    doms = [
        { 'name': infty,
          'phys': 1,
          'union': [-surfs[i] for i in range(len(surfs))],
      }]
    for ii in range(N):
        d =  {
            'name': ii+offset,
            'phys': phys[ii],
            'union': surfs[ii],
        }
        doms.append(d)
    return doms

#####

def generate_concentric_dict(N, phys=2, infty=('0', 1), offset=2):
    if isinstance(infty, tuple):
        infty, the_tag = infty
    else:
        the_tag = 1
    if str(the_tag) == str(offset):
        raise ValueError('infty and offset MUST be different.')
    if not isinstance(phys, list):
        v = phys
        phys = [ v for i in range(N) ]
    surfs = [the_tag]
    surfs.extend([ the_tag + i +offset for i in range(N) ])
    doms = [
        { 'name': infty,
          'phys': 1,
          'union': -surfs[0],
      }]
    for ii in range(N-1):
        d =  {
            'name': ii+offset,
            'phys': phys[ii],
            'union': [surfs[ii], -surfs[ii+1]],
        }
        doms.append(d)
    ii = N-1
    d =  {
        'name': ii+offset,
        'phys': phys[ii],
        'union': surfs[ii],
    }
    doms.append(d)
    return doms

#####

def write_params_geo(config):
    try:
        k = config['kRef']
        eps = config['eps']
        rad = config['rad']
        L = config['L']
    except:
        raise TypeError('Missing argument: k, eps, rad, L')
    try:
        alpha = config['alpha']
    except:
        alpha = 10.
    try:
        filename = config['file']
    except:
        filename='geo/params.geo'
    try:
        name = config['meshname']
        # ugly! TODO: fix this mess
        name = name.replace('geo/', '')
    except:
        name = 'my.msh'

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
        fp.write("\nname = '{}';\n".format(name))
        fp.write("\nk = {};\n".format(k))
        fp.write("alpha = {}; //point per wavelenght\n\n".format(alpha))
        fp_write("eps", eps)
        fp_write("rad", rad)
        fp_write("L", L)
    config['meshname'] = 'geo/' + name
    return config

############

if __name__ == "__main__":

    # N = 4
    # surfs = [ i+1 for i in range(N) ]

    # dd = [
    #     { 'name': '0',
    #       'phys': 1,
    #       'union': [-surfs[i] for i in range(len(surfs))],
    #   },
    #     { 'name': 'A',
    #       'phys': 2,
    #       'union': surfs[0],
    #   },
    #     { 'name': 'B',
    #       'phys': 2,
    #       'union': surfs[1],
    #   },
    #     { 'name': 'C',
    #       'phys': 2,
    #       'union': surfs[2],
    #   },
    #     { 'name': 'D',
    #       'phys': 2,
    #       'union': surfs[3],
    #   }
    # ]

    # domains = Domains(dd)
    # i = domains.getIndex('C')
    # n = domains.getName(2)

    # print('# first check: ok')

    # doms = Domains(generate_disjoint_Domains(8))
    # doms.write2dot("my-graph.dot")

    # print('# second check: ok')

    dd = [
        { 'name': '0',
          'phys': 1,
          'union': [-100],
      },
        { 'name': 'A',
          'phys': 2,
          'union': [100, -200],
      },
        { 'name': 'B',
          'phys': 1,
          'union': [200, -300],
      },
        { 'name': 'C',
          'phys': 1,
          'union': 300,
      }
    ]
    domains = Domains(dd)
    domains.write2dot("my-graph.dot")

    N = 2
    geoconf = {
        'kRef': 0.1,
        'eps': [ i+2 for i in range(N) ],
        'rad': [ 1. for i in range(N) ],
        'L': [ 1. for i in range(N) ]
        }
    geoconf['L'][0] = 0
    geoconf = write_params_geo(geoconf)
    my_d = generate_disjoint_dict(N+3) #, geoconf['eps'])
    my_c = generate_concentric_dict(N, geoconf['eps'], infty=(1, 1), offset=11)

    d, c = Domains(my_d), Domains(my_c)
    r = d + c
    r.write2dot("my-graph.dot")
    from os import system
    system('dot -Teps my-graph.dot > graph.eps')
