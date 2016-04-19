#!/usr/bin/env python
# coding: utf8

debug = False
if debug:
    print('Debug: {}'.format(debug))

GMSH = 'gmsh'

from copy import deepcopy
import warnings as warning

from os.path import isfile
import uuid
import itertools

MYID = itertools.count()
EXPLICIT_NAME = True

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
                if isinstance(v, list) or isinstane(v, tuple):
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

def sanitize_config(config=None,
                    Nints=1,
                    kRef=0.3,
                    phys=(2, 1, 1),
                    rad=1,
                    L=1,
                    names='0',
                    AB=(1, 1.5),
                    tag=1,
                    nlambda=10,
                    center=(0, 0, 0),
                    meshname=False,
                    file_params='geo/params_tmp.geo',
                    file_offset='geo/offset_tmp.geo',
                    init_offset=False,
                    inc_id=True):

    if config is None:
        config = {}

    if not isinstance(config, dict):
        raise TypeError('Require dict.')

    if EXPLICIT_NAME and inc_id:
        myid = next(MYID)
    else:
        myid = uuid.uuid1()

    if not meshname:
        meshname = str(myid)
    meshname += '.msh'

    if init_offset:
        offset = int(init_offset)
    elif isfile(file_offset):
        with open(file_offset, 'r') as fp:
            offset = int(float(fp.readline().split('=')[1].split(';')[0].strip()))
            # int(float()) because Gmsh does not know int, only float
    else:
        offset = 0
    with open(file_offset, 'w') as fp:
        fp.write("OFFSET = {}; // fix about Gmsh confusion ?".format(offset))
    config['offset'] = offset

    def counter(keys):
        n = -1
        for key in keys:
            try:
                tmp = config[key]
                if n != len(tmp) and n > 0:
                    raise ValueError("Incoherent parameters: check number of domain/interface")
                else:
                    n = len(tmp)
            except:
                pass
        return n

    ni = counter(['rad', 'L', 'AB'])
    if ni > 0:
        Nints = ni

    m = counter(['phys', 'names'])
    if m > 0:
        Ndoms = m
    else:
        Ndoms = Nints + 1

    if Nints + 1 != Ndoms:
        raise ValueError('Incoherent number of interface/domain')


    def check(key, default):
        try:
            val = config[key]
        except:
            val = default
        return val

    keys = ['kRef', 'tag', 'nlambda', 'center', 'file_params', 'meshname', 'file_offset']
    vals = [kRef, tag, nlambda, center, file_params, meshname, file_offset]
    for key, val in zip(keys, vals):
        config[key] = check(key, val)

    try:
        eps = [ e for e, a, b in config['phys'] ]
    except:
        config['phys'] = [ phys for i in range(Ndoms) ]
        config['phys'][0] = (1, 1, 1)
    try:
        rad = config['rad']
    except:
        config['rad'] = [ rad for i in range(Nints) ]
    try:
        L = config['L']
        if L[0] != 0:
            raise ValueError("L must start with 0, not {}".format(L[0]))
    except:
        LL = [ L for i in range(Nints) ]
        LL[0] = 0
        config['L'] = LL
    try:
        names = config['names']
    except:
        names = [names]
        if EXPLICIT_NAME:
            names.extend([ str(myid) + '_' + str(i) for i in range(1, Ndoms) ])
        else:
            names.extend([ str(uuid.uuid3(uuid.uuid1(), str(i))) for i in range(1, Ndoms) ])
        config['names'] = names
    finally:
        rm = set(names)
        if len(names) > len(rm):
            raise ValueError('Duplicate name.')
    try:
        AB = config['AB']
    except:
        config['AB'] = [ AB for i in range(Nints) ]

    return config

#####

def generate_concentric_dict(one_dict):
    dgen = {}
    config = sanitize_config(one_dict, inc_id=False)
    dgen['config'] = config

    names = config['names']
    phys = config['phys']

    tag = int(config['tag'])

    Ndoms = len(names)
    surfs = [tag]
    surfs.extend([ tag + i for i in range(Ndoms) ])
    doms = [
        { 'name': names[0],
          'phys': phys[0],
          'union': [-surfs[0]],
      }]
    for ii in range(1, Ndoms-1):
        d =  {
            'name': names[ii],
            'phys': phys[ii],
            'union': [surfs[ii], -surfs[ii+1]],
        }
        doms.append(d)
    ii = Ndoms-1
    d =  {
        'name': names[ii],
        'phys': phys[ii],
        'union': [surfs[ii]],
    }
    doms.append(d)
    dgen['doms'] = doms
    return dgen

#####

def write_params_geo(dictconf, file_geo='geo/sphere-concentric.script.geo'):
    config = sanitize_config(dictconf, inc_id=False)

    k = config['kRef']
    eps = [ e for e, _, _ in config['phys'] ]
    nlambda = config['nlambda']

    # TODO: fix in geo
    eps = eps[1:]

    rad = config['rad']
    L = config['L']

    C = config['center']

    A = [ a for a, _ in config['AB'] ]
    B = [ b for _, b in config['AB'] ]

    file_params = config['file_params']
    file_offset = config['file_offset']

    meshname = config['meshname']
    # ugly! TODO: fix this mess
    meshname = meshname.replace('geo/', '')

    tag = int(config['tag'])

    with open(file_params, 'w') as fp:
        def fp_write(mystr, myvals):
            fp.write("{} = {{".format(mystr))
            for v in myvals[:-1]:
                fp.write(" {},".format(v))
            fp.write(" {} }};\n".format(myvals[-1]))
        fp.write('// Generated by Python script\n\n')
        fp.write("\nname = '{}';\n".format(meshname))
        fp.write("\nk = {};\n".format(k))
        fp.write("alpha = {}; //point per wavelenght\n\n".format(nlambda))
        fp_write("eps", eps)
        fp_write("rad", rad)
        fp_write("L", L)
        fp.write("xo = {0} ; yo = {1} ; zo = {2} ;\n".format(C[0], C[1], C[2]))
        fp.write('\n// Only used by ellipsis, small and large radii\n')
        fp_write("A", A)
        fp_write("B", B)
        fp.write('\n// Physical tag: then consecutive Ndom numbering\n')
        fp.write("tag = {};".format(tag))
        fp.write("\n\n////done///\n\n\n")

        if isfile(file_offset):
            with open(file_offset, 'r') as of:
                fp.write(of.readline())
        else:
            OFFSET = 0
            fp.write("OFFSET = {}; // fix about Gmsh confusion ?".format(OFFSET))

    return [GMSH, file_geo, '-']

#####

def merge_msh(dalls, out='all.msh', tmp='geo/Merge-all-meshes_tmp.script.geo'):
    with open(tmp, 'w') as fp:
        for dgen in dalls:
            name = dgen['config']['meshname']
            fp.write("Merge '{}';\n".format(name))
        fp.write("Save 'all.msh';")
    call([GMSH, tmp, '-'])

    # TODO: fix this not robust merge because now I am tired !!
    doms = [
        {
            'name': '0',
            'phys': 1,
            'union': []
        }
    ]
    for d in dalls:
        print(d)
        for dd in d['doms']:
            print(dd)
            if dd['name'] == '0':
                doms[0]['union'].extend(dd['union'])
            else:
                doms.append(dd)

    return doms


#####


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

    # dd = [
    #     { 'name': '0',
    #       'phys': 1,
    #       'union': [-100],
    #   },
    #     { 'name': 'A',
    #       'phys': 2,
    #       'union': [100, -200],
    #   },
    #     { 'name': 'B',
    #       'phys': 1,
    #       'union': [200, -300],
    #   },
    #     { 'name': 'C',
    #       'phys': 1,
    #       'union': 300,
    #   }
    # ]
    # domains = Domains(dd)
    # domains.write2dot("my-graph.dot")

    # N, NN = 2, 3
    # geoconf = {
    #     'kRef': 0.1 * 3.1415,
    #     'eps': [ i+2 for i in range(N) ],
    #     'rad': [ 1. for i in range(N) ],
    #     'L': [ 1. for i in range(N) ]
    #     }
    # geoconf['L'][0] = 0

    # from subprocess import call

    # geoconf['meshname'] = 'm1.msh'
    # geoconf = write_params_geo(geoconf)
    # my_d1 = generate_concentric_dict(N) #, geoconf['eps'])

    # #call(['gmsh', 'geo/sphere-concentric.script.geo', '-'])
    # call(['gmsh', 'geo/ellipse-concentric.script.geo', '-'])

    # geoconf['meshname'] = 'm2.msh'
    # geoconf['center'] = [5, 0, 0]
    # geoconf['tag'] = 10
    # geoconf['rad'] = [ 2. for i in range(N) ]
    # geoconf = write_params_geo(geoconf)
    # my_d2 = generate_concentric_dict(N) #, geoconf['eps'])

    # #call(['gmsh', 'geo/sphere-concentric.script.geo', '-'])
    # call(['gmsh', 'geo/ellipse-concentric.script.geo', '-'])

    # geoconf['meshname'] = 'm3.msh'
    # geoconf['center'] = [10, 0, 0]
    # geoconf['tag'] = 100
    # geoconf['rad'] = [ 2. for i in range(N) ]
    # geoconf = write_params_geo(geoconf)
    # my_d3 = generate_concentric_dict(N) #, geoconf['eps'])

    # call(['gmsh', 'geo/sphere-concentric.script.geo', '-'])

    # merge_msh(['m1.msh', 'm2.msh', 'm3.msh'])

    # from random import randint

    # jj = 1
    # x, y, z = 0, 0, 0
    # names = []
    # ds = []
    # for ii in range(NN):
    #     name = 'm{}.msh'.format(ii)
    #     geoconf['meshname'] = name
    #     geoconf['center'] = [x, y, z]
    #     geoconf['tag'] = jj
    #     geoconf['rad'] = [ 2. for i in range(N) ]
    #     geoconf = write_params_geo(geoconf)
    #     my_d = generate_concentric_dict(N) #, geoconf['eps'])

    #     call(['gmsh', 'geo/sphere-concentric.script.geo', '-'])

    #     jj += N
    #     x += 5 + randint(0, 10)
    #     y += 10 + randint(0, 10)
    #     z += 2 + randint(0, 10)
    #     names.append(name)
    #     ds.append(my_d)

    # merge_msh(names)

    # # d, c = Domains(my_d), Domains(my_c)
    # # r = d + c
    # # r.write2dot("my-graph.dot")
    # # try:
    # #     call(['dot', '-Teps', 'my-graph.dot'], stdout=open('graph.eps', 'wb'))
    # # except OSError:
    # #     print('install graphviz to give a look to the adjacent graph.')

    from subprocess import call

    conf = sanitize_config(init_offset=True)
    dgen = generate_concentric_dict(conf)
    cmds = write_params_geo(conf)
    call(cmds)

    conff = sanitize_config(Nints=5, tag=2)
    conff['center'] = (3, 0, 0)
    dgenn = generate_concentric_dict(conff)
    cmds = write_params_geo(conff)
    call(cmds)

    dalls = [dgen, dgenn]
    doms = merge_msh(dalls)
