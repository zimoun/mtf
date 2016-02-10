#!/usr/bin/env python

N = 4
surfs = [ i+1 for i in range(N) ]

Doms = [
    { 'name': '0',
      'eps': 1,
      'union': [-surfs[i] for i in range(len(surfs))],
      'neighbor': ['A', 'B', 'C', 'D']
  },
    { 'name': 'A',
      'eps': 2,
      'union': surfs[0],
      'neighbor': '0'
  },
    { 'name': 'B',
      'eps': 2,
      'union': surfs[1],
      'neighbor': '0'
  },
    { 'name': 'C',
      'eps': 2,
      'union': surfs[2],
      'neighbor': '0'
  },
    { 'name': 'D',
      'eps': 2,
      'union': surfs[3],
      'neighbor': '0'
  }
]

from copy import deepcopy

def unsigned(l):
    ll = []
    for x in l:
        if x < 0:
            ll.append(-x)
    else:
        ll.append(x)
    return ll

def sanitize(doms):
    if not isinstance(doms, list):
        raise TypeError
    ddoms = deepcopy(doms)
    for d in ddoms:
        if not isinstance(d, dict):
            raise TypeError
        try:
            v = d['name']
            v = d['eps']
            v = d['union']
            if not isinstance(v, list):
                print("Trying to convert {0} into {1}".format(type(v), type([])))
                try:
                    d['union'] = list(v)
                except:
                    d['union'] = [v]
        except:
            raise SyntaxError
    return ddoms


def doms2graph(doms):
    graph = {}

    for dom in doms:
        name = dom['name']
        surfs = dom['union']

        temp = doms[:]
        temp.remove(dom)

        for d in temp:
            n = d['name']
            ss = unsigned(d['union'])
            for s in ss:
                if s in unsigned(surfs):
                    t = (n, s)
                    try:
                        graph[name].append(t)
                    except:
                        graph[name] = [t]
    return graph

def write2dot(filename, graph, label=False):
    with open(filename, 'w') as f:
        f.write('graph {\n')
        f.write('\n')
        for name, vals in graph.items():
            for n, i in vals:
                if label:
                    f.write(name + ' -- ' + n + ' [label={}]'.format(i) + ';\n')
                else:
                    f.write(name + ' -- ' + n + ' ;\n')
            f.write('\n')
        f.write('}\n')
