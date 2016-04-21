#!/usr/bin/env python
# coding: utf8

from subprocess import call

import domains as d


dgens = []

N, tag = 3, 1
conf = d.sanitize_config(Nints=N, init_offset=True, tag=tag)
conf['center'] = (-2, 0, 0)
dgen = d.generate_concentric_dict(conf)
cmds = d.write_params_geo(conf, file_geo='geo/ellipse-concentric.script.geo')
call(cmds)
dgens.append(dgen)


tag += N
N = 3
conf = d.sanitize_config(Nints=N, init_offset=False, tag=tag)
conf['center'] = (2, 0, 0)
dgen = d.generate_concentric_dict(conf)
cmds = d.write_params_geo(conf, file_geo='geo/sphere-concentric.script.geo')
call(cmds)
dgens.append(dgen)


dicts = d.merge_msh(dgens)


tag += N
N = 1
conf = d.sanitize_config(Nints=N, init_offset=False, tag=tag)
conf['center'] = (0, 0, 0)
conf['rad'] = [5]
dgen = d.generate_concentric_dict(conf)
cmds = d.write_params_geo(conf, file_geo='geo/sphere-concentric.script.geo')
call(cmds)
dgens.append(dgen)


dicts = d.merge_msh(dgens)
