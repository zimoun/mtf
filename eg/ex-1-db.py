#!/usr/bin/env python
# coding: utf8

import numpy as np

from assemb import MultiTrace, checker

from time import time
import scipy.linalg as la

meshname = "./geo/sphere-disjoint.msh"

kRef = 0.1 * np.pi

dd = [
    { 'name': '0',
      'phys': 1,
      'union': [-1, -2, -3],
  },
    { 'name': 'A',
      'phys': 2,
      'union': 1,
  },
    { 'name': 'B',
      'phys': 4,
      'union': 2,
  },
    { 'name': 'C',
      'phys': 16,
      'union': 3,
  }
]


mtf = MultiTrace(kRef, meshname, dd)
mtf.collecting()
