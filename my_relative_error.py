# coding: utf-8

import numpy as np

from bempp.api.integration import gauss_triangle_points_and_weights

import multiprocessing as mp


class Process(mp.Process):
    def __init__(self, i, queue, fcompute, num, den):
        super(self.__class__, self).__init__()
        self.i = i
        self.queue = queue
        self.fcompute = fcompute
        self.num = num
        self.den = den

    def run(self):
        i = None
        while i != 'done':
            i = self.queue.get()
            if isinstance(i, int):
                nu, de = self.fcompute(i)
                self.num[self.i] += nu
                self.den[self.i] += de
            self.queue.task_done()

def relative_error(gf, fun, element=None):

    def compute(i):
        element = elements[i]
        integration_elements = element.geometry.integration_elements(points)
        global_dofs = element.geometry.local2global(points)
        fun_vals = np.zeros((gf.component_count, npoints), dtype=gf.dtype)

        for j in range(npoints):
            fun_vals[:, j] = fun(global_dofs[:, j])

        diff = np.sum(np.abs(gf.evaluate(element, points) - fun_vals)**2, axis=0)
        global_diff = np.sum(diff * integration_elements * weights)
        abs_fun_squared = np.sum(np.abs(fun_vals)**2, axis=0)
        fun_l2_norm = np.sum(abs_fun_squared * integration_elements * weights)

        return global_diff, fun_l2_norm

    accuracy_order = gf.parameters.quadrature.far.single_order
    points, weights = gauss_triangle_points_and_weights(accuracy_order)
    npoints = points.shape[1]

    if element is None:
        elements =  list(gf.grid.leaf_view.entity_iterator(0))
    elif not isinstance(element, list):
        elements = [element]
    else:
        elements = element

    nelems = len(elements)
    # print('#elem:', nelems)

    nprocs, procs = mp.cpu_count(), []
    jobs = mp.JoinableQueue()

    # print('#proc:', nprocs)

    num = mp.Array('d', np.zeros(nprocs))
    den = mp.Array('d', np.zeros(nprocs))

    for i in range(nprocs):
        proc = Process(i, jobs, compute, num, den)
        procs.append(proc)
        proc.start()

    for j in range(nelems):
        jobs.put(j)
    jobs.join()

    for proc in procs:
        jobs.put('done')
    jobs.join()

    nu, de = 0., 0.
    for i in range(nprocs):
        nu += num[i]
        de += den[i]
        procs[i].terminate()

    return np.sqrt(nu / de)

#########################################################

def relative_error_seq(gf, fun, element=None):

    global_diff = 0
    fun_l2_norm = 0

    accuracy_order = gf.parameters.quadrature.far.single_order
    points, weights = gauss_triangle_points_and_weights(accuracy_order)
    npoints = points.shape[1]

    element_list = [element] if element is not None else list(gf.grid.leaf_view.entity_iterator(0))

    for element in element_list:
        integration_elements = element.geometry.integration_elements(points)
        global_dofs = element.geometry.local2global(points)
        fun_vals = np.zeros((gf.component_count, npoints), dtype=gf.dtype)

        for j in range(npoints):
            fun_vals[:, j] = fun(global_dofs[:, j])

        diff = np.sum(np.abs(gf.evaluate(element, points) - fun_vals)**2, axis=0)
        global_diff += np.sum(diff * integration_elements * weights)
        abs_fun_squared = np.sum(np.abs(fun_vals)**2, axis=0)
        fun_l2_norm += np.sum(abs_fun_squared * integration_elements * weights)

    return np.sqrt(global_diff/fun_l2_norm)




if __name__ == '__main__':

    import time
    import numpy.linalg as la
    import bempp.api as bem

    h = 0.5
    grid = bem.shapes.sphere(h=h)
    space = bem.space.function_space(grid, "P", 1)

    print('#dof:', space.global_dof_count)
    print('#elems:', space.grid.leaf_view.elements.shape)
    print('#nodes:', space.grid.leaf_view.vertices.shape)
    print('', flush=True)

    def fdat(point, normal, dom_ind, result):
        x, y, z = point
        val = x**2
        result[0] = val

    def gfun(point):
        x, y, z = point
        val = y**2
        return y

    def gdat(point, normal, dom_ind, result):
        val = gfun(point)
        result[0] = val

    tt = time.time()
    f = bem.GridFunction(space, fun=fdat)
    tf = time.time() - tt
    print('time gf:', tf)
    print('', flush=True)

    tt = time.time()
    g = bem.GridFunction(space, fun=gdat)
    tg = time.time() - tt
    print('time gf:', tg)
    print('', flush=True)

    if tf < tg:
        print('double call slower')
    else:
        print('weird?!')
    print('', flush=True)

    tt = time.time()
    fn, gn = f.coefficients, g.coefficients
    en = la.norm(fn - gn) / la.norm(gn)
    t = time.time() - tt
    print('l2:', en)
    print('time:', t)

    print('', flush=True)

    tt = time.time()
    En = f.relative_error(gfun)
    T = time.time() - tt
    print('L2:', En)
    print('time:', T)

    if t < T:
        print('l2 faster than L2')
    else:
        print('l2 slower than L2')
    print('', flush=True)


    tt = time.time()
    seqEn = relative_error_seq(f, gfun)
    seqT = time.time() - tt
    print('seqL2:', seqEn)
    print('err:', En - seqEn)
    print('time:', seqT, T - seqT)
    if seqT < T:
        print('faster')
    else:
        print('slower')
    print('', flush=True)

    tt = time.time()
    parEn = relative_error(f, gfun)
    parT = time.time() - tt
    print('parL2:', parEn)
    print('err:', En - parEn)
    print('time:', parT, T - parT)
    if parT < T:
        print('faster')
    else:
        print('slower')

    print('', flush=True)
