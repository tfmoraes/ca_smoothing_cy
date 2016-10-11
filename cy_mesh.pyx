#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: nonecheck=False

import sys
cimport numpy as np

from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI
from libc.stdlib cimport abs as cabs
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.unordered_map cimport unordered_map
from libcpp.set cimport set
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.deque cimport deque as cdeque
from cython.parallel import prange

from cy_my_types cimport vertex_t, normal_t, vertex_id_t

import numpy as np
import vtk

from vtk.util import numpy_support

ctypedef float weight_t

cdef struct s_point:
    vertex_t x
    vertex_t y
    vertex_t z

ctypedef s_point Point

cdef class Mesh:
    cdef vertex_t[:, :] vertices
    cdef vertex_id_t[:, :] faces
    cdef normal_t[:, :] normals

    cdef bool _initialized

    cdef unordered_map[int, vector[vertex_id_t]] map_vface

    def __cinit__(self, pd=None, other=None):
        cdef int i
        if pd:
            self._initialized = True
            _vertices = numpy_support.vtk_to_numpy(pd.GetPoints().GetData())
            _vertices.shape = -1, 3

            _faces = numpy_support.vtk_to_numpy(pd.GetPolys().GetData())
            _faces.shape = -1, 4

            _normals = numpy_support.vtk_to_numpy(pd.GetCellData().GetArray("Normals"))
            _normals.shape = -1, 3

            self.vertices = _vertices
            self.faces = _faces
            self.normals = _normals


            for i in xrange(_faces.shape[0]):
                self.map_vface[self.faces[i, 1]].push_back(i)
                self.map_vface[self.faces[i, 2]].push_back(i)
                self.map_vface[self.faces[i, 3]].push_back(i)

        elif other:
            _other = <Mesh>other
            self._initialized = True
            self.vertices = _other.vertices.copy()
            self.faces = _other.faces.copy()
            self.normals = _other.normals.copy()
            self.map_vface = unordered_map[int, vector[vertex_id_t]](_other.map_vface)

        else:
            self._initialized = False


    cdef void copy_to(self, Mesh other):
        if self._initialized:
            other.vertices[:] = self.vertices
            other.faces[:] = self.faces
            other.normals[:] = self.normals
            other.map_vface = unordered_map[int, vector[vertex_id_t]](self.map_vface)
        else:
            other.vertices = self.vertices.copy()
            other.faces = self.faces.copy()
            other.normals = self.normals.copy()

            other.map_vface = self.map_vface

    cdef vector[vertex_id_t]* get_faces_by_vertex(self, int v_id) nogil:
        return &self.map_vface[v_id]

    cdef vector[vertex_id_t]* get_near_vertices_to_v(self, vertex_id_t v_id, float dmax) nogil:
        cdef vector[vertex_id_t]* idfaces
        cdef vector[vertex_id_t]* near_vertices = new vector[vertex_id_t]()

        cdef cdeque[vertex_id_t] to_visit
        cdef unordered_map[vertex_id_t, bool] status_v
        cdef unordered_map[vertex_id_t, bool] status_f

        cdef vertex_t *vip
        cdef vertex_t *vjp

        cdef float distance
        cdef int nf, nid, j
        cdef vertex_id_t f_id, vj

        vip = &self.vertices[v_id][0]
        to_visit.push_back(v_id)
        while(not to_visit.empty()):
            v_id = to_visit.front()
            to_visit.pop_front()

            status_v[v_id] = True

            idfaces = self.get_faces_by_vertex(v_id)
            nf = idfaces.size()

            for nid in xrange(nf):
                f_id = deref(idfaces)[nid]
                if status_f.find(f_id) == status_f.end():
                    status_f[f_id] = True

                    for j in xrange(3):
                        vj = self.faces[f_id][j+1]
                        if status_v.find(vj) == status_v.end():
                            status_v[vj] = True
                            vjp = &self.vertices[vj][0]
                            distance = sqrt((vip[0] - vjp[0]) * (vip[0] - vjp[0]) \
                                            + (vip[1] - vjp[1]) * (vip[1] - vjp[1]) \
                                            + (vip[2] - vjp[2]) * (vip[2] - vjp[2]))
                            if distance <= dmax:
                                near_vertices.push_back(vj)
                                to_visit.push_back(vj)


        return near_vertices

    cpdef get_near_vertices(self, vertex_id_t v_id, float dmax):
        cdef vector[vertex_id_t] *vertices = self.get_near_vertices_to_v(v_id, dmax)
        print "percorrendo", vertices.size(), v_id, dmax
        cdef vector[vertex_id_t].iterator it = vertices.begin()
        return deref(vertices)

cdef vector[weight_t]* calc_artifacts_weight(Mesh mesh, vector[vertex_id_t]* vertices_staircase, float tmax, float bmin) nogil:
    cdef int vi_id, vj_id, nnv, n_ids, i, j
    cdef vector[vertex_id_t]* near_vertices
    cdef weight_t value
    cdef float d
    n_ids = vertices_staircase.size()

    cdef vertex_t* vi
    cdef vertex_t* vj
    cdef size_t msize

    msize = mesh.vertices.shape[0]
    cdef vector[weight_t]* weights = new vector[weight_t](msize)
    weights.assign(msize, bmin)


    for i in xrange(n_ids):
        vi_id = deref(vertices_staircase)[i]
        vi = &mesh.vertices[vi_id][0]
        near_vertices = mesh.get_near_vertices_to_v(vi_id, tmax)
        nnv = near_vertices.size()

        for j in xrange(nnv):
            vj_id = deref(near_vertices)[j]
            vj = &mesh.vertices[vj_id][0]

            d = sqrt((vi[0] - vj[0]) * (vi[0] - vj[0])\
                    + (vi[1] - vj[1]) * (vi[1] - vj[1])\
                    + (vi[2] - vj[2]) * (vi[2] - vj[2]))
            value = (1.0 - d/tmax) * (1 - bmin) + bmin

            if value > deref(weights)[vj_id]:
                deref(weights)[vj_id] = value

        del near_vertices

    return weights


cdef Point calc_d(Mesh mesh, vertex_id_t v_id) nogil:
    cdef Point D
    cdef int nf, f_id, nid
    cdef float n=0
    cdef int i
    cdef vertex_t* vi
    cdef vertex_t* vj
    cdef set[vertex_id_t] vertices
    cdef set[vertex_id_t].iterator it
    cdef vertex_id_t vj_id

    cdef vector[vertex_id_t]* idfaces = mesh.get_faces_by_vertex(v_id)
    nf = idfaces.size()
    for nid in xrange(nf):
        f_id = deref(idfaces)[nid]
        for i in xrange(3):
            vj_id = mesh.faces[f_id][i+1]
            if (vj_id != v_id):
                vertices.insert(vj_id)
    #  del idfaces

    D.x = 0.0
    D.y = 0.0
    D.z = 0.0

    vi = &mesh.vertices[v_id][0]

    it = vertices.begin()
    while it != vertices.end():
        vj = &mesh.vertices[deref(it)][0]

        D.x = D.x + (vi[0] - vj[0])
        D.y = D.y + (vi[1] - vj[1])
        D.z = D.z + (vi[2] - vj[2])
        n += 1.0

        inc(it)

    D.x = D.x / n
    D.y = D.y / n
    D.z = D.z / n
    return D

cdef vector[vertex_id_t]* find_staircase_artifacts(Mesh mesh, double[3] stack_orientation, double T) nogil:
    cdef int nv, nf, f_id, v_id
    cdef double of_z, of_y, of_x, min_z, max_z, min_y, max_y, min_x, max_x;
    cdef vector[vertex_id_t]* f_ids
    cdef normal_t* normal

    cdef vector[vertex_id_t]* output = new vector[vertex_id_t]()
    cdef int i

    nv = mesh.vertices.shape[0]

    for v_id in xrange(nv):
        max_z = -10000
        min_z = 10000
        max_y = -10000
        min_y = 10000
        max_x = -10000
        min_x = 10000

        f_ids = mesh.get_faces_by_vertex(v_id)
        nf = deref(f_ids).size()

        for i in xrange(nf):
            f_id = deref(f_ids)[i]
            normal = &mesh.normals[f_id][0]

            of_z = 1 - fabs(normal[0]*stack_orientation[0] + normal[1]*stack_orientation[1] + normal[2]*stack_orientation[2]);
            of_y = 1 - fabs(normal[0]*0 + normal[1]*1 + normal[2]*0);
            of_x = 1 - fabs(normal[0]*1 + normal[1]*0 + normal[2]*0);

            if (of_z > max_z):
                max_z = of_z

            if (of_z < min_z):
                min_z = of_z

            if (of_y > max_y):
                max_y = of_y

            if (of_y < min_y):
                min_y = of_y

            if (of_x > max_x):
                max_x = of_x

            if (of_x < min_x):
                min_x = of_x


            if ((fabs(max_z - min_z) >= T) or (fabs(max_y - min_y) >= T) or (fabs(max_x - min_x) >= T)):
                output.push_back(v_id)
                break
    return output

cdef Mesh taubin_smooth(Mesh mesh, vector[weight_t]* weights, float l, float m, int steps):
    cdef Mesh new_mesh = Mesh(other=mesh)
    cdef vector[Point] D = vector[Point](mesh.vertices.shape[0])
    cdef vertex_t* vi
    cdef int s, i
    for s in xrange(steps):
        for i in prange(D.size(), nogil=True):
            D[i] = calc_d(new_mesh, i)

        for i in prange(D.size(), nogil=True):
            mesh.vertices[i][0] += deref(weights)[i]*l*D[i].x;
            mesh.vertices[i][1] += deref(weights)[i]*l*D[i].y;
            mesh.vertices[i][2] += deref(weights)[i]*l*D[i].z;

        for i in prange(D.size(), nogil=True):
            D[i] = calc_d(new_mesh, i)

        for i in prange(D.size(), nogil=True):
            mesh.vertices[i][0] += deref(weights)[i]*m*D[i].x;
            mesh.vertices[i][1] += deref(weights)[i]*m*D[i].y;
            mesh.vertices[i][2] += deref(weights)[i]*m*D[i].z;

    return new_mesh

def ca_smoothing(Mesh mesh, double T, double tmax, double bmin, int n_iters):
    cdef double[3] stack_orientation = [0.0, 0.0, 1.0]
    cdef vector[vertex_id_t]* vertices_staircase =  find_staircase_artifacts(mesh, stack_orientation, T)

    cdef vector[weight_t]* weights = calc_artifacts_weight(mesh, vertices_staircase, tmax, bmin)
    print deref(weights)

    return taubin_smooth(mesh, weights, 0.5, -0.53, n_iters).vertices
