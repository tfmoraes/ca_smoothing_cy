cimport numpy as np

from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI
from libc.stdlib cimport abs as cabs
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.deque cimport deque as cdeque

from cy_my_types cimport vertex_t, normal_t, face_t

import numpy as np
import vtk

from vtk.util import numpy_support

cdef class Mesh:
    cdef vertex_t[:, :] vertices
    cdef face_t[:, :] faces
    cdef normal_t[:, :] normals

    cdef unordered_map[int, vector[face_t]] map_vface

    def __init__(self, pd):
        vertices = numpy_support.vtk_to_numpy(pd.GetPoints().GetData())
        vertices.shape = -1, 3

        faces = numpy_support.vtk_to_numpy(pd.GetPolys().GetData())
        faces.shape = -1, 4

        normals = numpy_support.vtk_to_numpy(pd.GetCellData().GetArray("Normals"))
        normals.shape = -1, 3

        print ">>>", normals.dtype

        self.vertices = vertices
        self.faces = faces
        self.normals = normals

        cdef int i

        for i in xrange(faces.shape[0]):
            self.map_vface[self.faces[i, 1]].push_back(i)
            self.map_vface[self.faces[i, 2]].push_back(i)
            self.map_vface[self.faces[i, 3]].push_back(i)

    cdef vector[face_t]* get_faces_by_vertex(self, int v_id) nogil:
        return &self.map_vface[v_id]

    cdef vector[face_t]* get_near_vertices_to_v(self, face_t v_id, float dmax) nogil:
        cdef vector[face_t]* idfaces
        cdef vector[face_t]* near_vertices = new vector[face_t]()

        cdef cdeque[face_t] to_visit
        cdef unordered_map[face_t, bool] status_v
        cdef unordered_map[face_t, bool] status_f

        cdef vertex_t *vip
        cdef vertex_t *vjp

        cdef float distance
        cdef int nf, nid, j
        cdef face_t f_id, vj

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

    cpdef get_near_vertices(self, face_t v_id, float dmax):
        cdef vector[face_t] *vertices = self.get_near_vertices_to_v(v_id, dmax)
        print "percorrendo", vertices.size(), v_id, dmax
        cdef vector[face_t].iterator it = vertices.begin()
        return deref(vertices)


cdef vector[face_t] find_staircase_artifacts(Mesh mesh, double[3] stack_orientation, double T) nogil:
    cdef int nv, nf, f_id, v_id
    cdef double of_z, of_y, of_x, min_z, max_z, min_y, max_y, min_x, max_x;
    cdef vector[face_t]* f_ids
    cdef normal_t* normal

    cdef vector[face_t] output
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

def ca_smoothing(Mesh mesh, double T, double tmax, double bmin, int n_iters):
    cdef double[3] stack_orientation = [0.0, 0.0, 1.0]
    print find_staircase_artifacts(mesh, stack_orientation, T)
