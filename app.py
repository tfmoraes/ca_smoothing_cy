import sys

import numpy as np
import vtk

from vtk.util import numpy_support

import cy_mesh

def view(pd):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.Update()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)

    rwin = vtk.vtkRenderWindow()
    rwin.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(rwin)
    iren.Start()

def main():
    fname = sys.argv[1]
    foutput = sys.argv[2]

    r = vtk.vtkPLYReader()
    r.SetFileName(fname)
    r.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(r.GetOutput())
    normals.ComputeCellNormalsOn()
    normals.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(normals.GetOutputPort())
    clean.Update()

    pd = clean.GetOutput()
    mesh = cy_mesh.Mesh(pd)

    cy_mesh.ca_smoothing(mesh, 0.7, 3, 0.7, 10)

    pd.Modified()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(pd)
    normals.ComputeCellNormalsOn()
    normals.Update()

    view(normals.GetOutput())

    w = vtk.vtkPolyDataWriter()
    w.SetInputData(pd)
    w.SetFileName(foutput)
    w.Update()
    w.Write()

if __name__ == "__main__":
    main()
