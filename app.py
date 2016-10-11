import sys

import numpy as np
import vtk

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

    print pd.GetCellData().GetNumberOfArrays()
    print pd.GetCellData().GetArray("Normals")

    mesh = cy_mesh.Mesh(pd)

    print mesh.get_near_vertices(2, 1.5)
    cy_mesh.ca_smoothing(mesh, 0.7, 3, 0.2, 10)

    #  view(pd)

if __name__ == "__main__":
    main()
