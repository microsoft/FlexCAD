import os
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_XYZ, gp_Ax3, gp_Trsf, gp_Pln
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer

def create_xyz(xyz):
    return gp_XYZ(xyz["x"], xyz["y"], xyz["z"])


def get_ax3(transform_dict):
    origin = create_xyz(transform_dict["origin"])
    x_axis = create_xyz(transform_dict["x_axis"])
    y_axis = create_xyz(transform_dict["y_axis"])
    z_axis = create_xyz(transform_dict["z_axis"])
    # Create new coord (orig, Norm, x-axis)
    axis3 = gp_Ax3(gp_Pnt(origin), gp_Dir(z_axis), gp_Dir(x_axis))
    return axis3


def get_transform(transform_dict):
    axis3 = get_ax3(transform_dict)
    transform_to_local = gp_Trsf()
    transform_to_local.SetTransformation(axis3)
    return transform_to_local.Inverted()


def create_sketch_plane(transform_dict):
    axis3 = get_ax3(transform_dict)
    return gp_Pln(axis3)


def create_point(point_dict, transform):
    pt2d = gp_Pnt(point_dict["x"], point_dict["y"], point_dict["z"])
    return pt2d.Transformed(transform)


def create_unit_vec(vec_dict, transform):
    vec2d = gp_Dir(vec_dict["x"], vec_dict["y"], vec_dict["z"])
    return vec2d.Transformed(transform)


def write_stl_file(a_shape, filename, mode="ascii", linear_deflection=0.001, angular_deflection=0.5):
    """ export the shape to a STL file
    Be careful, the shape first need to be explicitely meshed using BRepMesh_IncrementalMesh
    a_shape: the topods_shape to export
    filename: the filename
    mode: optional, "ascii" by default. Can either be "binary"
    linear_deflection: optional, default to 0.001. Lower, more occurate mesh
    angular_deflection: optional, default to 0.5. Lower, more accurate_mesh
    """
    if a_shape.IsNull():
        raise AssertionError("Shape is null.")
    if mode not in ["ascii", "binary"]:
        raise AssertionError("mode should be either ascii or binary")
    if os.path.isfile(filename):
        print("Warning: %s file already exists and will be replaced" % filename)
    # first mesh the shape
    mesh = BRepMesh_IncrementalMesh(a_shape, linear_deflection, False, angular_deflection, True)
    #mesh.SetDeflection(0.05)
    mesh.Perform()
    if not mesh.IsDone():
        raise AssertionError("Mesh is not done.")

    stl_exporter = StlAPI_Writer()
    if mode == "ascii":
        stl_exporter.SetASCIIMode(True)
    else:  # binary, just set the ASCII flag to False
        stl_exporter.SetASCIIMode(False)
    stl_exporter.Write(a_shape, filename)

    if not os.path.isfile(filename):
        raise IOError("File not written to disk.")
