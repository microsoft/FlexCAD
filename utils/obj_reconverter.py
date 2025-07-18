import numpy as np
from collections import OrderedDict
from utils import create_point, create_unit_vec, get_transform, create_sketch_plane

# OCC
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeEdge,
)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire
from OCC.Core.gp import gp_Vec, gp_Ax2, gp_Dir, gp_Circ
from OCC.Extend.DataExchange import write_stl_file


class OBJReconverter:
    """OBJ Data Reconverter"""

    def __init__(self):
        self.vertex_dict = OrderedDict()
        self.PRECISION = 1e-5
        self.eps = 1e-7
        self.x_axis = gp_Dir(1.0, 0.0, 0.0)

    def convert_curve(self, curve):
        """
        convert to json dict format
        """
        json_curve = {}

        if curve.type == "circle":
            json_curve["type"] = "Circle3D"
            json_curve["center_point"] = {
                "x": curve.center[0],
                "y": curve.center[1],
                "z": 0,
            }
            json_curve["radius"] = curve.radius

        if curve.type == "line":
            json_curve["type"] = "Line3D"
            json_curve["start_point"] = {
                "x": curve.start[0],
                "y": curve.start[1],
                "z": 0,
            }
            json_curve["end_point"] = {"x": curve.end[0], "y": curve.end[1], "z": 0}

        if curve.type == "arc":
            json_curve["type"] = "Arc3D"
            json_curve["start_point"] = {
                "x": curve.start[0],
                "y": curve.start[1],
                "z": 0,
            }
            json_curve["end_point"] = {"x": curve.end[0], "y": curve.end[1], "z": 0}
            json_curve["mid_point"] = {"x": curve.mid[0], "y": curve.mid[1], "z": 0}
            json_curve["center_point"] = {
                "x": curve.center[0],
                "y": curve.center[1],
                "z": 0,
            }

        json_curve["is_outer"] = curve.is_outer
        return json_curve

    def convert_vertices(self):
        """Convert all the vertices to .obj format"""
        vertex_strings = ""
        for pt in self.vertex_dict.values():
            # e.g. v 0.123 0.234 0.345 1.0
            vertex_string = f"v {pt[0]} {pt[1]}\n"
            vertex_strings += vertex_string
        return vertex_strings

    def parse_obj(self, faces, meta_info):
        """
        reconstruct brep from obj file
        """
        # At least one needs to match
        for face in faces:
            for loop in face:
                if len(loop) > 1:
                    for idx, curve in enumerate(loop[:-1]):
                        next_curve = np.vstack([loop[idx + 1].start, loop[idx + 1].end])
                        diff1 = np.sum(np.abs(curve.start - next_curve), 1)
                        diff2 = np.sum(np.abs(curve.end - next_curve), 1)

                        if min(diff2) == 0 or min(diff1) == 0:
                            continue  # edge connected

                        assert (
                            min(diff1) < 1e-3 or min(diff2) < 1e-3
                        )  # difference should be small

                        if min(diff1) > min(diff2):
                            min_idx = np.argmin(diff2)
                            if min_idx == 0:
                                loop[idx + 1].start_idx = curve.end_idx
                                loop[idx + 1].start = curve.end
                            else:
                                loop[idx + 1].end_idx = curve.end_idx
                                loop[idx + 1].end = curve.end
                        else:
                            min_idx = np.argmin(diff1)
                            if min_idx == 0:
                                loop[idx + 1].start_idx = curve.start_idx
                                loop[idx + 1].start = curve.start
                            else:
                                loop[idx + 1].end_idx = curve.start_idx
                                loop[idx + 1].end = curve.start

                    # Solve start / end connection
                    shared_idx = list(
                        set([loop[-2].start_idx, loop[-2].end_idx]).intersection(
                            set([loop[-1].start_idx, loop[-1].end_idx])
                        )
                    )

                    assert len(shared_idx) >= 1

                    if len(shared_idx) == 2:
                        assert len(loop) == 2  # do nothing
                    else:
                        if shared_idx[0] == loop[-1].start_idx:
                            do_start = False
                        else:
                            do_start = True
                        start_curve = np.vstack([loop[0].start, loop[0].end])

                        if do_start:
                            diff = np.sum(np.abs(loop[-1].start - start_curve), 1)
                        else:
                            diff = np.sum(np.abs(loop[-1].end - start_curve), 1)
                        assert min(diff) < 1e-3

                        min_idx = np.argmin(diff)
                        if min_idx == 0:
                            if do_start:
                                loop[-1].start_idx = loop[0].start_idx
                                loop[-1].start = loop[0].start
                            else:
                                loop[-1].end_idx = loop[0].start_idx
                                loop[-1].end = loop[0].start
                        else:
                            if do_start:
                                loop[-1].start_idx = loop[0].end_idx
                                loop[-1].start = loop[0].end
                            else:
                                loop[-1].end_idx = loop[0].end_idx
                                loop[-1].end = loop[0].end

        # Parse groups to json loop/curve profile
        extrusion = {}
        extrusion["profiles"] = []
        for face in faces:
            profile = {}
            profile["loops"] = []
            for loop in face:
                pl = {}
                pl["profile_curves"] = []
                for curve in loop:
                    # convert to json format
                    pl["profile_curves"].append(self.convert_curve(curve))
                profile["loops"].append(pl)
            extrusion["profiles"].append(profile)

        # Parse transform
        sketch = {}
        transform = {}
        transform["origin"] = {
            "x": meta_info["t_orig"][0],
            "y": meta_info["t_orig"][1],
            "z": meta_info["t_orig"][2],
        }
        transform["x_axis"] = {
            "x": meta_info["t_x"][0],
            "y": meta_info["t_x"][1],
            "z": meta_info["t_x"][2],
        }
        transform["y_axis"] = {
            "x": meta_info["t_y"][0],
            "y": meta_info["t_y"][1],
            "z": meta_info["t_y"][2],
        }
        transform["z_axis"] = {
            "x": meta_info["t_z"][0],
            "y": meta_info["t_z"][1],
            "z": meta_info["t_z"][2],
        }
        sketch["transform"] = transform

        # Parse extrude
        extrude_params = {}
        extrude_params["extrude_type"] = meta_info["set_op"]
        extrude_params["extrude_values"] = meta_info["extrude_value"]

        # Create sketch
        all_faces = []
        curve_strings = ""
        curve_count = 0
        for profile in extrusion["profiles"]:
            ref_face, face, curve_string, c_count = self.parse_sketch(sketch, profile)
            curve_strings += curve_string
            curve_count += c_count
            all_faces.append(face)

        # Merge all faces in the same plane
        plane_face = all_faces[0]
        for face in all_faces[1:]:
            plane_face = self.my_op(plane_face, face, "fuse")
        solid = self.extrude_face(ref_face, plane_face, extrude_params)
        return solid, curve_strings, curve_count

    def my_op(self, big, small, op_name):
        if op_name == "cut":
            op = BRepAlgoAPI_Cut(big, small)
        elif op_name == "fuse":
            op = BRepAlgoAPI_Fuse(big, small)
        elif op_name == "common":
            op = BRepAlgoAPI_Common(big, small)
        op.SetFuzzyValue(self.PRECISION)
        op.Build()
        return op.Shape()

    def build_body(self, face, normal, value):
        extrusion_vec = gp_Vec(normal).Multiplied(value)
        make_prism = BRepPrimAPI_MakePrism(face, extrusion_vec)
        make_prism.Build()
        prism = make_prism.Prism()
        return prism.Shape()

    def extrudeBasedOnType(self, face, normal, distance):
        # Extrude based on the two bound values
        if not (distance[0] < distance[1]):
            raise Exception("incorrect distance")
        large_value = distance[1]
        small_value = distance[0]

        if large_value == 0:
            return self.build_body(face, -normal, -small_value)
        elif small_value == 0:
            return self.build_body(face, normal, large_value)
        elif np.sign(large_value) == np.sign(small_value):
            if large_value < 0:
                body1 = self.build_body(face, -normal, -small_value)
                body2 = self.build_body(face, -normal, -large_value)
                return self.my_op(body1, body2, "cut")
            else:
                assert large_value > 0
                body1 = self.build_body(face, normal, small_value)
                body2 = self.build_body(face, normal, large_value)
                return self.my_op(body2, body1, "cut")
        else:
            assert np.sign(large_value) != np.sign(small_value)
            body1 = self.build_body(face, normal, large_value)
            body2 = self.build_body(face, -normal, -small_value)
            return self.my_op(body1, body2, "fuse")

    def extrude_face(self, ref_face, face, extrude_params):
        distance = extrude_params["extrude_values"]
        surf = BRepAdaptor_Surface(ref_face).Plane()
        normal = surf.Axis().Direction()
        extruded_shape = self.extrudeBasedOnType(face, normal, distance)
        return extruded_shape

    def parse_sketch(self, sketch, profile):
        """
        Sketch in one closed loop (one out, multiple ins)
        """
        # Transformation from local to global xyz coord
        transform = get_transform(sketch["transform"])

        # Create face region (automatically infer from all wires)
        outer_facelist = []
        inner_facelist = []
        curve_count = 0
        outer_string = []
        inner_string = []
        plane = create_sketch_plane(sketch["transform"])

        for idx, pl in enumerate(profile["loops"]):
            # Create loop
            loop, curve_string, num_curve = self.parse_loop(
                pl["profile_curves"], transform
            )
            # Create face
            face_builder = BRepBuilderAPI_MakeFace(plane, loop)
            if not face_builder.IsDone():
                raise Exception("face builder not done")
            face = face_builder.Face()
            # Fix face
            fixer = ShapeFix_Face(face)
            fixer.SetPrecision(self.PRECISION)
            fixer.FixOrientation()

            analyzer = BRepCheck_Analyzer(fixer.Face())
            if not analyzer.IsValid():
                raise Exception("face check failed")

            curve_count += num_curve

            if pl["profile_curves"][0]["is_outer"]:
                outer_facelist.append(fixer.Face())
                outer_string.append(curve_string)
            else:
                inner_facelist.append(fixer.Face())
                inner_string.append(curve_string)

        # Create final closed loop face
        assert len(outer_facelist) > 0
        final_face = outer_facelist[0]
        for face in outer_facelist[1:]:
            final_face = self.my_op(final_face, face, "fuse")
        for face in inner_facelist:
            final_face = self.my_op(final_face, face, "cut")

        # Append inner outer information to string
        assert len(outer_string) == 1
        out_str = ""
        in_str = ""
        for c_str in outer_string:
            out_str += "out\n" + c_str + "\n"
        for c_str in inner_string:
            in_str += "in\n" + c_str + "\n"
        final_str = "face\n" + out_str + in_str

        return outer_facelist[0], final_face, final_str, curve_count

    def parse_loop(self, profile_loop, transform):
        """Create face in one closed loop"""
        topo_wire = BRepBuilderAPI_MakeWire()
        curve_strings = ""
        curve_count = 0

        # Loop through all the curves in one loop
        for profile_curve in profile_loop:
            curve_edge, curve_string = self.parse_curve(profile_curve, transform)
            topo_wire.Add(curve_edge)
            if not topo_wire.IsDone():
                raise Exception("wire builder not done")

            curve_string += "\n"
            curve_count += 1
            curve_strings += curve_string

        fixer = ShapeFix_Wire()
        fixer.Load(topo_wire.Wire())
        fixer.SetPrecision(self.PRECISION)
        fixer.FixClosed()
        fixer.Perform()
        return fixer.Wire(), curve_strings, curve_count

    def parse_curve(self, curve, transform):
        if curve["type"] == "Line3D":
            return self.create_line(curve, transform)
        elif curve["type"] == "Circle3D":
            return self.create_circle(curve, transform)
        elif curve["type"] == "Arc3D":
            return self.create_arc(curve, transform)
        else:
            raise Exception("unknown curve type")

    def create_line(self, line, transform):
        start = create_point(line["start_point"], transform)
        end = create_point(line["end_point"], transform)
        if start.Distance(end) == 0:
            raise Exception("start/end point same location")
        topo_edge = BRepBuilderAPI_MakeEdge(start, end)

        # Save pre-transform
        star_idx = self.save_vertex(
            line["start_point"]["x"] + 0.0, line["start_point"]["y"] + 0.0, "p"
        )
        end_idx = self.save_vertex(
            line["end_point"]["x"] + 0.0, line["end_point"]["y"] + 0.0, "p"
        )
        curve_string = f"l {star_idx} {end_idx}"
        return topo_edge.Edge(), curve_string

    def create_arc(self, arc, transform):
        start = create_point(arc["start_point"], transform)
        mid = create_point(arc["mid_point"], transform)
        end = create_point(arc["end_point"], transform)
        arc_occ = GC_MakeArcOfCircle(start, mid, end).Value()
        topo_edge = BRepBuilderAPI_MakeEdge(arc_occ)

        # Save pre-transform
        start_idx = self.save_vertex(
            arc["start_point"]["x"] + 0.0, arc["start_point"]["y"] + 0.0, "p"
        )
        end_idx = self.save_vertex(
            arc["end_point"]["x"] + 0.0, arc["end_point"]["y"] + 0.0, "p"
        )
        center_idx = self.save_vertex(
            arc["center_point"]["x"] + 0.0, arc["center_point"]["y"] + 0.0, "p"
        )
        mid_idx = self.save_vertex(
            arc["mid_point"]["x"] + 0.0, arc["mid_point"]["y"] + 0.0, "p"
        )
        curve_string = f"a {start_idx} {mid_idx} {center_idx} {end_idx}"
        return topo_edge.Edge(), curve_string

    def create_circle(self, circle, transform):
        center = create_point(circle["center_point"], transform)
        radius = circle["radius"]
        normal = create_unit_vec({"x": 0.0, "y": 0.0, "z": 1.0}, transform)
        ref_vector3d = self.x_axis.Transformed(transform)
        axis = gp_Ax2(center, normal, ref_vector3d)
        gp_circle = gp_Circ(axis, abs(float(radius)))
        topo_edge = BRepBuilderAPI_MakeEdge(gp_circle)

        center_idx = self.save_vertex(
            circle["center_point"]["x"] + 0.0, circle["center_point"]["y"] + 0.0, "p"
        )
        radius_idx = self.save_vertex(abs(float(radius)) + 0.0, 0, "r")
        curve_string = f"c {center_idx} {radius_idx}"
        return topo_edge.Edge(), curve_string

    def save_vertex(self, h_x, h_y, text):
        unique_key = f"{text}:x{h_x}y{h_y}"
        index = 0
        for key in self.vertex_dict.keys():
            # Vertex location already exist in dict
            if unique_key == key:
                return index
            index += 1
        # Vertex location does not exist in dict
        self.vertex_dict[unique_key] = [h_x, h_y]
        return index
