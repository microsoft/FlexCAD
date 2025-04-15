import numpy as np
from collections import OrderedDict
import re
from pathlib import Path
import argparse
import os
import json
import math
import pickle

# hyperparameters from SkexGen project
SKETCH_R = 1
RADIUS_R = 1
EXTRUDE_R = 1.0
SCALE_R = 1.4
OFFSET_R = 0.9
PIX_PAD = 4
CMD_PAD = 3
COORD_PAD = 4
EXT_PAD = 1
EXTRA_PAD = 1
R_PAD = 2


class CADparser:
    """Parse CAD sequence to CAD object."""

    def __init__(self, bit):
        self.vertex_dict = OrderedDict()
        self.bit = bit

    def perform(self, cad_seq):
        # divide into sketch and extrude
        sketches, extrudes = self.get_SE(cad_seq)
        if sketches is None or extrudes is None:
            return None
        # sequentially parse each pair of SE into obj
        se_datas = []
        for sketch, extrude in zip(sketches, extrudes):
            extrude_param, scale, offset = self.parse_extrude(extrude)
            if extrude_param is None or scale is None or offset is None:
                return None
            vertex_str, se_str = self.parse_sketch(sketch, scale, offset)
            if vertex_str is None or se_str is None:
                return None
            se_datas.append(
                {"vertex": vertex_str, "curve": se_str, "extrude": extrude_param}
            )
            self.vertex_dict.clear()

        return se_datas

    def parse_sketch(self, sketch, scale, offset):
        faces = self.get_faces(sketch)
        if len(faces) == 0:
            return None, None
        se_str = ""
        for face_idx, face in enumerate(faces):  # each face
            face_str = "face\n"
            loops = self.get_loops(face)
            if len(loops) == 0:
                return None, None
            for loop_idx, loop in enumerate(loops):  # each loop
                curves = self.get_curves(loop)
                if len(curves) == 0:
                    return None, None
                next_curves = curves[1:]
                next_curves += curves[:1]
                cur_str = []
                for curve, next_curve in zip(curves, next_curves):  # each curve
                    if not self.obj_curve(curve, next_curve, cur_str, scale, offset):
                        return None, None
                loop_str = ""
                for c in cur_str:
                    loop_str += f"{c}\n"
                if loop_idx == 0:
                    face_str += f"out\n{loop_str}\n"
                else:
                    face_str += f"in\n{loop_str}\n"
            se_str += face_str
        vertex_str = self.convert_vertices()
        return vertex_str, se_str

    def parse_extrude(self, extrude):
        ext = extrude.split(",")
        if len(ext) != 18:
            return None, None, None

        # operation str to int
        ext_op = {"add": 1, "cut": 2, "intersect": 3}.get(ext[0], None)
        if ext_op is None:
            return None, None, None
        # dequantize ext_v, ext_T, scale and offset
        ext_v, ext_T, scale, offset = self.dequantize_extrude_params(ext)
        # get ext_R
        ext_R = np.array(ext[6:15], dtype=int)

        extrude_param = {"value": ext_v, "T": ext_T, "R": ext_R, "op": ext_op}
        return extrude_param, scale, offset

    def obj_curve(self, curve, next_curve, cur_str, scale, offset):
        cur = curve.split(",")
        next_cur = next_curve.split(",")
        if cur[0] == "circle":
            if len(cur) != 9:
                return False
            p1, p2, p3, p4 = self.dequantize_circle_points(
                cur, next_cur, scale, offset)
            center = np.asarray([0.5 * (p4[0] + p2[0]), 0.5 * (p3[1] + p1[1])])
            radius = (np.linalg.norm(p4- p2) + np.linalg.norm(p3 - p1)) / 4.0

            center = center * scale + offset
            radius = radius * scale

            center_idx = self.save_vertex(center[0], center[1], "p")
            radius_idx = self.save_vertex(radius, 0.0, "r")
            cur_str.append(f"c {center_idx} {radius_idx}")
        elif cur[0] == "arc":
            if len(cur) != 5:
                return False
            if (
                cur[1:3] == cur[3:5]
                or cur[1:3] == next_cur[1:3]
                or cur[3:5] == next_cur[3:5]
            ):  # invalid arc
                return False
            start_v, mid_v, end_v = self.dequantize_arc_points(
                cur, next_cur, scale, offset
            )
            try:
                center, _, _, _ = find_arc_geometry(start_v, mid_v, end_v)
            except Exception:
                return False
            start_v = start_v * scale + offset
            mid_v = mid_v * scale + offset
            end_v = end_v * scale + offset
            center = center * scale + offset

            center_idx = self.save_vertex(center[0], center[1], "p")
            start_idx = self.save_vertex(start_v[0], start_v[1], "p")
            mid_idx = self.save_vertex(mid_v[0], mid_v[1], "p")
            end_idx = self.save_vertex(end_v[0], end_v[1], "p")
            cur_str.append(f"a {start_idx} {mid_idx} {center_idx} {end_idx}")
        elif cur[0] == "line":
            if len(cur) != 3:
                return False
            if cur[1:3] == next_cur[1:3]:
                return False
            start_v, end_v = self.dequantize_line_points(
                cur, next_cur, scale, offset)
            start_v = start_v * scale + offset
            end_v = end_v * scale + offset

            start_idx = self.save_vertex(start_v[0], start_v[1], "p")
            end_idx = self.save_vertex(end_v[0], end_v[1], "p")
            cur_str.append(f"l {start_idx} {end_idx}")
        else:
            return False
        return True

    def get_SE(self, cad_seq):
        # sketches: 1) between sequence start and sketch_end,
        sketches_from_start = re.findall(r"^(.+?)(?=<sketch_end>)", cad_seq)
        # sketches: 2) between extrude_end and sketch_end
        sketches_after_extrude = re.findall(
            r"(?<=<extrusion_end>)(.+?)(?=<sketch_end>)", cad_seq
        )
        sketches = [x.strip() for x in sketches_from_start] + [
            x.strip() for x in sketches_after_extrude
        ]
        # extrudes: between sketch_end and extrude_end
        extrudes = [
            x.strip() for x in re.findall(r"<sketch_end>(.+?)<extrusion_end>", cad_seq)
        ]
        if len(sketches) != len(extrudes):
            return None, None
        return sketches, extrudes

    def get_faces(self, sketch):
        faces = sketch.split("<face_end>")
        return [x.strip() for x in faces if x.strip() != ""]

    def get_loops(self, face):
        loops = face.split("<loop_end>")
        return [x.strip() for x in loops if x.strip() != ""]

    def get_curves(self, loop):
        curves = loop.split("<curve_end>")
        return [x.strip() for x in curves if x.strip() != ""]

    def dequantize_circle_points(self, curve, next_curve, scale, offset):
        p1 = dequantize_verts(
            np.array(curve[1:3], dtype=int),
            n_bits=self.bit,
            min_range=-SKETCH_R,
            max_range=SKETCH_R,
            add_noise=False,
        )
        p2 = dequantize_verts(
            np.array(curve[3:5], dtype=int),
            n_bits=self.bit,
            min_range=-SKETCH_R,
            max_range=SKETCH_R,
            add_noise=False,
        )
        p3 = dequantize_verts(
            np.array(curve[5:7], dtype=int),
            n_bits=self.bit,
            min_range=-SKETCH_R,
            max_range=SKETCH_R,
            add_noise=False,
        )
        p4 = dequantize_verts(
            np.array(curve[7:9], dtype=int),
            n_bits=self.bit,
            min_range=-SKETCH_R,
            max_range=SKETCH_R,
            add_noise=False,
        )
        return p1, p2, p3, p4

    def dequantize_arc_points(self, curve, next_curve, scale, offset):
        start_v = dequantize_verts(
            np.array(curve[1:3], dtype=int),
            n_bits=self.bit,
            min_range=-SKETCH_R,
            max_range=SKETCH_R,
            add_noise=False,
        )
        mid_v = dequantize_verts(
            np.array(curve[3:5], dtype=int),
            n_bits=self.bit,
            min_range=-SKETCH_R,
            max_range=SKETCH_R,
            add_noise=False,
        )
        end_v = dequantize_verts(
            np.array(next_curve[1:3], dtype=int),
            n_bits=self.bit,
            min_range=-SKETCH_R,
            max_range=SKETCH_R,
            add_noise=False,
        )
        return start_v, mid_v, end_v

    def dequantize_line_points(self, curve, next_curve, scale, offset):
        start_v = dequantize_verts(
            np.array(curve[1:3], dtype=int),
            n_bits=self.bit,
            min_range=-SKETCH_R,
            max_range=SKETCH_R,
            add_noise=False,
        )
        end_v = dequantize_verts(
            np.array(next_curve[1:3], dtype=int),
            n_bits=self.bit,
            min_range=-SKETCH_R,
            max_range=SKETCH_R,
            add_noise=False,
        )
        return start_v, end_v

    def dequantize_extrude_params(self, extrude):
        ext_v = dequantize_verts(
            np.array(extrude[1:3], dtype=int),
            n_bits=self.bit,
            min_range=-EXTRUDE_R,
            max_range=EXTRUDE_R,
            add_noise=False,
        )
        ext_T = dequantize_verts(
            np.array(extrude[3:6], dtype=int),
            n_bits=self.bit,
            min_range=-EXTRUDE_R,
            max_range=EXTRUDE_R,
            add_noise=False,
        )
        scale = dequantize_verts(
            np.array(extrude[15], dtype=int),
            n_bits=self.bit,
            min_range=0.0,
            max_range=SCALE_R,
            add_noise=False,
        )
        offset = dequantize_verts(
            np.array(extrude[16:18], dtype=int),
            n_bits=self.bit,
            min_range=-OFFSET_R,
            max_range=OFFSET_R,
            add_noise=False,
        )
        return ext_v, ext_T, scale, offset

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

    def convert_vertices(self):
        """Convert all the vertices to .obj format"""
        vertex_strings = ""
        for pt in self.vertex_dict.values():
            # e.g. v 0.123 0.234 0.345 1.0
            vertex_string = f"v {pt[0]} {pt[1]}\n"
            vertex_strings += vertex_string
        return vertex_strings


def find_arc_geometry(a, b, c):
    A = b[0] - a[0]
    B = b[1] - a[1]
    C = c[0] - a[0]
    D = c[1] - a[1]

    E = A*(a[0] + b[0]) + B*(a[1] + b[1])
    F = C*(a[0] + c[0]) + D*(a[1] + c[1])

    G = 2.0*(A*(c[1] - b[1])-B*(c[0] - b[0]))

    if G == 0:
        raise Exception("zero G")

    p_0 = (D*E - B*F) / G
    p_1 = (A*F - C*E) / G

    center = np.array([p_0, p_1])
    radius = np.linalg.norm(center - a)

    angles = []
    for xx in [a, b, c]:
        angle = angle_from_vector_to_x(xx - center)
        angles.append(angle)

    ab = b-a
    ac = c-a
    cp = np.cross(ab, ac)
    if cp >= 0:
        start_angle_rads = angles[0]
        end_angle_rads = angles[2]
    else:
        start_angle_rads = angles[2]
        end_angle_rads = angles[0]

    return center, radius, start_angle_rads, end_angle_rads


def angle_from_vector_to_x(vec):
    assert vec.size == 2
    # We need to find a unit vector
    angle = 0.0

    l = np.linalg.norm(vec)
    uvec = vec/l

    # 2 | 1
    # -------
    # 3 | 4
    if uvec[0] >= 0:
        if uvec[1] >= 0:
            # Qadrant 1
            angle = math.asin(uvec[1])
        else:
            # Qadrant 4
            angle = 2.0*math.pi - math.asin(-uvec[1])
    else:
        if vec[1] >= 0:
            # Qadrant 2
            angle = math.pi - math.asin(uvec[1])
        else:
            # Qadrant 3
            angle = math.pi + math.asin(-uvec[1])
    return angle


def dequantize_verts(verts, n_bits=8, min_range=-0.5, max_range=0.5, add_noise=False):
    """Convert quantized vertices to floats."""
    range_quantize = 2**n_bits - 1
    verts = verts.astype("float32")
    verts = verts * (max_range - min_range) / range_quantize + min_range
    return verts


def write_obj_sample(save_folder, data):
    for idx, write_data in enumerate(data):
        obj_name = Path(save_folder).stem + "_" + \
            str(idx).zfill(3) + "_param.obj"
        obj_file = Path(save_folder) / obj_name
        extrude_param = write_data["extrude"]
        vertex_strings = write_data["vertex"]
        curve_strings = write_data["curve"]

        """Write an .obj file with the curves and verts"""
        if extrude_param["op"] == 1:  # 'add'
            set_op = "NewBodyFeatureOperation"
        elif extrude_param["op"] == 2:  # 'cut'
            set_op = "CutFeatureOperation"
        elif extrude_param["op"] == 3:  # 'cut'
            set_op = "IntersectFeatureOperation"

        with open(obj_file, "w") as fh:
            # Write Meta info
            fh.write("# WaveFront *.obj file\n")
            fh.write("# ExtrudeOperation: " + set_op + "\n")
            fh.write("\n")

            # Write vertex and curve
            fh.write(vertex_strings)
            fh.write("\n")
            fh.write(curve_strings)
            fh.write("\n")

            # Write extrude value
            extrude_string = "Extrude "
            for value in extrude_param["value"]:
                extrude_string += str(value) + " "
            fh.write(extrude_string)
            fh.write("\n")

            # Write refe plane value
            p_orig = parse3d_sample(extrude_param["T"])
            x_axis = parse3d_sample(extrude_param["R"][0:3])
            y_axis = parse3d_sample(extrude_param["R"][3:6])
            z_axis = parse3d_sample(extrude_param["R"][6:9])
            fh.write("T_origin " + p_orig)
            fh.write("\n")
            fh.write("T_xaxis " + x_axis)
            fh.write("\n")
            fh.write("T_yaxis " + y_axis)
            fh.write("\n")
            fh.write("T_zaxis " + z_axis)


def parse3d_sample(point3d):
    x = point3d[0]
    y = point3d[1]
    z = point3d[2]
    return str(x) + " " + str(y) + " " + str(z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()
    print(args.in_path)
    with open(args.in_path, "r") as f:
        data = f.readlines()
    # print(data)
    data = [json.loads(x.replace('\n', '')) for x in data]


    num_valid_str = 0

    for idx, item in enumerate(data):
        try:

            cad_parser = CADparser(bit=6)
            parsed_data = cad_parser.perform(item["output"])  # ["output"]
            out_path = os.path.join(args.out_path, str(idx).zfill(5))
            os.makedirs(out_path, exist_ok=True)
            if parsed_data is not None:
                num_valid_str += 1
                write_obj_sample(out_path, parsed_data)
        except:
            print(item["output"])
    print(f"Number of valid CAD strings: {num_valid_str}/{len(data)}")

