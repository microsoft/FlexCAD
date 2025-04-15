import pickle
import argparse
import numpy as np
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

def create_curve_str(se_xy, se_cmd):
    curve_str = ""
    xy_offset = 0
    if se_cmd == 0:  # line
        curve_str = " line," + ",".join(str(x) for x in se_xy[0])
        xy_offset = 2
    elif se_cmd == 1:  # arc
        curve_str = " arc," + ",".join(str(x) for x in se_xy[0:2].flatten())
        xy_offset = 3
    elif se_cmd == 2:  # circle
        if args.circle_type == 'udlr':
            curve_str = " circle," + ",".join(str(x) for x in se_xy[0:4].flatten())
        elif args.circle_type == 'ldru':
            se_xyx = np.concatenate([se_xy[3], se_xy[1], se_xy[2], se_xy[0]], axis=0)
            curve_str = " circle," + ",".join(str(x) for x in se_xyx) # counter-clockwise
        elif args.circle_type == 'diam':
            curve_str = " circle," + ",".join(str(x) for x in se_xy[0:4].flatten()) # diameter
        elif args.circle_type == 'or':
            se_xyx = np.concatenate([(se_xy[0] + se_xy[1]) // 2, np.array([(se_xy[0, 1] - se_xy[1, 1]) // 2])], axis=0)
            curve_str = " circle," + ",".join(str(x) for x in se_xyx.flatten()) # centern & radius
        xy_offset = 5
    curve_str += " <curve_end>"
    return curve_str, xy_offset


def create_sketch_str(se_xy, se_cmd):
    sketch_str = ""
    len_xy, len_cmd = len(se_xy), len(se_cmd)
    xy_idx = 0
    for cmd_item in se_cmd:  # for each command
        if 0 <= cmd_item <= 2:  # curve
            curve_str, xy_offset = create_curve_str(se_xy[xy_idx:], cmd_item)
            sketch_str += curve_str
            xy_idx += xy_offset
        elif cmd_item == -1:  # loop
            sketch_str += " <loop_end>"
            xy_idx += 1
        elif cmd_item == -2:  # face
            sketch_str += " <face_end>"
            xy_idx += 1
        elif cmd_item == -3:  # sketch
            sketch_str += " <sketch_end>"
            xy_idx += 1
        else:
            raise ValueError("Invalid command: " + str(cmd_item))
    if xy_idx != len_xy:
        raise ValueError("xy_idx != len_xy")
    return sketch_str


def create_extrude_str(se_ext):
    extrude_str = ""
    # extrude operation
    if se_ext[14] == 1:
        extrude_str += "add"
    elif se_ext[14] == 2:
        extrude_str += "cut"
    elif se_ext[14] == 3:
        extrude_str += "intersect"
    else:
        raise ValueError("Invalid extrude operation: " + str(se_ext[14]))
    # other extrude parameters
    extrude_str = (
        extrude_str + "," + ",".join(str(x - EXT_PAD) for x in se_ext[0:5])
    )  # ext_v, ext_T
    extrude_str = (
        extrude_str + "," + ",".join(str(x - R_PAD) for x in se_ext[5:14])
    )  # ext_R
    extrude_str = (
        extrude_str + "," + ",".join(str(x - EXT_PAD) for x in se_ext[15:18])
    )  # scale, offset
    # extrude end
    extrude_str += " <extrusion_end>"
    return extrude_str


def convert(in_path, out_path):
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    print("Data loaded: " + str(len(data)) + " samples")

    results = []
    for item in data:  # for each data
        se_str = ""
        num_se = item["num_se"]
        for se_idx in range(num_se):  # for each sketch-extrude
            xy, cmd, ext = (
                item["se_xy"][se_idx] - COORD_PAD,
                item["se_cmd"][se_idx] - CMD_PAD,
                item["se_ext"][se_idx],
            )
            se_str = se_str + " " + create_sketch_str(xy, cmd).strip()
            se_str = se_str + " " + create_extrude_str(ext).strip()
        results.append(se_str.strip())

    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print("Data converted: " + str(len(results)) + " samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--circle_type", type=str, required=True)
    args = parser.parse_args()

    convert(args.in_path, args.out_path)
