import numpy as np
import cv2
import math
from pyquaternion import Quaternion

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def read_thumb(obj_file):

    orig_file = []
    np_verts = []

    with open(obj_file) as f:
        vert_num = 0
        for line in f:
            if not line.startswith('v '):
                orig_file.append(line)
                continue
            orig_file.append(line)
            line = line.split(' ')
            coords = line[1:4]
            coords = [float(x) for x in coords]
            np_verts.append(coords)
            vert_num+=1

    np_verts = np.asarray(np_verts, dtype = np.float32)

    rot180 = np.array([[-1,0,0],[0,-1,0], [0,0,1]])
    #ref_coord = np_verts[102]
    np_verts = np.matmul(np_verts,rot180)
    #np_verts = np.dot(np_verts, rotation_matrix(ref_coord, math.pi))
    #print (np_verts[102])
    ref_coord = np_verts[102]
    #np_verts = np.matmul(np_verts,rot180)


    return np_verts, ref_coord, orig_file

def adjust_thumb_pts(obj_file, output_obj = 'thumb_over_phone.obj',
                    key_coord = [[0, 420]]):
    #key_coord is the pixel of a letter

    pixel_pts = np.asarray([
        [0,0],
        [0,632],
        [312, 632],
        [312, 0]],
        dtype = np.float32)

    meter_pts = np.asarray([
        [0,0],
        [0,-.1509],
        [-0.0757, -0.1509 ],
        [-0.0757, 0]],
        dtype = np.float32)

    M = cv2.getPerspectiveTransform(pixel_pts, meter_pts)

    in_pt = np.array([key_coord], dtype=np.float32)

    out_pt = cv2.transform(in_pt, M)[0][0]
    out_pt[0] -= .0757#.082





    thumb_pts, thumb_tip, orig_file = read_thumb(obj_file)
    out_pt[2]= 0
    offset = out_pt - thumb_tip
    thumb_pts += offset
    print (thumb_pts[102])

    ctr = 0

    new_obj = []

    for line in orig_file:
        if not line.startswith('v '):
            new_obj.append(line)
            continue
        line = line.split(' ')

        new_pt = thumb_pts[ctr]
        new_pt = new_pt.tolist()
        new_pt = [str(x) for x in new_pt]
        line[1:4] = new_pt
        line = " ".join(line)
        new_obj.append(line)
        ctr+=1

    with open(output_obj, 'w') as of:
        for line in new_obj:
            of.write(line)



thumb_obj = 'left_thumb.obj'
adjust_thumb_pts(thumb_obj)
