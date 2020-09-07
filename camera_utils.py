import numpy as np
from PIL import Image
##Camera info for iPhone 6plus
# http://lcdtech.info/en/data/pixel.size.htm
from pyquaternion import Quaternion

def print_obj_verts(points_3d):
    #prints out the obj format of the 3d points
    row_nums = points_3d.shape[0]


    points_3d = points_3d.tolist()
    return_str = ''

    for idx, row in enumerate(points_3d):
        row_str =  "".join(format(x, "10.4f") for x in row)
        row_str = 'v ' + row_str
        if idx == row_nums-1:
            return_str = return_str + row_str
        else:
            return_str = return_str + row_str + '\n'

    return return_str

def back_2_3d(points_4d):
    #divides w if not 0
    #Turn homogeneous points to non

    assert(points_4d.shape[1] == 4)

    #idxs where w (last col) has zeros
    zero_idxs = np.where(points_4d[:,3] == 0)
    points_4d[zero_idxs, 3] = 1.
    points_3d = points_4d[:,:3]
    w_coords = points_4d[:,3]
    points_3d = points_3d / w_coords[:,None]
    print_obj_verts(points_3d)
    return points_3d

def create_extrinsic_matrix(rotation_vector, center_vector):
    last_row = np.array([0., 0., 0., 1.])

    center_vector = np.array([center_vector[0]*-1 , center_vector[1]*-1 , center_vector[2]*-1])
    center_vector = np.array(center_vector)[np.newaxis].T
    q1 = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(rotation_vector[0]))
    q2 = Quaternion(axis=[0, 1, 0], angle=np.deg2rad(rotation_vector[1]))
    q3 = Quaternion(axis=[0, 0, 1], angle=np.deg2rad(rotation_vector[2]))

    q4 = q1*q2*q3
    rotation_matrix =  q4.rotation_matrix.T
    
    eye_3 = np.eye(3)
    center_mat = np.hstack((eye_3, center_vector))
    center_mat = np.vstack((center_mat, last_row))
    rotation_matrix = np.hstack((rotation_matrix, np.zeros(3)[np.newaxis].T))
    rotation_matrix = np.vstack((rotation_matrix, last_row))
    ext_matrix = np.matmul(rotation_matrix, center_mat)

    return ext_matrix

def create_intrinsic_matrix(fx, fy, cx, cy, skew=0.):

    int_matrix = np.array((
        [fx, skew, cx ,0.],
        [0., fy, cy, 0.],
        [0., 0., 1., 0.]
    ))

    return int_matrix

#create method for dictionary
