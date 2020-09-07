import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import ConfigParser

import camera_utils


config = ConfigParser.ConfigParser()
config.readfp(open(r'env_config.txt'))
rotation_vector = [float(config.get('CAM_P3D_SETTINGS', x)) for x in ['Rx', 'Ry', 'Rz']]
center_vector = [float(config.get('CAM_P3D_SETTINGS', x))/2. for x in ['tx', 'ty']]
center_vector.append(float(config.get('CAM_P3D_SETTINGS', 'tz')))
fx = float(config.get('CAM_P3D_SETTINGS', 'fx'))
fy = float(config.get('CAM_P3D_SETTINGS', 'fy'))
height = float(config.get('CAM_P3D_SETTINGS', 'height'))
width = float(config.get('CAM_P3D_SETTINGS', 'width'))

center_vector = np.array([-1.73472347598e-18, -3.46944695195e-18,  float(config.get('CAM_P3D_SETTINGS', 'tz'))])

def adjust_cv2_color(cv2_im):
    red = cv2_im[:,:,2].copy()
    blue = cv2_im[:,:,0].copy()
    cv2_im[:,:,0] = red
    cv2_im[:,:,2] = blue

    return cv2_im

def display_images(list_of_ims, corners = []):

    num_subplots = len(list_of_ims)
    fig,ax = plt.subplots(1,num_subplots)

    for idx, im_type in enumerate(list_of_ims):
        img = cv2.imread(im_type)
        img = adjust_cv2_color(img)
        ax[idx].imshow(img)

    if len(corners) > 0:
        print corners
        plt.scatter(corners[:,0], corners[:,1])
    plt.show()

reference_im = 'xr_.png'
capture_im = 'render.png'
homography_im = 'im_homography.png'

phone_pts = np.array([
        [0.,0.,0., 1.],
        [0,-0.1509,0., 1.],
        [-0.0757 ,-0.1509 , 0., 1.],
        [-0.0757, 0., 0., 1.]
        ])

phone_pts = np.array([
    [ 0.0550  , -0.0017 ,  -0.0640,1],
    [-0.0015  , -0.0518  ,  0.0666,1],
    [-0.0550  ,  0.0017  ,  0.0640,1],
    [ 0.0015  ,  0.0518 ,  -0.0666,1]
])

ext_matrix = camera_utils.create_extrinsic_matrix(
                            rotation_vector,
                            center_vector)

int_matrix = camera_utils.create_intrinsic_matrix(
                fx, fy, width/2., height/2.)


camera_matrix = np.matmul(int_matrix, ext_matrix)
points_2d = np.matmul(camera_matrix, phone_pts.T).T
w_coords = points_2d[:,2]
points_2d = points_2d / w_coords[:,None]

phone_pts = np.asarray(phone_pts[:,:2], dtype = np.float32)
points_2d = np.asarray( points_2d[:,:2], dtype = np.float32)

phone_pts = np.asarray([
    [0,0],
    [0,632],
    [312, 632],
    [312, 0]],
    dtype = np.float32)

h, status = cv2.findHomography(points_2d, phone_pts)

im_src = cv2.imread(capture_im)
im_src = adjust_cv2_color(im_src)

im_dest = cv2.imread(reference_im)
im_dest = adjust_cv2_color(im_dest)

im_out = cv2.warpPerspective(im_src, h, (im_dest.shape[1], im_dest.shape[0]))
im_out = adjust_cv2_color(im_out)

cv2.imwrite(homography_im, im_out)

display_images([reference_im, homography_im ,capture_im], points_2d)
