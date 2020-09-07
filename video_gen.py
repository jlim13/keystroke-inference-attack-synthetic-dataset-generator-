import numpy as np
import configparser
import cv2
import matplotlib.pyplot as plt
import threading
import os
import glob
from PIL import Image
import panda3d as p3d
import panda3d.core
import random
from os.path import relpath
import tifffile
from PIL import Image
import csv
#
import camera_utils
import adjusted_render


config = configparser.ConfigParser()
config.readfp(open(r'env_config.txt'))

WIDTH = int(config.get('CAM_P3D_SETTINGS', 'width'))
HEIGHT = int(config.get('CAM_P3D_SETTINGS', 'height'))
FX = float(config.get('CAM_P3D_SETTINGS', 'fx'))
FY = float(config.get('CAM_P3D_SETTINGS', 'fy'))
CAM_CEN_X = float(config.get('CAM_P3D_SETTINGS', 'tx'))/2.
CAM_CEN_Y = float(config.get('CAM_P3D_SETTINGS', 'ty'))/2.


############################################### VARIABLES
template_pts = np.array([
        [0.,0.,0., 1.],
        [0,-0.1509, 0., 1.],
        [-0.0757 ,-0.1509 , 0., 1.],
        [-0.0757, 0., 0., 1.]
        ])

############################################### METHODS
def create_random_position():
    '''
    randomly will output how the
    phone is rotated and the adversary's distance

    ranges below:
    Rx = [0, 75]
    Ry = [-55, 55]
    Rz = [-90, 90]
    tz = [-1., -8.]
    '''

    new_rx = random.uniform(0, 65)
    new_ry = random.uniform(-40,40)
    new_rz = random.uniform(-75, 75)
    new_tz = random.uniform(-1, -11)

    return new_rx, new_ry, new_rz, new_tz

def write_obj(rotated_pts, new_obj_file, base_obj = 'sample.obj'):

    found_verts = False
    with open(base_obj) as base_file, open(new_obj_file, "w+") as new_file:
        for line in base_file:
            if line == '\n':
                continue
            elif line.startswith('v '):
                if not found_verts:
                    new_file.write(rotated_pts)
                    new_file.write('\n')
                    found_verts = True
                else:
                    continue
            else:
                new_file.write(line)

def update_mtl_file(new_img_path, mtl):
    to_replace = ''
    curr_path = os.getcwd()
    image_rel_path = relpath( new_img_path,curr_path)

    with open(mtl, 'r') as file :
        for line in file:
            if line.startswith('map_Kd'):
                to_replace = line
        file.close()

    with open(mtl, 'r') as file :
        filedata = file.read()

    map_kd = 'map_Kd ' + image_rel_path + '\n'
    # Replace the target string
    filedata = filedata.replace(to_replace, map_kd)

    # Write the file out again
    with open(mtl, 'w') as file:
        file.write(filedata)

def rotate_phone(phone_pts, rotation_vector, camera_vector):

    ext = camera_utils.create_extrinsic_matrix(rotation_vector, camera_vector)
    new_pts = np.matmul(  ext, phone_pts.T).T
    new_pts = camera_utils.back_2_3d(new_pts)
    obj_pts = camera_utils.print_obj_verts(new_pts)

    #new_pts_list = new_pts.tolist()
    #print obj_pts
    #print new_pts_list

    top_left = new_pts[0]
    bottom_right = new_pts[2]
    center_x = (top_left[0] + bottom_right[0]) / 2.
    center_y = (top_left[1] + bottom_right[1]) / 2.

    return obj_pts, new_pts, (center_x, center_y)

def render_rotated_pts(rotated_obj_file,
                        height, width,
                        fx, fy,
                        center,
                        z_distance,
                        render_im_name):

    # # get image dimensions

    p3d.core.loadPrcFileData("", "window-type offscreen")
    p3d.core.loadPrcFileData("", "win-size {} {}".format(width, height))
    p3d.core.loadPrcFileData("", "audio-library-name null")
    #---------------------------------------------------------------------------


    adjusted_render.Renderer(fx, fy,
                            width,height,
                            center[0], center[1],
                            z_distance,
                            rotated_obj_file,
                            render_im_name)

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
        print (corners)
        plt.scatter(corners[:,0], corners[:,1])
    plt.show()

def project_points(rotated_pts,
                    center_vector,
                    height, width,
                    fx, fy,
                    adversary_metadata,
                    render_im_name,
                    homography_im,
                    display):

    reference_im = 'xr_.png'
    capture_im = render_im_name

    ones = np.ones(4)[np.newaxis].T
    rotated_pts = np.hstack((rotated_pts, ones))

    ext_matrix = camera_utils.create_extrinsic_matrix(
                                np.array([0., 180., 0.]),
                                center_vector)

    int_matrix = camera_utils.create_intrinsic_matrix(
                    fx, fy, width/2., height/2.)

    camera_matrix = np.matmul(int_matrix, ext_matrix)
    points_2d = np.matmul(camera_matrix, rotated_pts.T).T
    w_coords = points_2d[:,2]
    points_2d = points_2d / w_coords[:,None]

    rotated_pts = np.asarray(rotated_pts[:,:2], dtype = np.float32)
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


    im_out = cv2.warpPerspective(im_src, h,
                (im_dest.shape[1], im_dest.shape[0]),
                flags=cv2.INTER_LINEAR )
    im_out = np.fliplr(im_out)


    #save image below
    #need to write metadata here!!
    #metadata = json.dumps(metadata)
    #tifffile.imsave('microscope.tif', data, description=metadata)

    #homography_im_tiff = homography_im.split('.')[0] + '.tif'
    #tifffile.imsave(homography_im_tiff, im_out, description=adversary_metadata)
    image_name = homography_im.split('/')[-1]
    csv_filename = "/".join(homography_im.split('/')[:len(homography_im.split('/'))-1]) + '/metadata.csv'



    homography_IM = Image.fromarray(im_out)

    #homography_IM.show()



    homography_IM.save(homography_im)

    write_meta_to_csv(csv_filename, image_name, adversary_metadata )

    if display:
        display_images([reference_im, homography_im ,capture_im], points_2d)

def write_meta_to_csv(csv_name, image_name, metadata):

    file_exists = os.path.isfile(csv_name)

    with open(csv_name, 'a') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames = ["image_name", "rx", "ry", "rz", "z"])

        if not file_exists:
            print ("Writing header, image_name, rx, ry, rz, z")
            writer.writeheader()

        writer.writerow({'image_name' : image_name,
                        'rx' : str(metadata['x_rotation']),
                        'ry' : str(metadata['y_rotation']),
                        'rz' : str(metadata['z_rotation']),
                        'z' : str(metadata['distance_away'])})


if __name__ == "__main__":

    data_dir = '../data/video_words/'
    aligned_data_dir = "../data/aligned_video/"

    for keypress_dir in os.listdir(data_dir):

        Rx, Ry, Rz, distance_away = create_random_position()
        video_path = os.path.join(data_dir,keypress_dir)

        for video_idx in os.listdir(video_path):

            video_idx_path = os.path.join(video_path, video_idx)

            # sample 20 frames from here

            for image_path in glob.glob(video_idx_path + '/*'):

                intermediate_obj_file = 'intermediate.obj'
                render_im_name = 'render.png'
                mtl_f = 'mtl.mtl'

                aligned_data_path = aligned_data_dir + keypress_dir + '/'
                aligned_data_path = aligned_data_dir + keypress_dir+ '/' + video_idx + '/'

                if not os.path.exists(aligned_data_path):
                    os.makedirs(aligned_data_path)
                aligned_image_name = aligned_data_path + image_path.split('/')[-1]


                #double check with hardrive path here

                #function to define rotations and distance

                #if we are working with words, we assume that the camera will stay in place

                adversary_metadata = {'x_rotation': Rx,
                                    'y_rotation': Ry,
                                    'z_rotation': Rz,
                                    'distance_away': distance_away }

                #ROTATION_VECTOR
                #CAMERA_VECTOR
                ROTATION_VECTOR = [Rx, Ry, Rz]

                #camera position
                CAMERA_VECTOR = [CAM_CEN_X, CAM_CEN_Y, distance_away]

                # in meters
                rotated_pts_str, new_pts, center_pts = rotate_phone(template_pts,
                                                                    ROTATION_VECTOR,
                                                                    CAMERA_VECTOR)

                #Step 2 write rotated points to obj
                write_obj(rotated_pts_str, intermediate_obj_file)

                #adjust mtl file
                update_mtl_file(image_path, mtl_f)


                #Step 3 render the points
                render_rotated_pts(intermediate_obj_file,
                                    HEIGHT,WIDTH,FX,FY,
                                    center_pts,
                                    CAMERA_VECTOR[2],
                                    render_im_name)
                #step 4 project the rotated corners to the rendered image
                center_vector = np.array([center_pts[0], center_pts[1], CAMERA_VECTOR[2]])
                project_points(new_pts, center_vector,
                            HEIGHT,WIDTH,FX,FY,
                            adversary_metadata,
                            render_im_name,
                            homography_im =aligned_image_name,
                            display = False)
