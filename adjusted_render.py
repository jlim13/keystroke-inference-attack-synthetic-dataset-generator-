import imageio
import numpy as np
import os
from pyquaternion import Quaternion
import panda3d as p3d
import panda3d.core
from direct.showbase.ShowBase import ShowBase
import matplotlib.pyplot as plt
import os
import threading
import queue as Queue


import camera_utils


#variables


class OBJNode(p3d.core.GeomNode):
    # TODO (True): large point clouds will overrun the buffer; so, we'll have to
    # split up into smaller
    MAX_NUM_VERTICES = 2**32

    def __init__(self, obj_file,new_obj_pts,  name=""):
        super(OBJNode, self).__init__(name)

        vertices = []
        texcoord = []
        normals = []
        faces = []
        face_texcoord = []
        face_normals = []

        faces = [[1,4,2], [3,2,4]]
        face_texcoord = [[3,4,2], [1,2,4]]
        texcoord = [[-1.0, -1.0], [0.0, -1.0], [0.0, 0.0], [-1.0, 0.0]]

        vertices = new_obj_pts

        self.has_texture = (len(texcoord) > 0)
        if self.has_texture:
            texcoord = np.array(texcoord)

        self.has_normals = (len(normals) > 0)
        if self.has_normals:
            normals = np.array(normals)

        self.has_faces = (len(faces) > 0)
        if self.has_faces:
            faces = np.row_stack(faces)

        self.has_face_texture = (len(face_texcoord) > 0)
        if self.has_face_texture:
            face_texcoord = np.row_stack(face_texcoord)

        self.has_face_normals = (len(face_normals) > 0)
        if self.has_face_normals:
            face_normals = np.row_stack(face_normals)

        if self.has_faces:
            # set up vertex normals from faces

            # if not self.has_normals:
            #     normals = compute_vertex_normals(vertices, faces)
            #     self.has_normals = True

            # if the faces have their own textures or normals, we'll have to
            # duplicate all the vertices
            if self.has_face_texture or self.has_face_normals:

                vertices = vertices[faces-1].reshape(-1, 3)

                faces = np.arange(len(vertices)).reshape(-1, 3)

                if self.has_face_texture:
                    texcoord = texcoord[face_texcoord-1].reshape(-1, 2)

                if self.has_face_normals:
                    normals = normals[face_normals].reshape(-1, 3)


        # set up data in chunks
        for i in range(0, len(vertices), OBJNode.MAX_NUM_VERTICES):
            stop = min(i + OBJNode.MAX_NUM_VERTICES, len(vertices))
            n = stop - i

            if self.has_texture and self.has_normals:
                p3d_data_format = p3d.core.GeomVertexFormat().getV3n3t2()
            elif self.has_texture:
                p3d_data_format = p3d.core.GeomVertexFormat().getV3t2()
            elif self.has_normals:
                p3d_data_format = p3d.core.GeomVertexFormat().getV3n3()
            else:
                p3d_data_format = p3d.core.GeomVertexFormat().getV3()

            p3d_data = p3d.core.GeomVertexData(
                name, p3d_data_format, p3d.core.Geom.UHStatic)
            p3d_data.setNumRows(n)

            # load vertex positions
            v_writer = p3d.core.GeomVertexWriter(p3d_data, "vertex")
            for vertex in vertices[i:stop]:
                v_writer.addData3f(*vertex)

            # load colors
            if self.has_texture:
                t_writer = p3d.core.GeomVertexWriter(p3d_data, "texcoord")
                for coord in texcoord[i:stop]:
                    t_writer.addData2f(coord[0], coord[1])

            # load normals
            if self.has_normals:
                n_writer = p3d.core.GeomVertexWriter(p3d_data, "normal")
                for normal in normals[i:stop]:
                    n_writer.addData3f(*normal)

            # add faces, if available
            if self.has_faces:
                p3d_primitives = p3d.core.GeomTriangles(p3d.core.Geom.UHStatic)
                mask = np.all(faces >= i, axis=1) & np.all(faces < stop, axis=1)
                for f in faces[mask]:
                    p3d_primitives.addVertices(*(f - i))

            # otherwise, render a point cloud
            else:
                p3d_primitives = p3d.core.GeomPoints(p3d.core.Geom.UHStatic)
                p3d_primitives.addNextVertices(n)

            geom = p3d.core.Geom(p3d_data)
            geom.addPrimitive(p3d_primitives)
            self.addGeom(geom)

#-------------------------------------------------------------------------------

class OBJNodePath(p3d.core.NodePath):
    def __init__(self, obj_file,new_obj_pts, texture_image):
        self.geom_node = OBJNode(obj_file, new_obj_pts)
        super(OBJNodePath, self).__init__(self.geom_node)


        texture = p3d.core.Texture()
        texture.read(texture_image)
        self.setTexture(texture)



#-------------------------------------------------------------------------------
#
# classes for panda3d rendering
#
#-------------------------------------------------------------------------------

#
# Renderer class
#
#-------------------------------------------------------------------------------

class Renderer(ShowBase):
    def __init__(self,
                queue,
                fx, fy,
                width, height,
                center_x, center_y,
                camera_dist,
                rotated_obj_file,
                texture_image,
                new_obj_pts):

        ShowBase.__init__(self)



        base.disableMouse()
        lens = p3d.core.MatrixLens()
        # Set the OpenGL projection matrix for the lens.
        w = width
        h = height
        cx, cy = w/2., h/2.
        z0, z1 = 1e-2, 1e2 # near, far

        # Set the OpenGL projection matrix for the lens.
        lens.setUserMat(p3d.core.Mat4(
           2. * fx / w, 0., 0., 0.,
           0., 2. * fy / h, 0., 0.,
           2. * cx / w - 1., 2. * cy / h - 1., (z1 + z0) / (z1 - z0), 1.,
           0., 0., -2. * z0 * z1 / (z1 - z0), 0.))

        base.camNode.setLens(lens)
        self.camera.setPos(center_x, center_y, camera_dist)

        q1 = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(0.))
        q2 = Quaternion(axis=[0, 1, 0], angle=np.deg2rad(0.))
        q3 = Quaternion(axis=[0, 0, 1], angle=np.deg2rad(0.))

        q4 = q1*q2*q3
        self.camera.setQuat(p3d.core.Quat(q4.elements[0],
                                            q4.elements[1],
                                            q4.elements[2],
                                            q4.elements[3] ))

        # get depth image; see
        # gist.github.com/alexlee-gk/b28fb962c9b2da586d1591bac8888f1f
        winprops = p3d.core.WindowProperties.size(
            self.win.getXSize(), self.win.getYSize())

        fbprops = p3d.core.FrameBufferProperties()
        fbprops.setDepthBits(1)
        self.depthBuffer = self.graphicsEngine.makeOutput(
            self.pipe, "depth buffer", -2,
            fbprops, winprops,
            p3d.core.GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(), self.win)
        self.depthTex = p3d.core.Texture()
        self.depthTex.setFormat(p3d.core.Texture.FDepthComponent)
        self.depthBuffer.addRenderTexture(self.depthTex,
            p3d.core.GraphicsOutput.RTMCopyRam,
            p3d.core.GraphicsOutput.RTPDepth)

        self.depthCam = self.makeCamera(self.depthBuffer,
            lens=lens,
            scene=render)
        self.depthCam.reparentTo(self.cam)

        print ("Rendering")
        ###adjust here

        obj_node_path = OBJNodePath(rotated_obj_file, new_obj_pts, texture_image)
        obj_node_path.reparentTo(render)

        self.graphicsEngine.renderFrame()

        # get image; see
        # gist.github.com/alexlee-gk/b28fb962c9b2da586d1591bac8888f1f

        texture = self.win.getScreenshot()

        data = self.depthTex.getRamImage()
        depth_image = np.frombuffer(data.get_data(), np.float32)
        depth_image.shape = [self.depthTex.getYSize(), self.depthTex.getXSize(), self.depthTex.getNumComponents()]
        depth_image = np.flipud(depth_image)

        data = texture.getRamImageAs('RGB')
        image = np.frombuffer(data.get_data(), np.uint8)

        image.shape = [texture.getYSize(), texture.getXSize(), 3]
        image = np.flipud(image)#.astype(np.float32)

        #self.mesh_node.hide()

        print ("Saving")
        queue.put(image)


        #exit()
        # imageio.imwrite(
        #     os.path.join(render_im_name),
        #     (image).astype(np.uint8))

        print ("here")
        print ("Done rendering")

        #del obj_node_path
        self.destroy()
        #exit()

    @staticmethod
    def start(queue,
            fx, fy, width, height,
            center_x, center_y,
            camera_dist,
            rotated_obj_file,
            texture_image,
            new_obj_pts):

        p3d.core.loadPrcFileData("", "window-type offscreen")
        p3d.core.loadPrcFileData("", "win-size {} {}".format(width, height))
        p3d.core.loadPrcFileData("", "audio-library-name null")

        # render using Panda3D

        renderer = Renderer(
                            queue,
                            fx, fy,
                            width, height,
                            center_x, center_y,
                            camera_dist,
                            rotated_obj_file,
                            texture_image,
                            new_obj_pts)
        renderer.run()



def render_points(fx, fy, width, height,
                    center_x, center_y,
                    camera_dist,
                    rotated_obj_file,
                    texture_image,
                    new_obj_pts):




    queue = Queue.Queue()

    # Renderer.start(queue,
    #             fx, fy,
    #             width, height,
    #             center_x, center_y, camera_dist,
    #             rotated_obj_file,
    #             texture_image,
    #             new_obj_pts,
    #             render_im_name)

    t = threading.Thread(
        target=Renderer.start, args=(
                                queue,
                                fx, fy,
                                width, height,
                                center_x, center_y, camera_dist,
                                rotated_obj_file,
                                texture_image,
                                new_obj_pts))
    t.daemon = True
    t.start()

    render_image_data = queue.get()

    return render_image_data

#-------------------------------------------------------------------------------
#
# program main
#
#-------------------------------------------------------------------------------
