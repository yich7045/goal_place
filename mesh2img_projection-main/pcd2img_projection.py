#!/usr/bin/env python
import copy
import cv2, json, os
import numpy as np
import open3d as o3d
import numpy
import math
import pickle

#### Parameters ####
PATH_IMAGE = ['projection_test.jpg'] ##: your image
PATH_MESH = [ 'cube.ply'] ## TODO : A Mesh you want to project
PATH_CAMERA_CONFIG = 'config/realsense_param.json' # instrinsic
x, y, z = 0.0, 0.0, 0.0 ## target position in world frame
scale = 1/40.
color = [0, 180, 255]
## extrinsic matrix
t = [1.530, -0.009, 0.444]
q = [-0.549, -0.552, 0.444, 0.444]
PIXEL = 0.001 # unit in mm

##############
_EPS = numpy.finfo(float).eps * 4.0
def quaternion_matrix(quaternion):
    # """Return homogeneous rotation matrix from quaternion.
    # >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    # >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    # True
    # """
    q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)

##############
def BuildMatrix(translation, quaternion):
    tfmatrix = quaternion_matrix(quaternion)
    tfmatrix[0][3] = translation[0]
    tfmatrix[1][3] = translation[1]
    tfmatrix[2][3] = translation[2]
    return tfmatrix

class overlaid_image():
    def __init__(self, vis = False):

        self.camera_info = {}
        self.rgb = None
        self.mesh = None
        self.depth_image = None
        self.combined_rgb = None
        self.transform = False
        self.rgb_resolution = None
        self.vis = vis

    def assign_camera_param(self, path_config):
        f = open(path_config)
        data = json.load(f)
        f.close()
        P = np.array(data["P"]).reshape(3,4)
        K = np.array(data["K"]).reshape(3,3)
        D = np.array(data["D"])
        self.camera_info = {"P": P, "K": K, "D": D}

    def read_image(self, path_img):
        self.rgb = cv2.imread(path_img)
        h, w, _ = self.rgb.shape
        self.rgb_resolution = (h, w)

    def read_mesh(self, path_mesh):
        self.mesh = o3d.geometry.TriangleMesh.create_sphere()
        self.mesh = self.mesh.subdivide_loop(number_of_iterations=3)
        self.mesh.paint_uniform_color([1, 0.706, 0])

    def transform_mesh(self, t, q):
        '''
        :param t: Translation from source to Target frame
        :param q: Rotation from source to Target frame
        :param pcd: open3d pointcloud from Source frame
        :return: array of pointcloud
        '''
        self.transform = True
        tf_mat = BuildMatrix(t, q)
        self.mesh.scale(scale, (x,y,z))
        mesh_  = self.mesh.transform(np.linalg.inv(tf_mat))
        self.transformed_pcd = np.array(mesh_.vertices)
        # print(self.transformed_pcd)
        frame_ori = o3d.geometry.TriangleMesh.create_coordinate_frame()


    def project_mesh(self):
        pcd_array = self.transformed_pcd

        # Read RS camera parameters
        rvec = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]) #self.camera_info['P'][:3, :3]
        tvec = np.array([0., 0., 0.]) #self.camera_info['P'][:3, 3]

        camera_mat =self.camera_info['K']
        dist_coeff = self.camera_info['D']

        # Project 3D points to 2D image in Realsense
        imgpts, jac = cv2.projectPoints(pcd_array, rvec, tvec, camera_mat, dist_coeff)
        imgpts = imgpts.squeeze()

        if self.vis:
            import matplotlib.pyplot as plt
            plt.scatter(imgpts[:,0], imgpts[:,1])
            plt.show()

        imgpts = np.rint(imgpts/PIXEL).astype(int)
        imgpts[:, 0] = np.clip(imgpts[:,0], 0, self.rgb_resolution[1]-1)
        imgpts[:, 1] = np.clip(imgpts[:, 1], 1, self.rgb_resolution[0]-1)

        # overlay 2D image with the existing rgb image
        self._pcd_2_depth(imgpts)
        return imgpts  # 2D array

    ## PCD -> 2D image. Input : VIRDO pointcloud
    def init_latest_projected_pcd(self):
        h, w, _ = self.rgb.shape

        # print("realsense resolution", h, w)
        # img_depth = np.zeros((h, w, 3))
        img_depth = copy.deepcopy(self.rgb)
        self.projected_pcd = img_depth

    def _pcd_2_depth(self, imgpts):
        self.init_latest_projected_pcd()
        pcd_projected = self.projected_pcd
        # print(imgpts)
        for dx, dy in imgpts:
            self.projected_pcd[dy, dx, :] = color  # Paint color

        if self.vis:
            cv2.imshow("empty", self.projected_pcd)
            cv2.waitKey(0)

    def overlay_pcd_on_raw(self):
        src1 = self.rgb # realsense image
        src2 = self.projected_pcd
        # blend two images
        alpha = 0.8
        beta = (1.0 - alpha)
        self.combined_rgb = cv2.addWeighted(src1, alpha, src2, beta, 0.0)

    def save_combined_rgb(self, filename):
        cv2.imwrite(filename, self.projected_pcd)


for image_idx, path_image in enumerate(PATH_IMAGE):
    path_mesh = os.path.join(PATH_MESH[image_idx])
    oli = overlaid_image(vis=False)
    oli.assign_camera_param(PATH_CAMERA_CONFIG)
    oli.read_image(path_image)
    oli.read_mesh(path_mesh)
    oli.transform_mesh(t, q)
    oli.project_mesh()
    oli.overlay_pcd_on_raw()

    PATH_SAVE_RESULT = f'result/test.png'
    # print(PATH_SAVE_RESULT)
    oli.save_combined_rgb(PATH_SAVE_RESULT)
