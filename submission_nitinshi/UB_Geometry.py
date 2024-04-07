import numpy as np
from typing import List, Tuple
import cv2

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    rot_xyz2XYZ = np.eye(3).astype(float)
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    # Your implementation
    rotate_1 = [[np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1]] 
    

    rotate_2 = [[1, 0, 0],
                [0, np.cos(beta), -np.sin(beta)],
                [0, np.sin(beta), np.cos(beta)]]
    

    rotate_3 = [[np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1]] 
    
    new_rotate = np.dot(rotate_3, rotate_2)
    rot_xyz2XYZ = np.dot(new_rotate, rotate_1)

    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    findRot_XYZ2xyz = np.eye(3).astype(float)
    straight_rotation = findRot_xyz2XYZ(alpha, beta, gamma)
    findRot_XYZ2xyz = np.transpose(straight_rotation)
    
    return findRot_XYZ2xyz

"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1



#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)
    gray_image = cvtColor(image, COLOR_BGR2GRAY)
    something, new = findChessboardCorners(gray_image, [4,  9], None)
    if something == True:
        criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 68, 0.0001)
        new = cornerSubPix(gray_image, new, (11, 11), (-5, -5), criteria)
        new = new.reshape(36, 2)    
        img_coord = np.delete(new, np.arange(16, 20), axis=0)
        return img_coord
    else:
        new = new.reshape(36, 2)    
        img_coord = np.delete(new, np.arange(16, 20), axis=0)
        return img_coord
        # sa = drawChessboardCorners(image, (4, 8), img_coord, something)
        # import cv2
        # cv2.imwrite('sa.png', sa)
    


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    # world_coord = np.zeros([32, 3], dtype=float)
    world_coord = [[40, 0, 40],
                   [40, 0, 30],
                   [40, 0, 20],
                   [40, 0, 10],
                   [30, 0, 40],
                   [30, 0, 30],
                   [30, 0, 20],
                   [30, 0, 10],
                   [20, 0, 40],
                   [20, 0, 30],
                   [20, 0, 20],
                   [20, 0, 10],
                   [10, 0, 40],
                   [10, 0, 30],
                   [10, 0, 20],
                   [10, 0, 10],
                   [0, 10, 40],
                   [0, 10, 30],
                   [0, 10, 20],
                   [0, 10, 10],
                   [0, 20, 40],
                   [0, 20, 30],
                   [0, 20, 20],
                   [0, 20, 10],
                   [0, 30, 40],
                   [0, 30, 30],
                   [0, 30, 20],
                   [0, 30, 10],
                   [0, 40, 40],
                   [0, 40, 30],
                   [0, 40, 20],
                   [0, 40, 10]]
    world_coord = np.array(world_coord)
    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    # Your implementation
    A = np.zeros((2 * len(img_coord), 12))
    p = compute_eigen(A, world_coord, img_coord)
    p = p[:, :-1]
    l = p[2:, :]
    l = l.reshape(1, 3)
    scale = np.linalg.norm(l)
    scale = 1/scale
    p = p * scale
    R, _ = rq_decomposition(p)
    fx = R[0][0]
    fy = R[1][1]
    cx = R[0][2]
    cy = R[1][2]
    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)

    # Your implementation
    A = np.zeros((2 * len(img_coord), 12))
    p = compute_eigen(A, world_coord, img_coord)
    new_p = p[:, :-1]
    l = new_p[2:, :]
    l = l.reshape(1, 3)
    scale = np.linalg.norm(l)
    scale = 1/scale
    p = p * scale
    p2 = p[:, 3]
    p = p[:, :-1]
    Q, R = rq_decomposition(p)
    inv_Q = np.linalg.inv(Q)
    print(p2)
    T = inv_Q @ p2
    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2

def compute_eigen(A, world_coord, img_coord):
    len_A = len(A)
    for i in range(len_A):
        if i%2 == 0:
            j = i
            i = i // 2
            A[j] = [world_coord[i][0], world_coord[i][1], world_coord[i][2], 1, 0, 0, 0, 0, -img_coord[i][0] * world_coord[i][0], -img_coord[i][0] * world_coord[i][1], - img_coord[i][0] * world_coord[i][2], -img_coord[i][0]]
        else:
            j = i
            i = i // 2
            A[j] = [0, 0, 0, 0, world_coord[i][0], world_coord[i][1], world_coord[i][2], 1, -img_coord[i][1] * world_coord[i][0], -img_coord[i][1] * world_coord[i][1], - img_coord[i][1] * world_coord[i][2], -img_coord[i][1]]
    A2 = np.dot(np.transpose(A), A)
    eig_val, k = np.linalg.eigh(A2)
    eig_vec = k[:, 0]
    eig_vec = eig_vec.reshape(3, 4)
    return eig_vec

def rq_decomposition(A):
    from scipy import linalg
    R, Q = linalg.rq(A)
    if R[0, 0] < 0:
        R = -R
        Q = -Q
    return R, Q





#---------------------------------------------------------------------------------------------------------------------