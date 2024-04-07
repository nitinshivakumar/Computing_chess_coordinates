'''
Camera Calibration: 
Please read the instructions before you start task1.

Please do NOT make any change to this file.
'''
import sys
import argparse
import cv2
import json
import numpy as np
from UB_Geometry import findRot_xyz2XYZ, findRot_XYZ2xyz

def parse_args():
    parser = argparse.ArgumentParser(description="CSE 473/573 project Geometry.")
    parser.add_argument("--alpha", type=float, default=45)
    parser.add_argument("--beta",  type=float, default=30)
    parser.add_argument("--gamma", type=float, default=50)
    args = parser.parse_args()
    return args

def save_result(rot_xyz2XYZ: np.ndarray, rot_XYZ2xyz: np.ndarray, save_path='result_task1.json'):
    result = {}
    result['rot_xyz2XYZ'] = rot_xyz2XYZ.tolist()
    result['rot_XYZ2xyz'] = rot_XYZ2xyz.tolist()
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    if cv2.__version__ != '4.5.4':
        print("Please use OpenCV 4.5.4")
        sys.exit(1)
    args = parse_args()
    rot_xyz2XYZ = findRot_xyz2XYZ(args.alpha, args.beta, args.gamma)
    rot_XYZ2xyz = findRot_XYZ2xyz(args.alpha, args.beta, args.gamma)
    save_result(rot_xyz2XYZ, rot_XYZ2xyz)

    print('rot_xyz2XYZ:')
    print(rot_xyz2XYZ.tolist())
    print('rot_XYZ2xyz:')
    print(rot_XYZ2xyz.tolist())
