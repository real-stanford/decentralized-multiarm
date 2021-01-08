import os
import pickle
import numpy as np
import pybullet_utils as pu
import time
from collections import namedtuple
from transforms3d.affines import compose, decompose
from transforms3d.quaternions import quat2mat, mat2quat
import pybullet as p

mico_configs = {
    'GRASPIT_LINK_TO_MOVEIT_LINK': ([0.0, 0.0, -0.16], [-0.7071067811882787, -0.7071067811848163, 0.0, 0.0]),
    'GRASPIT_LINK_TO_PYBULLET_LINK': ([0, 0, 0], [0.0, 0.0, 0.0, 1.0]),
    'PYBULLET_LINK_TO_COM': ([-0.002216, -0.000001, -0.06], [0.0, 0.0, 0.0, 1.0]),
    'PYBULLET_LINK_COM': [-0.002216, -0.000001, -0.06],

    'robot_urdf': os.path.abspath('assets/mico/mico.urdf'),
    'gripper_urdf': os.path.abspath('assets/mico/mico_hand.urdf'),
    'EEF_LINK_INDEX': 0,
    'GRIPPER_JOINT_NAMES': ['m1n6s200_joint_finger_1', 'm1n6s200_joint_finger_tip_1', 'm1n6s200_joint_finger_2',
                            'm1n6s200_joint_finger_tip_2'],
    'OPEN_POSITION': [0.0, 0.0, 0.0, 0.0],
    'CLOSED_POSITION': [1.1, 0.0, 1.1, 0.0],

    'graspit_approach_dir': 'z'
}

ur5_robotiq_configs = {
    'GRASPIT_LINK_TO_MOVEIT_LINK': ([0, 0, 0], [0.7071067811865475, 0.0, 0.0, 0.7071067811865476]),
    'GRASPIT_LINK_TO_PYBULLET_LINK': ([0.0, 0.0, 0.0], [0.0, 0.706825181105366, 0.0, 0.7073882691671998]),
    'PYBULLET_LINK_TO_COM': ([0.0, 0.0, 0.031451], [0.0, 0.0, 0.0, 1.0]),
    'PYBULLET_LINK_COM': [0.0, 0.0, 0.031451],

    'robot_urdf': os.path.abspath('assets/NONE'),
    'gripper_urdf': os.path.abspath('assets/robotiq_2f_85_hand/robotiq_arg2f_85_model.urdf'),
    'EEF_LINK_INDEX': 0,
    'GRIPPER_JOINT_NAMES': ['finger_joint', 'left_inner_knuckle_joint', 'left_inner_finger_joint',
                            'right_outer_knuckle_joint', 'right_inner_knuckle_joint', 'right_inner_finger_joint'],
    'OPEN_POSITION': [0] * 6,
    'CLOSED_POSITION': (0.72 * np.array([1, 1, -1, 1, 1, -1])).tolist(),

    'graspit_approach_dir': 'x'
}

robot_configs = {'mico': namedtuple('RobotConfigs', mico_configs.keys())(*mico_configs.values()),
                 'ur5_robotiq': namedtuple('RobotConfigs', ur5_robotiq_configs.keys())(*ur5_robotiq_configs.values())
                 }


def reshape_quaternion(q):
    return [q[-1], q[0], q[1], q[2]]


def back_off(grasp_pose, offset=.05, approach_dir='z'):
    if approach_dir == 'x':
        translation = (-offset, 0, 0)
        rotation = (0, 0, 0, 1)
    if approach_dir == 'z':
        translation = (0, 0, -offset)
        rotation = (0, 0, 0, 1)
    pos, quat = p.multiplyTransforms(grasp_pose[0], grasp_pose[1], translation, rotation)
    pre_grasp_pose = [list(pos), list(quat)]
    return pre_grasp_pose


def change_end_effector_link_pose(grasp_pose, old_link_to_new_link):
    pos, quat = p.multiplyTransforms(grasp_pose[0], grasp_pose[1], old_link_to_new_link[0], old_link_to_new_link[1])
    pre_grasp_pose = [list(pos), list(quat)]
    return pre_grasp_pose


def load_grasp_database_new(grasp_database_path, object_name):
    actual_grasps = np.load(os.path.join(grasp_database_path, object_name, 'actual_grasps.npy'))
    graspit_grasps = np.load(os.path.join(grasp_database_path, object_name, 'graspit_grasps.npy'))
    return actual_grasps, graspit_grasps


def convert_grasp_in_object_to_world(object_pose, grasp_in_object):
    """
    :param object_pose: 2d list
    :param grasp_in_object: 2d list
    """
    pos, quat = p.multiplyTransforms(object_pose[0], object_pose[1], grasp_in_object[0], grasp_in_object[1])
    grasp_in_world = [list(pos), list(quat)]
    return grasp_in_world
