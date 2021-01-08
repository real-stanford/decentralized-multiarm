import pybullet as p
import pybullet_data
import pybullet_utils as pu
import grasp_utils as gu
from math import radians
from os.path import join
from trimesh import load_mesh
import quaternion
import numpy as np
from collections import namedtuple, OrderedDict
import os
import csv


ConfigInfo = namedtuple(
    'ConfigInfo', ['joint_config', 'ee_pose', 'pos_distance', 'quat_distance'])


def configure_pybullet(rendering=False, debug=False, yaw=46.39, pitch=-55.00, dist=1.9, target=(0.0, 0.0, 0.0)):
    if not rendering:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if not debug:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    pu.reset_camera(yaw=yaw, pitch=pitch, dist=dist, target=target)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)


def load_object(object_name, xy_position, surface_height, rpy):
    """" rpy is in degrees """
    model_dir = 'assets/models/'
    rpy = [radians(i) for i in rpy]
    object_mesh_filepath = join(model_dir, '{}'.format(
        object_name), '{}.obj'.format(object_name))
    target_urdf = join(model_dir, '{}'.format(
        object_name), '{}.urdf'.format(object_name))
    target_mesh = load_mesh(object_mesh_filepath)
    target_z = -target_mesh.bounds.min(0)[2] + surface_height
    target_initial_pose = [
        [xy_position[0], xy_position[1], target_z],
        pu.quaternion_from_euler(rpy)]
    return p.loadURDF(target_urdf,
                      basePosition=target_initial_pose[0],
                      baseOrientation=target_initial_pose[1])


class TargetObject:
    def __init__(self, object_name, xy_position, surface_height, rpy=(0, 0, 0)):
        self.object_name = object_name
        self.id = load_object(object_name, xy_position, surface_height, rpy)
        self.robot_configs = gu.robot_configs['ur5_robotiq']
        self.back_off = 0.1
        self.initial_pose = self.get_pose()
        self.urdf_path = join('assets/models/', '{}'.format(
            object_name), '{}.urdf'.format(object_name))

        # p = robot_1.get_eef_pose()
        # pu.create_frame_marker(p)
        # pu.create_frame_marker(gu.back_off(p, approach_dir='x'))
        grasp_database_path = 'assets/filtered_grasps_noise_robotiq_100_1.00'
        actual_grasps, graspit_grasps = gu.load_grasp_database_new(
            grasp_database_path, self.object_name)
        use_actual = False
        self.graspit_grasps = actual_grasps if use_actual else graspit_grasps

        self.graspit_pregrasps = [
            pu.merge_pose_2d(gu.back_off(
                pu.split_7d(g),
                self.back_off,
                self.robot_configs.graspit_approach_dir))
            for g in self.graspit_grasps]
        self.grasps_eef = [pu.merge_pose_2d(
            gu.change_end_effector_link_pose(
                pu.split_7d(g),
                self.robot_configs.GRASPIT_LINK_TO_MOVEIT_LINK))
            for g in self.graspit_grasps]
        self.grasps_link6_ref = [pu.merge_pose_2d(
            gu.change_end_effector_link_pose(
                pu.split_7d(g),
                self.robot_configs.GRASPIT_LINK_TO_PYBULLET_LINK))
            for g in self.graspit_grasps]
        self.pre_grasps_eef = [pu.merge_pose_2d(
            gu.change_end_effector_link_pose(
                pu.split_7d(g),
                self.robot_configs.GRASPIT_LINK_TO_MOVEIT_LINK))
            for g in self.graspit_pregrasps]
        self.pre_grasps_link6_ref = [pu.merge_pose_2d(
            gu.change_end_effector_link_pose(
                pu.split_7d(g),
                self.robot_configs.GRASPIT_LINK_TO_PYBULLET_LINK))
            for g in self.graspit_pregrasps]

    def get_pose(self):
        return pu.get_body_pose(self.id)

    def check_lift(self):
        return pu.get_body_pose(self.id)[0][2] > 0.1


class OtherObject:
    def __init__(self, urdf, initial_pose):
        self.urdf = urdf
        self.initial_pose = initial_pose
        self.id = p.loadURDF(
            urdf, basePosition=initial_pose[0], baseOrientation=initial_pose[1])

    def get_pose(self):
        return pu.get_body_pose(self.id)


class Robotiq2F85Target:
    def __init__(self, pose=[[0, 0, 0], [0, 0, 0, 1]]):
        self.body_id = p.loadURDF(
            'assets/gripper/robotiq_2f_85_no_colliders.urdf',
            pose[0],
            pose[1],
            useFixedBase=1)
        self.set_pose(pose)
        for i in range(p.getNumJoints(self.body_id)):
            p.changeVisualShape(
                self.body_id,
                i,
                textureUniqueId=-1,
                rgbaColor=(0, 0, 0, 0.5))

    def transform_orientation(self, orientation):
        A = np.quaternion(*orientation)
        B = np.quaternion(*p.getQuaternionFromEuler([np.pi / 2, 0, np.pi / 2]))
        C = B * A
        return quaternion.as_float_array(C)

    def set_pose(self, pose):
        p.resetBasePositionAndOrientation(
            self.body_id,
            pose[0],
            self.transform_orientation(pose[1]))

    def __del__(self):
        p.removeBody(self.body_id)


def write_csv_line(result_file_path, result):
    """ write a line in a csv file; create the file and write the first line if the file does not already exist """
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(result)
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
