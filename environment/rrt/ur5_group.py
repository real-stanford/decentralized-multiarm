import numpy as np
from .pybullet_utils import remove_all_markers
from .rrt_connect import birrt
import time
import pybullet as p
from random import uniform
from environment import UR5
from os.path import isfile
from json import load, dump


def random_point_in_workspace(radius=0.5):
    i = uniform(0, 1)
    j = uniform(0, 1) ** 0.5
    k = uniform(0, 1)
    return np.array([
        radius * j * np.cos(i * np.pi * 2) * np.cos(k * np.pi / 2),
        radius * j * np.sin(i * np.pi * 2) * np.cos(k * np.pi / 2),
        radius * j * np.sin(k * np.pi / 2),
    ])


def reached(controllers, targets):
    dist = [np.linalg.norm(
        np.array(t) - np.array(c.get_end_effector_pose()[0]))
        for c, t in zip(controllers, targets)]
    reached = [d < 0.1 for d in dist]
    return all(reached)


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


class UR5Group:
    def __init__(self, create_ur5s_fn, collision_distance):
        self.all_controllers = create_ur5s_fn()
        self.active_controllers = []
        self.collision_distance = collision_distance

    def setup(self, start_poses, start_joints):
        self.disable_all_ur5s()
        self.enable_ur5s(count=len(start_poses))
        for c, pose, joints in zip(
                self.active_controllers, start_poses, start_joints):
            c.set_arm_joints(joints)
            c.set_pose(pose)
        p.stepSimulation()
        return None

    def disable_all_ur5s(self):
        for i, ur5 in enumerate(self.all_controllers):
            ur5.disable(idx=i)
        self.active_controllers = []

    def enable_ur5s(self, count=None):
        self.disable_all_ur5s()
        for i, ur5 in enumerate(self.all_controllers):
            if count is not None and i == count:
                break
            ur5.enable()
            self.active_controllers.append(ur5)

    def set_joint_positions(self, joint_values):
        assert len(joint_values) == self.compute_dof()
        robot_joint_values = split(joint_values, len(self.active_controllers))
        for c, jv in zip(self.active_controllers, robot_joint_values):
            c.set_arm_joints(jv)

    def get_joint_positions(self):
        joint_values = []
        for c in self.active_controllers:
            joint_values += c.get_arm_joint_values()
        return joint_values

    def compute_dof(self):
        return sum([len(c.GROUP_INDEX['arm'])
                    for c in self.active_controllers])

    def difference_fn(self, q1, q2):
        difference = []
        split_q1 = split(q1, len(self.active_controllers))
        split_q2 = split(q2, len(self.active_controllers))
        for ctrl, q1_, q2_ in zip(self.active_controllers, split_q1, split_q2):
            difference += ctrl.arm_difference_fn(q1_, q2_)
        return difference

    def distance_fn(self, q1, q2):
        diff = np.array(self.difference_fn(q2, q1))
        return np.sqrt(np.dot(diff, diff))

    def sample_fn(self):
        values = []
        for ctrl in self.active_controllers:
            values += ctrl.arm_sample_fn()
        return values

    def get_extend_fn(self, resolutions=None):
        dof = self.compute_dof()
        if resolutions is None:
            resolutions = 0.05 * np.ones(dof)

        def fn(q1, q2):
            steps = np.abs(np.divide(self.difference_fn(q2, q1), resolutions))
            num_steps = int(max(steps))
            waypoints = []
            diffs = self.difference_fn(q2, q1)
            for i in range(num_steps):
                waypoints.append(
                    list(((float(i) + 1.0) /
                          float(num_steps)) * np.array(diffs) + q1))
            return waypoints

        return fn

    def get_collision_fn(self, log=False):
        # Automatically check everything
        def collision_fn(q=None):
            if q is not None:
                self.set_joint_positions(q)
            return any([c.check_collision(
                collision_distance=self.collision_distance)
                for c in self.active_controllers])
        return collision_fn

    def forward_kinematics(self, q):
        """ return a list of eef poses """
        poses = []
        split_q = split(q, len(self.active_controllers))
        for ctrl, q_ in zip(self.active_controllers, split_q):
            poses.append(ctrl.forward_kinematics(q_))
        return poses
