from torch import FloatTensor
from itertools import chain
from numpy.linalg import norm
from math import acos, cos, sin
import numpy as np
from .UR5 import UR5
import pybullet as p
import quaternion


class Target:
    NORMAL = 0
    TOUCHED = 1

    def __init__(self,
                 pose=[(0, 0, 0), (0, 0, 0, 0)],
                 radius=0.02,
                 mass=0.0,
                 color=[0.2, 0.2, 0.2]):
        self.radius = radius
        self.color = color
        self.vs_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=0.1,
            rgbaColor=(
                self.color[0],
                self.color[1],
                self.color[2], 0.7))
        self.body_id = p.createMultiBody(
            baseMass=mass,
            basePosition=pose[0],
            baseOrientation=pose[1],
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=self.vs_id)
        self.constraint_id = self.create_constraint()
        self._mode = Target.NORMAL
        self.normal()

    def create_constraint(self):
        pass

    def get_pose(self):
        return p.getBasePositionAndOrientation(self.body_id)

    def set_pose(self, pose):
        p.resetBasePositionAndOrientation(
            self.body_id,
            pose[0],
            pose[1])

    def normal(self):
        if self._mode != Target.NORMAL:
            p.changeVisualShape(self.body_id, -1,
                                rgbaColor=(
                                    self.color[0],
                                    self.color[1],
                                    self.color[2], 0.5))
            self._mode = Target.NORMAL

    def touched(self):
        if self._mode != Target.TOUCHED:
            p.changeVisualShape(self.body_id, -1,
                                rgbaColor=(0.4, 1.0, 0.4, 0.8))
            self._mode = Target.TOUCHED


class ContrainedTarget(Target):
    def __init__(self,
                 ur5,
                 radius=0.02,
                 color=[0.4, 0.4, 0.4]):
        self.ur5 = ur5
        self._mode = Target.NORMAL
        self.normal()
        super().__init__(
            radius=radius,
            color=color,
            mass=1.0)

    def create_constraint(self):
        return None
        return p.createConstraint(
            parentBodyUniqueId=self.ur5.body_id,
            parentLinkIndex=UR5.EEF_LINK_INDEX,
            childBodyUniqueId=self.body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0])

    def get_pose(self):
        position, orientation = p.getBasePositionAndOrientation(self.body_id)
        return [list(position), list(orientation)]

    def step(self):
        pass

    def update_eef_pose(self):
        # self.set_pose(self.ur5.get_end_effector_pose())
        pass


def get_observation_dimensions(observations_config):
    observation_dim = 0
    for observation_item in observations_config['items']:
        observation_dim += observation_item['dimensions'] * \
            (observation_item['history'] + 1)
    return observation_dim


def ur5_reached_target_pose(ur5, target_pose,
                            position_tolerance,
                            orientation_tolerance):
    target_position = np.array(target_pose[0])
    target_orientation = np.quaternion(*target_pose[1])

    ur5_end_eff = ur5.get_end_effector_pose()
    eef_position = np.array(ur5_end_eff[0])
    eef_orientation = np.quaternion(*ur5_end_eff[1])

    dist = np.linalg.norm(
        target_position - eef_position)
    reached_position = dist < position_tolerance
    angle_disp = (target_orientation *
                  eef_orientation.inverse()).angle()
    reached_orientation = angle_disp < orientation_tolerance
    return reached_position and reached_orientation


"""
RRT Supervision Utility funtions
"""


def ur5s_at_waypoint(ur5s, waypoint, threshold=5):
    waypoints = np.split(np.array(waypoint), len(ur5s))
    return all([max((
        np.array(target_joints) -
        np.array(ur5.get_arm_joint_values()))) < threshold
        for ur5, target_joints in zip(ur5s, waypoints)])


def compute_actions_to_waypoint(
        ur5s,
        waypoint,
        centralized_policy=False,
        action_type='delta'):
    if not centralized_policy:
        waypoints = np.split(np.array(waypoint), len(ur5s))
        if action_type == 'target-norm':
            # Normalize waypoint w.r.t UR5 joint limits
            waypoints = [UR5.normalize_joint_values(wp)
                         for wp in waypoints]
            return [(FloatTensor(target_joint))
                    for ur5, target_joint in zip(ur5s, waypoints)]
        return [FloatTensor(
            np.array(target_joint) -
            np.array(ur5.get_arm_joint_values()))
            for ur5, target_joint in zip(ur5s, waypoints)]
    else:
        waypoint = np.array(waypoint)
        current_joint_values = np.array(list(chain.from_iterable(
            [ur5.get_arm_joint_values() for ur5 in ur5s])))
        return [FloatTensor(waypoint - current_joint_values)]


def angle(a, b):
    # Angle between two vectors1
    return acos(min(np.dot(a, b) / (norm(a) * norm(b)), 1.0))


def perform_expert_actions_variable_threshold(
        env,
        expert_waypoints,
        joint_tolerance,
        threshold,
        max_action,
        action_type,
        log=False):
    def done():
        return ur5s_at_waypoint(
            env.active_ur5s,
            expert_waypoints[-1],
            threshold=0.001)
    next_wp_idx = 0
    curr_j = np.array(list(chain.from_iterable(
        [ur5.get_arm_joint_values() for ur5 in env.active_ur5s])))
    rv = None
    while (not env.terminate_episode) and \
            (not done()):

        # Closest waypoint ahead of current
        next_j = expert_waypoints[next_wp_idx]

        # If already at waypoint, then skip
        if ur5s_at_waypoint(
            env.active_ur5s,
            next_j,
            threshold=threshold) and\
                next_wp_idx < len(expert_waypoints) - 1:
            next_wp_idx += 1
            continue

        # next direction to move in joint space
        next_dir_j = next_j - curr_j

        # Find next target joint that is within action
        # magnitude and joint direction tolerance
        target_wp_idx = next_wp_idx
        while True:
            target_j = expert_waypoints[target_wp_idx]
            target_dir_j = target_j - curr_j
            if target_wp_idx < len(expert_waypoints) - 1 and \
                all([delta_j < max_action for delta_j in abs(target_dir_j)])\
                    and angle(next_dir_j, target_dir_j) < joint_tolerance:
                target_wp_idx += 1
            else:
                break
        actions = compute_actions_to_waypoint(
            ur5s=env.active_ur5s,
            waypoint=target_j,
            centralized_policy=env.centralized_policy,
            action_type=action_type)

        rv = env.step(actions)

        curr_j = np.array(list(chain.from_iterable(
            [ur5.get_arm_joint_values() for ur5 in env.active_ur5s])))
        # Set current waypoint index to the waypoint closest to curr_j
        result = [(idx, norm(wp - curr_j))
                  for idx, wp in enumerate(expert_waypoints)]
        result.sort(
            key=lambda v: v[1])
        next_wp_idx = result[0][0]
        if next_wp_idx < len(expert_waypoints) - 1:
            next_wp_idx += 1
    if env.current_step >= env.episode_length and log:
        print("expert out of time!")
    return rv


def perform_expert_actions_fixed_threshold(
        env,
        expert_waypoints,
        expert_action_threshhold,
        action_type):
    threshold = expert_action_threshhold
    rv = None
    for i, wp in enumerate(expert_waypoints):
        if i == len(expert_waypoints) - 1:
            threshold = 0.001
        while not ur5s_at_waypoint(env.active_ur5s, wp, threshold=threshold) \
                and not env.terminate_episode:
            rrt_action = compute_actions_to_waypoint(
                ur5s=env.active_ur5s,
                centralized_policy=env.centralized_policy,
                waypoint=wp,
                action_type=action_type)
            rv = env.step(rrt_action)
        if env.terminate_episode:
            break
    return rv


def perform_expert_actions(
        env,
        expert_waypoints,
        expert_config):
    if expert_config['waypoint_conversion_mode'] == 'tolerance':
        # Variable threshold
        tolerance_config = expert_config['tolerance_config']
        return perform_expert_actions_variable_threshold(
            env=env,
            expert_waypoints=expert_waypoints,
            joint_tolerance=tolerance_config['tolerance'],
            threshold=tolerance_config['threshold'],
            max_action=tolerance_config['max_magnitude'],
            action_type=env.action_type)
    elif expert_config['action_type'] == 'threshold':
        threshold_config = expert_config['threshold_config']
        return perform_expert_actions_fixed_threshold(
            env=env,
            expert_waypoints=expert_waypoints,
            expert_action_threshhold=threshold_config['threshold'],
            action_type=env.action_type)
    else:
        raise NotImplementedError


def create_circular_poses(radius, count):
    return [[
        [radius *
         np.cos(i * np.pi * 2 / count),
         radius *
         np.sin(i * np.pi * 2 / count),
         0],
        p.getQuaternionFromEuler(
            [0, 0, i * np.pi * 2 / count])]
        for i in range(count)]


def create_ur5s(radius, count, speed, training=True):
    return [
        UR5(
            pose=pose,
            enabled=False,
            velocity=speed,
            training=training
        )
        for pose in create_circular_poses(radius, count)
    ]


def pos_to_high_freq_pos(pos):
    """
    10 sines and cosines encoding
    """
    pos_high_freq = []
    for i in range(1, 11):
        freq = np.pi*2**i
        pos_high_freq.extend([
            sin(freq * pos[0]),
            sin(freq * pos[1]),
            sin(freq * pos[2]),
        ])
        pos_high_freq.extend([
            cos(freq * pos[0]),
            cos(freq * pos[1]),
            cos(freq * pos[2]),
        ])
    return pos_high_freq


def pose_to_high_freq_pose(pose):
    pos, orn = pose
    return pos_to_high_freq_pos(pos), orn
