import ray
from .baseEnv import BaseEnv
from random import randint, seed, uniform
from .tasks import Task, ProxyTaskManager
from .utils import create_circular_poses
import numpy as np
from itertools import product
import pybullet as p


@ray.remote
class TargetEnv(BaseEnv):
    def __init__(self, env_config, training_config, gui, logger):
        self.logger = logger
        self.setup(env_config, training_config, gui)
        self.task_manager = ProxyTaskManager(
            colors=[ur5.color for ur5 in self.active_ur5s])
        seed()

    def setup(self, env_config, training_config, gui):
        super().setup(env_config, training_config, gui)
        self.task_generation_method = env_config['task']['generation_method']
        self.task_difficulty = env_config['task']['difficulty']

    def step(self, actions):
        pass

    def reset(self):
        # If variable length, sample a random count
        ur5s_count = randint(1, self.max_ur5s_count)
        self.enable_ur5s(count=ur5s_count)
        if self.task_generation_method == 'circular':
            separation = uniform(
                self.task_difficulty['min_separation'],
                self.task_difficulty['max_separation'])
            radius = separation * ur5s_count / (2 * np.pi)
            for pose, ur5 in zip(create_circular_poses(
                    radius=radius, count=len(self.active_ur5s)),
                    self.active_ur5s):
                ur5.set_pose(pose)
        else:
            self.randomize_positions()
        for ur5 in self.active_ur5s:
            ur5.reset()
        self.randomize_configs()

    def sample_random_position(self):
        return np.array([
            uniform(-self.task_difficulty['max_distance_between_bases'],
                    self.task_difficulty['max_distance_between_bases']),
            uniform(-self.task_difficulty['max_distance_between_bases'],
                    self.task_difficulty['max_distance_between_bases']),
            0])

    def randomize_positions(self):
        positions = None
        assert len(self.active_ur5s) > 0
        while True:
            positions = []
            for _ in range(len(self.active_ur5s)):
                if len(positions) == 0:
                    positions.append(self.sample_random_position())
                else:
                    positions.append(
                        positions[-1] + self.sample_random_position())

            distances = [np.linalg.norm(pos1 - pos2)
                         for pos1, pos2 in product(positions, repeat=2)]
            if all([d == 0.0 or
                    (d > self.task_difficulty[
                        'min_distance_between_bases'])
                    for d in distances]):
                for ur5, position in zip(self.active_ur5s, positions):
                    ur5.set_pose([
                        position,
                        p.getQuaternionFromEuler([
                            0, 0, uniform(0, np.pi * 2)])])
                return

    def randomize_configs(self):
        while True:
            for ur5 in self.active_ur5s:
                ur5.reset()
                target_pose = ur5.workspace.point_in_workspace()
                ur5.set_target_end_eff_pos(target_pose)

            for _ in range(50):
                p.stepSimulation()
                for ur5 in self.active_ur5s:
                    ur5.step()

            if not any([(ur5.check_collision(
                collision_distance=self.collision_distance
            ) or ur5.violates_limits())
                    for ur5 in self.active_ur5s]):
                # Collision free initial states within joint limits
                break

    def sample_collision_free_config(self):
        while True:
            for ur5 in self.active_ur5s:
                ur5.set_arm_joints(
                    joint_values=ur5.arm_sample_fn())
            collision_mask = [ur5.check_collision()
                              for ur5 in self.active_ur5s]
            if not any(collision_mask):
                break

    def create_task(self):
        self.reset()
        start_config = [ur5.get_arm_joint_values()
                        for ur5 in self.active_ur5s]
        self.randomize_configs()
        goal_config = [ur5.get_arm_joint_values()
                       for ur5 in self.active_ur5s]
        self.randomize_configs()
        start_goal_config = [ur5.get_arm_joint_values()
                             for ur5 in self.active_ur5s]
        return Task(
            base_poses=[ur5.get_pose() for ur5 in self.active_ur5s],
            goal_config=goal_config,
            start_goal_config=start_goal_config,
            start_config=start_config,
            target_eff_poses=[ur5.get_end_effector_pose()
                              for ur5 in self.active_ur5s])

    def create_task_remote(self):
        return self.create_task(), self.ray_id['val']

    def forward_kinematics(self, joint_values):
        self.enable_ur5s(count=len(joint_values))
        rv = [ur5.forward_kinematics(joint_value)
              for ur5, joint_value in zip(self.active_ur5s, joint_values)]
        return rv

    def set_ur5_poses(self, base_poses):
        self.enable_ur5s(count=len(base_poses))
        [ur5.set_pose(base_pose)
         for ur5, base_pose in zip(self.active_ur5s, base_poses)]
