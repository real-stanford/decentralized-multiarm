from .baseEnv import BaseEnv
import pybullet as p
from time import time, sleep
from threading import Thread
from .utils import perform_expert_actions
import ray
from .rrt import RRTWrapper


class RealTimeEnv(BaseEnv):
    def __init__(self, env_config, training_config, gui, logger):
        env_config['ur5_speed'] = 0.008
        super().__init__(env_config, training_config, gui, logger)
        p.setRealTimeSimulation(0)
        self.episode_start_time = None
        self.episode_time_limit = 10

        self.is_successful = False
        self.success_check_thread = Thread(target=self.update_success_daemon,
                                           name='success checker')
        self.success_check_thread.setDaemon(True)
        self.success_check_thread.start()
        self.time_of_success = None
        while not self.success_check_thread.isAlive():
            sleep(0.001)

        self.in_collision = False
        self.collision_check_thread = Thread(
            target=self.update_collision_daemon,
            name='collision checker')
        self.collision_check_thread.setDaemon(True)
        self.collision_check_thread.start()
        while not self.collision_check_thread.isAlive():
            sleep(0.001)

        self.use_policy = 'policy' in training_config

        if self.use_policy:
            self.policy = training_config['policy']
        else:
            self.rrt = RRTWrapper.remote(
                env_config,
                gui=False)
            self.centralized_policy = True
        self.expert_config = env_config['expert']

    def update_success_daemon(self):
        while True:
            self.update_success()
            sleep(0.01)

    def on_target_reach(self, ur5, idx):
        pass

    def update_success(self):
        if self.is_currently_in_episode() and \
                not self.is_successful:
            self.task_manager.set_timer(
                float(time() - self.episode_start_time) /
                self.episode_time_limit)
            self.is_successful = all([
                self.check_ur5_reached_target(i, ur5, target_eef_pose)
                for i, (ur5, target_eef_pose) in enumerate(
                    zip(
                        self.active_ur5s,
                        self.task_manager.get_target_end_effector_poses()
                    ))])
            if self.is_successful:
                self.time_of_success = time()
                self.terminate_episode = True

    def is_currently_in_episode(self):
        return self.episode_start_time is not None

    def update_collision_daemon(self):
        while True:
            self.update_collision()
            sleep(0.1)

    def update_collision(self):
        if self.is_currently_in_episode() and \
                not self.in_collision:
            self.in_collision = any([
                ur5.check_collision()
                for ur5 in self.active_ur5s])
            if self.in_collision:
                self.terminate_episode = True

    def setup_task(self, task):
        self.task_manager.current_task = task
        self.enable_ur5s(count=self.get_current_task().ur5_count)
        for ur5, ur5_task in zip(self.active_ur5s,
                                 self.get_current_task()):
            ur5.set_pose(ur5_task['base_pose'])
            ur5.set_arm_joints(ur5_task['start_config'])
            ur5.step()

    def start(self):
        self.history = []
        self.episode_policy_instance_keys = [
            key for key in self.memory_cluster_map] * len(self.active_ur5s)
        p.setRealTimeSimulation(1)
        self.terminate_episode = False
        self.is_successful = False
        self.in_collision = False
        self.task_manager.set_timer(0.0)
        self.time_of_success = None
        observation = self.obs_to_policy([self.preprocess_obs(o)
                                          for o in self.get_observations()])
        self.episode_start_time = time()
        return observation

    def has_time(self):
        return float(time() - self.episode_start_time) \
            < self.episode_time_limit

    def should_continue(self):
        return self.has_time() and \
            not self.is_successful and \
            not self.in_collision

    def get_curr_conf(self):
        return [ur5.get_arm_joint_values() for ur5 in self.active_ur5s]

    def get_goal_conf(self):
        return self.task_manager.get_goal_confs()

    def step(self, actions):
        if self.centralized_policy:
            self.action_to_robots(actions[0].split(6))
        else:
            self.action_to_robots(actions)
        return self.obs_to_policy([self.preprocess_obs(o)
                                   for o in self.get_observations()])

    def get_target_end_effector_poses(self):
        return self.task_manager.get_target_end_effector_poses()

    def on_all_ur5s_reach_target(self):
        pass

    def get_poses(self):
        return [ur5.get_pose() for ur5 in self.active_ur5s]

    def solve_task(self, task):
        self.setup_task(task)
        observation = self.start()
        while self.should_continue():
            if self.use_policy:
                actions = self.policy.act(
                    observation['multiarm_motion_planner'])
                observation = self.step(actions)
            else:
                waypoints = ray.get(self.rrt.birrt.remote(
                    start_conf=self.get_curr_conf(),
                    goal_conf=self.get_goal_conf(),
                    ur5_poses=self.get_poses(),
                    target_eff_poses=self.get_target_end_effector_poses(),
                    timeout=self.episode_time_limit
                ))
                if waypoints is not None:
                    perform_expert_actions(
                        env=self,
                        expert_waypoints=waypoints,
                        expert_config=self.expert_config)
        return {
            'success_rate': float(self.is_successful),
            'collision_rate': float(self.in_collision),
            'collision_failure_rate': 1. if
            (self.in_collision and not self.is_successful)
            else 0.,
            'task': task.to_json()
        }
