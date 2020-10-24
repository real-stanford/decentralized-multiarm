from .baseEnv import BaseEnv
import ray
import numpy as np
from .tasks import Task
from time import sleep
from .utils import create_ur5s


class BenchmarkEnv(BaseEnv):
    EPISODE_CLOCK_TIME_LENGTH = 20

    def __init__(self, env_config, training_config, gui, logger):
        env_config['episode_length'] = int(
            BenchmarkEnv.EPISODE_CLOCK_TIME_LENGTH /
            env_config['action_interval'])
        env_config['ur5_speed'] = 0.008
        env_config['max_ur5s_count'] = 10
        env_config['min_ur5s_count'] = 1
        print("[BenchmarkEnv] config:")
        print("\tepisode_length:{}".format(env_config['episode_length']))
        print("\taction_interval:{}".format(env_config['action_interval']))
        print("\tsimulation_time:{}".format(
            BenchmarkEnv.EPISODE_CLOCK_TIME_LENGTH))
        super().__init__(
            env_config=env_config,
            training_config=training_config,
            gui=gui,
            logger=logger)
        self.reset_score()
        self.position_tolerance = 0.02
        self.orientation_tolerance = 0.1
        self.terminate_on_collectively_reach_target = True
        self.terminate_on_collision = True
        self.stop_ur5_after_reach = False
        print("\tposition_tolerance:{}".format(
            self.position_tolerance))
        print("\torientation_tolerance:{}".format(
            self.orientation_tolerance))

    def on_reset(self):
        super().on_reset()
        self.reset_score()

    def reset_score(self):
        self.current_episode_score = {
            'time': 0.0
        }

    def set_level(self, level):
        pass

    def setup_ur5s(self, env_config):
        # Add UR5s
        self.all_ur5s = create_ur5s(
            radius=0.5,
            count=self.max_ur5s_count,
            speed=env_config['ur5_speed'])
        self.active_ur5s = []
        self.enable_ur5s()

    def on_episode_end(self):
        if self.gui:
            sleep(1)
        stats = self.get_stats_to_log()
        for key in stats.keys():
            self.current_episode_score[key] = stats[key]
        collision_free = self.current_episode_score['collision'] == 0
        self.current_episode_score['success'] =\
            self.current_episode_score['collective_reach']\
            if collision_free\
            else 0
        if self.current_episode_score['success'] == 0:
            self.current_episode_score['collision'] =\
                0 if collision_free else 1
        task = self.get_current_task()
        self.current_episode_score['task'] = {
            'task_path': task.task_path,
            'id': task.id,
            'ur5_count': task.ur5_count,
            'difficulty': task.difficulty
        }
        pos_residuals, orn_residuals = self.get_ur5_eef_residuals()
        target_poses = self.task_manager.get_target_end_effector_poses()
        transformed_target_poses = [
            ur5.global_to_ur5_frame(pose[0], pose[1])
            for pose, ur5 in
            zip(target_poses, self.active_ur5s)
        ]
        self.current_episode_score['debug'] = {
            'target_ee_poses': target_poses,
            'transformed_target_ee_poses': transformed_target_poses,
            'pos_residuals': pos_residuals,
            'orn_residuals': orn_residuals,
            'pos_reached': [res < self.position_tolerance
                            for res in pos_residuals],
            'orn_reached': [res < self.orientation_tolerance
                            for res in orn_residuals],
            'target_joint_config':
            self.task_manager.current_task.start_goal_config,
            'initial_joint_config':
            self.task_manager.current_task.start_config,
            'final_joint_config': [list(ur5.get_arm_joint_values())
                                   for ur5 in self.active_ur5s],
            'final_ee_poses': [list(ur5.get_end_effector_pose())
                               for ur5 in self.active_ur5s],
        }
        ray.get(self.logger.add_stats.remote(
                self.current_episode_score))

    def reset_stats(self):
        super().reset_stats()

    def on_step_simulation(self, curr, max, state):
        super().on_step_simulation(curr, max, state)
        self.current_episode_score['time'] = \
            BenchmarkEnv.EPISODE_CLOCK_TIME_LENGTH * curr / max


@ray.remote
class ParallelBenchmarkEnv(BenchmarkEnv):
    def __init__(self, env_config, training_config, gui, logger):
        super().__init__(env_config, training_config, gui=gui, logger=logger)
