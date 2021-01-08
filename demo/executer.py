import numpy as np
from time import sleep
from itertools import chain
from torch import FloatTensor
import pybullet as p
from typing import List
from task import UR5AsyncTaskRunner as TaskRunner, PolicyTask
import pybullet_utils as pu
from recorder import PybulletRecorder
from random import randint
import os


class Executer:
    def __init__(self,
                 task_runners: List[TaskRunner],
                 recorder: PybulletRecorder,
                 recorder_dir,
                 synchronous=False, limit=100000):
        self.task_runners = task_runners
        self.ur5s = [tr.ur5 for tr in task_runners]
        if synchronous:
            raise Exception('synchronous execution not supported')
        self.synchronous = synchronous
        self.limit = limit
        self.recorder = recorder
        self.recorder_dir = recorder_dir
        self.simulation_output_path = os.path.join(self.recorder_dir, f'simulation_{randint(0,1000)}.pkl')
        self.observation_items = [
            {
                "name": "joint_values",
                "dimensions": 6,
                "history": 1
            },
            {
                "name": "end_effector_pose",
                "dimensions": 7,
                "history": 1
            },
            {
                "name": "target_pose",
                "dimensions": 7,
                "history": 1
            },
            {
                "name": "link_positions",
                "dimensions": 30,
                "history": 1
            },
            {
                "name": "pose",
                "dimensions": 7,
                "history": 0
            }
        ]
        self.obs_key = [
            obs_item['name']
            for obs_item in self.observation_items
        ]
        self.history = []

        for tr in self.task_runners:
            for t in tr.tasks:
                if type(t) == PolicyTask:
                    t.get_observation = self.get_observation

    def get_target_end_effector_poses(self):
        return [tr.current_task().target_pose
                if type(tr.current_task()) == PolicyTask
                else tr.ur5.get_end_effector_pose()
                for tr in self.task_runners
                ]

    def get_state(self):
        """
        Collect states of entire environment, and updates variables,
        such as whether or not to terminate episode
        """
        self.state = {
            "ur5s": []
        }

        self.state['ur5s'] = [{
            'end_effector_pose': ur5.get_end_effector_pose(),
            'joint_values': ur5.get_arm_joint_values(),
            'link_positions': ur5.get_link_global_positions(),
            'ur5': ur5,
            'pose': ur5.get_pose(),
            'target_pose': target_pose
        } for i, (ur5, target_pose) in
            enumerate(zip(self.ur5s,
                          self.get_target_end_effector_poses()))]
        return self.state

    def update_history(self, limit=10):
        state = self.get_state()
        if len(self.history) == 0:
            self.history = [state] * limit
        else:
            self.history.append(state)
        if len(self.history) > limit:
            del self.history[0]

    def is_done(self):
        return all([tr.is_done() for tr in self.task_runners])

    def run(self):
        for step_count in range(self.limit):
            if self.is_done():
                self.try_save()
                return True, step_count, None
            self.update_history()
            for tr in self.task_runners:
                success, info = tr.step()
                if not success:
                    self.try_save()
                    return False, step_count, info
            p.stepSimulation()
            self.recorder.add_keyframe()
        self.try_save()
        return False, step_count, 'out of time'

    def get_observation(self, this_ur5, history=None):
        if history is None:
            history = self.history
        workspace_radius = 0.85
        obs = {
            'ur5s': [],
        }
        # Sequence Observation
        pos = np.array(this_ur5.get_pose()[0])
        sorted_ur5s = [ur5 for ur5 in self.ur5s
                       if np.linalg.norm(
                           pos - np.array(ur5.get_pose()[0]))
                       < 2 * workspace_radius]
        # Sort by base distance, furthest to closest
        sorted_ur5s.sort(reverse=True, key=lambda ur5:
                         np.linalg.norm(pos - np.array(ur5.get_pose()[0])))
        for ur5 in sorted_ur5s:
            obs['ur5s'].append({})
            ur5_idx = self.ur5s.index(ur5)
            for item in self.observation_items:
                key = item['name']
                val = None
                if key == 'joint_values':
                    val = [state['ur5s'][ur5_idx][key]
                           for state in history[-(item['history'] + 1):]]
                elif key == 'link_positions':
                    # get flatten link positions in ur5's frame of reference
                    val = [list(chain.from_iterable(
                        [this_ur5.global_to_ur5_frame(
                            position=np.array(link_pos),
                            rotation=None)[0]
                         for link_pos in state['ur5s'][ur5_idx][key]]))
                        for state in history[-(item['history'] + 1):]]
                elif key == 'end_effector_pose' \
                        or key == 'target_pose'\
                        or key == 'pose':
                    val = [list(chain.from_iterable(
                        this_ur5.global_to_ur5_frame(
                            position=state['ur5s'][ur5_idx][key][0],
                            rotation=state['ur5s'][ur5_idx][key][1])))
                           for state in history[-(item['history'] + 1):]]
                else:
                    val = [this_ur5.global_to_ur5_frame(
                        state['ur5s'][ur5_idx][key])
                        for state in history[-(item['history'] + 1):]]
                obs['ur5s'][-1][key] = val
        return self.preprocess_obs(obs)

    def preprocess_obs(self, observation):
        output = []
        for ur5_obs in observation['ur5s']:
            ur5_output = np.array([])
            for key in self.obs_key:
                item = ur5_obs[key]
                for history_frame in item:
                    ur5_output = np.concatenate((
                        ur5_output,
                        history_frame))
            output.append(ur5_output)
        output = FloatTensor(output)
        return output

    def try_save(self):
        if self.recorder:
            self.recorder.save(
                self.simulation_output_path)
            self.recorder.reset()
