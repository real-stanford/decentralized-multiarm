import pybullet as p
import pybullet_data
import numpy as np
from itertools import chain
import ray
import time
import torch
import traceback
from .UR5 import UR5
from .utils import (
    get_observation_dimensions,
    create_ur5s,
    pose_to_high_freq_pose,
    pos_to_high_freq_pos
)
from .tasks import TaskManager
from policy import Memory


class BaseEnv:
    SIMULATION_TIMESTEP = 1.0 / 240.0

    def __init__(self, env_config, training_config, gui, logger):
        self.logger = logger
        self.setup(env_config, training_config, gui)
        self.task_manager = TaskManager(
            config={
                'environment': env_config,
                'training': training_config
            },
            task_loader=training_config['task_loader'],
            colors=[ur5.color for ur5 in self.all_ur5s])

    def setup_action_observation(self, observations_config):
        self.obs_key = [
            obs_item['name']
            for obs_item in observations_config['items']
        ]
        self.action_dim = UR5.joints_count

        self.observation_items = observations_config['items']
        observation_dim = get_observation_dimensions(observations_config)

        self.actor_observation_dim = observation_dim
        self.critic_observation_dim = observation_dim

    def set_memory_cluster_map(self, memory_cluster_map):
        self.memory_cluster_map = memory_cluster_map

    def setup(self, env_config, training_config, gui):
        self.action_interval = env_config["action_interval"]
        self.episode_length = env_config["episode_length"]
        self.simulation_steps_per_action_step = int(
            self.action_interval / BaseEnv.SIMULATION_TIMESTEP)
        self.episode_counts = 0
        self.action_type = 'delta'\
            if 'action_type' not in env_config \
            else env_config['action_type']
        self.observations = None
        self.gui = gui
        self.ray_id = None
        # Get variables from config
        self.max_ur5s_count = env_config['max_ur5s_count']
        self.max_task_ur5s_count = env_config['max_ur5s_count']
        self.min_task_ur5s_count = env_config['min_ur5s_count']
        self.survival_penalty = env_config['reward']['survival_penalty']
        self.workspace_radius = env_config['workspace_radius']
        self.individually_reach_target = \
            env_config['reward']['individually_reach_target']
        self.collectively_reach_target = \
            env_config['reward']['collectively_reach_target']
        self.cooperative_individual_reach_target = \
            env_config['reward']['cooperative_individual_reach_target']
        self.collision_penalty = env_config['reward']['collision_penalty']
        proximity_config = env_config['reward']['proximity_penalty']
        self.proximity_penalty_distance = proximity_config['max_distance']
        self.proximity_penalty = proximity_config['penalty']
        self.delta_reward = env_config['reward']['delta']
        self.terminate_on_collectively_reach_target = env_config[
            'terminate_on_collectively_reach_target']
        self.terminate_on_collision = env_config[
            'terminate_on_collision']
        self.position_tolerance = env_config['position_tolerance']
        self.orientation_tolerance = env_config['orientation_tolerance']
        self.stop_ur5_after_reach = False\
            if 'stop_ur5_after_reach' not in env_config \
            else env_config['stop_ur5_after_reach']
        self.finish_task_in_episode = False
        self.centralized_policy = False \
            if 'centralized_policy' not in training_config\
            else training_config['centralized_policy']
        self.centralized_critic = False  \
            if 'centralized_critic' not in training_config\
            else training_config['centralized_critic']

        self.curriculum = env_config['curriculum']
        self.retry_on_fail = env_config['retry_on_fail']
        self.failed_in_task = False
        self.task_manager = None
        self.episode_policy_instance_keys = None
        self.memory_cluster_map = {}
        self.collision_distance = \
            env_config['collision_distance']
        self.curriculum_level = -1
        self.min_task_difficulty = 0.
        self.max_task_difficulty = 100.0
        if self.logger is not None:
            self.set_level(ray.get(self.logger.get_curriculum_level.remote()))

        self.expert_trajectory_threshold = 0.4\
            if 'expert_trajectory_threshold' not in env_config\
            else env_config['expert_trajectory_threshold']

        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf",
                   [0, 0, -self.collision_distance - 0.01])
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -9.81)

        self.all_ur5s = []  # reference to all UR5's in scene
        self.ur5_episode_memories = None

        # Keep track of episode progress
        self.current_step = 0
        self.terminate_episode = False

        # Visualization
        if self.gui:
            self.real_time_debug = p.addUserDebugParameter(
                'real-time', 0.0, 1.0, 1.0)
        self.viewer = None
        self.on_setup(env_config, training_config)
        self.setup_action_observation(training_config['observations'])
        self.setup_ur5s(env_config)

    def setup_ur5s(self, env_config):
        # Add UR5s
        if env_config['ur5s_position_picker'] == 'evenly_spaced':
            pass
        else:
            print("[BaseEnv]" +
                  " Position Picker not supported: {}".format(
                      env_config['ur5s_position_picker']))
            exit(-1)
        self.all_ur5s = create_ur5s(
            radius=0.5,
            count=self.max_ur5s_count,
            speed=env_config['ur5_speed'])
        self.active_ur5s = []
        self.enable_ur5s()

    def disable_all_ur5s(self):
        for i, ur5 in enumerate(self.all_ur5s):
            ur5.disable(idx=i)
        self.active_ur5s = []

    def enable_ur5s(self, count=None):
        if count == len(self.active_ur5s):
            return
        self.disable_all_ur5s()
        for i, ur5 in enumerate(self.all_ur5s):
            if count is not None and i == count:
                break
            ur5.enable()
            self.active_ur5s.append(ur5)

    def reset_stats(self):
        self.stats = {
            # number of steps arm is collided in each episode
            "collisions": [0] * len(self.active_ur5s),
            # number of steps arm spends in reached state in each episode
            "reached": [0] * len(self.active_ur5s),
            "collective_reach_count": 0
        }

    def get_pose_residuals(self, poseA, poseB):
        posA = np.array(poseA[0])
        ornA = np.quaternion(*poseA[1])

        posB = np.array(poseB[0])
        ornB = np.quaternion(*poseB[1])

        pos_residual = np.linalg.norm(posA - posB)
        orn_residual = (ornA * ornB.inverse()).angle()
        # Get smallest positive angle
        orn_residual = orn_residual % (np.pi * 2)
        if orn_residual > np.pi:
            orn_residual = 2 * np.pi - orn_residual
        return pos_residual, orn_residual

    def get_ur5_eef_residuals(self):
        residuals = [self.get_pose_residuals(
            target_pose,
            ur5.get_end_effector_pose()
        ) for ur5, target_pose in zip(
            self.active_ur5s,
            self.task_manager.get_target_end_effector_poses()
        )]
        pos_residuals = np.array([res[0] for res in residuals])
        orn_residuals = np.array([res[1] for res in residuals])
        return pos_residuals, orn_residuals

    def check_ur5_reached_target(self, i, ur5, target_pose):
        pos_residual, orn_residual = self.get_pose_residuals(
            target_pose,
            ur5.get_end_effector_pose())
        reached_position = pos_residual < self.position_tolerance
        reached_orientation = orn_residual < self.orientation_tolerance
        reached = reached_position and reached_orientation
        if reached:
            self.on_target_reach(ur5, i)
        else:
            ur5.on_untouch_target()
            self.task_manager.targets_visuals[i].normal()
        return reached

    def get_state(self):
        """
        Collect states of entire environment, and updates variables,
        such as whether or not to terminate episode
        """
        colliding = any([ur5.check_collision() for ur5 in self.active_ur5s])
        if colliding:
            self.on_collision()
        self.state = {
            "ur5s": []
        }

        self.state['ur5s'] = [{
            'end_effector_pose': ur5.get_end_effector_pose(),
            'joint_values': ur5.get_arm_joint_values(),
            'link_positions': ur5.get_link_global_positions(),
            'ur5': ur5,
            'pose': ur5.get_pose(),
            'colliding': False if not colliding else ur5.check_collision(),
            'target_pose': target_eef_pose,
            'reached_target': self.check_ur5_reached_target(
                i, ur5, target_eef_pose),
        } for i, (ur5, target_eef_pose) in
            enumerate(zip(self.active_ur5s,
                          self.task_manager.get_target_end_effector_poses()))]

        self.state['reach_count'] = sum([
            1 if ur5_state['reached_target']
            else 0 for ur5_state in self.state['ur5s']])
        if self.state['reach_count'] == len(self.active_ur5s):
            self.on_all_ur5s_reach_target()

        return self.state

    def get_observations(self, state=None, limit=10):
        if state is None:
            state = self.get_state()
        if len(self.history) == 0:
            self.history = [state] * limit
        else:
            self.history.append(state)
        if len(self.history) > limit:
            del self.history[0]
        if self.centralized_policy:
            return [self.centralized_policy_get_observation(self.history)]
        else:
            return [self.get_observation(
                this_ur5=ur5,
                history=self.history)
                for ur5 in self.active_ur5s]

    def centralized_policy_get_observation(self, history):
        obs = {
            'ur5s': []
        }
        for ur5_idx in range(len(self.active_ur5s)):
            obs['ur5s'].append({})
            for item in self.observation_items:
                key = item['name']
                val = None
                if key == 'link_positions':
                    val = [list(chain.from_iterable(
                        [np.array(link_pos)
                         for link_pos in state['ur5s'][ur5_idx][key]]))
                        for state in history[-(item['history'] + 1):]]
                elif key == 'end_effector_pose' \
                        or key == 'target_pose'\
                        or key == 'pose':
                    val = [list(chain.from_iterable(
                        state['ur5s'][ur5_idx][key]))
                        for state in
                        history[-(item['history'] + 1):]]
                else:
                    val = [state['ur5s'][ur5_idx][key]
                           for state in history[-(item['history'] + 1):]]
                obs['ur5s'][-1][key] = val
        return obs

    def get_observation(self, this_ur5, history):
        obs = {
            'ur5s': [],
        }
        # Sequence Observation
        pos = np.array(this_ur5.get_pose()[0])
        sorted_ur5s = [ur5 for ur5 in self.active_ur5s
                       if np.linalg.norm(
                           pos - np.array(ur5.get_pose()[0]))
                       < 2 * self.workspace_radius]
        # Sort by base distance, furthest to closest
        sorted_ur5s.sort(reverse=True, key=lambda ur5:
                         np.linalg.norm(pos - np.array(ur5.get_pose()[0])))
        for ur5 in sorted_ur5s:
            obs['ur5s'].append({})
            ur5_idx = self.active_ur5s.index(ur5)
            for item in self.observation_items:
                key = item['name']
                val = None
                high_freq = 'high_freq' in key
                key = key.split('_high_freq')[0]
                if key == 'joint_values':
                    val = [state['ur5s'][ur5_idx][key]
                           for state in history[-(item['history'] + 1):]]
                elif 'link_positions' in key:
                    # get flatten link positions in ur5's frame of reference
                    val = [list(chain.from_iterable(
                        [pos_to_high_freq_pos(this_ur5.global_to_ur5_frame(
                            position=np.array(link_pos),
                            rotation=None)[0])
                         if high_freq else
                            this_ur5.global_to_ur5_frame(
                            position=np.array(link_pos),
                            rotation=None)[0]
                         for link_pos in state['ur5s'][ur5_idx][key]]))
                        for state in history[-(item['history'] + 1):]]
                elif 'end_effector_pose' in key or \
                        'target_pose' in key\
                        or key == 'pose' or key == 'pose_high_freq':
                    val = [list(chain.from_iterable(
                        pose_to_high_freq_pose(
                            this_ur5.global_to_ur5_frame(
                                position=state['ur5s'][ur5_idx][key][0],
                                rotation=state['ur5s'][ur5_idx][key][1]))
                        if high_freq else
                        this_ur5.global_to_ur5_frame(
                            position=state['ur5s'][ur5_idx][key][0],
                            rotation=state['ur5s'][ur5_idx][key][1])))
                           for state in history[-(item['history'] + 1):]]
                else:
                    val = [pose_to_high_freq_pose(this_ur5.global_to_ur5_frame(
                        state['ur5s'][ur5_idx][key]))
                        if high_freq else
                        this_ur5.global_to_ur5_frame(
                        state['ur5s'][ur5_idx][key])
                        for state in history[-(item['history'] + 1):]]
                obs['ur5s'][-1][key] = val
        return obs

    def get_rewards(self, state):
        current_ur5_ee_residuals = self.get_ur5_eef_residuals()
        if self.prev_ur5_ee_residuals is None:
            self.prev_ur5_ee_residuals = current_ur5_ee_residuals
        survival_penalties = np.array([
            (self.survival_penalty
             if not ur5_state['reached_target']
             else 0)
            for ur5_state in state['ur5s']
        ])

        collision_penalties = np.array([
            (self.collision_penalty if ur5_state['colliding'] else 0)
            for ur5_state in state['ur5s']
        ])

        if self.cooperative_individual_reach_target:
            individually_reached_target_rewards = np.array(
                [self.individually_reach_target * state['reach_count']
                    for _ in range(len(self.active_ur5s))])
        else:
            individually_reached_target_rewards = np.array([
                (self.individually_reach_target
                    if ur5_state['reached_target']
                 else 0)
                for ur5_state in state['ur5s']
            ])
        # Only give delta rewards if ee is within a radius
        # away from the target ee
        delta_position_rewards = \
            [(prev - curr) * self.delta_reward['position']
             if curr < self.delta_reward['activation_radius']
             else 0.0
             for curr, prev in zip(
                current_ur5_ee_residuals[0],
                self.prev_ur5_ee_residuals[0])]
        delta_orientation_rewards = \
            [(prev - curr) * self.delta_reward['orientation']
             if curr_pos_res < self.delta_reward['activation_radius']
             else 0.0
             for curr, prev, curr_pos_res in zip(
                current_ur5_ee_residuals[1],
                self.prev_ur5_ee_residuals[1],
                current_ur5_ee_residuals[0])]
        # proximity penalty
        proximity_penalties = np.array([
            sum([(1 - closest_points_to_other[0][8]
                  / self.proximity_penalty_distance)
                 * self.proximity_penalty
                 for closest_points_to_other in ur5.closest_points_to_others
                 if len(closest_points_to_other) > 0 and
                 closest_points_to_other[0][8] <
                 self.proximity_penalty_distance])
            for ur5 in self.active_ur5s])

        collectively_reached_targets = (
            state['reach_count'] == len(self.active_ur5s))
        collective_reached_targets_rewards = np.array(
            [(self.collectively_reach_target
                if collectively_reached_targets
              else 0)
             for _ in range(len(self.active_ur5s))])
        self.prev_ur5_ee_residuals = current_ur5_ee_residuals
        ur5_rewards_sum = \
            collision_penalties + individually_reached_target_rewards +\
            collective_reached_targets_rewards + survival_penalties + \
            proximity_penalties + \
            delta_position_rewards + delta_orientation_rewards
        if self.centralized_policy:
            return np.array([ur5_rewards_sum.sum()])
        else:
            return ur5_rewards_sum

    def step_simulation(self):
        p.stepSimulation()

    def finish_up_step(self):
        pass

    def step(self, actions):
        if self.terminate_episode:
            return self.reset()
        self.current_step += 1
        rewards = np.zeros(len(self.ur5_episode_memories))
        self.handle_actions(actions)

        for t_sim in range(self.simulation_steps_per_action_step):
            p.stepSimulation()

            for ur5 in self.active_ur5s:
                ur5.step()

            self.state = self.get_state()
            rewards += self.get_rewards(self.state)

            self.on_step_simulation(
                self.current_step *
                self.simulation_steps_per_action_step + t_sim,
                self.episode_length * self.simulation_steps_per_action_step,
                self.state)

            if self.gui and \
                    p.readUserDebugParameter(self.real_time_debug) == 1.0:
                time.sleep(BaseEnv.SIMULATION_TIMESTEP)

            # Check if user wants to end current episode
            rKey = ord('r')
            keys = p.getKeyboardEvents()
            if rKey in keys and keys[rKey] & p.KEY_WAS_TRIGGERED:
                self.terminate_episode = True

            if self.terminate_episode:
                break

        # Check if ur5 reached target positions
        self.terminate_episode = self.terminate_episode\
            or self.current_step >= self.episode_length

        self.finish_up_step()

        self.observations = [self.preprocess_obs(o)
                             for o in self.get_observations(state=self.state)]

        self.episode_reward_sum += rewards

        for o, r, m in zip(
                self.observations,
                rewards,
                self.ur5_episode_memories):
            m.add_rewards_and_termination(r, self.terminate_episode)
            m.add_value('next_observations', o)
            if self.centralized_critic:
                critic_next_obs = []
                for obs in m.data['next_observations'][-1]:
                    critic_next_obs.append(torch.cat((
                        obs,
                        torch.FloatTensor([0.]*6))))
                m.data['critic_next_observations'].append(
                    torch.stack(critic_next_obs))
            if not self.terminate_episode:
                m.add_observation(o)

        if self.terminate_episode:
            self.on_episode_end()

        return self.obs_to_policy(self.observations), self.ray_id

    def obs_to_policy(self, obs):
        retval = {}
        for ob, policy_key in zip(obs, self.episode_policy_instance_keys):
            if policy_key not in retval:
                retval[policy_key] = []
            retval[policy_key].append(ob)
        return retval

    def send_memory_to_clusters(self):
        ray.get([
            self.memory_cluster_map[
                policy_instance_key].submit_memory.remote(memory)
            for memory, policy_instance_key in zip(
                self.ur5_episode_memories, self.episode_policy_instance_keys)])

    def get_stats_to_log(self):
        pos_residuals, orn_residuals = self.get_ur5_eef_residuals()
        return {
            'rewards': np.mean(self.episode_reward_sum),
            'individual_reach': np.mean(self.stats['reached']),
            'collective_reach': np.mean(self.stats['collective_reach_count']),
            'collision': np.mean(self.stats['collisions']) /
            (self.episode_length * self.simulation_steps_per_action_step),
            'success': self.stats['collective_reach_count']
            if sum(self.stats['collisions']) == 0
            else 0,
            'curriculum_level': self.curriculum_level,
            'mean_pos_residual': pos_residuals.mean(),
            'mean_orn_residual': orn_residuals.mean(),
            'max_pos_residual': pos_residuals.max(),
            'max_orn_residual': orn_residuals.max(),
        }

    def send_stats_to_logger(self):
        if self.logger is not None:
            logger_curriculum_level = ray.get(self.logger.add_stats.remote(
                self.get_stats_to_log()))
            self.set_level(logger_curriculum_level)

    def on_success(self):
        self.task_manager.on_success()

    def on_episode_end(self):
        success = self.stats['collective_reach_count']\
            if sum(self.stats['collisions']) == 0\
            else 0
        if success != 0:
            self.on_success()
        self.failed_in_task = \
            self.stats['collective_reach_count'] == 0\
            or sum(self.stats['collisions']) != 0
        self.send_memory_to_clusters()
        self.send_stats_to_logger()

    def reset_memories(self):
        self.ur5_episode_memories = [
            Memory(memory_field)
            for memory_field in self.get_memory_fields()]

    def get_memory_fields(self):
        memory_fields = []
        for policy_key in self.episode_policy_instance_keys:
            if self.memory_cluster_map[policy_key] is None:
                memory_fields.append([])
            else:
                memory_fields.append(
                    ray.get(
                        self.memory_cluster_map[policy_key]
                        .get_memory_fields.remote()))
        return memory_fields

    def reset(self):
        self.history = []
        self.current_step = 0
        self.episode_counts += 1
        self.finish_task_in_episode = False
        self.setup_task()
        self.reset_memories()
        self.reset_stats()
        self.prev_ur5_ee_residuals = None
        self.on_reset()
        self.episode_reward_sum = np.zeros(len(self.ur5_episode_memories))
        for _ in range(50):
            p.stepSimulation()
        self.observations = [self.preprocess_obs(o)
                             for o in self.get_observations()]
        assert len(self.observations) == len(self.ur5_episode_memories)
        for o, m in zip(self.observations, self.ur5_episode_memories):
            m.add_observation(o)

        self.terminate_episode = False
        return self.obs_to_policy(self.observations), self.ray_id

    def should_get_next_task(self):
        if not self.retry_on_fail:
            return True
        elif not self.failed_in_task:
            return True
        # Only retry failed task if
        # graduate from final curriculum level a while ago
        return False

    def get_current_task(self):
        return self.task_manager.current_task

    def setup_task(self):
        if self.should_get_next_task() and self.task_manager is not None:
            self.task_manager.setup_next_task(
                max_task_ur5s_count=self.max_task_ur5s_count,
                min_task_ur5s_count=self.min_task_ur5s_count,
                max_task_difficulty=self.max_task_difficulty,
                min_task_difficulty=self.min_task_difficulty)
        self.enable_ur5s(count=self.get_current_task().ur5_count)
        for ur5, ur5_task in zip(self.active_ur5s,
                                 self.get_current_task()):
            ur5.set_pose(ur5_task['base_pose'])
            ur5.set_arm_joints(ur5_task['start_config'])
            ur5.step()
        assert len(self.memory_cluster_map) == 1
        num_policies = 1 if self.centralized_policy else len(self.active_ur5s)
        self.episode_policy_instance_keys = [
            key for key in self.memory_cluster_map] * num_policies

    def handle_actions(self, actions):
        if self.stop_ur5_after_reach:
            for i, (action, ur5, target_eef_pose) in enumerate(
                    zip(actions,
                        self.active_ur5s,
                        self.task_manager.get_target_end_effector_poses())):
                if self.check_ur5_reached_target(i, ur5, target_eef_pose):
                    action = np.zeros(6)
        if self.centralized_policy:
            actions = actions[0]
            self.ur5_episode_memories[0].add_action(actions)
            actions = list(torch.split(actions, 6))
        else:
            for action, m in zip(actions, self.ur5_episode_memories):
                m.add_action(action)
        if self.centralized_critic:
            for this_ur5, memory in zip(self.active_ur5s,
                                        self.ur5_episode_memories):
                # Sort actions based on base distance
                pos = np.array(this_ur5.get_pose()[0])
                sorted_ur5s = [(action, ur5) for action, ur5 in
                               zip(actions, self.active_ur5s)
                               if np.linalg.norm(
                                   pos - np.array(ur5.get_pose()[0]))
                               < 2 * self.workspace_radius]
                # Sort by base distance, furthest to closest
                sorted_ur5s.sort(reverse=True, key=lambda item:
                                 np.linalg.norm(
                                     pos - np.array(item[1].get_pose()[0])))
                sorted_actions = [action for (action, ur5) in sorted_ur5s]
                # Last aciton is self, set to zero
                sorted_actions[-1] = torch.FloatTensor([0.]*6)
                # concat actions to previous observations
                critic_obs = []
                for obs, action in zip(
                        memory.data['observations'][-1],
                        sorted_actions):
                    critic_obs.append(torch.cat((obs, action)))
                memory.data['critic_observations'].append(
                    torch.stack(critic_obs))
                # also update next_observations
                if len(memory.data['critic_next_observations']):
                    memory.data['critic_next_observations'][-1] = \
                        memory.data['critic_observations'][-1]
        self.action_to_robots(actions)

    def action_to_robots(self, actions):
        if len(actions) != len(self.active_ur5s):
            print("Wrong action dimensions: {} (received) vs {} (correct)"
                  .format(len(actions), len(self.active_ur5s)))
            for line in traceback.format_stack():
                print(line.strip())
            exit()
        for ur5, action in zip(self.active_ur5s, actions):
            if type(action) != np.ndarray:
                action = action.data.numpy()
            if self.action_type == 'delta':
                ur5.control_arm_joints_delta(action)
            elif self.action_type == 'target-norm':
                ur5.control_arm_joints_norm(action)
            else:
                print("[BaseEnv] Unsupported action type:", self.action_type)

    def preprocess_obs(self, obs):
        output = []
        for ur5_obs in obs['ur5s']:
            ur5_output = np.array([])
            for key in self.obs_key:
                key = key.split('_high_freq')[0]
                item = ur5_obs[key]
                for history_frame in item:
                    ur5_output = np.concatenate((
                        ur5_output,
                        history_frame))
            output.append(ur5_output)
        output = torch.FloatTensor(output)
        if self.centralized_policy:
            output = output.view(-1)
        return output

    def setup_ray(self, id):
        print("[BaseEnv] Setting up ray: {}".format(id))
        self.ray_id = {"val": id}

    # Call backs
    def on_setup(self, env_config, training_config):
        pass

    def on_collision(self):
        visualize_collision = True
        if visualize_collision:
            self.visualize_collision()

        if self.terminate_on_collision:
            self.terminate_episode = True

    def visualize_collision(self):
        if self.gui:
            points = set([ur5.prev_collided_with[5]
                          for ur5 in self.active_ur5s
                          if ur5.prev_collided_with])
            sphere_vs_id = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.05,
                rgbaColor=(1, 0, 0, 0.5))
            point_visuals = [p.createMultiBody(
                basePosition=point,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=sphere_vs_id)
                for point in points]
            input('\tCollision: Press ENTER to continue')
            for point_visual in point_visuals:
                p.removeBody(point_visual)

    def on_step_simulation(self, curr, max, state):
        """
        :param curr: current simulation step in episode
        :param max: maximum number of simulation steps in episode
        :param state: full state of environment at curr time
        :return: None
        """
        for i, ur5 in enumerate(self.active_ur5s):
            if state['ur5s'][i]['colliding']:
                self.stats['collisions'][i] += + 1
        self.task_manager.set_timer(curr / max)

    def on_reset(self):
        pass

    def on_all_ur5s_reach_target(self):
        self.stats["collective_reach_count"] += 1
        self.terminate_episode = self.terminate_episode or\
            self.terminate_on_collectively_reach_target
        self.finish_task_in_episode = True

    def on_target_reach(self, ur5, idx):
        ur5.on_touch_target()
        self.task_manager.targets_visuals[idx].touched()
        self.stats["reached"][self.active_ur5s.index(ur5)] += 1
        if self.stop_ur5_after_reach:
            ur5.control_arm_joints(ur5.get_arm_joint_values())

    def set_level(self, level):
        if self.curriculum is None:
            return
        if not (level < len(self.curriculum['levels'])):
            return
        elif self.curriculum_level >= level:
            return
        for level_idx in range(self.curriculum_level + 1, level + 1):
            level_config = self.curriculum['levels'][level_idx]
            updated_stats = {}
            for key in level_config:
                if key == 'position_tolerance':
                    self.position_tolerance = level_config[key]
                elif key == 'orientation_tolerance':
                    self.orientation_tolerance = level_config[key]
                elif key == 'collision_penalty':
                    self.collision_penalty = level_config[key]
                elif key == 'max_task_ur5s_count':
                    self.max_task_ur5s_count = level_config[key]
                elif key == 'max_task_difficulty':
                    self.max_task_difficulty = level_config[key]
                elif key == 'min_task_difficulty':
                    self.min_task_difficulty = level_config[key]
                updated_stats[key] = level_config[key]
            output = '[BaseEnv] Level {} '.format(level_idx)
            for key in updated_stats:
                output += '| {}: {}'.format(
                    key, updated_stats[key])
            print(output)
        self.curriculum_level = level


@ ray.remote
class ParallelBaseEnv(BaseEnv):
    def __init__(self, env_config, training_config, gui, logger):
        super().__init__(env_config, training_config, gui, logger)
