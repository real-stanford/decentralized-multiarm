from .BaseRLAlgo import BaseRLAlgo
from torch.optim import Adam
import torch
from copy import deepcopy
from .Memory import MemoryCluster
import ray
from .utils import ReplayBufferDataset
from time import time
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

torch.multiprocessing.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')


class SACLearner(BaseRLAlgo):
    def __init__(
            self,
            policy_key,
            network_fns,
            algo_config,
            writer,
            load_path=None,
            device='cuda',
            training=True,
            save_interval=500):
        super().__init__(
            policy_key=policy_key,
            writer=writer,
            training=training,
            load_path=load_path,
            device=device)
        self.network_fns = network_fns
        self.pi_lr = algo_config["pi_lr"]
        self.q_lr = algo_config["q_lr"]
        self.tau = algo_config["tau"]
        self.discount = algo_config["discount"]
        self.alpha = algo_config["alpha"]
        self.reparametrize = True
        self.reward_scale = 1.0
        self.num_updates_per_train = algo_config["num_updates_per_train"]
        self.batch_size = algo_config["batch_size"]
        self.memory_cluster_capacity = algo_config["memory_cluster_capacity"]
        self.warmup_timesteps = algo_config["warmup_timesteps"]
        self.minimum_replay_buffer_freshness = algo_config["minimum_replay_buffer_freshness"]
        self.action_scaling = algo_config["action_scaling"]
        self.save_interval = save_interval
        self.replay_buffer = None
        self.setup(load_path)

    def setup(self, load_path):
        self.stats = {
            'time_steps': 0,
            'update_steps': 0
        }

        self.policy = self.network_fns['policy']().to(self.device)
        self.Q1 = self.network_fns['Q1']().to(self.device)
        self.Q2 = self.network_fns['Q2']().to(self.device)
        self.Q1_target = deepcopy(self.Q1)
        self.Q2_target = deepcopy(self.Q2)
        self.policy_opt = Adam(
            self.policy.parameters(),
            lr=self.pi_lr,
            betas=(0.9, 0.999))
        self.Q1_opt = Adam(
            self.Q1.parameters(),
            lr=self.q_lr,
            betas=(0.9, 0.999))
        self.Q2_opt = Adam(
            self.Q2.parameters(),
            lr=self.q_lr,
            betas=(0.9, 0.999))

        if load_path is not None:
            print("[SAC] loading networks from ", load_path)
            checkpoint = torch.load(load_path, map_location=self.device)
            networks = checkpoint['networks']
            self.policy.load_state_dict(networks['policy'])
            self.Q1.load_state_dict(networks['Q1'])
            self.Q2.load_state_dict(networks['Q2'])
            self.Q1_target.load_state_dict(networks['Q1_target'])
            self.Q2_target.load_state_dict(networks['Q2_target'])
            self.policy_opt.load_state_dict(networks['policy_opt'])
            self.Q1_opt.load_state_dict(networks['Q1_opt'])
            self.Q2_opt.load_state_dict(networks['Q2_opt'])
            self.stats['time_steps'] = checkpoint['stats']['time_steps']
            self.stats['update_steps'] = checkpoint['stats']['update_steps']
            print("[SAC] Continuing from time step ", self.stats['time_steps'],
                  " and update step ", self.stats['update_steps'])
        self.memory_cluster = MemoryCluster.remote(
            memory_fields=['next_observations'],
            init_time_steps=self.stats['time_steps'])

        def time_step_callback_fn(value):
            self.stats['time_steps'] = value
        self.register_async_task(
            name='time_steps',
            fn=self.memory_cluster.get_total_timesteps.remote,
            callback=time_step_callback_fn
        )

        def memory_cluster_callback_fn(data):
            if data is None:
                return
            if self.replay_buffer is None:
                self.replay_buffer = ReplayBufferDataset(
                    data=data,
                    device=self.device,
                    capacity=self.memory_cluster_capacity)
                self.train_loader = DataLoader(
                    self.replay_buffer,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=0)
            else:
                self.replay_buffer.extend(data)
        self.register_async_task(
            name='memory_cluster',
            fn=self.memory_cluster.get_data.remote,
            callback=memory_cluster_callback_fn
        )
        self.Q_criterion = torch.nn.MSELoss()
        print(self.policy)
        print(self.Q1)
        self.prev_update_time = time()

    def get_policy_parameters(self):
        return self.policy.state_dict()

    def get_update(self, version):
        self.check_async()
        if self.should_update():
            self.train()
        if version < self.stats['update_steps']:
            return self.stats['update_steps'], self.policy.state_dict()
        else:
            return None

    def get_inference_node(self):
        class SACInference:
            def __init__(self,
                         device,
                         policy,
                         policy_version,
                         reparametrize,
                         learner,
                         deterministic=False):
                self.device = device
                self.policy = policy.to(self.device)
                self.reparametrize = reparametrize
                self.deterministic = deterministic
                self.current_version = policy_version
                self.learner_handle = learner
                self.check_update_task = None

            def act(self, obs):
                self.check_update()
                obs = pad_sequence(obs, batch_first=True).to(self.device)
                if len(obs) == 1:
                    obs = torch.unsqueeze(obs[0], dim=0)
                actions, _ = self.policy(
                    obs, deterministic=self.deterministic,
                    reparametrize=self.reparametrize)
                return actions.detach().cpu()

            def check_update(self):
                if self.check_update_task is None:
                    self.check_update_task = \
                        self.learner_handle.get_update.remote(
                            self.current_version)
                else:
                    ready, remaining = ray.wait(
                        [self.check_update_task],
                        num_returns=1,
                        timeout=1e-8)
                    if len(ready) > 0:
                        retval = ray.get(ready)[0]
                        self.check_update_task = None
                        if retval is None:
                            return
                        self.current_version, new_params = retval
                        self.policy.load_state_dict(new_params)

        inference_policy = deepcopy(self.policy)
        return SACInference(
            device=self.device,
            policy=inference_policy,
            policy_version=self.stats['update_steps'],
            reparametrize=self.reparametrize,
            learner=self.ray_id)

    def check_async(self):
        for task_key in self.async_tasks:
            task = self.async_tasks[task_key]
            ready, remaining = ray.wait(
                [task['value']], num_returns=1, timeout=1e-5)
            if len(ready) > 0:
                task['callback'](ray.get(ready)[0])
                task['value'] = task['fn']()

    def update_targets(self):
        for Qf_target, Qf in zip(
            [self.Q1_target, self.Q2_target],
                [self.Q1, self.Q2]):
            for target_param, param in zip(
                    Qf_target.parameters(), Qf.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) +
                                        param.data * self.tau)

    def train(self):
        if not self.training:
            return 0.0
        policy_loss_sum = 0.0
        q1_loss_sum = 0.0
        q2_loss_sum = 0.0
        policy_entropy_sum = 0.0
        policy_value_sum = 0.0

        train_loader_it = iter(self.train_loader)
        for i in range(self.num_updates_per_train):
            batch = next(train_loader_it, None)
            if batch is None:
                train_loader_it = iter(self.train_loader)
                batch = next(train_loader_it, None)
            policy_loss, q1_loss, q2_loss, policy_entropy, policy_value = \
                self.train_batch(batch=batch)
            policy_loss_sum += policy_loss
            q1_loss_sum += q1_loss
            q2_loss_sum += q2_loss
            policy_entropy_sum += policy_entropy
            policy_value_sum += policy_value
            self.update_targets()
            self.stats['update_steps'] += 1
            if self.stats['update_steps'] % self.save_interval == 0:
                self.save()

        policy_loss_sum /= self.num_updates_per_train
        q1_loss_sum /= self.num_updates_per_train
        q2_loss_sum /= self.num_updates_per_train
        policy_entropy_sum /= self.num_updates_per_train
        policy_value_sum /= self.num_updates_per_train

        self.log_scalars({
            'Training/Policy_Loss': policy_loss_sum,
            'Training/Policy_Entropy': policy_entropy_sum,
            'Training/Policy_Value': policy_value_sum,
            'Training/Q1_Loss': q1_loss_sum,
            'Training/Q2_Loss': q2_loss_sum,
            'Training/Freshness': self.replay_buffer.freshness,
        })

        output = "\r[SAC] pi: {0:.4f} | pi_entropy: {1:.4f}".format(
            policy_loss_sum, policy_entropy_sum)
        output += "| pi_value: {0:.4f} ".format(policy_value_sum)
        output += "| Q1: {0:.4f} | Q2: {1:.4f} ".format(
            q1_loss_sum, q2_loss_sum)
        output += "| freshness: {0:.3f} ".format(self.replay_buffer.freshness)
        output += "| time: {0:.2f}".format(
            float(time() - self.prev_update_time))
        print(output, end='')
        self.prev_update_time = time()
        self.replay_buffer.freshness = 0.0

    def should_update(self):
        return self.replay_buffer is not None and \
            len(self.replay_buffer) > self.warmup_timesteps\
            and self.replay_buffer.freshness > self.minimum_replay_buffer_freshness

    def train_batch(self, batch):
        if len(batch) == 5:
            obs, actions, rewards, next_obs, terminals = batch
            critic_obs = obs
            critic_next_obs = next_obs
        else:
            critic_obs, obs, actions, rewards, critic_next_obs,\
                next_obs, terminals = batch

        rewards = torch.unsqueeze(rewards, dim=1)
        terminals = torch.unsqueeze(terminals, dim=1)

        new_obs_actions, new_obs_action_logprobs = self.policy(
            obs=obs,
            deterministic=False,
            reparametrize=self.reparametrize)
        new_obs_action_logprobs = torch.unsqueeze(
            new_obs_action_logprobs, dim=1)

        new_next_obs_action, next_obs_action_logprobs = self.policy(
            obs=next_obs,
            deterministic=False,
            reparametrize=self.reparametrize)
        next_obs_action_logprobs = torch.unsqueeze(
            next_obs_action_logprobs, dim=1)

        q_new_actions = torch.min(
            self.Q1(obs=critic_obs, actions=new_obs_actions *
                    self.action_scaling),
            self.Q2(obs=critic_obs, actions=new_obs_actions *
                    self.action_scaling))

        # Train policy
        policy_loss = (self.alpha * new_obs_action_logprobs -
                       q_new_actions).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_opt.step()

        # Train Q networks
        q1_pred = self.Q1(obs=critic_obs, actions=actions *
                          self.action_scaling)
        q2_pred = self.Q2(obs=critic_obs, actions=actions *
                          self.action_scaling)

        target_q_values = torch.min(
            self.Q1_target(critic_next_obs, new_next_obs_action *
                           self.action_scaling),
            self.Q2_target(critic_next_obs, new_next_obs_action *
                           self.action_scaling)) \
            - (self.alpha * next_obs_action_logprobs)
        with torch.no_grad():
            q_target = self.reward_scale * rewards + (1. - terminals) * \
                self.discount * target_q_values

        q1_loss = self.Q_criterion(q1_pred, q_target)
        q2_loss = self.Q_criterion(q2_pred, q_target)

        self.Q1_opt.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.Q1_opt.step()

        self.Q2_opt.zero_grad()
        q2_loss.backward()
        self.Q2_opt.step()

        return (
            policy_loss.item(),
            q1_loss.item(),
            q2_loss.item(),
            -new_obs_action_logprobs.mean().item(),
            q_new_actions.detach().mean().item()
        )

    def get_state_dicts_to_save(self):
        return {
            'policy': self.policy.state_dict(),
            'Q1': self.Q1.state_dict(),
            'Q2': self.Q2.state_dict(),
            'Q1_target': self.Q1_target.state_dict(),
            'Q2_target': self.Q2_target.state_dict(),
            'policy_opt': self.policy_opt.state_dict(),
            'Q1_opt': self.Q1_opt.state_dict(),
            'Q2_opt': self.Q2_opt.state_dict(),
        }

    def log_scalars(self, scalars_dict):
        ray.get(self.writer.add_scalars.remote(
            scalars_dict, self.stats['time_steps']))
