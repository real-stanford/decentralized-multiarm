# Copyright (c) 2019 Huy Ha
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from .Actor import (Actor4, Actor8, Actor12)
from .Critic import (Critic4, Critic8, Critic12)


class ActorCritic(nn.Module):
    def __init__(self,
                 actor_observation_dim,
                 critic_observation_dim,
                 action_dim,
                 depth=8,
                 load_path=None,
                 device='cpu'):
        super(ActorCritic, self).__init__()

        if depth == 4:
            Actor = Actor4
            Critic = Critic4
        elif depth == 8:
            Actor = Actor8
            Critic = Critic8
        elif depth == 12:
            Actor = Actor12
            Critic = Critic12
        else:
            print("Unsupported Depth: {}".format(depth))
            exit()

        self.action_dim = action_dim

        # Contruct actor and critic
        self.actor = Actor(
            observation_dim=actor_observation_dim,
            # +1 dim for each dim for variance
            action_dim=self.action_dim * 2)

        self.critic = Critic(
            observation_dim=critic_observation_dim,
            action_dim=action_dim)

        if load_path is not None:
            # trained_policy = torch.load(load_path)
            trained_policy = torch.load(
                load_path, map_location=device)
            print("[ActorCritic] loading policy from {}".format(load_path))
            self.actor = trained_policy['actor']
            self.critic = trained_policy['critic']
        else:
            def weights_init_uniform_rule(m):
                classname = m.__class__.__name__
                # for every Linear layer in a model..
                if classname.find('Linear') != -1:
                    # get the number of the inputs
                    n = m.in_features
                    y = 1.0 / np.sqrt(n)
                    m.weight.data.uniform_(-y, y)
                    m.bias.data.fill_(0)
            self.actor.apply(weights_init_uniform_rule)
            self.critic.apply(weights_init_uniform_rule)

    def forward(self):
        raise NotImplementedError

    def act(self, observation, device, grad=False, return_dist=False):
        # Sample from a distribution of actions
        output = self.actor(observation)

        single_process = output.size() == torch.Size([self.action_dim * 2])

        if single_process:
            action_means, action_variances = torch.split(
                output, self.action_dim, dim=0)
        else:
            action_means, action_variances = torch.split(
                output, self.action_dim, dim=1)
        # Scale action variance between 0 and 1
        action_variances = torch.clamp_min((action_variances + 1) / 2, 1e-8)

        if single_process:
            action_variances = [action_variances]
        action_variances = torch.stack(
            [torch.diag(action_variance)
             for action_variance in action_variances])
        try:
            dist = MultivariateNormal(action_means, action_variances)
        except Exception as e:
            print(e)
            print("Action Means")
            print(action_means)
            print("Action Variances")
            print(action_variances)
            print("Observations")
            print(observation)
            exit()

        if return_dist:
            return dist
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if not grad:
            action = action.detach()
            action_logprob = action_logprob.detach()
        return action, action_logprob

    def evaluate(self, observation, action, device):
        dist = self.act(observation, device, return_dist=True)

        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(observation)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def save(self, path, stats=None):
        output = {
            'actor': self.actor,
            'critic': self.critic
        }
        if stats is not None:
            if 'time_steps' in stats:
                output['time_steps'] = stats['time_steps']
            if 'update_steps' in stats:
                output['update_steps'] = stats['update_steps']
            if 'curriculum_level' in stats:
                output['curriculum_level'] = stats['curriculum_level']
            if 'epoch' in stats:
                output['epoch'] = stats['epoch']
        torch.save(output, path)
