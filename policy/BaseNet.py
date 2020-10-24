import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal
from math import log


class BaseNet(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 network_config,
                 activation_func=nn.Tanh):
        super(BaseNet, self).__init__()
        self.activation_func = activation_func
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.network_config = network_config
        self.sequence_input = False

        if 'lstm' in self.network_config['modules'] or \
                'rnn' in self.network_config['modules']:
            self.hidden_size = self.network_config['lstm_hidden_size']
            self.num_layers = self.network_config['lstm_num_layers']
            self.bidirectional = self.network_config['lstm_bidirectional']
            self.seq_net_class = nn.LSTM\
                if 'lstm' in self.network_config['modules']\
                else nn.RNN
            self.seq_net = self.seq_net_class(
                input_size=self.input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                batch_first=True)
            self.init_hidden_fn = None
            if self.network_config['initial_hidden_state'] == 'random':
                def init_random_hidden_fn(batch_size, device):
                    return torch.randn(
                        self.num_layers,
                        batch_size,
                        self.hidden_size).to(device)
                if self.seq_net_class == nn.LSTM:
                    self.init_hidden_fn = lambda batch_size, device:\
                        (init_random_hidden_fn(batch_size, device),
                         init_random_hidden_fn(batch_size, device))
                else:
                    self.init_hidden_fn = lambda batch_size, device:\
                        init_random_hidden_fn(batch_size, device)
            elif self.network_config['initial_hidden_state'] == 'zero':
                def init_zero_hidden_fn(batch_size, device):
                    return torch.zeros(
                        self.num_layers,
                        batch_size,
                        self.hidden_size).to(device)
                if self.seq_net_class == nn.LSTM:
                    self.init_hidden_fn = lambda batch_size, device:\
                        (init_zero_hidden_fn(batch_size, device),
                         init_zero_hidden_fn(batch_size, device))
                else:
                    self.init_hidden_fn = lambda batch_size, device:\
                        init_zero_hidden_fn(batch_size, device)
            elif self.network_config['initial_hidden_state'] == 'parameter':
                self.init_h = nn.Parameter(
                    torch.randn(
                        self.num_layers,
                        1,
                        self.hidden_size).type(torch.FloatTensor),
                    requires_grad=True)
                if self.seq_net_class == nn.LSTM:
                    self.init_c = nn.Parameter(
                        torch.randn(
                            self.num_layers,
                            1,
                            self.hidden_size).type(torch.FloatTensor),
                        requires_grad=True)
                    self.init_hidden_fn = lambda batch_size, device:\
                        (torch.cat(batch_size * [self.init_h], dim=1),
                         torch.cat(batch_size * [self.init_c], dim=1))
                else:
                    self.init_hidden_fn = lambda batch_size, device:\
                        torch.cat(batch_size * [self.init_h], dim=1)
            self.sequence_input = True
        self.net = nn.Sequential(*self.get_layers())

    def get_mlp_input_dim(self):
        if self.sequence_input:
            return self.hidden_size
        else:
            return self.input_dim

    def get_layers(self):
        mlp_input_dim = self.get_mlp_input_dim()
        if len(self.network_config['mlp_layers']) == 0:
            return [nn.Linear(mlp_input_dim, self.output_dim)]
        else:
            layers = [
                nn.Linear(
                    mlp_input_dim,
                    self.network_config['mlp_layers'][0]),
                self.activation_func()
            ]
            for i in range(len(self.network_config['mlp_layers']) - 1):
                layers.append(
                    nn.Linear(
                        self.network_config['mlp_layers'][i],
                        self.network_config['mlp_layers'][i + 1]))
                layers.append(self.activation_func())
            layers.append(
                nn.Linear(
                    self.network_config['mlp_layers'][-1],
                    self.output_dim))
        return layers

    def process_sequence(self, input):
        self.seq_net.flatten_parameters()
        _, h_t = self.seq_net(
            input,
            self.init_hidden_fn(
                batch_size=input.size(0),
                device=input.device))
        if self.seq_net_class == nn.LSTM:
            (h_t, _) = h_t
        return torch.squeeze(h_t, dim=0)

    def forward(self, input):
        if self.sequence_input:
            input = self.process_sequence(input)
        return self.net(input)


class StochasticActor(BaseNet):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 action_variance_bounds,
                 network_config):
        self.action_dim = action_dim
        self.max_action_var = action_variance_bounds['max']
        self.min_action_var = action_variance_bounds['min']
        self.log_action_var = action_variance_bounds['log']
        if self.log_action_var:
            self.max_action_var = log(self.max_action_var)
            self.min_action_var = log(self.min_action_var)

        super(StochasticActor, self).__init__(
            input_dim=obs_dim,
            output_dim=action_dim * 2,
            network_config=network_config)

    def get_layers(self):
        layers = super().get_layers()
        layers.append(nn.Tanh())
        return layers

    def get_action_mean_and_variance(self, obs):
        output = super().forward(obs)
        action_means, action_variances = torch.split(
            output, int(self.action_dim), dim=1)
        action_variances = (self.max_action_var - self.min_action_var) * \
            (action_variances + 1.0) / 2.0 + self.min_action_var
        if self.log_action_var:
            action_variances = action_variances.exp()
        return action_means, action_variances

    def generate_dist(self, means, variances):
        variances = torch.stack(
            [torch.diag(variance)
             for variance in variances])
        try:
            dist = MultivariateNormal(means, variances)
            return dist
        except Exception as e:
            print(e)
            print("mean:\n{}\nvariances:\n{}".format(
                means, variances))

    def forward(self, obs, deterministic, reparametrize, return_dist=False):
        actions = None
        action_logprobs = None
        action_means, action_variances = self.get_action_mean_and_variance(obs)
        if deterministic:
            actions = action_means
        else:
            dist = self.generate_dist(action_means, action_variances)
            if return_dist:
                return dist
            if reparametrize:
                actions = dist.rsample()
            else:
                actions = dist.sample()
            action_logprobs = dist.log_prob(actions)

        return actions, action_logprobs


class Critic(BaseNet):
    def __init__(self,
                 obs_dim,
                 network_config):
        self.network_config = network_config
        super(Critic, self).__init__(
            input_dim=obs_dim,
            output_dim=1)


class Q(BaseNet):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 network_config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        super(Q, self).__init__(
            input_dim=obs_dim,
            output_dim=1,
            network_config=network_config)

    def get_mlp_input_dim(self):
        if self.sequence_input:
            return self.hidden_size + self.action_dim
        return self.action_dim + self.obs_dim

    def forward(self, obs, actions):
        if self.sequence_input:
            obs = self.process_sequence(obs)
            obs = torch.squeeze(obs, dim=1)
        return self.net(torch.cat((obs, actions), 1))
