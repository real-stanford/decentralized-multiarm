from torch.nn.utils.rnn import pad_sequence
from nets import StochasticActor
from torch import load


class InferencePolicy:
    def __init__(self,
                 reparametrize=True,
                 deterministic=True,
                 device='cpu'):
        self.device = device
        self.net = StochasticActor(
            obs_dim=107,
            action_dim=6,
            action_variance_bounds={
                "max": 1e-1,
                "min": 1e-8,
                "log": True
            },
            network_config={
                "modules": [
                    "lstm",
                    "mlp"
                ],
                "mlp_layers": [
                    128,
                    64
                ],
                "lstm_hidden_size": 256,
                "lstm_num_layers": 1,
                "lstm_bidirectional": False,
                "initial_hidden_state": "zero"
            }).to(self.device)
        load_path = 'checkpoint.pth'
        checkpoint = load(load_path, map_location=self.device)
        networks = checkpoint['networks']
        self.net.load_state_dict(networks['policy'])

        self.reparametrize = reparametrize
        self.deterministic = deterministic

    def act(self, obs):
        obs = pad_sequence(obs, batch_first=True).to(self.device)
        if len(obs) == 1:
            obs = obs[0].unsqueeze(dim=0)
        actions, _ = self.net(
            obs, deterministic=self.deterministic,
            reparametrize=self.reparametrize)
        return actions.detach().cpu()
