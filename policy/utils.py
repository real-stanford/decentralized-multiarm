from torch.utils.data import Dataset
from torch import tensor, cat, stack, FloatTensor, Tensor, save
import ray
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from .BaseNet import StochasticActor
from torch.optim import Adam
from torch.nn.functional import mse_loss
from time import time
from tensorboardX import SummaryWriter
from os import mkdir
from os.path import exists, abspath


class PolicyManager:
    def __init__(self):
        self.policies = {}
        self.memory_map = {}

    def register_policy(self, name, entry):
        self.policies[name] = entry
        self.memory_map[name] = ray.get(
            entry.get_memory_cluster.remote())['handle']
        ray.get(entry.set_handle.remote({'handle': entry}))

    def get_inference_node(self, key):
        return ray.get(self.policies[
            key].get_inference_node.remote())

    def get_inference_nodes(self):
        return dict([(p,
                      self.get_inference_node(p))
                     for p in self.policies])


class ReplayBufferDataset(Dataset):
    def __init__(self, data, device, capacity):
        self.keys = list(data)
        self.data = data
        self.device = device
        self.capacity = int(capacity)
        self.observations_padded = None
        self.next_observations_padded = None
        self.freshness = 0.0

    def __len__(self):
        return len(self.data['observations'])

    def extend(self, data):
        len_new_data = len(data['observations'])
        for key in self.data:
            self.data[key].extend(data[key])
        if len(self) != 0:
            self.freshness += len_new_data / len(self)
        else:
            self.freshness = 0.0

        if len(self.data['observations']) > self.capacity:
            amount = len(self.data['observations']) - self.capacity
            for key in self.data:
                del self.data[key][:amount]

        self.padded_observations = None
        self.next_observations_padded = None

    def pad_observations(self, input):
        return pad_sequence(input, batch_first=True)

    def __getitem__(self, idx):
        if self.observations_padded is None or\
                self.next_observations_padded is None:
            self.observations_padded = self.pad_observations(
                self.data['observations'])
            self.next_observations_padded = self.pad_observations(
                self.data['next_observations'])
            if 'critic_observations' in self.data\
                    and len(self.data['critic_observations']) > 0:
                self.critic_observations_padded = self.pad_observations(
                    self.data['critic_observations'])
                self.critic_next_observations_padded = self.pad_observations(
                    self.data['critic_next_observations'])
            else:
                self.critic_observations_padded = \
                    self.observations_padded
                self.critic_next_observations_padded =\
                    self.next_observations_padded
        if 'critic_observations' in self.data:
            return (
                self.critic_observations_padded[idx].to(
                    self.device, non_blocking=True),
                self.observations_padded[idx].to(
                    self.device, non_blocking=True),
                self.data['actions'][idx].to(
                    self.device, non_blocking=True),
                tensor(float(self.data['rewards'][idx])).to(
                    self.device, non_blocking=True),
                self.critic_next_observations_padded[idx].to(
                    self.device, non_blocking=True),
                self.next_observations_padded[idx].to(
                    self.device, non_blocking=True),
                tensor(float(self.data['is_terminal'][idx])).to(
                    self.device, non_blocking=True)
            )
        else:
            return (
                self.observations_padded[idx].to(
                    self.device, non_blocking=True),
                self.data['actions'][idx].to(
                    self.device, non_blocking=True),
                tensor(float(self.data['rewards'][idx])).to(
                    self.device, non_blocking=True),
                self.next_observations_padded[idx].to(
                    self.device, non_blocking=True),
                tensor(float(self.data['is_terminal'][idx])).to(
                    self.device, non_blocking=True)
            )


class BehaviourCloneDataset(Dataset):
    def __init__(self, path):
        self.observations = []
        self.actions = []
        for file_name in tqdm(
                Path(path).rglob('*.pkl'),
                desc='importing trajectories'):
            with open(file_name, 'rb') as file:
                try:
                    trajectories = pickle.load(file)
                    self.observations.extend(
                        FloatTensor(trajectories['observations']))
                    self.actions.extend(trajectories['actions'])
                except:
                    pass
        self.observations = pad_sequence(
            self.observations,
            batch_first=True)
        self.actions = FloatTensor(self.actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (self.observations[idx], self.actions[idx])


def setup_behaviour_clone(args, config, obs_dim, device):
    dataset = BehaviourCloneDataset(args.expert_trajectories)
    train_loader = DataLoader(
        dataset,
        batch_size=config['behaviour-clone']['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=0)
    policy = StochasticActor(
        obs_dim=obs_dim,
        action_dim=6,
        action_variance_bounds=config['training']['action_variance'],
        network_config=config['training']['network']['actor']).to(device)
    optimizer = Adam(
        policy.parameters(),
        lr=config['behaviour-clone']['lr'],
        weight_decay=config['behaviour-clone']['weight_decay'],
        betas=(0.9, 0.999))
    print(policy)
    print(optimizer)

    def train():
        batch_count = 0
        logdir = "runs/" + args.name
        if not exists(logdir):
            mkdir(logdir)
        writer = SummaryWriter(logdir)
        for epoch in range(config['behaviour-clone']['epochs']):
            pbar = tqdm(train_loader)
            epoch_loss = 0.0
            start = time()
            for batch_idx, (observations, expert_actions) in enumerate(pbar):
                policy_action_dist = policy(
                    observations.to(device, non_blocking=True),
                    deterministic=False,
                    reparametrize=True,
                    return_dist=True)
                loss = - \
                    policy_action_dist.log_prob(
                        expert_actions.to(device, non_blocking=True)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_count += 1
                writer.add_scalar(
                    'Training/Policy_Loss',
                    loss.item(),
                    batch_count)
                epoch_loss += loss.item()
                pbar.set_description(
                    'Epoch {} | Batch {} | Loss: {:.05f}'.format(
                        epoch, batch_idx,
                        loss.item()))
            pbar.set_description('Epoch {} | Loss: {:.05f} | {:.01f} seconds'.format(
                epoch, epoch_loss / (batch_idx + 1),
                float(time() - start)))
            output_path = abspath("{}/ckpt_{:05d}".format(
                writer.logdir,
                epoch))
            save({
                'networks': {
                    'policy': policy.state_dict(),
                    'opt': optimizer.state_dict()
                },
                'stats': {
                    'batches': batch_count,
                    'epochs': epoch
                }
            }, output_path)
            print("Saved checkpoint at " + output_path)
    return train
