import os
import ray
from json import load, dump
from os.path import dirname
import argparse
import numpy as np
from environment import (
    BaseEnv,
    BenchmarkEnv,
    RRTWrapper,
    RRTSupervisionEnv,
    TaskLoader
)
from environment.utils import get_observation_dimensions
import torch
from tensorboardX import SummaryWriter
from policy import (
    PolicyManager,
    StochasticActor,
    Q,
    SACLearner,
    setup_behaviour_clone
)
from copy import deepcopy
from torch.utils.data import Dataset
from pathlib import Path
from os.path import abspath
from os.path import exists
from os import mkdir
import pickle


def merge(a, b):
    if isinstance(b, dict) and isinstance(a, dict):
        a_and_b = a.keys() & b.keys()
        every_key = a.keys() | b.keys()
        return {k: merge(a[k], b[k]) if k in a_and_b else
                deepcopy(a[k] if k in a else b[k]) for k in every_key}
    return deepcopy(b)


def exit_handler(exit_handlers):
    print("Gracefully terminating")
    if exit_handlers is not None:
        for exit_handler in exit_handlers:
            if exit_handler is not None:
                exit_handler()
    ray.shutdown()
    exit(0)


def load_config(path, merge_with_default=True):
    base_dirname = os.path.basename(os.path.dirname(path))
    merge_with_default = (base_dirname == 'configs')

    if os.path.splitext(os.path.basename(path))[0] != 'default'\
            and merge_with_default:
        config = load(open(dirname(path) + '/default.json'))
        additional_config = load(open(path))
        config = merge(config, additional_config)
    else:
        config = load(open(path))
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        "Collaborative Multi Arm")
    parser.add_argument("--name", type=str,
                        help="name of run", default=None)
    parser.add_argument("--config", type=str,
                        help="path of config json", default=None)
    parser.add_argument("--load", type=str, default=None,
                        help="path of policy to load")
    parser.add_argument("--tasks_path", type=str, default=None,
                        help="path of directory containing tasks")
    parser.add_argument('--gui', action='store_true',
                        default=False, help='Run headless or render')
    parser.add_argument('--num_processes', type=int,
                        default=16, help='How many processes to parallelize')
    parser.add_argument('--curriculum_level', type=int,
                        default=0,
                        help='Which level of the curriculum to start training')
    parser.add_argument('--mode',
                        choices=[
                            # 1. Train behaviour clone on expert trajectories
                            'behaviour-clone',
                            # 2. Run RL
                            'train',
                            # 3. View policy behaviour without training
                            'enjoy',
                            # 4. Load pretrained policy
                            'benchmark',
                        ],
                        default='train')
    parser.add_argument('--expert_waypoints', type=str,
                        default=None,
                        help='path to expert waypoints directory' +
                        'for trajectory generation')
    parser.add_argument('--expert_trajectories', type=str,
                        default=None,
                        help='path to expert trajectories directory' +
                        'for behaviour cloning')

    args = parser.parse_args()

    def require_tasks():
        if args.tasks_path is None:
            print("Please supply tasks with --tasks_path")
            exit()

    def require_name():
        if args.name is None:
            print("Please supply experiment name with --name")
            exit()

    def require_config():
        if args.config is None:
            print("Please supply path to config file with --config")
            exit()
    if args.mode == 'benchmark' or args.mode == 'enjoy':
        if args.load is None:
            print("load a policy to benchmark")
            parser.print_help()
            exit()
        if args.config is None:
            args.config = "{}/config.json".format(dirname(args.load))

    if args.mode == 'behaviour-clone':
        if args.expert_trajectories is None:
            parser.print_help()
            exit()
        if args.config is None:
            args.config = "{}/config.json".format(
                dirname(args.expert_trajectories))

    if args.mode == 'enjoy':
        args.gui = True
        args.num_processes = 1

    if (args.mode == 'train' or
            args.mode == 'expert' or
            args.mode == 'tasks'):
        require_name()
        if args.config is None:
            args.config = "{}/config.json".format(dirname(args.load))
        require_config()

    if args.mode == 'expert':
        require_tasks()

    if args.mode == 'trajectories' and (
        args.config is None or args.name is None
        or args.expert_waypoints is None
        or args.tasks_path is None
    ):
        parser.print_help()
        exit()
    return args


def get_device():
    if not torch.cuda.is_available():
        return 'cpu'
    return 'cuda'


def create_policies(args,
                    training_config,
                    action_dim,
                    actor_obs_dim,
                    critic_obs_dim,
                    training,
                    logger,
                    device=None):
    if device is None:
        device = get_device()
    hyperparams = training_config['hyperparameters']

    def create_policy():
        return StochasticActor(
            obs_dim=actor_obs_dim,
            action_dim=action_dim,
            action_variance_bounds=training_config['action_variance'],
            network_config=training_config['network']['actor'])

    def create_qf():
        return Q(obs_dim=critic_obs_dim + 6
                 if training_config['centralized_critic']
                 else critic_obs_dim,
                 action_dim=action_dim,
                 network_config=training_config['network']['critic'])

    policy_manager = PolicyManager()
    learner_class = None
    if training_config['algo'] == 'sac':
        learner_class = ray.remote(SACLearner)
    else:
        print(training_config['algo'], ' not supported')
        exit()

    multiarm_motion_planner_learner = None
    num_gpus = 0 if device == 'cpu' else 1
    network_fns = {
        'policy': create_policy,
        'Q1': create_qf,
        'Q2': create_qf,
    }
    multiarm_motion_planner_learner = learner_class.options(
        num_gpus=num_gpus).remote(
        policy_key="multiarm_motion_planner",
        network_fns=network_fns,
        algo_config=hyperparams,
        writer=logger,
        device=device,
        load_path=args.load,
        training=training)
    policy_manager.register_policy(
        "multiarm_motion_planner", multiarm_motion_planner_learner)
    return policy_manager


def load_json(path):
    try:
        item = load(open(path, 'r'))
    except Exception as e:
        print(e)
        print(path)
    item['observations'] = [torch.tensor(
        obs) for obs in item['observations']]
    item['actions'] = [torch.tensor(
        action) for action in item['actions']]
    return item


class RRTTrajectoryDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [abspath(file_name)
                      for file_name in Path(self.root_dir).rglob('*.json')
                      if 'config' not in str(file_name)]
        print('[RRTTrajectoryDataset] found {} files'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = load_json(self.files[idx])
        item['path'] = self.files[idx]
        return item


@ray.remote
class Logger:
    def __init__(self,
                 logdir,
                 curriculum_level=0,
                 graduation_threshold=0.9,
                 benchmark_mode=False,
                 benchmark_name=None):
        self.benchmark_mode = benchmark_mode
        self.benchmark_name = benchmark_name
        if self.benchmark_mode:
            self.logdir = logdir
            self.benchmark_scores = []
        else:
            self.writer = SummaryWriter(logdir)
            self.stats_history = {}
            self.success_rate_history = []
            self.curriculum_level = curriculum_level
            self.graduation_threshold = graduation_threshold

    def add_scalar(self, key, val, timestamp):
        if not self.benchmark_mode:
            self.writer.add_scalar(key, val, timestamp)

    def add_scalars(self, scalars_dict, timestamp, flush_stats=True):
        if self.benchmark_mode:
            return
        for key in scalars_dict:
            self.writer.add_scalar(key, scalars_dict[key], timestamp)
        for key in self.stats_history:
            if len(self.stats_history[key]) == 0:
                continue
            print(key, ":", np.mean(self.stats_history[key]))
            self.writer.add_scalar(
                key + "/mean",
                np.mean(self.stats_history[key]), timestamp)
            self.writer.add_scalar(
                key + "/std",
                np.std(self.stats_history[key]), timestamp)
        self.stats_history = {}

    def get_logdir(self):
        if self.benchmark_mode:
            return None
        return self.writer.logdir

    def add_stats(self, stats):
        if self.benchmark_mode:
            self.benchmark_scores.append(stats)
            if len(self.benchmark_scores) % 50 == 0:
                print(len(self.benchmark_scores))
                self.save()
                self.print_summary()
            return
        for key in stats:
            if key not in self.stats_history:
                self.stats_history[key] = []
            self.stats_history[key].append(stats[key])
            if key == 'success':
                self.success_rate_history.append(stats[key])
                if len(self.success_rate_history) > 100:
                    self.success_rate_history.pop(0)
        if self.get_average_success_rate() > self.graduation_threshold:
            self.curriculum_level += 1
            self.success_rate_history = []
        return self.curriculum_level

    def get_average_success_rate(self):
        if len(self.success_rate_history) < 50:
            return 0.0
        return np.mean(self.success_rate_history)

    def get_curriculum_level(self):
        if self.benchmark_mode:
            return 0
        return self.curriculum_level

    def atexit(self):
        if self.benchmark_mode:
            self.save()

    def save(self):
        benchmark_name = self.benchmark_name\
            if self.benchmark_name is not None\
            else 'benchmark_score'
        output_path = self.logdir + '/{}.pkl'.format(benchmark_name)
        print('[Logger] Saving benchmark scores to ',
              output_path)
        pickle.dump(
            self.benchmark_scores,
            open(output_path, 'wb'))

    def print_summary(self):
        for key in self.benchmark_scores[0].keys():
            if key == 'task' or key == 'debug':
                continue
            scores = [score[key] for score in self.benchmark_scores]
            print(key)
            print('\tmean:', np.mean(scores))
            print('\tstd:', np.std(scores))


def mkdir_and_save_config(args, dir_path, config):
    if not exists(dir_path):
        mkdir(dir_path)
    config_path = dir_path + '/config.json'
    if 'targets_provider' in config['environment']:
        del config['environment']['targets_provider']
    if exists(config_path):
        # Make sure the configs are the same
        prev_config = load(open(config_path))
        if prev_config != config:
            print("Different config files!")
            exit()
    else:
        # Save the config
        print("No config file yet. Saving {} to {}".format(
            args.config,
            config_path
        ))
        dump(config, open(config_path, 'w'), indent=4)
    return abspath(dir_path)


def initialize_ray():
    ray.init(
        # log_to_driver=False,
        # local_mode=False,
        # logging_level=ray.logging.ERROR
    )


def compute_expert_setup(args, config):
    initialize_ray()
    return ([RRTWrapper.remote(
        env_config=config['environment'],
        gui=args.gui)
        for _ in range(args.num_processes)],
        TaskLoader(root_dir=args.tasks_path,
                   shuffle=True,
                   repeat=False))


def prepare_logger(args, config):
    logger = None
    if args.mode == 'train':
        if args.name is None:
            print('[Setup] Set experiment name using --name.' +
                  ' Otherwise, disable training with --enjoy')
            exit(0)
        logdir = "runs/" + args.name
        if not exists(logdir):
            mkdir(logdir)
        logger = Logger.remote(
            logdir=logdir,
            graduation_threshold=config[
                'environment']['curriculum']['graduation_threshold'],
            curriculum_level=args.curriculum_level)
        dump(config, open(logdir + '/config.json', 'w'), indent=4)
    elif args.mode == 'benchmark':
        logger = Logger.remote(
            logdir=dirname(args.load),
            benchmark_mode=True,
            benchmark_name=args.name
        )
    return logger


def setup(args, config):
    if args.mode == 'behaviour-clone':
        return setup_behaviour_clone(
            args, config,
            obs_dim=get_observation_dimensions(
                config['training']['observations']),
            device=get_device())
    env_config = config['environment']
    training_config = config['training']

    Env = None
    if args.mode == 'enjoy':
        Env = BaseEnv
    elif args.mode == 'benchmark':
        Env = BenchmarkEnv
    elif env_config['expert']['expert_on_fail']:
        Env = RRTSupervisionEnv
        env_config['expert_root_dir'] = args.expert_waypoints
    else:
        Env = BaseEnv

    num_processes = config['training']['num_processes']
    if num_processes < 0:
        num_processes = os.sysconf('SC_NPROCESSORS_ONLN')
    if args.num_processes is not None:
        num_processes = args.num_processes
    print("[Setup] Parallelizing across {} processes".format(num_processes))
    initialize_ray()

    logger = prepare_logger(args, config)

    # create targetLoader
    task_loader = None
    if args.tasks_path is not None:
        print("[Setup] Creating task loader...")
        task_loader = ray.remote(TaskLoader).remote(
            root_dir=args.tasks_path,
            repeat=args.mode != 'benchmark',
            shuffle=True)
    else:
        print("[Setup] No tasks loaded. Generating tasks online.")
    config['training']['task_loader'] = task_loader
    print("[Setup] Creating environments...")
    envs = [Env.remote(
            env_config=env_config,
            training_config=config['training'],
            gui=args.gui,
            logger=logger)
            for _ in range(num_processes)]
    ray.get([e.setup_ray.remote(e) for e in envs])

    del config['training']['task_loader']

    if args.mode == 'tasks':
        for e in envs:
            ray.get(
                e.set_memory_cluster_map.remote({
                    'multiarm_motion_planner': None
                }))
        return envs

    if 'centralized_policy' not in training_config:
        training_config['centralized_policy'] = False

    obs_dim = get_observation_dimensions(training_config['observations'])
    action_dim = 6
    if training_config['centralized_policy']:
        obs_dim *= env_config['max_ur5s_count']
        action_dim *= env_config['max_ur5s_count']
    policy_manager = create_policies(
        args=args,
        training_config=config['training'],
        action_dim=action_dim,
        actor_obs_dim=obs_dim,
        critic_obs_dim=obs_dim,
        training=args.mode == 'train',
        logger=logger)
    for e in envs:
        ray.get(e.set_memory_cluster_map.remote(
            policy_manager.memory_map))

    keep_alive = {
        'task_loader': {
            'obj': task_loader,
            'exit_handler': None
        },
        'logger': {
            'obj': logger,
            'exit_handler': lambda:
            ray.get(logger.atexit.remote())
        }
    }
    return envs, policy_manager, keep_alive
