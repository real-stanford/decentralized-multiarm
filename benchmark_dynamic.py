import ray
from environment.rrt import RRTWrapper
from environment import utils
from environment import RealTimeEnv
from utils import (
    parse_args,
    load_config,
    create_policies,
    exit_handler
)
from environment import TaskLoader
import pickle
from signal import signal, SIGINT
from numpy import mean
from distribute import Pool
from os.path import exists
from tqdm import tqdm


if __name__ == "__main__":
    args = parse_args()
    args.gui = True
    config = load_config(args.config)
    env_conf = config['environment']
    training_conf = config['training']
    env_conf['min_ur5s_count'] = 1
    env_conf['max_ur5s_count'] = 10
    env_conf['task']['type'] = 'dynamic'
    ray.init()
    signal(SIGINT, lambda sig, frame: exit())
    output_path = 'rrt_dynamic_benchmark_score.pkl'
    if args.load:
        output_path = 'policy_dynamic_benchmark_score.pkl'
    benchmark_results = []

    continue_benchmark = False

    if exists(output_path):
        # continue benchmark
        benchmark_results = pickle.load(open(output_path, 'rb'))
        continue_benchmark = True
        finished_task_paths = [r['task']['task_path']
                               for r in benchmark_results]

    task_loader = TaskLoader(
        root_dir=args.tasks_path,
        shuffle=True,
        repeat=False)
    training_conf['task_loader'] = task_loader
    # set up policy if loaded
    if args.load:
        obs_dim = utils.get_observation_dimensions(
            training_conf['observations'])
        action_dim = 6
        policy_manager = create_policies(
            args=args,
            training_config=config['training'],
            action_dim=action_dim,
            actor_obs_dim=obs_dim,
            critic_obs_dim=obs_dim,
            training=args.mode == 'train',
            logger=None)
        policy = policy_manager.get_inference_nodes()[
            'multiarm_motion_planner']

        policy.policy.to('cpu')
        training_conf['policy'] = policy
        env = RealTimeEnv(
            env_config=env_conf,
            training_config=training_conf,
            gui=args.gui,
            logger=None)
        env.set_memory_cluster_map(policy_manager.memory_map)
    else:
        RealTimeEnv = ray.remote(RealTimeEnv)
        envs = [RealTimeEnv.remote(
            env_config=env_conf,
            training_config=training_conf,
            gui=args.gui,
            logger=None)
            for _ in range(args.num_processes)]
        env_pool = Pool(envs)

    def callback(result):
        benchmark_results.append(result)
        if len(benchmark_results) % 100 == 0\
                and len(benchmark_results) > 0:

            print('Saving benchmark scores to ',
                  output_path)
            with open(output_path, 'wb') as f:
                pickle.dump(benchmark_results, f)

    def pbar_update(pbar):
        pbar.set_description(
            'Average Success Rate : {:.04f}'.format(
                mean([r['success_rate']
                      for r in benchmark_results])))
    tasks = [t for t in task_loader
             if not continue_benchmark
             or t.task_path not in finished_task_paths]
    if args.load:
        with tqdm(tasks, dynamic_ncols=True, smoothing=0.01) as pbar:
            for task in pbar:
                callback(env.solve_task(task))
                pbar_update(pbar)
    else:
        benchmark_results = env_pool.map(
            exec_fn=lambda env, task: env.solve_task.remote(task),
            iterable=tasks,
            pbar_update=pbar_update,
            callback_fn=callback
        )
