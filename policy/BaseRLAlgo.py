import torch
import ray


class BaseRLAlgo:
    def __init__(
            self,
            policy_key,
            writer,
            training=True,
            load_path=None,
            device='cuda'):
        self.writer = writer
        self.policy_key = policy_key
        self.training = training
        if self.writer is not None:
            self.logdir = ray.get(self.writer.get_logdir.remote())
        else:
            self.training = False
        self.device = device
        self.load_path = load_path
        self.stats = {}
        self.memory_cluster = None
        self.ray_id = None
        self.async_tasks = {}

    def should_update(self):
        raise NotImplementedError

    """
    Statistics + Performance tracking
    """

    def get_stats_list(self):
        return ['rewards']

    """
    Training Utils
    """

    def get_memory_cluster(self):
        return {
            'handle': self.memory_cluster
        }

    def set_handle(self, ray_id):
        self.ray_id = ray_id['handle']

    def get_state_dicts_to_save(self):
        return {}

    def get_stats_to_save(self):
        return self.stats

    def save(self):
        if self.training and self.logdir is not None:
            output_path = "{}/ckpt_{}_{:05d}".format(
                self.logdir,
                self.policy_key,
                int(self.stats['update_steps'] / self.save_interval))
            torch.save({
                'networks': self.get_state_dicts_to_save(),
                'stats': self.get_stats_to_save()
            }, output_path)
            print("[BaseRlAlgo] saved checkpoint at {}".format(output_path))

    def register_async_task(self, name, fn, callback):
        self.async_tasks[name] = {
            'value': None,
            'fn': fn,
            'callback': callback
        }
        self.async_tasks[name]['value'] = self.async_tasks[name]['fn']()
