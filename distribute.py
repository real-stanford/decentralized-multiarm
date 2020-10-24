import ray
from tqdm import tqdm


class Pool:
    def __init__(self, actors):
        """
        actors: list of ray actor handles
        """
        self.actors = actors
        assert len(self.actors) > 0

    def map(self, exec_fn, iterable,
            callback_fn=None,
            desc=None,
            pbar_update=None,
            use_tqdm: bool = True):
        """
        exec_fn: function to execute actor on each item of iterable
        callback_fn: function to process each result
        """
        arg_it = iter(iterable)
        actor_it = iter(self.actors)
        pending_tasks = []
        results = []

        while True:
            arg = next(arg_it, None)
            if arg is None:
                break
            actor = next(actor_it, None)
            if actor is None:
                actor_it = iter(self.actors)
                actor = next(actor_it, None)
            pending_tasks.append(exec_fn(actor, arg))
        if use_tqdm:
            pbar = tqdm(total=len(pending_tasks), desc=desc,
                        dynamic_ncols=True, smoothing=0.01)
        while len(pending_tasks) > 0:
            finished_tasks, pending_tasks = ray.wait(pending_tasks)
            for finished_task in finished_tasks:
                results.append(ray.get(finished_task))
                if callback_fn is not None:
                    callback_fn(results[-1])
                if use_tqdm:
                    pbar.update()
                    if pbar_update is not None:
                        pbar_update(pbar)
        return results
