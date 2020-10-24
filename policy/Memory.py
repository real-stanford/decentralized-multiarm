import ray


class Memory:
    log = False
    base_keys = [
        'observations',
        'critic_observations',
        'actions',
        'rewards',
        'is_terminal',
        'next_observations',
        'critic_next_observations',
    ]

    def __init__(self, memory_fields=[]):
        self.data = {}
        for key in Memory.base_keys:
            self.data[key] = []
        for memory_field in memory_fields:
            self.data[memory_field] = []
        self.to_delete = False

    @staticmethod
    def concat(memories):
        output = Memory()
        for memory in memories:
            for key in memory.data:
                if key not in output.data:
                    output.data[key] = []
                output.data[key].extend(memory.data[key])
        return output

    def clear(self):
        for key in self.data:
            del self.data[key][:]

    def print_length(self):
        output = "[Memory] "
        for key in self.data:
            output += f" {key}: {len(self.data[key])} |"
        print(output)

    def assert_length(self):
        key_lens = [len(self.data[key]) for key in self.data]

        same_length = key_lens.count(key_lens[0]) == len(key_lens)
        if not same_length:
            self.print_length()

    def add_rewards_and_termination(self, reward, termination):
        assert len(self.data['rewards']) \
            == len(self.data['is_terminal'])\
            == len(self.data['actions']) - 1\
            == len(self.data['observations']) - 1
        self.data['rewards'].append(reward)
        self.data['is_terminal'].append(float(termination))

    def add_observation(self, observation):
        assert len(self.data['rewards']) \
            == len(self.data['is_terminal'])\
            == len(self.data['actions'])\
            == len(self.data['observations'])
        self.data['observations'].append(observation)

    def add_action(self, action):
        assert len(self.data['rewards']) \
            == len(self.data['is_terminal'])\
            == len(self.data['actions'])\
            == len(self.data['observations']) - 1
        self.data['actions'].append(action)

    def add_value(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def keys(self):
        return [key for key in self.data]

    def count(self):
        return len(self.data['observations'])

    def done(self):
        if len(self.data['is_terminal']) == 0:
            return False
        return self.data['is_terminal'][-1]

    def get_data(self):
        return self.data

    def mark_for_delete(self):
        self.to_delete = True

    def should_delete(self):
        return self.to_delete


@ray.remote
class MemoryCluster:
    def __init__(self, memory_fields=[], init_time_steps=0):
        self.nodes = []
        self.memory_fields = memory_fields
        self.total_timesteps = init_time_steps
        self.clear()

    def get_total_timesteps(self):
        return self.total_timesteps

    def get_memory_fields(self):
        return self.memory_fields

    def submit_memory(self, memory):
        if self.to_clear:
            self.clear()
        if memory.to_delete or memory.count() == 0:
            return
        self.total_timesteps += len(memory.data['observations'])
        for key in memory.data:
            if key not in self.data:
                self.data[key] = []
            self.data[key].extend(memory.data[key])

    def get_data(self):
        if len(self.data['observations']) == 0:
            return None
        self.to_clear = True
        return self.data

    def clear(self):
        self.data = {}
        for key in Memory.base_keys:
            self.data[key] = []
        self.to_clear = False
