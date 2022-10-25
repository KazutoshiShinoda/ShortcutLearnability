import json
import copy


def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)


class Statistics(object):
    def __init__(self):
        self.global_update = 0
        self.previous_update = 0
        self.global_stat = {}
        self.previous_stat = {}

    def update_dict(self, stat_dict):
        for key, value in stat_dict.items():
            self.global_stat[key] = self.global_stat.get(key, 0) + value
        self.global_update += 1

    def mean(self):
        if self.global_update == 0:
            return {}
        mean_stat = {}
        for key, value in self.global_stat.items():
            mean_stat[key] = value / self.global_update
        return mean_stat

    def local_mean(self):
        if self.global_update - self.previous_update == 0:
            return {}
        mean_stat = {}
        for key, value in self.global_stat.items():
            mean_stat[key] = (value - self.previous_stat.get(key, 0)) / \
                (self.global_update - self.previous_update)

        self.previous_update = self.global_update
        self.previous_stat = copy.deepcopy(self.global_stat)

        return mean_stat

    def reset(self):
        self.global_update = 0
        self.previous_update = 0
        self.global_stat = {}
        self.previous_stat = {}
