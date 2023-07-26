import itertools
import numpy as np
from numpy import random

seed = 0
random.seed(seed)
np.random.seed(seed)


class DataUtils:
    def __init__(self):
        pass

    @staticmethod
    def get_permutations(el_list):
        permutations = list(itertools.permutations(el_list))
        try:
            for idx, permutation in enumerate(permutations):
                permutations[idx] = sum(list(permutation), [])
        except Exception:
            pass

        return permutations

    @staticmethod
    def get_random_permutations(el_list, num):
        permutations = []
        for _ in range(num):
            permutation = random.permutation(el_list)
            permutations.append(list(itertools.chain(*permutation)))

        return permutations

    @staticmethod
    def avg_across_dicts(dicts):
        """

        Args:
            dicts: list of dictionaries with same keys and lists as elements

        Returns:
            single dict with the keys and average of lists

        """
        avg_dict = dict()
        keys = dicts[0].keys()
        for k in keys:
            values = []
            for d in dicts:
                values.append(d[k])
            avg_dict[k] = np.mean(values, axis=0)

        return avg_dict

