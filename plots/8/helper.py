import random
import jellyfish
from matplotlib import pyplot as plt
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)


def sort_dict(unsorted_dict, reverse=True):
    return dict(sorted(unsorted_dict.items(), key=lambda item: item[1], reverse=reverse))


class BinnedStringComparator:
    def __init__(self, fscore_list, n_bins=5):
        self.fscore_list = fscore_list
        self.hamming = jellyfish.hamming_distance
        self.n_bins = n_bins
        self.len_curriculum = len(list(fscore_list[0].keys())[0])
        self.num_curriculums = len(fscore_list[0])

        for idx, fscore in enumerate(self.fscore_list):
            fscore_list[idx] = sort_dict(fscore)
        self.fscore_strings = [self.get_string(
            fscore) for fscore in self.fscore_list]
        if n_bins is not None:
            self.fscore_bins = [self.bin_fscores(
                fscore, n_bins) for fscore in self.fscore_list]

    def get_string(self, fscore):
        fscore_str = ""
        fscore_curriculums = list(fscore.keys())
        len_curriculum = len(fscore_curriculums[0])
        num_curriculums = len(fscore)

        for i in range(len_curriculum):
            for j in range(num_curriculums):
                fscore_str += fscore_curriculums[j][i]

        return fscore_str

    def get_random_string(self, curriculums):
        random_curriculum_list = np.random.permutation(curriculums)
        curriculum_str = ""
        for i in range(self.len_curriculum):
            for j in range(self.num_curriculums):
                curriculum_str += random_curriculum_list[j][i]

        return curriculum_str

    def get_bins(self, fscore):
        _, bins, __ = plt.hist(list(fscore.values()), bins=self.n_bins)

        return bins

    def bin_fscores(self, fscore, n_bins):
        binned_fscore = {i: [] for i in range(n_bins)}
        bins = self.get_bins(fscore)
        for permutation, score in fscore.items():
            for i in range(self.n_bins):
                lo, hi = bins[i], bins[i+1]
                if i == self.n_bins - 1:
                    if lo <= score <= hi:
                        binned_fscore[i].append(permutation)
                        break
                else:
                    if lo <= score < hi:
                        binned_fscore[i].append(permutation)
                        break

        for k, v in binned_fscore.items():
            binned_fscore[k] = len(v)

        return binned_fscore

    def get_consistency(self):
        score_f, score_random = 0.0, 0.0
        alg_list, rand_list = [], []
        if self.n_bins is not None:
            for idx_a, fscore_a in enumerate(self.fscore_list):
                fscore_a_str, fscore_a_bin = self.get_string(
                    fscore_a), self.bin_fscores(fscore_a, n_bins=self.n_bins)
                n_a = fscore_a_bin[4]
                curriculums_a = list(fscore_a.keys())[:n_a]
                for idx_b, fscore_b in enumerate(self.fscore_list):
                    if (idx_a == idx_b):
                        continue
                    fscore_b_str = self.get_string(fscore_b)
                    score = self.hamming(fscore_a_str[:self.len_curriculum*n_a],
                                         fscore_b_str[:self.len_curriculum*n_a])
                    score_f += score
                    alg_list.append(score)
                for _ in range(100):
                    random_str = self.get_random_string(curriculums_a)
                    score = self.hamming(fscore_a_str[:self.len_curriculum*n_a],
                                         random_str)
                    score_random += score
                    rand_list.append(score)
            print("alg: {}".format(alg_list))
            print("random: {}".format(rand_list))
            score_f, score_random = score_f / \
                len(self.fscore_list), score_random / 100
        else:
            for idx_a, fscore_a in enumerate(self.fscore_list):
                fscore_a_str = self.get_string(fscore_a)
                curriculums_a = list(fscore_a.keys())
                for idx_b, fscore_b in enumerate(self.fscore_list):
                    if (idx_a == idx_b):
                        continue
                    fscore_b_str = self.get_string(fscore_b)
                    score = self.hamming(fscore_a_str, fscore_b_str)
                    score_f += score
                    alg_list.append(score)
                for _ in range(100):
                    random_str = self.get_random_string(curriculums_a)
                    score = self.hamming(fscore_a_str, random_str)
                    score_random += score
                    rand_list.append(score)
            print("alg: {}".format(alg_list))
            print("random: {}".format(rand_list))
            score_f, score_random = score_f / \
                len(self.fscore_list), score_random / 100

        return score_f, score_random

    def __call__(self):
        return self.get_consistency()
