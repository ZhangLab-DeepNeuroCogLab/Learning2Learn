import random
import numpy as np
import pandas as pd
import jellyfish
import matplotlib.pyplot as plt
import seaborn as sns

seed = 0
random.seed(seed)
np.random.seed(seed)


class StringComparator:
    """
    Note:
        works for ewc and naive only, does not extend to other strategies
    """
    def __init__(self, fscore_a, fscore_b, ascore, save_loc):
        self.fscore_a = StringComparator.sort_dict(fscore_a)
        self.fscore_b = StringComparator.sort_dict(fscore_b)
        self.ascore = StringComparator.sort_dict(ascore)
        self.save_loc = save_loc
        self.len_curriculum = len(list(fscore_a.keys())[0])
        self.num_curriculums = len(fscore_a)

        self.fscore_a_str = ""
        self.fscore_b_str = ""
        self.ascore_str = ""
        self.random_strs = []

    @staticmethod
    def sort_dict(unsorted_dict):
        return dict(sorted(unsorted_dict.items(), key=lambda item: item[1], reverse=True))

    def get_strings(self):
        fscore_a_curriculums, fscore_b_curriculums, ascore_curriculums = (
            list(self.fscore_a.keys()), list(self.fscore_b.keys()),
            list(self.ascore.keys())
        )
        for i in range(self.len_curriculum):
            for j in range(self.num_curriculums):
                self.fscore_a_str += fscore_a_curriculums[j][i]
                self.fscore_b_str += fscore_b_curriculums[j][i]
                self.ascore_str += ascore_curriculums[j][i]

        for _ in range(1000):
            random_curriculum_list = np.random.permutation(
                ascore_curriculums
            )
            curriculum_str = ""
            for i in range(self.len_curriculum):
                for j in range(self.num_curriculums):
                    curriculum_str += random_curriculum_list[j][i]
            self.random_strs.append(curriculum_str)

    def plot_distances(self):
        levenshtein = jellyfish.levenshtein_distance
        damerau_levenshtein = jellyfish.damerau_levenshtein_distance
        hamming = jellyfish.hamming_distance

        x = [
            "levenshtein",
            "damerau-levenshtein",
            "hamming"
        ]

        # a vs b
        y = [
            levenshtein(self.fscore_a_str, self.fscore_b_str),
            damerau_levenshtein(self.fscore_a_str, self.fscore_b_str),
            hamming(self.fscore_a_str, self.fscore_b_str),
        ]
        plt.plot(x, y, 'g^-', linewidth=2, label='naive vs ewc')

        # a vs a-score
        y = [
            levenshtein(self.fscore_a_str, self.ascore_str),
            damerau_levenshtein(self.fscore_a_str, self.ascore_str),
            hamming(self.fscore_a_str, self.ascore_str)
        ]
        plt.plot(x, y, 'bv--', linewidth=2, label='naive vs a-score')

        # b vs a-score
        y = [
            levenshtein(self.fscore_b_str, self.ascore_str),
            damerau_levenshtein(self.fscore_b_str, self.ascore_str),
            hamming(self.fscore_b_str, self.ascore_str),
        ]
        plt.plot(x, y, 'ys:', linewidth=2, label='ewc vs a-score')

        # a vs random
        y_ar, y_br = [], []
        for random_str in self.random_strs:
            y_ar.append(
                [
                    levenshtein(self.fscore_a_str, random_str),
                    damerau_levenshtein(self.fscore_a_str, random_str),
                    hamming(self.fscore_a_str, random_str),
                ]
            )
            y_br.append(
                [
                    levenshtein(self.fscore_b_str, random_str),
                    damerau_levenshtein(self.fscore_b_str, random_str),
                    hamming(self.fscore_b_str, random_str),
                ]
            )
        y_ar, y_br = np.mean(y_ar, axis=0), np.mean(y_br, axis=0)
        plt.plot(x, y_ar, 'r*--', linewidth=2, label='naive vs random')
        plt.plot(x, y_br, 'mo-.', linewidth=2, label='ewc vs random')

        plt.ylabel('distance')
        plt.legend(loc='upper left', fontsize='x-small')
        plt.title('string distance overlap')
        plt.savefig('string_plots/{}.png'.format(self.save_loc), dpi=100)
        plt.close()

    def __call__(self):
        self.get_strings()
        self.plot_distances()


class BinnedStringComparator:
    def __init__(self, fscore_list, ascore, metric, save_loc, n_bins, strategies, hscore=None):
        self.fscore_list = fscore_list
        self.ascore = ascore
        self.metric = metric
        self.save_loc = save_loc
        self.n_bins = n_bins
        self.strategies = strategies
        self.len_curriculum = len(list(ascore.keys())[0])
        self.num_curriculums = len(ascore)
        self.hamming = jellyfish.hamming_distance
        self.fscore_strings = []
        self.hscore_strings = []
        self.fscore_bins = []
        self.hscore_bins = []
        self.avg_dists = [[] for _ in range(self.n_bins)]
        self.avg_legends = ['strategies', 'ascore', 'random']

        # self.strategies = list(map(lambda x: x.replace('ewc_alt', 'ewc//2'), self.strategies))
        if len(self.strategies) != len(self.fscore_list):
            self.strategies = [i for i in range(len(self.fscore_list))]

        for idx, fscore in enumerate(self.fscore_list):
            sorted_fscore = BinnedStringComparator.sort_dict(fscore)
            self.fscore_list[idx] = sorted_fscore
            self.fscore_strings.append(BinnedStringComparator.get_string(sorted_fscore))
            self.fscore_bins.append(self.bin_fscores(sorted_fscore, self.get_bins(sorted_fscore)))
        self.ascore = BinnedStringComparator.sort_dict(ascore)

        self.hscore = hscore
        if self.hscore is not None:
            self.hscore_list = [self.hscore]
            for idx, _hscore in enumerate(self.hscore_list):
                sorted_hscore = BinnedStringComparator.sort_dict(_hscore)
                self.hscore_list[idx] = sorted_hscore
                self.hscore_strings.append(BinnedStringComparator.get_string(sorted_hscore))
                self.hscore_bins.append(self.bin_fscores(sorted_hscore, self.get_bins(sorted_hscore)))


    @staticmethod
    def sort_dict(unsorted_dict):
        return dict(sorted(unsorted_dict.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def get_string(fscore):
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

    def bin_fscores(self, fscore, bins):
        binned_fscore = {i: [] for i in range(self.n_bins)}
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

    def plot_bins(self):
        y = [[] for _ in range(self.n_bins)]
        columns = []

        if self.hscore is None:
            avg_counter = [[] for _ in range(self.n_bins)]
            for i, fscore_a in enumerate(self.fscore_list):
                fscore_a_str = self.fscore_strings[i]
                fscore_a_bin = self.fscore_bins[i]
                for j, fscore_b in enumerate(self.fscore_list):
                    if j <= i:
                        continue
                    fscore_b_str = self.fscore_strings[j]
                    fscore_b_bin = self.fscore_bins[j]
                    str_idx_prev_a, str_idx_next_a = 0, 0
                    str_idx_prev_b, str_idx_next_b = 0, 0
                    for idx, (bin_size_a, bin_size_b) in enumerate(zip(fscore_a_bin.values(), fscore_b_bin.values())):
                        if idx > 0:
                            str_idx_prev_a = str_idx_next_a
                            str_idx_prev_b = str_idx_next_b
                        str_idx_next_a += self.len_curriculum * bin_size_a
                        str_idx_next_b += self.len_curriculum * bin_size_b
                        score_a = self.hamming(
                            fscore_a_str[str_idx_prev_a: str_idx_next_a],
                            fscore_b_str[str_idx_prev_a: str_idx_next_a]
                        )
                        score_b = self.hamming(
                            fscore_a_str[str_idx_prev_b: str_idx_next_b],
                            fscore_b_str[str_idx_prev_b: str_idx_next_b]
                        )
                        score = (score_a + score_b) / 2
                        y[idx].append(score)
                        avg_counter[idx].append(score)
                    columns.append("{} vs {}".format(self.strategies[i], self.strategies[j]))
            for i in range(self.n_bins):
                print("algo, bin idx {}:".format(avg_counter[i]))
                self.avg_dists[i].append(np.mean(avg_counter[i]))
        else:
            avg_counter = [[] for _ in range(self.n_bins)]
            for i, fscore_a in enumerate(self.fscore_list):
                fscore_a_str = self.fscore_strings[i]
                fscore_a_bin = self.fscore_bins[i]
                for j, fscore_b in enumerate(self.hscore_list):
                    fscore_b_str = self.fscore_strings[j]
                    fscore_b_bin = self.fscore_bins[j]
                    str_idx_prev_a, str_idx_next_a = 0, 0
                    str_idx_prev_b, str_idx_next_b = 0, 0
                    for idx, (bin_size_a, bin_size_b) in enumerate(zip(fscore_a_bin.values(), fscore_b_bin.values())):
                        if idx > 0:
                            str_idx_prev_a = str_idx_next_a
                            str_idx_prev_b = str_idx_next_b
                        str_idx_next_a += self.len_curriculum * bin_size_a
                        str_idx_next_b += self.len_curriculum * bin_size_b
                        score_a = self.hamming(
                            fscore_a_str[str_idx_prev_a: str_idx_next_a],
                            fscore_b_str[str_idx_prev_a: str_idx_next_a]
                        )
                        score_b = self.hamming(
                            fscore_a_str[str_idx_prev_b: str_idx_next_b],
                            fscore_b_str[str_idx_prev_b: str_idx_next_b]
                        )
                        score = (score_a + score_b) / 2
                        y[idx].append(score)
                        avg_counter[idx].append(score)
                    columns.append("{} vs hscore".format(self.strategies[i]))
            for i in range(self.n_bins):
                print("algo, bin idx {}:".format(avg_counter[i]))
                self.avg_dists[i].append(np.mean(avg_counter[i]))

        if self.hscore is None:
            avg_counter = [[] for _ in range(self.n_bins)]
            ascore_str = BinnedStringComparator.get_string(self.ascore)
            for i, fscore_a in enumerate(self.fscore_list):
                fscore_a_str = self.fscore_strings[i]
                fscore_a_bin = self.fscore_bins[i]
                str_idx_prev_a, str_idx_next_a = 0, 0
                for idx, bin_size_a in enumerate(fscore_a_bin.values()):
                    if idx > 0:
                        str_idx_prev_a = str_idx_next_a
                    str_idx_next_a += self.len_curriculum * bin_size_a
                    score = self.hamming(
                        fscore_a_str[str_idx_prev_a: str_idx_next_a],
                        ascore_str[str_idx_prev_a: str_idx_next_a]
                    )
                    y[idx].append(score)
                    avg_counter[idx].append(score)
                columns.append("{} vs ascore".format(self.strategies[i]))
            for i in range(self.n_bins):
                print("cd, bin idx {}:".format(avg_counter[i]))
                self.avg_dists[i].append(np.mean(avg_counter[i]))
        else:
            avg_counter = [[] for _ in range(self.n_bins)]
            ascore_str = BinnedStringComparator.get_string(self.ascore)
            for i, fscore_a in enumerate(self.hscore_list):
                fscore_a_str = self.fscore_strings[i]
                fscore_a_bin = self.fscore_bins[i]
                str_idx_prev_a, str_idx_next_a = 0, 0
                for idx, bin_size_a in enumerate(fscore_a_bin.values()):
                    if idx > 0:
                        str_idx_prev_a = str_idx_next_a
                    str_idx_next_a += self.len_curriculum * bin_size_a
                    score = self.hamming(
                        fscore_a_str[str_idx_prev_a: str_idx_next_a],
                        ascore_str[str_idx_prev_a: str_idx_next_a]
                    )
                    y[idx].append(score)
                    avg_counter[idx].append(score)
                columns.append("hscore vs ascore")
            for i in range(self.n_bins):
                print("cd, bin idx {}:".format(avg_counter[i]))
                self.avg_dists[i].append(np.mean(avg_counter[i]))

        if self.hscore is None:
            avg_counter = [[] for _ in range(self.n_bins)]
            ascore_curriculums = list(self.ascore.keys())
            for i, fscore_a in enumerate(self.fscore_list):
                score = [[] for _ in range(self.n_bins)]
                for _ in range(100):
                    random_str = self.get_random_string(ascore_curriculums)
                    fscore_a_str = self.fscore_strings[i]
                    fscore_a_bin = self.fscore_bins[i]
                    str_idx_prev_a, str_idx_next_a = 0, 0
                    for idx, bin_size_a in enumerate(fscore_a_bin.values()):
                        if idx > 0:
                            str_idx_prev_a = str_idx_next_a
                        str_idx_next_a += self.len_curriculum * bin_size_a
                        _score = self.hamming(
                            fscore_a_str[str_idx_prev_a: str_idx_next_a],
                            random_str[str_idx_prev_a: str_idx_next_a]
                        )
                        score[idx].append(_score)
                for idx, score_list in enumerate(score):
                    y[idx].append(np.mean(score_list))
                    avg_counter[idx].append(y[idx][-1])
                columns.append("{} vs random".format(self.strategies[i]))
            for i in range(self.n_bins):
                print("random, bin idx {}:".format(avg_counter[i]))
                self.avg_dists[i].append(np.mean(avg_counter[i]))
        else:
            avg_counter = [[] for _ in range(self.n_bins)]
            ascore_curriculums = list(self.ascore.keys())
            for i, fscore_a in enumerate(self.hscore_list):
                score = [[] for _ in range(self.n_bins)]
                for _ in range(100):
                    random_str = self.get_random_string(ascore_curriculums)
                    fscore_a_str = self.fscore_strings[i]
                    fscore_a_bin = self.fscore_bins[i]
                    str_idx_prev_a, str_idx_next_a = 0, 0
                    for idx, bin_size_a in enumerate(fscore_a_bin.values()):
                        if idx > 0:
                            str_idx_prev_a = str_idx_next_a
                        str_idx_next_a += self.len_curriculum * bin_size_a
                        _score = self.hamming(
                            fscore_a_str[str_idx_prev_a: str_idx_next_a],
                            random_str[str_idx_prev_a: str_idx_next_a]
                        )
                        score[idx].append(_score)
                for idx, score_list in enumerate(score):
                    y[idx].append(np.mean(score_list))
                    avg_counter[idx].append(y[idx][-1])
                columns.append("hscore vs random")
            for i in range(self.n_bins):
                print("random, bin idx {}:".format(avg_counter[i]))
                self.avg_dists[i].append(np.mean(avg_counter[i]))


        df = pd.DataFrame(y)
        df = df.div(df.sum(axis=1), axis=0)  # normalize
        df.columns = columns
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(df.transpose(), cmap="PiYG", square=True)
        ax.figure.savefig('string_plots/binned_{}_{}.png'.format(self.metric, self.save_loc), dpi=150)
        plt.close()

        # avg-plot
        df = pd.DataFrame(self.avg_dists)
        df.columns = self.avg_legends
        ax = df.plot.bar()
        ax.figure.savefig('string_plots/categorical_{}_{}.png'.format(self.metric, self.save_loc), dpi=150)
        plt.close()

        return df

    def __call__(self):
        df_avg = self.plot_bins()
        if self.metric == 'otdd':
            df_avg.to_pickle('paper-plots: binning/categorical_{}_{}.pkl'.format(self.metric, self.save_loc))












