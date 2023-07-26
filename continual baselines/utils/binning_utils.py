import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import sem

seed = 0
random.seed(seed)
np.random.seed(seed)


class BinningUtils:
    def __init__(self, fscore, ascore, save_loc, n_bins=5, percentile=75):
        self.fscore = fscore
        self.ascore = ascore
        self.save_loc = save_loc
        self.n_bins = n_bins
        self.percentile = percentile
        self.bins = None

        self.binned_fscore = {i: [] for i in range(self.n_bins)}
        self.sampling_size = int(len(self.fscore) * (1 - self.percentile/100))

        # graph params
        self.max = False
        self.random = True

    def get_bins(self):
        _, self.bins, __ = plt.hist(list(self.fscore.values()), bins=self.n_bins)
        plt.savefig('binning_plots/bins_{}.png'.format(self.save_loc), dpi=100)
        plt.close()

    def bin_fscores(self):
        for permutation, score in self.fscore.items():
            for i in range(self.n_bins):
                lo, hi = self.bins[i], self.bins[i+1]
                if i == self.n_bins - 1:
                    if lo <= score <= hi:
                        self.binned_fscore[i].append(permutation)
                        break
                else:
                    if lo <= score < hi:
                        self.binned_fscore[i].append(permutation)
                        break

    def get_fscore_stddev(self):
        stddev = []
        for permutations in self.binned_fscore.values():
            fscores = [self.fscore[permutation] for permutation in permutations]
            stddev.append(round(np.std(fscores), 3))

        return stddev

    def hypothesis_plots(self):
        ascore_threshold_75, ascore_threshold_80, ascore_threshold_90 = (
            np.percentile(list(self.ascore.values()), self.percentile),
            np.percentile(list(self.ascore.values()), 80),
            np.percentile(list(self.ascore.values()), 90)
        )
        ascore_75, ascore_80, ascore_90 = (
            {k: v for k, v in self.ascore.items() if v > ascore_threshold_75},
            {k: v for k, v in self.ascore.items() if v > ascore_threshold_80},
            {k: v for k, v in self.ascore.items() if v > ascore_threshold_90}
        )
        best_ascore_curriculums = [
            list(ascore_75.keys()),
            list(ascore_80.keys()),
            list(ascore_90.keys())
        ]

        fscore_stddev = self.get_fscore_stddev()
        y_max = [len(itr) for itr in list(self.binned_fscore.values())]
        x = ["bin {}: {}\nstd dev: {}".format(i, y_max[i], fscore_stddev[i]) for i in range(self.n_bins)]

        if self.max:
            plt.plot(x, y_max, 'g^-', linewidth=2, label='num: bin curriculums')

        y_ascore_75, y_ascore_80, y_ascore_90 = [], [], []
        for bin, curriculums in self.binned_fscore.items():
            y_ascore_75.append(
                len(
                    set(best_ascore_curriculums[0]) &
                    set(curriculums)
                )
            )
            y_ascore_80.append(
                len(
                    set(best_ascore_curriculums[1]) &
                    set(curriculums)
                )
            )
            y_ascore_90.append(
                len(
                    set(best_ascore_curriculums[2]) &
                    set(curriculums)
                )
            )
        plt.plot(x, y_ascore_75, 'bv--', linewidth=2, label='algo.{}'.format(self.percentile))
        plt.plot(x, y_ascore_80, 'go--', linewidth=2, label='algo.{}'.format(80))
        plt.plot(x, y_ascore_90, 'r*--', linewidth=2, label='algo.{}'.format(90))

        if self.random:
            y_random_common, y_stderr = [], []
            for bin, curriculums in self.binned_fscore.items():
                y_temp = []
                for _ in range(100):
                    y_temp.append(
                        len(
                            set(random.sample(list(self.fscore.keys()), self.sampling_size)) &
                            set(curriculums)
                        )
                    )
                y_random_common.append(np.mean(y_temp))
                y_stderr.append(sem(y_temp) * 1.96)
            y_random_common, y_stderr = np.array(y_random_common), np.array(y_stderr)
            plt.plot(x, y_random_common, 'ys:', linewidth=2, label='random')
            plt.fill_between(x, (y_random_common - y_stderr), (y_random_common + y_stderr), color='b', alpha=.1)

        plt.ylim(ymin=0)
        plt.ylabel('num: curriculums')
        plt.legend(loc='upper right', fontsize='x-small')
        plt.title('f-score ∩ algo-score')
        plt.savefig('binning_plots/{}_{}_{}.png'.format(self.n_bins, self.percentile, self.save_loc), dpi=100)
        plt.close()

    def __call__(self):
        self.get_bins()
        self.bin_fscores()
        self.hypothesis_plots()


