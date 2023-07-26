import pickle
import sys
import numpy as np
import random
from scipy.stats import sem
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from .distconfig.squeezenet import *
from .distconfig.squeezenet_samples import *
from .binning_utils import BinningUtils
from .string_comparator import StringComparator, BinnedStringComparator
from .curriculum_vis import CurriculumVis
import scienceplots
plt.style.use(['science', 'no-latex'])

seed = 0
random.seed(seed)
np.random.seed(seed)


def save_res(inp_list, save_loc):
    with open('results/{}.pkl'.format(save_loc), 'wb') as f:
        pickle.dump(inp_list, f)


def load_res(save_loc):
    with open('results/{}.pkl'.format(save_loc), 'rb') as f:
        res = pickle.load(f)

    return res


# noinspection PyShadowingNames
class CurriculumUtils:
    def __init__(self, result_dict, t1_dict, save_loc, layer=12, n_bins=5, percentile=75, result_dict_2=None, t1_dict_2=None):
        self.result_dict = result_dict
        self.t1_dict = t1_dict
        self.n_bins = n_bins
        self.percentile = percentile
        self.result_dict_2 = result_dict_2
        self.t1_dict_2 = t1_dict_2
        self.save_loc = save_loc
        self.layer = layer
        self.delta_dict = dict()
        self.favg_dict = dict()
        self.fscore = dict()
        self.custom_algorithm_score = dict()
        self.result_dict_alt = dict()
        self.subs = None
        self.best_curriculums, self.worst_curriculums = None, None
        self.dist_config = None
        self.sampling_config = None
        self.fscores_alt = None
        self.bin_strings = True
        self.all_strategy = ['naive', 'ewc', 'lwf']
        self.layer_comparison = True
        self.metric_comparison = True
        self.pretrained_comparison = True
        self.model_comparison = False
        self.samples_comparison = True
        self.get_algo_scores_alt = False

        self.current_strategy = self.save_loc.split('-', 1)[0]
        self.all_strategy.remove(self.current_strategy)
        self.all_strategy.insert(0, self.current_strategy)

        delimiter_1, delimiter_2 = None, None
        for (key, value), val_t1 in zip(self.result_dict.items(), self.t1_dict.values()):
            if delimiter_1 is None and delimiter_2 is None:
                delimiter_1 = key.find('[')
                delimiter_2 = key.find(']')

            subset = list(key[delimiter_1 + 1:delimiter_2].split(', '))
            try:
                value[0] = value[0].item()
                value[-1] = value[-1].item()
            except Exception:
                pass
            self.delta_dict[tuple(subset)] = val_t1[0] - val_t1[-1]
            self.favg_dict[tuple(subset)] = value[-1]
            self.fscore[tuple(subset)] = (2 * value[-1]) / \
                ((val_t1[0] - val_t1[-1]) * value[-1] + 1)
            self.result_dict_alt[tuple(subset)] = value

        self.human_score = load_res('human_score_top3bot3')
        if 'NovelNet' not in save_loc:
            self.human_score = None

        if self.result_dict_2 is not None:
            self.fscores_alt = []
            for results, t1 in zip(self.result_dict_2, self.t1_dict_2):
                fscore_2 = dict()
                for (key, value), val_t1 in zip(results.items(), t1.values()):
                    if delimiter_1 is None and delimiter_2 is None:
                        delimiter_1 = key.find('[')
                        delimiter_2 = key.find(']')

                    subset = list(key[delimiter_1 + 1:delimiter_2].split(', '))
                    try:
                        value[0], value[-1] = value[0].item(), value[-1].item()
                        val_t1[0], val_t1[-1] = val_t1[0].item(), val_t1[-1].item()
                    except Exception:
                        pass

                    fscore_2[tuple(subset)] = (2 * value[-1]) / \
                        ((val_t1[0] - val_t1[-1]) * value[-1] + 1)
                self.fscores_alt.append(fscore_2)

    @staticmethod
    def save_pkl(inp, loc):
        with open('curriculums/{}.pkl'.format(loc), 'wb') as f:
            pickle.dump(inp, f)

    @staticmethod
    def load_pkl(loc, res_type='int'):
        with open('curriculums/{}.pkl'.format(loc), 'rb') as f:
            res = pickle.load(f)

        if res_type == 'int':
            for idx, curriculum in enumerate(res):
                res[idx] = list(map(int, curriculum))
        return res

    def sub_classes(self, curriculum_list):
        rev_subs = {v: k for k, v in self.subs.items()}
        ret_list = [rev_subs.get(item, item) for item in curriculum_list]

        return ret_list

    def get_best_curriculums(self, topk):
        sorted_dict = dict(
            sorted(self.fscore.items(), key=lambda item: item[1], reverse=True))
        best_curriculums = list(sorted_dict.keys())[:topk]

        self.save_pkl(best_curriculums, "best_{}".format(self.save_loc))
        print("best curriculums:\n")
        for curriculum in best_curriculums:
            print(self.sub_classes(curriculum))

        self.best_curriculums = best_curriculums

    def get_worst_curriculums(self, topk):
        sorted_dict = dict(
            sorted(self.fscore.items(), key=lambda item: item[1], reverse=False))
        worst_curriculums = list(sorted_dict.keys())[:topk]

        self.save_pkl(worst_curriculums, "worst_{}".format(self.save_loc))
        print("worst curriculums:\n")
        for curriculum in worst_curriculums:
            print(self.sub_classes(curriculum))

        self.worst_curriculums = worst_curriculums

    def plot_top_curriculums(self):
        evenly_spaced_interval = np.linspace(0, 1, len(self.best_curriculums))
        colors_blues = [cm.Blues(x) for x in evenly_spaced_interval]
        colors_reds = [cm.Reds(x) for x in evenly_spaced_interval]

        for idx, curriculum in enumerate(self.best_curriculums):
            plt.plot(self.result_dict_alt[curriculum],
                     color=colors_blues[idx], label="best-rank {}".format(idx + 1))
        for idx, curriculum in enumerate(self.worst_curriculums):
            plt.plot(self.result_dict_alt[curriculum],
                     color=colors_reds[idx], label="worst-rank {}".format(idx + 1))

        plt.ylim(ymin=0)
        plt.ylabel('avg accuracy')
        plt.xticks([i for i in range(5)])
        plt.legend(loc='upper right', fontsize='x-small')
        plt.title('best vs worst curriculums')
        plt.savefig(
            'curriculum_plots/best_worst_curriculums_{}.png'.format(self.save_loc), dpi=100)
        plt.close()

    def plot_scatter(self):
        x_red, x_blue, y_red, y_blue = [], [], [], []
        for key, delta, favg in zip(self.delta_dict.keys(), self.delta_dict.values(), self.favg_dict.values()):
            if key in self.best_curriculums:
                x_red.append(delta), y_red.append(favg)
            else:
                x_blue.append(delta), y_blue.append(favg)

        plt.scatter(x_blue, y_blue, c='blue', label='other curriculums')
        plt.scatter(x_red, y_red, c='red', label='best curriculums')
        plt.xlabel('delta'), plt.ylabel('favg')
        plt.legend(loc='upper right', fontsize='x-small')
        plt.title('delta vs favg')
        plt.savefig(
            'curriculum_plots/scatter_{}.png'.format(self.save_loc), dpi=100)
        plt.close()

    def plot_bar(self):
        x_red, x_blue, y_red, y_blue = [], [], [], []
        for idx, (key, f_val) in enumerate(zip(self.fscore.keys(), self.fscore.values())):
            if key in self.best_curriculums:
                x_red.append(idx), y_red.append(f_val)
            else:
                x_blue.append(idx), y_blue.append(f_val)

        plt.bar(x_blue, y_blue, color='blue', label='other curriculums')
        plt.bar(x_red, y_red, color='red',
                linewidth=5, label='best curriculums')
        plt.xlabel('curriculums'), plt.ylabel('f-score')
        plt.legend(loc='upper right', fontsize='x-small')
        plt.title('F-Scores')
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False
        )
        plt.savefig(
            'curriculum_plots/bar_{}.png'.format(self.save_loc), dpi=100)
        plt.close()

    def hypothesis_test(self):
        def get_algo_scores(_scorer):
            custom_algorithm_score = dict()
            if not self.get_algo_scores_alt:
                for _subset in self.fscore.keys():
                    it = iter(_subset)
                    inp_curr = list(zip(it, it))
                    inp_curr = [class_num_map[x] for x in inp_curr]
                    custom_algorithm_score[_subset] = _scorer(inp_curr)
            else:
                for _subset in self.fscore.keys():
                    inp_curr = [class_num_map[i] for i in _subset]
                    custom_algorithm_score[_subset] = _scorer(inp_curr)
            return custom_algorithm_score

        def enum_config(_enum_dict, _var_dict, _sorted_adjacency_list):
            _var_dict = dict((_enum_dict[key], value)
                             for (key, value) in _var_dict.items())
            _sorted_adjacency_list = dict((_enum_dict[key], value) for (
                key, value) in _sorted_adjacency_list.items())
            for _, value in _sorted_adjacency_list.items():
                for idx, val in enumerate(value):
                    value[idx] = (value[idx][0], _enum_dict[value[idx][1]])
            return _var_dict, _sorted_adjacency_list

        class_num_map, enum_dict, var_dict, sorted_adjacency_list = (
            DistConfig.class_num_map,
            self.dist_config.enum_dict,
            self.dist_config.var_dict,
            self.dist_config.sorted_adjacency_list
        )

        var_dict, sorted_adjacency_list = enum_config(
            enum_dict, var_dict, sorted_adjacency_list)
        scorer = ScoreCurriculum(var_dict, sorted_adjacency_list)
        fscore_75 = np.percentile(list(self.fscore.values()), 75)
        x_green, x_blue, y_green, y_blue = [], [], [], []
        self.custom_algorithm_score = get_algo_scores(scorer)
        algoscore_75 = np.percentile(
            list(self.custom_algorithm_score.values()), 75)
        for idx, (subset, algoscore) in enumerate(self.custom_algorithm_score.items()):
            if algoscore > algoscore_75:
                x_green.append(idx), y_green.append(self.fscore[subset])
            else:
                x_blue.append(idx), y_blue.append(self.fscore[subset])

        # plot: f-score for verified curriculums
        plt.scatter(x_blue, y_blue, c='blue', label='other curriculums')
        plt.scatter(x_green, y_green, c='green', label='best curriculums')
        plt.plot(
            [i for i in range(len(self.fscore))],
            [fscore_75] * len(self.fscore),
            color='red', linestyle='-', linewidth=0.5
        )
        plt.xlabel('curriculums'), plt.ylabel('f-score')
        plt.legend(loc='upper right', fontsize='x-small')
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False
        )
        plt.savefig('algo_plots/verified_{}.png'.format(self.save_loc), dpi=100)
        plt.close()

        # string comp
        if self.result_dict_2 is not None:
            if self.bin_strings:
                metrics = ['otdd', 'cosine', 'euclidean']
                dist_class = type(self.dist_config)
                for metric in metrics:
                    temp_config = dist_class(dist_type=metric)
                    _var_dict, _sorted_adjacency_list = enum_config(
                        temp_config.enum_dict,
                        temp_config.var_dict,
                        temp_config.sorted_adjacency_list
                    )
                    _scorer = ScoreCurriculum(
                        _var_dict, _sorted_adjacency_list)
                    _algo_score = get_algo_scores(_scorer)
                    binned_string_comparator = BinnedStringComparator(
                        fscore_list=self.fscores_alt,
                        ascore=_algo_score,
                        hscore=self.human_score,
                        metric=metric,
                        save_loc=self.save_loc,
                        n_bins=self.n_bins,
                        strategies=self.all_strategy
                    )
                    binned_string_comparator()
            else:
                # works for any 2 at a time
                string_comparator = StringComparator(
                    self.fscore,
                    self.fscores_alt[0],
                    self.custom_algorithm_score,
                    self.save_loc
                )
                string_comparator()
            sys.exit()

        # visualizer
        visualizer = CurriculumVis(
            fscore=self.fscore,
            save_loc=self.save_loc,
            alt_mapping=self.get_algo_scores_alt
        )
        visualizer()

        # binning
        binner = BinningUtils(
            fscore=self.fscore,
            ascore=self.custom_algorithm_score,
            save_loc=self.save_loc,
            n_bins=self.n_bins,
            percentile=self.percentile
        )
        binner()

        # plot: f-score vs algo-score
        plt.scatter(list(self.fscore.values()), list(
            self.custom_algorithm_score.values()))
        plt.xlabel('f-score'), plt.ylabel('algo-score')
        plt.title('f-score vs algo-score')
        plt.savefig('algo_plots/scored_{}.png'.format(self.save_loc), dpi=100)
        plt.close()

        # plot: f-score vs algo-score
        _topk = 120
        _ascore = dict(sorted(self.custom_algorithm_score.items(),
                       key=lambda item: item[1], reverse=True))
        _fscore_asort = {k: self.fscore[k] for k, v in _ascore.items()}
        _val1, _val2 = list(_ascore.values())[:_topk], list(
            _fscore_asort.values())[:_topk]
        slope, intercept, r_value, _, _ = stats.linregress(_val1, _val2)
        line = slope * np.array(_val1) + intercept
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        plt.scatter(_val1, _val2)
        plt.plot(_val1, line, color='r', label='Line of Best Fit')
        plt.xlabel('CD-score'), plt.ylabel('F')
        r_value_str = 'R = {:.4f}'.format(r_value)
        plt.text(0.1, 0.9, r_value_str, transform=plt.gca().transAxes)
        plt.tick_params(axis="x", which="both", bottom=False, top=False)
        plt.tick_params(axis="y", which="both", left=True, right=False)
        plt.savefig(
            'algo_plots/topk_{}.scored_{}.png'.format(_topk, self.save_loc), dpi=200)
        plt.legend()
        plt.close()
        mpl.rcParams['axes.spines.right'] = True
        mpl.rcParams['axes.spines.top'] = True

        # plot: top-10, top-20, top-50
        sorted_fscore = dict(
            sorted(self.fscore.items(), key=lambda item: item[1], reverse=True))
        sorted_ascore = {
            k: v for k, v in self.custom_algorithm_score.items() if v > algoscore_75}
        x = ['top-10', 'top-20', 'top-50']
        y = [
            len(set(list(sorted_fscore.keys())[:10]) & set(
                list(sorted_ascore.keys()))),
            len(set(list(sorted_fscore.keys())[:20]) & set(
                list(sorted_ascore.keys()))),
            len(set(list(sorted_fscore.keys())[:50]) & set(
                list(sorted_ascore.keys())))
        ]
        plt.plot(x, y, 'g^-', linewidth=0.8)
        plots = sns.barplot(x=x, y=y, color='blue', fill=False, hatch='/')
        plt.yticks([10, 20, 50])
        plt.ylim(ymin=0)
        plt.ylabel('num: common curriculums')
        plt.title('f-score ∩ algo-score')
        for p in plots.patches:
            plots.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 9), textcoords='offset points')
        plt.savefig('algo_plots/topk_{}.png'.format(self.save_loc), dpi=100)
        plt.close()

        # layer-comparison
        if self.layer_comparison:
            layers = [3, 6, 9, 11, 12]
            dist_class = type(self.dist_config)
            x = ['top-10', 'top-20', 'top-50']
            colors = ['g', 'y', 'r', 'b', 'm']
            styles = ['^-', 'v--', '*-.', 's:', 'x--']
            for layer_num, color, style in zip(layers, colors, styles):
                temp_config = dist_class(layer=layer_num)
                try:
                    _var_dict, _sorted_adjacency_list = enum_config(
                        temp_config.enum_dict,
                        temp_config.var_dict,
                        temp_config.sorted_adjacency_list
                    )
                except AttributeError:
                    continue
                _scorer = ScoreCurriculum(_var_dict, _sorted_adjacency_list)
                _algo_score = get_algo_scores(_scorer)
                with open('paper-plots: fscore__bars/results/{}.{}.{}.pkl'.format('cd_score', self.save_loc, layer_num), 'wb') as f:
                    pickle.dump(_algo_score, f)
                _algoscore_75 = np.percentile(list(_algo_score.values()), 75)
                _algo_score = {k: v for k,
                               v in _algo_score.items() if v > _algoscore_75}
                y = [
                    len(set(list(sorted_fscore.keys())[:10]) & set(
                        list(_algo_score.keys()))),
                    len(set(list(sorted_fscore.keys())[:20]) & set(
                        list(_algo_score.keys()))),
                    len(set(list(sorted_fscore.keys())[:50]) & set(
                        list(_algo_score.keys())))
                ]
                plt.plot(x, y, '{}{}'.format(color, style),
                         linewidth=2, label=str(layer_num))
                # comment later
                print(
                    "layer - {}, 10: {}, 20: {}, 50: {}".format(layer_num, y[0], y[1], y[2]))
            plt.yticks([10, 20, 50])
            plt.ylim(ymin=0)
            plt.ylabel('num: common curriculums')
            plt.legend(loc='upper right', fontsize='x-small')
            plt.title('f-score ∩ algo-score - different layers')
            plt.savefig(
                'algo_plots/layers_{}.png'.format(self.save_loc), dpi=100)
            plt.close()

        # metric-comparison
        if self.metric_comparison:
            metrics = ['otdd', 'cosine', 'euclidean']
            dist_class = type(self.dist_config)
            x = ['top-10', 'top-20', 'top-50']
            colors = ['g', 'y', 'r']
            styles = ['^-', 'v--', 'o-.']
            for metric, color, style in zip(metrics, colors, styles):
                temp_config = dist_class(dist_type=metric)
                _var_dict, _sorted_adjacency_list = enum_config(
                    temp_config.enum_dict,
                    temp_config.var_dict,
                    temp_config.sorted_adjacency_list
                )
                _scorer = ScoreCurriculum(_var_dict, _sorted_adjacency_list)
                _algo_score = get_algo_scores(_scorer)
                with open('paper-plots: fscore__bars/results/{}.{}.{}.pkl'.format('cd_score', self.save_loc, metric), 'wb') as f:
                    pickle.dump(_algo_score, f)
                _algoscore_75 = np.percentile(list(_algo_score.values()), 75)
                _algo_score = {k: v for k,
                               v in _algo_score.items() if v > _algoscore_75}
                y = [
                    len(set(list(sorted_fscore.keys())[:10]) & set(
                        list(_algo_score.keys()))),
                    len(set(list(sorted_fscore.keys())[:20]) & set(
                        list(_algo_score.keys()))),
                    len(set(list(sorted_fscore.keys())[:50]) & set(
                        list(_algo_score.keys())))
                ]
                plt.plot(x, y, '{}{}'.format(color, style),
                         linewidth=2, label=metric)
                print("metric - {}, 10: {}, 20: {}, 50: {}".format(metric,
                      y[0], y[1], y[2]))  # comment later
            plt.yticks([10, 20, 50])
            plt.ylim(ymin=0)
            plt.ylabel('num: common curriculums')
            plt.legend(loc='upper right', fontsize='x-small')
            plt.title('f-score ∩ algo-score - different metrics')
            plt.savefig(
                'algo_plots/metrics_{}.png'.format(self.save_loc), dpi=100)
            plt.close()
        
        # pretrained-comparison
        if self.pretrained_comparison:
            pretrained = ['ResNet34_ImageNet100', 'ResNet18_ImageNet100', 'ResNet18_random', 'SqueezeNet_MNIST']
            dist_class = type(self.dist_config)
            for _pretrained in pretrained:
                temp_config = dist_class(dist_type='cosine', model=_pretrained)
                _var_dict, _sorted_adjacency_list = enum_config(
                    temp_config.enum_dict,
                    temp_config.var_dict,
                    temp_config.sorted_adjacency_list
                )
                _scorer = ScoreCurriculum(_var_dict, _sorted_adjacency_list)
                _algo_score = get_algo_scores(_scorer)
                with open('paper-plots: fscore__bars/results/{}.{}.{}.pkl'.format('cd_score', self.save_loc, _pretrained), 'wb') as f:
                    pickle.dump(_algo_score, f)


        # model-comparison
        if self.model_comparison:
            models = ['SqueezeNet100', 'random']
            dist_class = type(self.dist_config)
            x = ['top-10', 'top-20', 'top-50']
            colors = ['g', 'r']
            styles = ['^-', 'v--']
            for model, color, style in zip(models, colors, styles):
                temp_config = dist_class(random=False)
                _var_dict, _sorted_adjacency_list = enum_config(
                    temp_config.enum_dict,
                    temp_config.var_dict,
                    temp_config.sorted_adjacency_list
                )
                _scorer = ScoreCurriculum(_var_dict, _sorted_adjacency_list)
                _algo_score = get_algo_scores(_scorer)
                _algoscore_75 = np.percentile(list(_algo_score.values()), 75)
                _algo_score = {k: v for k,
                               v in _algo_score.items() if v > _algoscore_75}
                # _algo_score = dict(more_itertools.take(20, _algo_score.items()))
                if model != 'random':
                    y = [
                        len(set(list(sorted_fscore.keys())[:10]) & set(
                            list(_algo_score.keys()))),
                        len(set(list(sorted_fscore.keys())[:20]) & set(
                            list(_algo_score.keys()))),
                        len(set(list(sorted_fscore.keys())[:50]) & set(
                            list(_algo_score.keys())))
                    ]
                    plt.plot(x, y, '{}{}'.format(color, style),
                             linewidth=2, label=model)
                    print("10: {}, 20: {}, 50: {}".format(
                        y[0], y[1], y[2]))  # comment later
                else:
                    y, y_temp = [], []
                    for _ in range(10):
                        num_random = len(set(list(_algo_score.keys())))
                        y_temp = [
                            len(set(list(sorted_fscore.keys())[:10]) & set(
                                random.sample(list(sorted_fscore.keys()), num_random))),
                            len(set(list(sorted_fscore.keys())[:20]) & set(
                                random.sample(list(sorted_fscore.keys()), num_random))),
                            len(set(list(sorted_fscore.keys())[:50]) & set(
                                random.sample(list(sorted_fscore.keys()), num_random)))
                        ]
                        y.append(y_temp)
                    y_mean = np.mean(y, axis=0)
                    y_stderr = sem(y, axis=0) * 1.96
                    plt.plot(x, y_mean, '{}{}'.format(
                        color, style), linewidth=2, label=model)
                    plt.fill_between(x, (y_mean - y_stderr),
                                     (y_mean + y_stderr), color='b', alpha=.1)
                    print("10: {}, 20: {}, 50: {}".format(
                        y_mean[0], y_mean[1], y_mean[2]))  # comment later
            plt.yticks([10, 20, 50])
            plt.ylim(ymin=0)
            plt.ylabel('num: common curriculums')
            plt.legend(loc='upper right', fontsize='x-small')
            plt.title('f-score ∩ algo-score - different models')
            plt.savefig(
                'algo_plots/models_{}.png'.format(self.save_loc), dpi=100)
            plt.close()

        # samples-comparison
        if self.samples_comparison:
            sample_size = [25, 50, 100, 200, 400]
            dist_class = self.sampling_config
            x = ['top-10', 'top-20', 'top-50']
            colors = ['g', 'y', 'r', 'b', 'm']
            styles = ['^-', 'v--', '*-.', 's:', 'x--']
            for sample_num, color, style in zip(sample_size, colors, styles):
                temp_config = dist_class(sample_size=sample_num)
                _var_dict, _sorted_adjacency_list = enum_config(
                    temp_config.enum_dict,
                    temp_config.var_dict,
                    temp_config.sorted_adjacency_list
                )
                _scorer = ScoreCurriculum(_var_dict, _sorted_adjacency_list)
                _algo_score = get_algo_scores(_scorer)
                _algoscore_75 = np.percentile(list(_algo_score.values()), 75)
                _algo_score = {k: v for k,
                               v in _algo_score.items() if v > _algoscore_75}
                y = [
                    len(set(list(sorted_fscore.keys())[:10]) & set(
                        list(_algo_score.keys()))),
                    len(set(list(sorted_fscore.keys())[:20]) & set(
                        list(_algo_score.keys()))),
                    len(set(list(sorted_fscore.keys())[:50]) & set(
                        list(_algo_score.keys())))
                ]
                plt.plot(x, y, '{}{}'.format(color, style),
                         linewidth=2, label=str(sample_num))
            plt.yticks([10, 20, 50])
            plt.ylim(ymin=0)
            plt.ylabel('num: common curriculums')
            plt.legend(loc='upper right', fontsize='x-small')
            plt.title('f-score ∩ algo-score - different sample size')
            plt.savefig(
                'algo_plots/samples_{}.png'.format(self.save_loc), dpi=100)
            plt.close()

    def __call__(self, topk=5, return_curriculums=False):
        self.get_best_curriculums(topk=topk)
        self.get_worst_curriculums(topk=topk)
        self.plot_top_curriculums()
        self.plot_scatter()
        self.plot_bar()
        if self.dist_config is not None:
            self.hypothesis_test()

        if return_curriculums:
            return self.best_curriculums, self.worst_curriculums


class ImageNetCurriculum(CurriculumUtils):
    def __init__(self, result_dict, t1_dict, save_loc, n_bins, percentile, layer=12, result_dict_2=None, t1_dict_2=None):
        super().__init__(result_dict, t1_dict, save_loc, layer,
                         n_bins, percentile, result_dict_2, t1_dict_2)
        self.subs = {
            'airplane': '0', 'car': '1', 'bird': '2', 'cat': '3',
            'elephant': '4', 'dog': '5', 'bottle': '6', 'knife': '7',
            'truck': '8', 'boat': '9'
        }
        self.dist_config = ImageNetDistConfig(layer=self.layer)
        self.sampling_config = ImageNetDistSamples


class StyleNetCurriculum(CurriculumUtils):
    def __init__(self, result_dict, t1_dict, save_loc, n_bins, percentile, layer=12, result_dict_2=None, t1_dict_2=None):
        super().__init__(result_dict, t1_dict, save_loc, layer,
                         n_bins, percentile, result_dict_2, t1_dict_2)
        self.subs = {
            'candy': '0', 'mosaic_ducks_massimo': '1', 'pencil': '2', 'seated-nude': '3',
            'shipwreck': '4', 'starry_night': '5', 'stars2': '6', 'strip': '7',
            'the_scream': '8', 'wave': '9'
        }
        self.dist_config = StyleNetDistConfig(layer=self.layer)
        self.sampling_config = StyleNetDistSamples


class CIFAR10Curriculum(CurriculumUtils):
    def __init__(self, result_dict, t1_dict, save_loc, n_bins, percentile, layer=12, result_dict_2=None, t1_dict_2=None):
        super().__init__(result_dict, t1_dict, save_loc, layer,
                         n_bins, percentile, result_dict_2, t1_dict_2)
        self.subs = {
            'airplane': '0', 'automobile': '1', 'bird': '2', 'cat': '3',
            'deer': '4', 'dog': '5', 'frog': '6', 'horse': '7',
            'ship': '8', 'truck': '9'
        }
        self.dist_config = CIFAR10DistConfig(layer=self.layer)
        self.sampling_config = CIFAR10DistSamples


class MNISTCurriculum(CurriculumUtils):
    def __init__(self, result_dict, t1_dict, save_loc, n_bins, percentile, layer=12, result_dict_2=None, t1_dict_2=None):
        super().__init__(result_dict, t1_dict, save_loc, layer,
                         n_bins, percentile, result_dict_2, t1_dict_2)
        self.subs = {
            '0': '0', '1': '1', '2': '2', '3': '3',
            '4': '4', '5': '5', '6': '6', '7': '7',
            '8': '8', '9': '9'
        }
        self.dist_config = MNISTDistConfig(layer=self.layer)
        self.sampling_config = MNISTDistSamples


class FashionMNISTCurriculum(CurriculumUtils):
    def __init__(self, result_dict, t1_dict, save_loc, n_bins, percentile, layer=12, result_dict_2=None, t1_dict_2=None):
        super().__init__(result_dict, t1_dict, save_loc, layer,
                         n_bins, percentile, result_dict_2, t1_dict_2)
        self.subs = {
            'top': '0', 'trouser': '1', 'pullover': '2', 'dress': '3',
            'coat': '4', 'sandal': '5', 'shirt': '6', 'sneaker': '7',
            'bag': '8', 'ankle boot': '9'
        }
        self.dist_config = FashionMNISTDistConfig(layer=self.layer)
        self.sampling_config = FashionMNISTDistSamples


class FashionMNIST5Curriculum(CurriculumUtils):
    def __init__(self, result_dict, t1_dict, save_loc, n_bins, percentile, layer=12, result_dict_2=None, t1_dict_2=None):
        super().__init__(result_dict, t1_dict, save_loc, layer,
                         n_bins, percentile, result_dict_2, t1_dict_2)
        self.subs = {
            'top': '0', 'trouser': '1', 'pullover': '2', 'dress': '3',
            'coat': '4'
        }
        DistConfig.class_num_map = {
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4
        }
        self.dist_config = FashionMNIST5DistConfig(layer=self.layer)
        self.layer_comparison = True
        self.samples_comparison = False
        self.models_comparison = True
        self.get_algo_scores_alt = True
        self.metric_comparison = True
        self.pretrained_comparison = True


class MNIST5Curriculum(CurriculumUtils):
    def __init__(self, result_dict, t1_dict, save_loc, n_bins, percentile, layer=12, result_dict_2=None, t1_dict_2=None):
        super().__init__(result_dict, t1_dict, save_loc, layer,
                         n_bins, percentile, result_dict_2, t1_dict_2)
        self.subs = {
            '0': '0', '1': '1', '2': '2', '3': '3',
            '4': '4'
        }
        DistConfig.class_num_map = {
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4
        }
        self.dist_config = MNIST5DistConfig(layer=self.layer)
        self.layer_comparison = True
        self.samples_comparison = False
        self.models_comparison = True
        self.get_algo_scores_alt = True
        self.metric_comparison = True
        self.pretrained_comparison = True


class ImageNet5Curriculum(CurriculumUtils):
    def __init__(self, result_dict, t1_dict, save_loc, n_bins, percentile, layer=12, result_dict_2=None, t1_dict_2=None):
        super().__init__(result_dict, t1_dict, save_loc, layer,
                         n_bins, percentile, result_dict_2, t1_dict_2)
        self.subs = {
            'airplane': '0', 'car': '1', 'bird': '2', 'cat': '3',
            'elephant': '4'
        }
        DistConfig.class_num_map = {
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4
        }
        self.dist_config = ImageNet5DistConfig(layer=self.layer)
        self.layer_comparison = False
        self.samples_comparison = False
        self.models_comparison = True
        self.get_algo_scores_alt = True
        self.metric_comparison = True


class CIFAR105Curriculum(CurriculumUtils):
    def __init__(self, result_dict, t1_dict, save_loc, n_bins, percentile, layer=12, result_dict_2=None, t1_dict_2=None):
        super().__init__(result_dict, t1_dict, save_loc, layer,
                         n_bins, percentile, result_dict_2, t1_dict_2)
        self.subs = {
            'airplane': '0', 'automobile': '1', 'bird': '2', 'cat': '3',
            'deer': '4'
        }
        DistConfig.class_num_map = {
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4
        }
        self.dist_config = CIFAR105DistConfig(layer=self.layer)
        self.layer_comparison = True
        self.samples_comparison = False
        self.models_comparison = True
        self.get_algo_scores_alt = True
        self.metric_comparison = True


class ImageNet2012Curriculum(CurriculumUtils):
    def __init__(self, result_dict, t1_dict, save_loc, n_bins, percentile, layer=12, result_dict_2=None, t1_dict_2=None):
        super().__init__(result_dict, t1_dict, save_loc, layer,
                         n_bins, percentile, result_dict_2, t1_dict_2)
        self.subs = dict()
        for i in range(900):
            self.subs[str(i)] = str(i)


class NovelNetCurriculum(CurriculumUtils):
    def __init__(self, result_dict, t1_dict, save_loc, n_bins, percentile, layer=12, result_dict_2=None, t1_dict_2=None):
        super().__init__(result_dict, t1_dict, save_loc, layer,
                         n_bins, percentile, result_dict_2, t1_dict_2)
        self.subs = {
            'fa1': '0', 'fa2': '1', 'fb1': '2', 'fb3': '3',
            'fc1': '4'
        }
        DistConfig.class_num_map = {
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4
        }
        self.dist_config = NovelNetDistConfig(layer=self.layer)
        self.layer_comparison = False
        self.samples_comparison = False
        self.models_comparison = True
        self.get_algo_scores_alt = True
        self.metric_comparison = True


class MultiTaskCurriculum(CurriculumUtils):
    def __init__(self, result_dict, save_loc):
        super().__init__(result_dict, save_loc)
        self.result_dict = result_dict
        self.delta_dict = dict()
        self.favg_dict = dict()
        self.fscore = dict()
        self.save_loc = save_loc
        self.obj_subs = {
            'airplane': '0', 'car': '1', 'bird': '2', 'cat': '3',
            'elephant': '4', 'dog': '5', 'bottle': '6', 'knife': '7',
            'truck': '8', 'boat': '9'
        }
        self.sty_subs = {
            'candy': '0', 'mosaic_ducks_massimo': '1', 'pencil': '2', 'seated-nude': '3',
            'shipwreck': '4', 'starry_night': '5', 'stars2': '6', 'strip': '7',
            'the_scream': '8', 'wave': '9'
        }
        self.best_curriculums, self.worst_curriculums = None, None

        delimiter_1, delimiter_2 = None, None
        for key, value in self.result_dict.items():
            if delimiter_1 is None and delimiter_2 is None:
                delimiter_1 = key.find('[')
                delimiter_2 = key.find(']')

            obj_subset, sty_subset = key.split('_')
            obj_subset = list(
                obj_subset[delimiter_1 + 1:delimiter_2].split(', '))
            sty_subset = list(
                sty_subset[delimiter_1 + 1:delimiter_2].split(', '))
            try:
                value[0] = value[0].item()
                value[-1] = value[-1].item()
            except Exception:
                pass
            self.delta_dict[(tuple(obj_subset), tuple(
                sty_subset))] = value[0] - value[-1]
            self.favg_dict[(tuple(obj_subset), tuple(sty_subset))] = value[-1]
            self.fscore[(tuple(obj_subset), tuple(sty_subset))] = 1 / \
                ((value[0] - value[-1]) + (1 / value[-1]))

    def sub_classes(self, curriculum_list):
        rev_obj_subs = {v: k for k, v in self.obj_subs.items()}
        rev_sty_subs = {v: k for k, v in self.sty_subs.items()}
        ret_list_obj = [rev_obj_subs.get(item, item)
                        for item in curriculum_list[0]]
        ret_list_sty = [rev_sty_subs.get(item, item)
                        for item in curriculum_list[1]]

        return ret_list_obj, ret_list_sty


class NoiseNetCurriculum(CurriculumUtils):
    def __init__(self, result_dict, save_loc):
        # super().__init__(result_dict=result_dict, t1_dict=None, save_loc=save_loc)
        self.result_dict = result_dict
        self.delta_dict = dict()
        self.favg_dict = dict()
        self.fscore = dict()
        self.save_loc = save_loc
        self.sty_subs = {
            'candy': '0', 'mosaic_ducks_massimo': '1', 'pencil': '2', 'seated-nude': '3',
            'shipwreck': '4', 'starry_night': '5', 'stars2': '6', 'strip': '7',
            'the_scream': '8', 'wave': '9'
        }
        self.nse_subs = {
            'n.pixelate': '0', 'n.gaussian_blur': '1', 'n.contrast': '2', 'n.speckle_noise': '3',
            'n.brightness': '4', 'n.defocus_blur': '5', 'n.saturate': '6', 'n.gaussian_noise': '7',
            'n.impulse_noise': '8', 'n.shot_noise': '9'
        }
        self.best_curriculums, self.worst_curriculums = None, None

        delimiter_1, delimiter_2 = None, None
        for key, value in self.result_dict.items():
            if delimiter_1 is None and delimiter_2 is None:
                delimiter_1 = key.find('[')
                delimiter_2 = key.find(']')

            nse_subset, sty_subset = key.split('_')
            nse_subset = list(
                nse_subset[delimiter_1 + 1:delimiter_2].split(', '))
            sty_subset = list(
                sty_subset[delimiter_1 + 1:delimiter_2].split(', '))
            try:
                value[0] = value[0].item()
                value[-1] = value[-1].item()
            except Exception:
                pass
            self.delta_dict[(tuple(nse_subset), tuple(
                sty_subset))] = value[0] - value[-1]
            self.favg_dict[(tuple(nse_subset), tuple(sty_subset))] = value[-1]
            self.fscore[(tuple(nse_subset), tuple(sty_subset))] = 1 / \
                ((value[0] - value[-1]) + (1 / value[-1]))

    def sub_classes(self, curriculum_list):
        rev_nse_subs = {v: k for k, v in self.nse_subs.items()}
        rev_sty_subs = {v: k for k, v in self.sty_subs.items()}
        ret_list_nse = [rev_nse_subs.get(item, item)
                        for item in curriculum_list[0]]
        ret_list_sty = [rev_sty_subs.get(item, item)
                        for item in curriculum_list[1]]

        return ret_list_nse, ret_list_sty


class GenerateCurriculum:
    def __init__(self, var_dict, sorted_adjacency_list, top_k=2):
        self.var_dict = var_dict
        self.sorted_adjacency_list = sorted_adjacency_list
        self.top_k = top_k
        self.curriculums = []

        self.first_tasks = list(
            dict(
                sorted(self.var_dict.items(), key=lambda item: item[1])
            ).keys()
        )[:self.top_k]

    def __call__(self):
        for task in self.first_tasks:
            curriculum = [''] * len(self.var_dict)
            curriculum[0] = task
            curr_task = task
            ptr_start, ptr_end = 1, -1
            while True:
                if '' not in curriculum:
                    break

                dist_ptr_start, dist_ptr_end = 0, len(self.var_dict) - 2
                while True:
                    if self.sorted_adjacency_list[curr_task][dist_ptr_start][1] not in curriculum:
                        curriculum[ptr_end] = self.sorted_adjacency_list[curr_task][dist_ptr_start][1]
                        ptr_end -= 1
                        break
                    dist_ptr_start += 1
                while True:
                    if self.sorted_adjacency_list[curr_task][dist_ptr_end][1] not in curriculum:
                        curriculum[ptr_start] = self.sorted_adjacency_list[curr_task][dist_ptr_end][1]
                        ptr_start += 1
                        break
                    dist_ptr_end -= 1

            self.curriculums.append(curriculum)

        return self.curriculums


class VerifyCurriculum:
    def __init__(self, var_dict, sorted_adjacency_list, top_k=2):
        self.var_dict = var_dict
        self.sorted_adjacency_list = sorted_adjacency_list
        self.top_k = top_k

        self.first_tasks = list(
            dict(
                sorted(self.var_dict.items(), key=lambda item: item[1])
            ).keys()
        )[:self.top_k]

    def __call__(self, inp_curr):
        if inp_curr[0] not in self.first_tasks:
            return False

        possible_curr = [-1] * len(self.var_dict)
        possible_curr[0] = [inp_curr[0]]
        ptr_start, ptr_end = 1, len(possible_curr) - 1
        current_idx = 0
        while True:
            if ptr_end == ptr_start:
                for task in possible_curr[current_idx]:
                    if possible_curr[ptr_end] == -1:
                        possible_curr[ptr_start] = self.sorted_adjacency_list[task][-2:]
                    else:
                        possible_curr[ptr_start] += self.sorted_adjacency_list[task][-2:]
                break

            if -1 not in possible_curr:
                break

            for task in possible_curr[current_idx]:
                if possible_curr[ptr_end] == -1:
                    possible_curr[ptr_end] = self.sorted_adjacency_list[task][0:2]
                else:
                    possible_curr[ptr_end] += self.sorted_adjacency_list[task][0:2]
                if possible_curr[ptr_start] == -1:
                    possible_curr[ptr_start] = self.sorted_adjacency_list[task][-2:]
                else:
                    possible_curr[ptr_start] += self.sorted_adjacency_list[task][-2:]

            ptr_end -= 1
            ptr_start += 1
            current_idx += 1

        for idx, el in enumerate(possible_curr):
            possible_curr[idx] = list(set(el))

        for el_inp, el_possible in zip(inp_curr, possible_curr):
            if el_inp not in [x[1] for x in el_possible]:
                return False

        return True


class ScoreCurriculum:
    def __init__(self, var_dict, sorted_adjacency_list, stype='float'):
        self.var_dict = var_dict
        self.sorted_adjacency_list = sorted_adjacency_list
        self.stype = stype

        if self.stype == 'int':
            for _, value in self.sorted_adjacency_list.items():
                for idx, val in enumerate(value):
                    value[idx] = value[idx][1]
        elif self.stype == 'float':
            for key, value in self.sorted_adjacency_list.items():
                self.sorted_adjacency_list[key] = {t[1]: t[0] for t in value}

    def __call__(self, inp_curr, stype='float'):
        len_curr = len(inp_curr)

        score = 0
        num_el_covered = 0
        sorted_first_task = list(
            dict(
                sorted(self.var_dict.items(), key=lambda item: item[1])
            ).keys()
        )
        if self.stype == 'int':
            score += len_curr - sorted_first_task.index(inp_curr[0])
        elif self.stype == 'float':
            score += 1 - self.var_dict[inp_curr[0]]
        num_el_covered += 1

        curr_task = inp_curr[0]
        for idx, el in enumerate(inp_curr[1:]):
            if self.stype == 'int':
                score += (len_curr - 1) - self.sorted_adjacency_list[curr_task].index(
                    inp_curr[len_curr - (idx + 1)])
                try:
                    score += self.sorted_adjacency_list[curr_task].index(
                        el) + 1
                except IndexError:
                    pass
            if self.stype == 'float':
                score += 1 - \
                    self.sorted_adjacency_list[curr_task][inp_curr[len_curr - (
                        idx + 1)]]
                try:
                    score += self.sorted_adjacency_list[curr_task][el]
                except IndexError:
                    pass
            num_el_covered += 2
            curr_task = inp_curr[idx + 1]

            if num_el_covered >= len_curr:
                break

        return score
