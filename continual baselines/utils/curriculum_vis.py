import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class CurriculumVis:
    def __init__(self, fscore, save_loc, alt_mapping):
        self.fscore = CurriculumVis.sort_dict(fscore)
        self.save_loc = save_loc
        self.alt_mapping = alt_mapping
        self.hops = 2
        if self.alt_mapping:
            self.hops=1

        self.curriculum_list = [list(curriculum) for curriculum in self.fscore.keys()]
        self.curriculum_list = [CurriculumVis.split_list(x, self.hops) for x in self.curriculum_list]


    @staticmethod
    def sort_dict(unsorted_dict):
        return dict(sorted(unsorted_dict.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def split_list(lst, item=2):
        return [lst[i: i + item] for i in range(0, len(lst), item)]

    def map_tasks(self):
        if not self.alt_mapping:
            mapping = {
                ('0', '1'): 0,
                ('2', '3'): 1,
                ('4', '5'): 2,
                ('6', '7'): 3,
                ('8', '9'): 4
            }
        else:
            mapping = {
                ('0',): 0,
                ('1',): 1,
                ('2',): 2,
                ('3',): 3,
                ('4',): 4
            }
        for idx, curriculum in enumerate(self.curriculum_list):
            self.curriculum_list[idx] = [mapping[tuple(x)] for x in curriculum]

    def __call__(self):
        self.map_tasks()
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(self.curriculum_list, cmap="PiYG", square=True, cbar_kws={'ticks': [i for i in range(5)], "shrink": 0.25})
        ax.set_yticks([])
        ax.set_xticks([])
        ax.figure.savefig('vis_plots/{}.png'.format(self.save_loc), dpi=150)
        plt.close()