import numpy as np
import math
import time
import statistics
import pickle


class LogParser(object):
    def __init__(self, dir, exp_type, dataset, permutation, run, num_tasks=5):
        self.dir = dir
        self.exp_type = exp_type
        self.dataset = dataset
        self.permutation = permutation
        self.run = run
        self.num_tasks = num_tasks

    def load_data(self):
        pass

    def __call__(self):
        pass


class AvalancheParser(LogParser):
    def __init__(self, dir, exp_type, dataset, permutation, run, num_tasks=5):
        super().__init__(dir, exp_type, dataset, permutation, run, num_tasks)

    def load_data(self):
        dbfile = open("{}/{}_{}_{}_{}".format(self.dir, self.exp_type, self.dataset, self.permutation, self.run), 'rb')
        db = pickle.load(dbfile)
        dbfile.close()

        return db

    def __call__(self):
        log_file = self.load_data()

        t1_accuracy_list = []
        avg_accuracy_list = []

        for num_task in range(self.num_tasks):
            t1_accuracy = log_file[num_task]['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000']
            t1_accuracy_list.append(t1_accuracy)

            avg_accuracy = []
            for sub_task in range(num_task + 1):
                avg_accuracy.append(
                    log_file[num_task]["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00{}".format(sub_task)])
            avg_accuracy = statistics.mean(avg_accuracy)
            avg_accuracy_list.append(avg_accuracy)

        return t1_accuracy_list, avg_accuracy_list


class ProgressBar:
    def __init__(self, max_step=150, fill="#"):
        self.max_step = max_step
        self.fill = fill
        self.barLength = 20
        self.barInterval = 0
        self.prInterval = 0
        self.count = 0
        self.progress = 0
        self.barlen_smaller = True
        self.gen_bar_cfg()

    def gen_bar_cfg(self):
        if self.max_step >= self.barLength:
            self.barInterval = math.ceil(self.max_step / self.barLength)
        else:
            self.barlen_smaller = False
            self.barInterval = math.floor(self.barLength / self.max_step)
        self.prInterval = 100 / self.max_step

    def reset_bar(self):
        self.count = 0
        self.progress = 0

    def update_bar(self, step, head_data={'head': 10}, end_data={'end_1': 2.2, 'end_2': 1.0}, keep=False):
        head_str = "\r"
        end_str = " "
        process = ""
        if self.barlen_smaller:
            if step != 0 and step % self.barInterval == 0:
                self.count += 1
        else:
            self.count += self.barInterval
        self.progress += self.prInterval
        for key in head_data.keys():
            head_str = head_str + key + ": " + str(head_data[key]) + " "
        for key in end_data.keys():
            end_str = end_str + key + ": " + str(end_data[key]) + " "
        if step == self.max_step:
            process += head_str
            process += "[%3s%%]: [%-20s]" % (100.0, self.fill * self.barLength)
            process += end_str
            if not keep:
                process += "\n"
        else:
            process += head_str
            process += "[%3s%%]: [%-20s]" % (round(self.progress, 1), self.fill * self.count)
            process += end_str
        print(process, end='', flush=True)
