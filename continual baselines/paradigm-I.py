import pickle
import json

import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from avalanche.benchmarks.generators import nc_benchmark
from avalanche.training.strategies import Naive, EWC, LwF, SynapticIntelligence
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin, ReplayPlugin
from avalanche.models import SimpleCNN
from avalanche.logging import (
    InteractiveLogger,
    TensorboardLogger
)
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    confusion_matrix_metrics,
    accuracy_metrics,
    loss_metrics
)
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin

from utils.argument_parser import P1ArgumentParser
from utils.data_utils import DataUtils
from utils.custom_datasets import (
    CustomImageNet, CustomStyleNet, CustomParadigmDataset,
    CustomNovelNet, ImageNet2012
)
from utils.log_utils import AvalancheParser
from utils import config
from utils.eval_utils import EvalF
from utils.curriculum_utils import (
    save_res, load_res, ImageNetCurriculum, StyleNetCurriculum,
    CIFAR10Curriculum, MNISTCurriculum, FashionMNISTCurriculum,
    NovelNetCurriculum, FashionMNIST5Curriculum, MNIST5Curriculum,
    ImageNet5Curriculum, CIFAR105Curriculum
)
from utils.graph_utils import plot_multiline
from utils.custom_transforms import Grayscale
from models.SqueezeNet import SqueezeNet, SqueezeNet_random
from models.TestNet import TestNet
from models.ResNet import ResNet, ResNet32_CIFAR100
from seed import seed_reproduce

parser = P1ArgumentParser()
args = parser.run()

# config
class_dict = {v: k for k, v in config.class_dict.items()}
seed = config.seed
alt_seed = config.alt_seed
data_dir = config.data_dir

# args
tr_batch_size = args.train_batch_size
ts_batch_size = args.eval_batch_size
epochs = args.epochs
lr = args.lr
data_path = args.data_path
log_dir = args.log_dir
save_path = args.save_path
model_name = args.model
load_weights = args.load_weights
pretrained = args.pretrained
dataset = args.dataset
num_workers = args.num_workers
logging_only = args.logging_only
map_type = args.map_type
_strategy = args.strategy
ewc_lambda = args.ewc_lambda
snet_layer = args.snet_layer
n_experiences = args.num_experiences
n_permutations = args.num_permutations
schedule = args.schedule
milestone = args.milestone
n_bins = args.n_bins
percentile = args.percentile
strategy_comp = args.strategy_comparison
lwf_alpha = args.lwf_alpha
lwf_temperature = args.lwf_temperature
si_lambda = args.si_lambda
si_eps = args.si_eps
initialization = args.initialization
num_subset_classes = args.num_subset_classes
no_avg_strategy = args.no_avg_strategy

# reproducibility
if _strategy[-3:] == 'alt':
    seed = alt_seed
seed_reproduce(seed)
torch.backends.cuda.benchmark = True

# additional args
device = torch.device(
    f"cuda:{args.cuda}"
    if torch.cuda.is_available() and
       args.cuda >= 0 else "cpu"
)
num_runs = args.num_runs
seeds = []
if num_runs > 1:
    seeds = [i for i in range(num_runs)]
else:
    seeds = [seed]
num_load, num_actual = 1000, 10
try:
    checkpoint_sn = torch.load(
        '{}/{}/ImageNet100.pth.tar'.format(config.weights_dir, 'SqueezeNet')
    )
except Exception as e:
    if logging_only:
        pass
    else:
        print(e)

download_data = False

class_order = args.class_order
per_exp_classes = None
if class_order is not None:
    permutations = [class_order]
else:
    if dataset == 'NovelNet':
        all_perms = DataUtils.get_permutations([[0], [1], [2], [3], [4]])
        permutations, limit_flag = [], 0
        for i in range(len(all_perms)):
            limit_flag = 0
            for j in range(i + 1, len(all_perms)):
                perm_1, perm_2 = all_perms[i], all_perms[j]
                if perm_1[1] == perm_2[0] and perm_1[0] == perm_2[1]:
                    limit_flag = 1
                    break
            if limit_flag == 0:
                permutations.append(all_perms[i])

        if n_experiences == 4:
            per_exp_classes = {0: 2}
    elif dataset == 'ImageNet2012':
        permutations = DataUtils.get_random_permutations(
            [config.inet_classes[x:x + 9] for x in range(0, len(config.inet_classes), 9)],
            num=n_permutations
        )
    elif num_subset_classes is not None:
        permutations = DataUtils.get_permutations([[i] for i in range(num_subset_classes)])
    else:
        permutations = DataUtils.get_permutations([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])

curriculum_util = None
if dataset == 'ImageNetTiny':
    tr_transform, ts_transform = CustomImageNet.std_transform['train'], CustomImageNet.std_transform['eval']
    dataset = CustomImageNet
    curriculum_util = ImageNetCurriculum
    if num_subset_classes == 5:
        curriculum_util = ImageNet5Curriculum
elif dataset == 'StyleNet':
    tr_transform, ts_transform = CustomStyleNet.std_transform['train'], CustomStyleNet.std_transform['eval']
    dataset = CustomStyleNet
    curriculum_util = StyleNetCurriculum
elif dataset == 'ParadigmDataset':
    tr_transform, ts_transform = CustomImageNet.std_transform['train'], CustomImageNet.std_transform['eval']
    dataset = CustomParadigmDataset
    curriculum_util = ImageNetCurriculum
elif dataset == 'CIFAR10':
    tr_transform, ts_transform = CustomImageNet.std_transform['train'], CustomImageNet.std_transform['eval']
    dataset = datasets.CIFAR10
    curriculum_util = CIFAR10Curriculum
    if num_subset_classes == 5:
        curriculum_util = CIFAR105Curriculum
    download_data = True
elif dataset == 'MNIST':
    mnist_transform = Grayscale.mnist_transform()
    tr_transform, ts_transform = mnist_transform, mnist_transform
    dataset = datasets.MNIST
    curriculum_util = MNISTCurriculum
    if num_subset_classes == 5:
        curriculum_util = MNIST5Curriculum
    download_data = True
elif dataset == 'FashionMNIST':
    fmnist_transform = Grayscale.fmnist_transform()
    tr_transform, ts_transform = fmnist_transform, fmnist_transform
    dataset = datasets.FashionMNIST
    curriculum_util = FashionMNISTCurriculum
    if num_subset_classes == 5:
        curriculum_util = FashionMNIST5Curriculum
    download_data = True
elif dataset == 'NovelNet':  # bs = 4, lr = 1e-4
    tr_transform, ts_transform = CustomNovelNet.std_transform['train'], CustomNovelNet.std_transform['eval']
    dataset = CustomNovelNet
    curriculum_util = NovelNetCurriculum
    num_actual = 5
elif dataset == 'ImageNet2012':
    tr_transform, ts_transform = ImageNet2012.std_transform['train'], ImageNet2012.std_transform['eval']
    dataset = ImageNet2012
    num_actual = 900
    data_dir = '/data'
else:
    tr_transform, ts_transform = None, None

if __name__ == '__main__':
    t1_accuracy_dict, avg_accuracy_dict = dict(), dict()
    if strategy_comp:
        avg_accuracy_dict_alt, t1_accuracy_dict_alt = [], []
    else:
        avg_accuracy_dict_alt, t1_accuracy_dict_alt = None, None
    if not logging_only:
        for seed, run in zip(seeds, range(num_runs)):
            print("RUN: {}".format(run))
            for permutation in permutations:
                print("PERMUTATION: {}".format(permutation))

                tr, ts = (
                    dataset(root=data_dir, transform=tr_transform, download=download_data),
                    dataset(root=data_dir, train=False, transform=ts_transform, download=download_data)
                )

                scenario = nc_benchmark(
                    tr, ts,
                    n_experiences=n_experiences,
                    shuffle=False,
                    fixed_class_order=permutation,
                    seed=seed,
                    task_labels=False,
                    per_exp_classes=per_exp_classes
                )

                tr_stream, ts_stream = scenario.train_stream, scenario.test_stream

                # model-select
                if args.model == 'SqueezeNet':
                    net = SqueezeNet(pretrained=pretrained)
                    model = net()
                elif args.model == 'ResNet':
                    net = ResNet(pretrained=pretrained, num_classes=num_actual)
                    model = net()
                elif args.model == 'ResNet32_CIFAR100':
                    net = ResNet32_CIFAR100(pretrained=pretrained, num_classes=num_actual)
                    model = net()
                elif args.model == 'SqueezeNet_random':
                    net = SqueezeNet_random(pretrained=False, num_classes=num_actual)
                    model = net()
                elif args.model == 'TestNet':
                    model = TestNet()
                elif args.model == 'SimpleCNN':
                    model = SimpleCNN(num_classes=num_actual)
                else:
                    model = None

                if model is not None:
                    model = nn.DataParallel(model)
                    model = model.to(device)
                else:
                    raise Exception('Error 404: model not found')

                if pretrained:
                    load_weights = False
                if load_weights and args.model == 'SqueezeNet':
                    model.module.classifier[1] = nn.Conv2d(512, num_load, kernel_size=(1, 1), stride=(1, 1))
                    model.load_state_dict(checkpoint_sn['state_dict'])

                    seed_reproduce(seed)
                    if num_actual != num_load:
                        model.module.classifier[1] = nn.Conv2d(512, num_actual, kernel_size=(1, 1), stride=(1, 1)).to(device)
                        if initialization == 'xavier':
                            torch.nn.init.xavier_uniform(model.module.classifier[1].weight)
                        elif initialization == 'gaussian':
                            torch.nn.init.normal_(model.module.classifier[1].weight)
                elif load_weights == False and args.model == 'SqueezeNet':
                    model_name = 'SqueezeNetFull'

                # logger
                interactive_logger = InteractiveLogger()
                tensorboard_logger = TensorboardLogger()
                eval_plugin = EvaluationPlugin(
                    accuracy_metrics(
                        minibatch=True, epoch=True, epoch_running=True,
                        experience=True, stream=True
                    ),
                    loss_metrics(
                        minibatch=True, epoch=True, epoch_running=True,
                        experience=True, stream=True
                    ),
                    forgetting_metrics(
                        experience=True, stream=True
                    ),
                    confusion_matrix_metrics(
                        stream=True
                    ),
                    loggers=[interactive_logger, tensorboard_logger],
                    benchmark=scenario
                )

                # optimization
                optimizer = Adam(model.parameters(), lr=lr)
                if schedule:
                    scheduler = LRSchedulerPlugin(
                        MultiStepLR(optimizer, milestone, gamma=0.2)
                    )
                    early_stopper = EarlyStoppingPlugin(
                        patience=epochs // 4,
                        val_stream_name="tr_stream",
                        metric_name="Top1_Acc_Epoch"
                    )
                    plugins = [scheduler, early_stopper]
                else:
                    plugins = None
                loss_fn = CrossEntropyLoss()

                # replay buffer
                if _strategy == 'replay':
                    mem_size = int((1/10) * len(tr))
                    if plugins is not None:
                        plugins.append(ReplayPlugin(mem_size=mem_size))
                    else:
                        plugins = [ReplayPlugin(mem_size=mem_size)]

                if _strategy in ['naive', 'replay']:
                    strategy = Naive(
                        model=model.module, optimizer=optimizer, criterion=loss_fn,
                        train_mb_size=tr_batch_size, train_epochs=epochs, eval_mb_size=ts_batch_size,
                        device=device, evaluator=eval_plugin, plugins=plugins
                    )
                elif _strategy in ['ewc', 'ewc_alt']:
                    strategy = EWC(
                        model=model.module, optimizer=optimizer, criterion=loss_fn,
                        ewc_lambda=ewc_lambda, mode='separate', train_mb_size=tr_batch_size,
                        train_epochs=epochs, eval_mb_size=ts_batch_size, device=device,
                        evaluator=eval_plugin, plugins=plugins
                    )
                elif _strategy == 'lwf':
                    strategy = LwF(
                        model=model.module, optimizer=optimizer, criterion=loss_fn,
                        alpha=lwf_alpha, temperature=lwf_temperature, train_mb_size=tr_batch_size,
                        train_epochs=epochs, eval_mb_size=ts_batch_size, device=device,
                        evaluator=eval_plugin, plugins=plugins
                    )
                elif _strategy == 'si':
                    strategy = SynapticIntelligence(
                        model=model.module, optimizer=optimizer, criterion=loss_fn,
                        si_lambda=si_lambda, train_mb_size=tr_batch_size, train_epochs=epochs,
                        eval_mb_size=ts_batch_size, device=device, evaluator=eval_plugin,
                        plugins=plugins, eps=si_eps
                    )
                else:
                    raise Exception('Strategy not defined')

                results = []
                for idx, train_task in enumerate(tr_stream):
                    print("classes in this experience: {}".format(train_task.classes_in_this_experience))
                    print("classes in the previous experience: {}".format(train_task.previous_classes))
                    print("classes seen so far: {}".format(train_task.classes_seen_so_far))
                    print("This experience contains {} patterns".format(len(train_task.dataset)))
                    strategy.train(train_task, num_workers=num_workers)
                    if dataset == ImageNet2012:
                        eval_exp = [e for e in ts_stream][:idx+1]
                        results.append(strategy.eval(eval_exp, num_workers=num_workers))
                    else:
                        results.append(strategy.eval(ts_stream, num_workers=num_workers))
                dbfile = open('{}/{}_{}_{}_{}'.format(log_dir, _strategy, dataset, permutation, run), 'wb')
                pickle.dump(results, dbfile)
                dbfile.close()
                torch.cuda.empty_cache()

                # log extraction
                log_parser = AvalancheParser(dir=log_dir, exp_type=_strategy, dataset=dataset,
                                             permutation=permutation, run=run, num_tasks=n_experiences)
                t1_accuracy_list, avg_accuracy_list = log_parser()
                t1_accuracy_dict["{}".format(permutation)] = t1_accuracy_list
                avg_accuracy_dict["{}".format(permutation)] = avg_accuracy_list

                print("[END OF PERMUTATION]\n")

            if len(permutations) == 1:
                print('t1_accuracy_dict: {}\n'.format(t1_accuracy_dict))
                print('avg_accuracy_dict: {}\n'.format(avg_accuracy_dict))
                try:
                    # noinspection PyUnboundLocalVariable
                    print('DELTA (proportional to forgetting): {}'.format(
                        avg_accuracy_list[0] - avg_accuracy_list[-1]
                    ))
                except Exception as e:
                    pass
                quit()

            _dataset = dataset
            if epochs > 1:
                _dataset = "{}_epochs.{}".format(_dataset, epochs)
            if lr != 1e-3:
                _dataset = "{}_lr.{}".format(_dataset, lr)
            if initialization != 'uniform':
                _dataset = "{}_init.{}".format(_dataset, initialization)
            if model_name != 'SqueezeNet':
                _dataset = "{}_model.{}".format(_dataset, model_name)
            if num_subset_classes is not None:
                _dataset = '{}_classes.{}'.format(_dataset, num_subset_classes)
            save_res(t1_accuracy_dict, '{}-{}-t1_accuracy_dict_{}'.format(run, _strategy, _dataset))
            save_res(avg_accuracy_dict, '{}-{}-avg_accuracy_dict_{}'.format(run, _strategy, _dataset))

    all_strategy = ['naive', 'ewc', 'lwf']
    avg_accuracy_dicts_alt, t1_accuracy_dicts_alt  = [[] for _ in range(len(all_strategy))], [[] for _ in range(len(all_strategy))]
    avg_accuracy_dicts, t1_accuracy_dicts = [], []
    _dataset = dataset
    if epochs > 1:
        _dataset = "{}_epochs.{}".format(_dataset, epochs)
    if lr != 1e-3:
        _dataset = "{}_lr.{}".format(_dataset, lr)
    if initialization != 'uniform':
        _dataset = "{}_init.{}".format(_dataset, initialization)
    if pretrained and args.model == 'SqueezeNet':
        model_name = 'SqueezeNetFull'
    if model_name != 'SqueezeNet':
        _dataset = "{}_model.{}".format(_dataset, model_name)
    if num_subset_classes is not None:
        _dataset = '{}_classes.{}'.format(_dataset, num_subset_classes)
    for run in range(num_runs):
        if strategy_comp:
            for idx, strat in enumerate(all_strategy):
                try:
                    if no_avg_strategy:
                        avg_accuracy_dict_alt.append(
                            load_res('{}-{}-avg_accuracy_dict_{}'.format(run, strat, _dataset))
                        )
                        t1_accuracy_dict_alt.append(
                            load_res('{}-{}-t1_accuracy_dict_{}'.format(run, strat, _dataset))
                        )
                    else:
                        avg_accuracy_dicts_alt[idx].append(
                            load_res('{}-{}-avg_accuracy_dict_{}'.format(run, strat, _dataset))
                        )
                        t1_accuracy_dicts_alt[idx].append(
                            load_res('{}-{}-avg_accuracy_dict_{}'.format(run, strat, _dataset))
                        )
                except Exception:
                    avg_accuracy_dicts_alt[idx].append(
                        load_res('{}-avg_accuracy_dict_{}'.format(strat, _dataset))
                    )
                    t1_accuracy_dicts_alt[idx].append(
                        load_res('{}-t1_accuracy_dict_{}'.format(strat, _dataset))
                    )

        try:
            t1_accuracy_dict = load_res('{}-{}-t1_accuracy_dict_{}'.format(run, _strategy, _dataset))
            avg_accuracy_dict = load_res('{}-{}-avg_accuracy_dict_{}'.format(run, _strategy, _dataset))
            avg_accuracy_dicts.append(avg_accuracy_dict)
            t1_accuracy_dicts.append(t1_accuracy_dict)
        except Exception:
            t1_accuracy_dict = load_res('{}-t1_accuracy_dict_{}'.format(_strategy, _dataset))
            avg_accuracy_dict = load_res('{}-avg_accuracy_dict_{}'.format(_strategy, _dataset))
            avg_accuracy_dicts.append(avg_accuracy_dict)
            t1_accuracy_dicts.append(t1_accuracy_dict)
            break

    t1_accuracy_dict, avg_accuracy_dict = (
        DataUtils.avg_across_dicts(t1_accuracy_dicts),
        DataUtils.avg_across_dicts(avg_accuracy_dicts)
    )
    if strategy_comp and _strategy != 'replay':
        if not no_avg_strategy:
            avg_accuracy_dict_alt = [
                DataUtils.avg_across_dicts(dict_alt) for dict_alt in avg_accuracy_dicts_alt if dict_alt
            ]
            t1_accuracy_dict_alt = [
                DataUtils.avg_across_dicts(dict_alt) for dict_alt in t1_accuracy_dicts_alt if dict_alt
            ]
    if args.dataset == 'NovelNet':
        avg_accuracy_dict_cp = {k: v for k, v in avg_accuracy_dict.items() if json.loads(k) in permutations}
        avg_accuracy_dict = avg_accuracy_dict_cp

        t1_accuracy_dict_cp = {k: v for k, v in t1_accuracy_dict.items() if json.loads(k) in permutations}
        t1_accuracy_dict = t1_accuracy_dict_cp

        if strategy_comp:
            for idx, _dict in enumerate(avg_accuracy_dict_alt):
                temp = {k: v for k, v in _dict.items() if json.loads(k) in permutations}
                avg_accuracy_dict_alt[idx] = temp
            for idx, _dict in enumerate(t1_accuracy_dict_alt):
                temp = {k: v for k, v in _dict.items() if json.loads(k) in permutations}
                t1_accuracy_dict_alt[idx] = temp


    if _strategy != 'replay':
        curriculum_finder = curriculum_util(avg_accuracy_dict, t1_accuracy_dict, "{}-curriculum_{}".format(_strategy, _dataset),
                                            layer=snet_layer, n_bins=n_bins, percentile=percentile,
                                            result_dict_2=avg_accuracy_dict_alt, t1_dict_2=t1_accuracy_dict_alt)
        curriculum_finder()

    plot_multiline(
        x=[i for i in range(n_experiences)],
        Y=list(t1_accuracy_dict.values()),
        x_label='Task',
        y_label='Accuracy',
        x_ticks=[i for i in range(n_experiences)],
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        title='Accuracy(Task-1)',
        map_type=map_type,
        save_loc='{}-t1_{}'.format(_strategy, _dataset)
    )
    
    plot_multiline(
        x=[i for i in range(n_experiences)],
        Y=list(avg_accuracy_dict.values()),
        x_label='Task',
        y_label='Accuracy',
        x_ticks=[i for i in range(n_experiences)],
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        map_type=map_type,
        title='Mean accuracy \n (Current + Previous tasks)',
        save_loc='{}-avg_{}'.format(_strategy, _dataset)
    )

    if _strategy == 'replay':
        replay_eval = EvalF(t1_accuracy_dict, avg_accuracy_dict)
        replay_eval()
