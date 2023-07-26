import argparse


class ArgumentParser:
    def __init__(self):
        pass

    @staticmethod
    def run():
        pass


class P1ArgumentParser(ArgumentParser):
    def __init__(self):
        super().__init__()

    @staticmethod
    def run():
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=int, default=0,
                            help='select zero-indexed cuda device. -1 to use CPU.')
        parser.add_argument('--num_runs', type=int, default=1,
                            help='choose number of complete code run-through')
        parser.add_argument('--train_batch_size', type=int, default=64,
                            help='train batch size (default: 64)')
        parser.add_argument('--eval_batch_size', type=int, default=64,
                            help='eval batch size (default: 64)')
        parser.add_argument('--epochs', type=int, default=1,
                            help='train epochs (default: 1)')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate (default: 0.001)')
        parser.add_argument('--data_path', type=str, default='./dataset',
                            help='dataset root path (default: ./dataset)')
        parser.add_argument('--log_dir', type=str, default='./log',
                            help='tensorboard log dir (default: ./log)')
        parser.add_argument('--save_path', type=str, default='./weights',
                            help='model save loc (default: ./weights)')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='choose number of workers')
        parser.add_argument('--model', type=str, choices=['SqueezeNet', 'TestNet', 'SimpleCNN', 'ResNet'],
                            default='SqueezeNet', help='choose between different available models')
        parser.add_argument('--pretrained', type=bool, default=False,
                            help='if true, download model pretrained on full-ImageNet')
        parser.add_argument('--dataset', type=str,
                            choices=['ImageNetTiny', 'StyleNet', 'ParadigmDataset', 'CIFAR10', 'MNIST', 'FashionMNIST',
                                     'NovelNet', 'ImageNet2012'],
                            default='ParadigmDataset', help='choose between different available datasets')
        parser.add_argument('--class_order', nargs='+', type=int,
                            default=None, help='choose a specific class curriculum for the model')
        parser.add_argument('--logging_only', type=bool, default=False,
                            help='if true, run through logs only (default: False)')
        parser.add_argument('--map_type', type=str, choices=['slope', 'delta'],
                            default='slope', help='choose between slope or delta for graph-mapping')
        parser.add_argument('--load_weights', type=bool, default=True,
                            help='if true, loads weights for SqueezeNet trained on ImageNet100')
        parser.add_argument('--strategy', type=str, choices=['ewc', 'naive', 'ewc_alt', 'si', 'lwf'],
                            default='naive', help='selects a strategy')
        parser.add_argument('--ewc_lambda', type=float, default=0.4,
                            help='penalty hyperparameter for ewc')
        parser.add_argument('--snet_layer', type=int, choices=[3, 6, 9, 11, 12],
                            default=12, help='selects which layer to probe in SqueezeNet')
        parser.add_argument('--num_experiences', type=int, default=5,
                            help='selects the number of experiences')
        parser.add_argument('--num_permutations', type=int, default=None,
                            help='randomly select n permutations to run')
        parser.add_argument('--world_size', type=int, default=4,
                            metavar='N', help='num GPUs')
        parser.add_argument('--schedule', type=bool, default=False,
                            help='if True, enables LR Step Scheduling')
        parser.add_argument('--milestone', nargs='+', type=int,
                            default=None, help='mention milestone epochs for LR schedule')
        parser.add_argument('--n_bins', type=int, default=5,
                            help='number of bins')
        parser.add_argument('--percentile', type=int, default=75,
                            help='percentile threshold')
        parser.add_argument('--strategy_comparison', type=bool, default=False,
                            help='runs a string based similarity comp between strategies')
        parser.add_argument('--lwf_alpha', nargs="+", type=float,
                            default=[0, 0.5, 1.333, 2.25, 3.2], help="Penalty hyperparameter for LwF")
        parser.add_argument('--lwf_temperature', type=float, default=1,
                            help='temperature for softmax used in distillation')
        parser.add_argument('--si_lambda', nargs="+", type=float,
                            default=[0, 0.5, 1.333, 2.25, 3.2], help='lambda for synaptic intelligence')
        parser.add_argument('--si_eps', type=float, default=1e-7,
                            help='damping parameter for synaptic intelligence')
        parser.add_argument('--initialization', type=str, choices=['uniform', 'xavier', 'gaussian'],
                            default='uniform', help='select weight initialization method')
        parser.add_argument('--num_subset_classes', type=int, default=None,
                            help='select a subset of the original number of classes')
        parser.add_argument('--no_avg_strategy', type=bool, default=False,
                            help='if False, treats each run independently for strategy comp')
        args = parser.parse_args()

        return args


class P3ArgumentParser(ArgumentParser):
    def __init__(self):
        super().__init__()

    @staticmethod
    def run():
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=int, default=0,
                            help='select zero-indexed cuda device. -1 to use CPU.')
        parser.add_argument('--num_runs', type=int, default=1,
                            help='choose number of complete code run-through')
        parser.add_argument('--train_batch_size', type=int, default=64,
                            help='train batch size (default: 64)')
        parser.add_argument('--eval_batch_size', type=int, default=64,
                            help='eval batch size (default: 64)')
        parser.add_argument('--epochs', type=int, default=1,
                            help='train epochs (default: 1)')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate (default: 0.001)')
        parser.add_argument('--data_path', type=str, default='./dataset',
                            help='dataset root path (default: ./dataset)')
        parser.add_argument('--log_dir', type=str, default='./log',
                            help='tensorboard log dir (default: ./log)')
        parser.add_argument('--save_path', type=str, default='./weights',
                            help='model save loc (default: ./weights)')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='choose number of workers')
        parser.add_argument('--model', type=str, choices=['SqueezeNet', 'TestNet', 'SimpleCNN'],
                            default='SqueezeNet', help='choose between different available models')
        parser.add_argument('--pretrained', type=bool, default=False,
                            help='if true, download model pretrained on full-ImageNet')
        parser.add_argument('--dataset', type=str, choices=['ImageNet', 'NoiseNet'],
                            default='ImageNet', help='choose between different available datasets')
        parser.add_argument('--class_order', nargs='+', type=int,
                            default=None, help='choose a specific class curriculum for the model')
        parser.add_argument('--style_order', nargs='+', type=int,
                            default=None, help='choose a specific style curriculum for the model')
        parser.add_argument('--logging_only', type=bool, default=False,
                            help='if true, run through logs only (default: False)')
        parser.add_argument('--map_type', type=str, choices=['slope', 'delta'],
                            default='slope', help='choose between slope or delta for graph-mapping')
        parser.add_argument('--load_weights', type=bool, default=True,
                            help='if true, loads weights for SqueezeNet trained on ImageNet100')
        parser.add_argument('--strategy', type=str, choices=['ewc', 'naive'],
                            default='naive', help='selects a strategy')
        parser.add_argument('--ewc_lambda', type=float, default=0.4,
                            help='penalty hyperparameter for ewc')
        parser.add_argument('--style_subset', type=str, choices=['first', 'last'],
                            default=None, help='choose if we want to train on a subset of possible style permutations')
        args = parser.parse_args()

        return args


class ComparatorLSTMArgumentParser(ArgumentParser):
    def __init__(self):
        super().__init__()

    @staticmethod
    def run():
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=int, default=0,
                            help='select zero-indexed cuda device. -1 to use CPU.')
        parser.add_argument('--strategy', type=str, choices=['ewc', 'naive'],
                            default='naive', help='selects a strategy')
        parser.add_argument('--pretrained', type=bool, default=False,
                            help='if true, download model pretrained on full-ImageNet')
        parser.add_argument('--samples_per_class', type=int, default=250,
                            help='select the number of samples that will be averaged')
        parser.add_argument('--hidden_size', type=int, default=10,
                            help='hidden size for LSTM')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='number of stacked LSTMs')
        parser.add_argument('--train_batch_size', type=int, default=64,
                            help='train batch size (default: 64)')
        parser.add_argument('--eval_batch_size', type=int, default=64,
                            help='eval batch size (default: 64)')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate (default: 0.001)')
        parser.add_argument('--epochs', type=int, default=100,
                            help='train epochs (default: 1)')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='choose number of workers')
        parser.add_argument('--patience', type=int, default=20,
                            help='patience for LR Scheduling')
        parser.add_argument('--model_type', type=str, choices=['SqueezeNetLSTM', 'LSTMClassifier'],
                            default='SqueezeNetLSTM')
        args = parser.parse_args()

        return args


class ComparatorConv3DArgumentParser(ArgumentParser):
    def __int__(self):
        super().__init__()

    @staticmethod
    def run():
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=int, default=0,
                            help='select zero-indexed cuda device. -1 to use CPU.')
        parser.add_argument('--strategy', type=str, choices=['ewc', 'naive'],
                            default='naive', help='selects a strategy')
        parser.add_argument('--pretrained', type=bool, default=False,
                            help='if true, download model pretrained on full-ImageNet')
        parser.add_argument('--samples_per_class', type=int, default=1,
                            help='select the number of samples that will be averaged')
        parser.add_argument('--train_batch_size', type=int, default=64,
                            help='train batch size (default: 64)')
        parser.add_argument('--eval_batch_size', type=int, default=64,
                            help='eval batch size (default: 64)')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate (default: 0.001)')
        parser.add_argument('--epochs', type=int, default=100,
                            help='train epochs (default: 1)')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='choose number of workers')
        parser.add_argument('--patience', type=int, default=20,
                            help='patience for LR Scheduling')
        parser.add_argument('--model_type', type=str, choices=['SqueezeNetConv3D', 'ConvNet3DClassifier'],
                            default='SqueezeNetLSTM')
        args = parser.parse_args()

        return args