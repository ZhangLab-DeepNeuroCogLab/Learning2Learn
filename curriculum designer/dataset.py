import os
import random
from PIL import Image
from typing import Optional, Callable, List
import numpy as np
from utils import Noise as n
from inetmapping import *

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

seed = 0
random.seed(seed)


class CustomStyleNet(Dataset):
    """ 10-stylized classes derived from 10-class ImageNet https://image-net.org/index.php

    styles:
    https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer/tree/master/experiments/images/21styles

    Args:
        root (string): Root directory where ImageNet and its stylized variants are stored
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set
        style (string, optional): If specified, selects a stylized version of ImageNet
        transform (callable, optional): A function/transform that accepts a JPEG/PIL and
            transforms it
        spec_target: If specified, returns data only belonging to the listed targets
        samples_per_class: (int, optional): number of samples per class
    """

    base_name = 'ImageNet'
    url = 'https://image-net.org/index.php'
    style_dict = {
        'candy': 0, 'mosaic_ducks_massimo': 1, 'pencil': 2, 'seated-nude': 3,
        'shipwreck': 4, 'starry_night': 5, 'stars2': 6, 'strip': 7,
        'the_scream': 8, 'wave': 9
    }

    std_transform = {
        'train': transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        ),
        'eval': transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            style: str = None,
            transform: Optional[Callable] = None,
            spec_target: List[int] = None,
            download: bool = False,
            samples_per_class: int = None
    ):
        self.root = root
        self.train = train
        self.style = style
        self.transform = transform
        self.spec_target = spec_target
        self.download = download
        self.samples_per_class = samples_per_class

        main_dir = '{}/{}'.format(root, CustomImageNet.base_name)
        if self.train:
            dir_type = 'train'
        else:
            dir_type = 'val'

        self.data: Any = []
        self.targets = []

        for _style in CustomStyleNet.style_dict.keys():
            data = []
            targets = []
            style_label = CustomStyleNet.style_dict[_style]
            temp_root = '{}_{}/{}'.format(main_dir, _style, dir_type)

            if (
                    self.spec_target is not None
                    and style_label not in self.spec_target
            ):
                continue

            for _class in CustomImageNet.class_dict.keys():
                file_list = os.listdir('{}/{}'.format(temp_root, _class))

                for _file in file_list:
                    im = os.path.join(temp_root, _class, _file)
                    data.append(im)
                    targets.append(style_label)

            if self.samples_per_class is not None:
                data, targets = zip(*random.sample(list(zip(data, targets)), self.samples_per_class))
            self.data.extend(data)
            self.targets.extend(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, target = Image.open(self.data[idx]), self.targets[idx]

        if self.transform is not None:
            if len(np.asarray(image).shape) == 2:
                np_img = np.asarray(image)
                np_img = np.stack((np_img,) * 3, axis=-1)
                image = Image.fromarray(np_img)
            elif np.asarray(image).shape[2] > 3:
                image = image.convert('RGB')
            image = self.transform(image)

        return image, target


class CustomImageNet(Dataset):
    """ 10-class subset of ImageNet https://image-net.org/index.php

    class aggregation:
    https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/16-class-ImageNet
    /MSCOCO_to_ImageNet_category_mapping.txt

    Args:
        root (string): Root directory where ImageNet and its stylized variants are stored
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set
        style (string, optional): If specified, selects a stylized version of ImageNet
        transform (callable, optional): A function/transform that accepts a JPEG/PIL and
            transforms it
        spec_target: If specified, returns data only belonging to the listed targets
        return_style_labels: If True, returns a vector of targets for the corresponding style
        samples_per_class: (int, optional): number of samples per class
        samples_per_class: (int, optional): number of samples per class
    """

    base_name = 'ImageNet'
    url = 'https://image-net.org/index.php'
    class_dict = {
        'airplane': 0, 'car': 1, 'bird': 2, 'cat': 3,
        'elephant': 4, 'dog': 5, 'bottle': 6, 'knife': 7,
        'truck': 8, 'boat': 9
    }

    std_transform = {
        'train': transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        ),
        'eval': transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            style: str = None,
            transform: Optional[Callable] = None,
            spec_target: List[int] = None,
            return_style_labels: bool = False,
            download: bool = False,
            samples_per_class: int = None
    ):
        self.root = root
        self.train = train
        self.style = style
        self.transform = transform
        self.spec_target = spec_target
        self.return_style_labels = return_style_labels
        self.download = download
        self.samples_per_class = samples_per_class

        main_dir = '{}/{}'.format(root, CustomImageNet.base_name)
        if self.style is not None:
            main_dir = '{}_{}'.format(main_dir, self.style)
        if train:
            main_dir = '{}/train'.format(main_dir)
        else:
            main_dir = '{}/val'.format(main_dir)

        self.data: Any = []
        self.targets = []

        for _class in CustomImageNet.class_dict.keys():
            data = []
            targets = []
            file_list = os.listdir('{}/{}'.format(main_dir, _class))
            class_label = CustomImageNet.class_dict[_class]

            if (
                    self.spec_target is not None
                    and class_label not in self.spec_target
            ):
                continue

            for _file in file_list:
                im = os.path.join(main_dir, _class, _file)
                data.append(im)
                targets.append(class_label)

            if self.samples_per_class is not None:
                data, targets = zip(*random.sample(list(zip(data, targets)), self.samples_per_class))
            self.data.extend(data)
            self.targets.extend(targets)

        if self.return_style_labels:
            self.style_targets = [CustomStyleNet.style_dict[self.style]] * self.__len__()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, target = Image.open(self.data[idx]), self.targets[idx]

        if self.transform is not None:
            if len(np.asarray(image).shape) == 2:
                np_img = np.asarray(image)
                np_img = np.stack((np_img,) * 3, axis=-1)
                image = Image.fromarray(np_img)
            elif np.asarray(image).shape[2] > 3:
                image = image.convert('RGB')
            image = self.transform(image)

        if self.return_style_labels:
            style_target = self.style_targets[idx]
            return image, target, style_target
        return image, target


class CustomMNISTNet(Dataset):
    """
    Args:
        root (string): Root directory where ImageNet and its stylized variants are stored
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set
        transform (callable, optional): A function/transform that accepts a JPEG/PIL and
            transforms it
        spec_target: If specified, returns data only belonging to the listed targets
        samples_per_class: (int, optional): number of samples per class
    """

    std_transform = {
        'train': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.expand(3, -1, -1))
            ]
        ),
        'eval': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.expand(3, -1, -1))
            ]
        )
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            spec_target: List[int] = None,
            download: bool = False,
            samples_per_class: int = None
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.spec_target = spec_target
        self.download = download
        self.samples_per_class = samples_per_class

        self.data, self.targets = [], []
        dataset = datasets.MNIST(root=self.root, train=self.train, download=self.download)
        for target in self.spec_target:
            temp_dataset = [(x[0], x[1]) for x in dataset if x[1] == target]
            data, targets = [x[0] for x in temp_dataset], [x[1] for x in temp_dataset]
            data, targets = zip(*random.sample(list(zip(data, targets)), self.samples_per_class))
            self.data.extend(data), self.targets.extend(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class CustomFashionMNISTNet(Dataset):
    """
    Args:
        root (string): Root directory where ImageNet and its stylized variants are stored
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set
        transform (callable, optional): A function/transform that accepts a JPEG/PIL and
            transforms it
        spec_target: If specified, returns data only belonging to the listed targets
        samples_per_class: (int, optional): number of samples per class
    """

    std_transform = {
        'train': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.expand(3, -1, -1))
            ]
        ),
        'eval': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.expand(3, -1, -1))
            ]
        )
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            spec_target: List[int] = None,
            download: bool = False,
            samples_per_class: int = None
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.spec_target = spec_target
        self.download = download
        self.samples_per_class = samples_per_class

        self.data, self.targets = [], []
        dataset = datasets.FashionMNIST(root=self.root, train=self.train, download=self.download)
        for target in self.spec_target:
            temp_dataset = [(x[0], x[1]) for x in dataset if x[1] == target]
            data, targets = [x[0] for x in temp_dataset], [x[1] for x in temp_dataset]
            data, targets = zip(*random.sample(list(zip(data, targets)), self.samples_per_class))
            self.data.extend(data), self.targets.extend(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class CustomCIFAR10Net(Dataset):
    """
    Args:
        root (string): Root directory where ImageNet and its stylized variants are stored
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set
        transform (callable, optional): A function/transform that accepts a JPEG/PIL and
            transforms it
        spec_target: If specified, returns data only belonging to the listed targets
        samples_per_class: (int, optional): number of samples per class
    """

    std_transform = {
        'train': CustomImageNet.std_transform['train'],
        'eval': CustomImageNet.std_transform['eval']
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            spec_target: List[int] = None,
            download: bool = False,
            samples_per_class: int = None
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.spec_target = spec_target
        self.download = download
        self.samples_per_class = samples_per_class

        self.data, self.targets = [], []
        dataset = datasets.CIFAR10(root=self.root, train=self.train, download=self.download)
        for target in self.spec_target:
            temp_dataset = [(x[0], x[1]) for x in dataset if x[1] == target]
            data, targets = [x[0] for x in temp_dataset], [x[1] for x in temp_dataset]
            data, targets = zip(*random.sample(list(zip(data, targets)), self.samples_per_class))
            self.data.extend(data), self.targets.extend(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class CustomNovelNet(Dataset):
    """
    Args:
        root (string): Root directory where ImageNet and its stylized variants are stored
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set
        transform (callable, optional): A function/transform that accepts a JPEG/PIL and
            transforms it
        spec_target: If specified, returns data only belonging to the listed targets
        samples_per_class: (int, optional): number of samples per class
    """

    base_name = 'NovelNet'
    class_dict = {
        'fa1': 0, 'fa2': 1, 'fb1': 2, 'fb3': 3,
        'fc1': 4
    }

    std_transform = {
        'train': transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        ),
        'eval': transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            spec_target: List[int] = None,
            download: bool = False,
            samples_per_class: int = None
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.spec_target = spec_target
        self.download = download
        self.samples_per_class = samples_per_class

        main_dir = os.path.join(root, CustomNovelNet.base_name)
        if train:
            main_dir = os.path.join(main_dir, 'train')
        else:
            main_dir = os.path.join(main_dir, 'val')

        self.data: Any = []
        self.targets = []

        for _class in CustomNovelNet.class_dict.keys():
            data, targets = [], []
            file_list = os.listdir(os.path.join(main_dir, _class))
            class_label = CustomNovelNet.class_dict[_class]

            if (
                self.spec_target is not None
                and class_label not in self.spec_target
            ):
                continue

            for _file in file_list:
                im = os.path.join(main_dir, _class, _file)
                data.append(im)
                targets.append(class_label)
            if self.samples_per_class is not None:
                data, targets = zip(*random.sample(list(zip(data, targets)), self.samples_per_class))
            self.data.extend(data)
            self.targets.extend(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, target = Image.open(self.data[idx]), self.targets[idx]

        if np.asarray(image).shape[2] > 3:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class ImageNet2012(Dataset):
    """
    Args:
        root_dir (string): Root directory where ImageNet and its stylized variants are stored
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set
        transform (callable, optional): A function/transform that accepts a JPEG/PIL and
            transforms it
        spec_target: If specified, returns data only belonging to the listed targets
        samples_per_class: (int, optional): number of samples per class
    """

    base_name = 'ImageNet'
    url = 'https://image-net.org/index.php'

    std_transform = {
        'train': transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        ),
        'eval': transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            spec_target: List[int] = None,
            samples_per_class: int = None,
            download: bool = False
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.spec_target = spec_target
        self.samples_per_class = samples_per_class
        self.download = download

        main_dir = os.path.join(self.root, ImageNet2012.base_name)
        if train:
            main_dir = '{}/train/train'.format(main_dir)
        else:
            main_dir = '{}/val/val'.format(main_dir)

        self.data: Any = []
        self.targets = []

        dir_list = os.listdir(main_dir)
        for _class in dir_list:
            if _class in exclusion_mapping:
                continue

            try:
                file_list = os.listdir('{}/{}'.format(main_dir, _class))
            except NotADirectoryError:
                continue
            try:
                class_label = imagenet_mapping[_class]
            except KeyError:
                continue

            if (
                self.spec_target is not None
                and class_label not in self.spec_target
            ):
                continue

            data = []
            targets = []
            for _file in file_list:
                im = os.path.join(main_dir, _class, _file)
                data.append(im)
                targets.append(class_label)

            if self.samples_per_class is not None:
                data, targets = zip(*random.sample(list(zip(data, targets)), self.samples_per_class))
            self.data.extend(data)
            self.targets.extend(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, target = Image.open(self.data[idx]), self.targets[idx]

        if self.transform is not None:
            if len(np.asarray(image).shape) == 2:
                np_img = np.asarray(image)
                np_img = np.stack((np_img,) * 3, axis=-1)
                image = Image.fromarray(np_img)
            elif np.asarray(image).shape[2] > 3:
                image = image.convert('RGB')
            image = self.transform(image)

        return image, target