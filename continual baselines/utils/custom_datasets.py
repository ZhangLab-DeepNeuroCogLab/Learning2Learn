import os
import random
from PIL import Image
from typing import Optional, Callable, List
import numpy as np
from pyparsing import Any
from .noise_utils import Noise as n
from .dataconfig.inetmapping import imagenet_mapping, exclusion_mapping, class2idx
from .config import data_dir

import torch
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
                data, targets = zip(
                    *random.sample(list(zip(data, targets)), self.samples_per_class))
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
                data, targets = zip(
                    *random.sample(list(zip(data, targets)), self.samples_per_class))
            self.data.extend(data)
            self.targets.extend(targets)

        if self.return_style_labels:
            self.style_targets = [
                CustomStyleNet.style_dict[self.style]] * self.__len__()

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


class CustomParadigmDataset(Dataset):
    """
    Derived from the CustomImageNet and Custom StyleNet, enables a class-wise extraction of multiple styles

    Args:
        root (string): Root directory where ImageNet and its stylized variants are stored
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set
        transform (callable, optional): A function/transform that accepts a JPEG/PIL and
            transforms it
        spec_target: If specified, returns data only belonging to the listed targets
    """

    base_name = 'ImageNet'

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            spec_target: List[int] = None,
            download: bool = False
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.spec_target = spec_target
        self.download = download

        main_dir = '{}/{}'.format(root, CustomImageNet.base_name)
        styles = list(CustomStyleNet.style_dict.keys())
        if not train:
            styles.append('no style')

        self.data: Any = []
        self.targets = []

        for _style in styles:
            if _style != 'no style':
                iter_dir = '{}_{}'.format(main_dir, _style)
            else:
                iter_dir = '{}'.format(main_dir)

            if train:
                iter_dir = '{}/train'.format(iter_dir)
            else:
                iter_dir = '{}/val'.format(iter_dir)

            for _class in CustomImageNet.class_dict.keys():
                file_list = os.listdir('{}/{}'.format(iter_dir, _class))
                class_label = CustomImageNet.class_dict[_class]

                if (
                        self.spec_target is not None
                        and class_label not in self.spec_target
                ):
                    continue

                for _file in file_list:
                    im = os.path.join(iter_dir, _class, _file)
                    self.data.append(im)
                    self.targets.append(class_label)

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


class CustomNoiseNet(Dataset):
    """ dataset arranged specifically for the multitask setting of noise and style

    Args:
        root (string): Root directory where all data is stored
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set
        style (string, optional): specifies the style
        noise: specifies the noise function
        transform (callable, optional): A function/transform that accepts a JPEG/PIL and
            transforms it
    """

    base_name = 'ImageNetNS'
    style_dict = CustomStyleNet.style_dict
    noise_dict = {
        n.pixelate: 0, n.gaussian_blur: 1, n.contrast: 2, n.speckle_noise: 3,
        n.brightness: 4, n.defocus_blur: 5, n.saturate: 6, n.gaussian_noise: 7,
        n.impulse_noise: 8, n.shot_noise: 9
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
            style: str,
            noise: Callable,
            train: bool = True,
            transform: Optional[Callable] = None,
            download: bool = False
    ):
        self.root = root
        self.style = style
        self.noise = noise
        self.train = train
        self.transform = transform
        self.download = download

        main_dir = os.path.join(root, CustomNoiseNet.base_name)
        if train:
            main_dir = os.path.join(main_dir, 'train')
        else:
            main_dir = os.path.join(main_dir, 'val')
        main_dir = os.path.join(main_dir, style)

        self.data: Any = []
        self.style_targets = []
        self.noise_targets = []

        file_list = os.listdir(main_dir)
        style_label = CustomNoiseNet.style_dict[self.style]
        noise_label = CustomNoiseNet.noise_dict[self.noise]
        for _file in file_list:
            im = os.path.join(main_dir, _file)
            self.data.append(im)
            self.style_targets.append(style_label)
            self.noise_targets.append(noise_label)

    def __len__(self):
        return len(self.style_targets)

    def __getitem__(self, idx):
        image, style_target, noise_target = Image.open(
            self.data[idx]), self.style_targets[idx], self.noise_targets[idx]

        if len(np.asarray(image).shape) == 2:
            np_img = np.asarray(image)
            np_img = np.stack((np_img,) * 3, axis=-1)
            image = Image.fromarray(np_img)
        elif np.asarray(image).shape[2] > 3:
            image = image.convert('RGB')

        image = self.noise(image)
        if self.transform is not None:
            image = self.transform(image)

        return image, style_target, noise_target


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
            file_list = os.listdir(os.path.join(main_dir, _class))
            class_label = CustomNovelNet.class_dict[_class]

            if (
                self.spec_target is not None
                and class_label not in self.spec_target
            ):
                continue

            for _file in file_list:
                im = os.path.join(main_dir, _class, _file)
                self.data.append(im)
                self.targets.append(class_label)

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
                class_label = class2idx[class_label]
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
                data, targets = zip(
                    *random.sample(list(zip(data, targets)), self.samples_per_class))
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


class ImageNetSC(Dataset):
    """ dataset arranged specifically to study categorical arrangement
    Args:
        root_dir (string): Root directory where ImageNet and its stylized variants are stored
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set
        transform (callable, optional): A function/transform that accepts a JPEG/PIL and
            transforms it
        spec_target: If specified, returns data only belonging to the listed targets
        samples_per_class: (int, optional): number of samples per class
    """

    base_name = 'ImageNet_SC'
    class_dict = {
        'n02091244': 0, 'n02091134': 1, 'n02093859': 2, 'n02099429': 3, 
        'n02106662': 4, 'n02105056': 5, 'n02701002': 6, 'n03345487': 7,
        'n03417042': 8, 'n04285008': 9, 'n04037443': 10, 'n03425413': 11
    }
    category_dict = {
        'n02091244': 0, 'n02093859': 0, 'n02106662': 0, 'n02091134': 0,
        'n02099429': 0, 'n02105056': 0,
        'n02701002': 1, 'n03417042': 1, 'n04037443': 1, 'n03345487': 1,
        'n04285008': 1, 'n03425413': 1

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
            samples_per_class: int = None,
            download: bool = False,
            category: bool = False
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.spec_target = spec_target
        self.samples_per_class = samples_per_class
        self.download = download
        self.category = category

        main_dir = os.path.join(self.root, ImageNetSC.base_name)
        if train:
            main_dir = '{}/train/'.format(main_dir)
        else:
            main_dir = '{}/val/'.format(main_dir)

        self.data: Any = []
        self.targets = []

        if not self.category:
            for _class in ImageNetSC.class_dict.keys():
                file_list = os.listdir(os.path.join(main_dir, _class))
                class_label = ImageNetSC.class_dict[_class]

                if (
                    self.spec_target is not None
                    and class_label not in self.spec_target
                ):
                    continue

                for _file in file_list:
                    im = os.path.join(main_dir, _class, _file)
                    self.data.append(im)
                    self.targets.append(class_label)
        else:
            for _class in ImageNetSC.category_dict.keys():
                file_list = os.listdir(os.path.join(main_dir, _class))
                class_label = ImageNetSC.category_dict[_class]

                if (
                    self.spec_target is not None
                    and class_label not in self.spec_target
                ):
                    continue

                for _file in file_list:
                    im = os.path.join(main_dir, _class, _file)
                    self.data.append(im)
                    self.targets.append(class_label)
            
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
        dataset = datasets.MNIST(
            root=self.root, train=self.train, download=self.download)
        for target in self.spec_target:
            temp_dataset = [(x[0], x[1]) for x in dataset if x[1] == target]
            data, targets = [x[0] for x in temp_dataset], [x[1]
                                                           for x in temp_dataset]
            data, targets = zip(
                *random.sample(list(zip(data, targets)), self.samples_per_class))
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
        dataset = datasets.FashionMNIST(
            root=self.root, train=self.train, download=self.download)
        for target in self.spec_target:
            temp_dataset = [(x[0], x[1]) for x in dataset if x[1] == target]
            data, targets = [x[0] for x in temp_dataset], [x[1]
                                                           for x in temp_dataset]
            data, targets = zip(
                *random.sample(list(zip(data, targets)), self.samples_per_class))
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
        dataset = datasets.CIFAR10(
            root=self.root, train=self.train, download=self.download)
        for target in self.spec_target:
            temp_dataset = [(x[0], x[1]) for x in dataset if x[1] == target]
            data, targets = [x[0] for x in temp_dataset], [x[1]
                                                           for x in temp_dataset]
            data, targets = zip(
                *random.sample(list(zip(data, targets)), self.samples_per_class))
            self.data.extend(data), self.targets.extend(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class ComparatorNet(Dataset):
    """
    Args:
        train (bool, optional): if True, returns 80% of the dataset at random
        samples_per_class: number of examples to sample for feature vector
        dataset: base dataset
        fscore: dict of curriculums and corresponding f-scores
        embedder: model for vector extraction
        hop_size: size of input for seq model
        device: cuda or cpu
    """

    def __init__(
            self,
            dataset,
            fscore,
            embedder,
            device,
            train: bool = None,
            samples_per_class: int = 250,
            hop_size: int = 10
    ):
        self.dataset = dataset
        self.fscore = fscore
        self.embedder = embedder
        self.device = device
        self.train = train
        self.samples_per_class = samples_per_class
        self.hop_size = hop_size

        class_groups = [[i] for i in range(10)]
        class_datasets = dict()
        for class_group in class_groups:
            _data = dataset(
                root=data_dir, spec_target=class_group, train=True, transform=dataset.std_transform['train'],
                samples_per_class=self.samples_per_class
            )
            _data = DataLoader(_data, batch_size=64,
                               shuffle=True, num_workers=0)
            mean_batch_vectors = []
            for batch in _data:
                mean_batch_vectors.append(torch.mean(
                    self.embedder(batch[0]), axis=0))
            class_datasets[class_group[0]] = torch.mean(torch.stack(
                mean_batch_vectors), axis=0).detach().cpu().numpy()

        self.data, self.targets = [], []
        self.curriculum_vectors = dict()
        for curriculum_i, score_i in fscore.items():
            for curriculum_j, score_j in fscore.items():
                if curriculum_j == curriculum_i:
                    continue
                if score_i >= score_j:
                    self.targets.append(1)
                else:
                    self.targets.append(0)

                datum, datum_i, datum_j = [], [], []
                if curriculum_i not in self.curriculum_vectors.keys():
                    for _class in curriculum_i:
                        datum_i += class_datasets[int(_class)].tolist()
                    self.curriculum_vectors[curriculum_i] = datum_i
                if curriculum_j not in self.curriculum_vectors.keys():
                    for _class in curriculum_j:
                        datum_j += class_datasets[int(_class)].tolist()
                    self.curriculum_vectors[curriculum_j] = datum_j
                datum = (curriculum_i, curriculum_j)
                self.data.append(datum)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]
        datum = self.curriculum_vectors[data[0]
                                        ] + self.curriculum_vectors[data[0]]
        datum = [datum[i: i + self.hop_size]
                 for i in range(0, len(datum), self.hop_size)]

        return torch.FloatTensor(datum), target


class ComparatorNet3D(Dataset):
    """
    Args:
        train (bool, optional): if True, returns 80% of the dataset at random
        samples_per_class: number of examples to sample for feature vector
        dataset: base dataset
        fscore: dict of curriculums and corresponding f-scores
    """

    def __init__(
            self,
            dataset,
            fscore,
            train: bool = None,
            samples_per_class: int = 1,
    ):
        self.dataset = dataset
        self.fscore = fscore
        self.train = train
        self.samples_per_class = samples_per_class

        class_groups = [[i] for i in range(10)]
        class_datasets = dict()
        for class_group in class_groups:
            _data = dataset(
                root=data_dir, spec_target=class_group, train=True, transform=dataset.std_transform['train'],
                samples_per_class=self.samples_per_class
            )
            class_datasets[class_group[0]] = _data[0][0]

        self.data, self.targets = [], []
        self.curriculum_representations = dict()
        for curriculum_i, score_i in fscore.items():
            for curriculum_j, score_j in fscore.items():
                if curriculum_j == curriculum_i:
                    continue
                if score_i >= score_j:
                    self.targets.append(1)
                else:
                    self.targets.append(0)

                datum, datum_i, datum_j = [], [], []
                if curriculum_i not in self.curriculum_representations.keys():
                    for _class in curriculum_i:
                        datum_i.append(class_datasets[int(_class)])
                    self.curriculum_representations[curriculum_i] = torch.stack(
                        datum_i, 1)
                if curriculum_j not in self.curriculum_representations.keys():
                    for _class in curriculum_j:
                        datum_j.append(class_datasets[int(_class)])
                    self.curriculum_representations[curriculum_j] = torch.stack(
                        datum_j, 1)
                datum = (curriculum_i, curriculum_j)
                self.data.append(datum)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]
        representation = torch.concat(
            [self.curriculum_representations[data[0]],
                self.curriculum_representations[data[1]]], 1
        )
        representation = representation.detach().cpu().numpy()

        return representation, target


class ComparatorNetV2(Dataset):
    """
    Args:
        train (bool, optional): if True, returns 80% of the dataset at random
        samples_per_class: number of examples to sample for feature vector
        dataset: base dataset
        fscore: dict of curriculums and corresponding f-scores
        device: cuda or cpu
    """

    def __init__(
            self,
            dataset,
            fscore,
            train: bool = None,
            samples_per_class: int = 1
    ):
        self.dataset = dataset
        self.fscore = fscore
        self.train = train
        self.samples_per_class = samples_per_class

        class_groups = [[i] for i in range(10)]
        class_datasets = dict()
        for class_group in class_groups:
            _data = dataset(
                root=data_dir, spec_target=class_group, train=True, transform=dataset.std_transform['train'],
                samples_per_class=self.samples_per_class
            )
            class_datasets[class_group[0]] = _data[0][0]

        self.data, self.targets = [], []
        self.curriculum_representations = dict()
        for curriculum_i, score_i in fscore.items():
            for curriculum_j, score_j in fscore.items():
                if curriculum_j == curriculum_i:
                    continue
                if score_i >= score_j:
                    self.targets.append(1)
                else:
                    self.targets.append(0)

                datum, datum_i, datum_j = [], [], []
                if curriculum_i not in self.curriculum_representations.keys():
                    for _class in curriculum_i:
                        datum_i.append(class_datasets[int(_class)])
                    self.curriculum_representations[curriculum_i] = torch.stack(
                        datum_i, 0)
                if curriculum_j not in self.curriculum_representations.keys():
                    for _class in curriculum_j:
                        datum_j.append(class_datasets[int(_class)])
                    self.curriculum_representations[curriculum_j] = torch.stack(
                        datum_j, 0)
                datum = (curriculum_i, curriculum_j)
                self.data.append(datum)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]
        representation = torch.cat(
            (self.curriculum_representations[data[0]],
             self.curriculum_representations[data[1]]), 0
        )
        representation = representation.detach().cpu().numpy()

        return representation, target
