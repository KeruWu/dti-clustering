from functools import lru_cache
import os
import mrcfile
from scipy import io
import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import Compose, ToTensor

from .torch_transforms import TensorResize
#from src.utils import coerce_to_path_and_check_exist
#from src.utils.path import DATASETS_PATH


class AffNISTTestDataset(TorchDataset):
    root = DATASETS_PATH
    name = 'affnist_test'
    n_classes = 10
    n_channels = 1
    img_size = (40, 40)
    n_val = 1000

    def __init__(self, split, **kwargs):
        self.data_path = coerce_to_path_and_check_exist(self.root / 'affNIST_test.mat')
        self.split = split
        data, labels = self.load_mat(self.data_path)
        if split == 'val':
            data, labels = data[:self.n_val], labels[:self.n_val]
        self.data, self.labels = data, labels
        self.size = len(self.labels)

        img_size = kwargs.get('img_size')
        if img_size is not None:
            self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
            assert len(self.img_size) == 2

    @staticmethod
    def load_mat(data_path):
        mat = io.loadmat(data_path)['affNISTdata']
        data = mat['image'][0][0].transpose().reshape(-1, 40, 40)
        labels = mat['label_int'][0][0][0]
        return data, labels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.labels[idx]

    @property
    @lru_cache()
    def transform(self):
        transform = [ToTensor()]
        if self.img_size != self.__class__.img_size:
            transform.append(TensorResize(self.img_size))
        return Compose(transform)


class EMPIAR_10406_DATASET(TorchDataset):
    #root = DATASETS_PATH
    name = '10406'
    n_classes = 10
    n_channels = 1
    img_size = (416, 416)
    n_val = 1000

    def __init__(self, split, **kwargs):
        self.data_path = './'#coerce_to_path_and_check_exist(self.root / 'name/')
        self.split = split
        data, labels = self.load_mrcs(self.data_path)
        if split == 'val':
            data, labels = data[:self.n_val], labels[:self.n_val]
        self.data, self.labels = data, labels
        self.size = len(self.labels)

        img_size = kwargs.get('img_size')
        if img_size is not None:
            self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
            assert len(self.img_size) == 2

    @staticmethod
    def load_mrcs(data_path):
        particles = []
        for filename in [f for f in os.listdir(data_path) if f.endswith('.mrcs')]:
            with mrcfile.open(data_path + filename) as mrc:
                particle = mrc.data
            particles.append(particle)
        particles = np.vstack(particles)
        labels = np.arange(0, particle.shape[0])
        return particles, labels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.labels[idx]

    @property
    @lru_cache()
    def transform(self):
        transform = [ToTensor()]
        if self.img_size != self.__class__.img_size:
            transform.append(TensorResize(self.img_size))
        return Compose(transform)
