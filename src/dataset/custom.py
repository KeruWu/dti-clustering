from functools import lru_cache
import os
import mrcfile
from scipy import io
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import Compose, ToTensor

from .torch_transforms import TensorResize
from utils import coerce_to_path_and_check_exist
from utils.path import DATASETS_PATH


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
    root = DATASETS_PATH
    name = '10406'
    n_classes = 10
    n_channels = 1
    img_size = (208, 208)
    n_val = 1000

    def __init__(self, split, **kwargs):
        # self.data_path = coerce_to_path_and_check_exist(self.root / '10406/')
        # self.data_path = coerce_to_path_and_check_exist('/content/drive/My Drive/cryoEM/project/datasets/10406/')
        
        self.data_path = coerce_to_path_and_check_exist('/content/drive/My Drive/cryoEM/project/datasets/particle_stack/')
        self.split = split
        data, labels = self.load_mrcs(self.data_path)
        data, labels = self.load_particle_stack(self.data_path)
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
        for filename in [f for f in os.listdir(data_path) if f.endswith('.mrcs')][:100]:
            with mrcfile.open(data_path / filename) as mrc:
                particle = mrc.data
            particles.append(particle[:,104:312,104:312])
            
        particles = np.vstack(particles)
        labels = np.zeros(particles.shape[0])
        print(particles.shape)
        return particles, labels
        
    @staticmethod
    def load_particle_stack(data_path):

        with mrcfile.open(data_path + 'particle_stack_0.mrc') as mrc:
            particles = mrc.data
        
        processed_particles = []
        for particle in particles:
            blured_img = gaussian_filter(particle, sigma=9)
            res = cv2.resize(blured_img, dsize=(100, 100))
            processed_particles,append(res)
            
        processed_particles = np.vstack(processed_particles)
        labels = np.zeros(processed_particles.shape[0])
        print(processed_particles.shape)
        return processed_particles, labels

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
