import os
import os.path
import cv2
import glob
import ipdb
import random
import numpy as np
from PIL import Image
import pickle

import torch.utils.data as data

from utils import np_to_torch


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def crop_cv2(img, patch):
    height, width, _ = img.shape
    start_x = random.randint(0, height - patch)
    start_y = random.randint(0, width - patch)
    return img[start_x:start_x + patch, start_y:start_y + patch]




class ImageNet(data.Dataset):
    def __init__(self, fns, mode, args):
        self.fns = fns
        self.mode = mode
        self.args = args
        self.get_image_list()

    def get_image_list(self):
        # self.fns = []
        # for fn in glob.iglob(self.path + '/*JPEG'):
        #     self.fns.append(fn)

        random.Random(4).shuffle(self.fns)
        num_images = len(self.fns)
        train_size = int(num_images // 1.25)
        eval_size = int(num_images // 10)
        if self.mode == 'TRAIN':
            self.fns = self.fns[:train_size]
        elif self.mode == 'VALIDATE':
            self.fns = self.fns[train_size:train_size+eval_size]
        elif self.mode == 'EVALUATE':
            self.fns = self.fns[train_size+eval_size:train_size+2*eval_size]
        print('Number of {} images loaded: {}'.format(self.mode, len(self.fns)))

    def __getitem__(self, index):
        image_fn = self.fns[index]
        image = cv2.imread('datasets/' + image_fn)

        height, width, _ = image.shape
        if height < 128 or width < 128:
            return None, image_fn

        image = crop_cv2(image, self.args.crop)
        image = np_to_torch(image)
        image = image / 255.0
        return image, image_fn

    def __len__(self):
        return len(self.fns)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--crop', default=128, type=int,
                            help='crop size of images')
        return parser


class CIFAR10(data.Dataset):
    def __init__(self, path, mode):
        train_data = np.empty((50000, 32, 32, 3), dtype=np.uint8)
        train_labels = np.empty(50000, dtype=np.uint8)
        for i in range(0, 5):
            data_train = unpickle(os.path.join(path, 'data_batch_{}'.format(i+1)))
            train_data[i*10000:(i+1)*10000] = data_train[b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
            train_labels[i * 10000:(i + 1) * 10000] = data_train[b'labels']
        self.train = train_data, train_labels
        data_test = unpickle(os.path.join(path, 'test_batch'))
        test_set = data_test[b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1)), data_test[b'labels']
        self.test = (test_set[0][:5000], test_set[1][:5000])
        self.valid = (test_set[0][5000:], test_set[1][5000:])

        if mode == 'TRAIN':
            self.dataset = self.train
        elif mode == 'VALIDATE':
            self.dataset = self.valid
        else:
            self.dataset = self.test

    def __getitem__(self, index):
        img, label = self.dataset[0][index], self.dataset[1][index]
        img = np_to_torch(img) / 255.
        return img, int(label)

    def __len__(self):
        return len(self.dataset[0])


class Kodak(data.Dataset):
    def __init__(self, path, args):
        self.path = path
        self.get_image_list()

    def get_image_list(self):
        self.fns = []
        for fn in glob.iglob(self.path + '/*.png', recursive=True):
            self.fns.append(fn)
        print('Number of images loaded: {}'.format(len(self.fns)))

    def __getitem__(self, index):
        image_fn = self.fns[index]
        image = cv2.imread(image_fn)

        image = np_to_torch(image)
        image = image / 255.0
        return image, image_fn

    def __len__(self):
        return len(self.fns)
