# -*- coding:utf-8 -*-
__author__ = 'Yi'

import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']


def get_cls_by_path(path):
    return os.path.basename(os.path.dirname(path))


def get_random_interpolation():
    return random.choice(cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR,
                         cv2.INTER_NEAREST)


def get_cls_idx(cls):
    return CATEGORIES.index(cls)


class Stdsize:
    mini_size = 64
    small_size = 128
    mid_size = 256
    large_size = 512


class DSDataset(Dataset):
    def __init__(self, paths, input_size=256, ds_ratio=0.5, tensorify=None):
        self.data = [[p, get_cls_idx(get_cls_by_path(p))] for p in paths]
        self.aug = aug_img
        self.tensorify = tensorify
        self.large_size = input_size
        self.small_size = int(input_size * ds_ratio)
        self.ds_ratio = ds_ratio

    def __getitem__(self, index):
        path, label = self.data[index]

        im = cv2.imread(path)
        im = self.aug(im)

        im_x = cv2.resize(im, (self.large_size, self.large_size))
        im_y = lossy_compress_img(im, self.small_size)

        if self.tensorify:
            im_x = self.tensorify(im_x)
            im_y = self.tensorify(im_y)

        return im_x, im_y, label

    def __len__(self):
        return len(self.data)


def classify_size_grade(size):
    if size >= Stdsize.large_size:
        return 'xl'  # [512,oo)
    if size >= Stdsize.mid_size:
        return 'l'  # [256, 512)
    if size >= Stdsize.small_size:
        return 'm'  # [128, 256)
    if size >= Stdsize.mini_size:
        return 's'  # [64, 128)
    return 'xs'


def get_precomputed_sizes_path(root, label):
    return os.path.join(root, label + '.size_path')


def compute_sizes_of_training_data(root, label='train'):
    import glob
    if label == 'train':
        images = glob.glob(os.path.join(root, 'train', '*/*'))
    else:
        images = glob.glob(os.path.join(root, 'test', '*'))

    print('Find {} images for {}'.format(len(images), os.path.join(root, label)))

    path_size = {}
    for path in tqdm(images):
        x = cv2.imread(path)
        path_size[path] = (x.shape[1], x.shape[0])

    with open(get_precomputed_sizes_path(root, label), 'wb') as f:
        pickle.dump(path_size, f)

    return path_size


def group_data_by_size(root, label, print_stats=True):
    path = get_precomputed_sizes_path(root, label)
    if not os.path.exists(path):
        path_size = compute_sizes_of_training_data(root, label)
    else:
        path_size = pickle.load(open(path, 'rb'))

    from collections import defaultdict
    grade_paths = defaultdict(list)

    for path in tqdm(path_size):
        w, h = path_size[path]
        g = classify_size_grade(min(w, h))
        grade_paths[g].append(path)

    if print_stats:
        '''
        /train:
        xs	60	1%
        s	903	19%
        m	1386	29%
        l	1233	26%
        xl	1168	25%

        /test:
        xs	0	0%
        s	0	0%
        m	330	42%
        l	464	58%
        xl	0	0%
        '''
        n_tot = len(path_size)
        for grade in ('xs', 's', 'm', 'l', 'xl'):
            t = len(grade_paths[grade])
            print("{}\t{}\t{:.0f}%".format(grade, t, t * 100.0 / n_tot))

    return grade_paths


def count_cls_distribution(paths):
    from collections import Counter
    clses = list(map(get_cls_by_path, paths))
    c = Counter(clses)
    print(c.most_common())


def save_multi_images_in_one_row(imgs, saveto, titles=None, figsize=(12, 6)):
    show_title = isinstance(titles, list) and len(titles) == len(imgs
                                                                 )
    fig = plt.figure(figsize=figsize)
    n_cols = len(imgs)
    for i, x in enumerate(imgs):
        plt.subplot(1, n_cols, i + 1)
        plt.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if x.ndim == 3 else x)
        if show_title:
            plt.title(titles[i])

    fig.savefig(saveto)
    plt.close()


def aug_img(im, alpha=0.1, debug=False):
    h, w, _ = im.shape
    h_cut = random.random() * alpha * h
    w_cut = random.random() * alpha * w

    top, bottom = int(h_cut), h - int(h_cut)
    left, right = int(w_cut), w - int(w_cut)

    crop = im[top:bottom, left:right]
    return crop


def lossy_compress_img(im, target_size=128):
    ds_size = random.randint(50, 128)
    downsample = cv2.resize(im, (ds_size, ds_size))
    square = cv2.resize(downsample, (target_size, target_size))
    return square


def tensorify_img(im, mean=None, std=None):
    if mean is None:
        mean = (.5, .5, .5)
    if std is None:
        std = (.5, .5, .5)

    im = im.astype(np.float32)
    im /= 255.0

    im -= mean
    im /= std

    return torch.Tensor(im.transpose(2, 0, 1))


def collate(batch):
    im_xs, im_ys, labels = zip(*batch)
    return torch.stack(im_xs), torch.stack(im_ys), torch.stack(labels).type(torch.long)


if __name__ == '__main__':
    root = "/home/Yi/seeds"
    sizegrade_paths = group_data_by_size(root, 'test', True)
    data = sizegrade_paths['l'] + sizegrade_paths['xl']
    print('{} data.'.format(len(data)))

    count_cls_distribution(data)

    # with open(os.path.join(root, 'large.paths'), 'wb') as f:
    #     pickle.dump(sizegrade_paths['l'] + sizegrade_paths['xl'], f)

    # with open(os.path.join(root, 'small.paths'), 'wb') as f:
    #     pickle.dump(sizegrade_paths['s'] + sizegrade_paths['m'], f)

    # for i in range(5):
    #     im, ds, label = dataset[i]
    #     h, w, _ = im.shape
    #     hh, ww, _ = ds.shape
    #     save_multi_images_in_one_row([im, ds], '//home/Yi/ds_vis/{}.png'.format(i),
    #                                  ['{}x{}'.format(w, h), '{}x{}'.format(ww, hh)])

    # dataset = DSDataset(data, 256, ds_ratio=0.5, tensorify=lambda x: tensorify_img(x, (.5, .5, .5), (.5, .5, .5)))
    #
    # from torch.utils.data import DataLoader
    #
    # dataloader = DataLoader(dataset, num_workers=4, pin_memory=True, batch_size=8)
    #
    # counter = 0
    # for im_x, im_y, label in dataloader:
    #     print(im_x.size(), im_y.size(), label.size())
    #     counter += 1
    #     if counter > 10:
    #         break
