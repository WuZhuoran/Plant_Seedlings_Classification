# -*- coding:utf-8 -*-
__author__ = 'Yi'

import os
import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Gx2(nn.Module):
    def __init__(self):
        super(Gx2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.merge = nn.Sequential(
            nn.Conv2d(32 + 3, 3, 1, 1, 0),
            nn.Tanh()
        )

    def forward(self, x):
        y = F.upsample(x, scale_factor=2)
        z = self.features(x)
        z = torch.cat([y, z], dim=1)
        return self.merge(z)


class Gx4(nn.Module):
    def __init__(self):
        super(Gx4, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.merge = nn.Sequential(
            nn.Conv2d(32 + 3, 3, 1, 1, 0),
            nn.Tanh()
        )

    def forward(self, x):
        y = F.upsample(x, scale_factor=4)
        z = self.features(x)
        z = torch.cat([y, z], dim=1)
        return self.merge(z)


def get_generator(model_dir, filename, cuda=True):
    if filename.startswith('Gx2'):
        generator = Gx2()
    else:
        generator = Gx4()

    wts_path = os.path.join(model_dir, filename)
    generator.load_state_dict(torch.load(wts_path))

    if cuda:
        generator = generator.cuda()

    return generator


def tensor_to_im(tensor, normalize=True):
    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    def norm_range(t, range):
        if range is not None:
            norm_ip(t, range[0], range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    if normalize:
        norm_range(tensor, None)

    x = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return x


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


def upsample_image(generator, im, output_path=None):
    x = tensorify_img(im, (.5, .5, .5), (.5, .5, .5))
    x = torch.Tensor(x)
    y = generator(x.cuda().unsqueeze(0))
    z = tensor_to_im(y[0])

    if output_path is not None:
        cv2.imwrite(output_path, z)

    return z


def get_training_imgs(data_dir, out_dir):
    dirs = os.listdir(data_dir)
    data = []

    for d in dirs:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
        imgs = glob.glob(os.path.join(data_dir, d, '*'))
        data += [(x, os.path.join(out_dir, d, os.path.basename(x))) for x in imgs]

    return data


def get_test_imgs(data_dir, out_dir):
    imgs = glob.glob(os.path.join(data_dir, '*'))
    return [(x, os.path.join(out_dir, os.path.basename(x))) for x in imgs]


def work(gx2, gx4, imgpath, savepath, target_size=256):
    im = cv2.imread(imgpath)
    h, w, _ = im.shape
    my_size = min(h, w)
    half_target_size = target_size >> 1

    if half_target_size < my_size < target_size:
        upsample_image(gx2, im)
    elif my_size < half_target_size:
        upsample_image(gx4, im)

    canvas = cv2.resize(im, (target_size, target_size))
    cv2.imwrite(savepath, canvas)


if __name__ == '__main__':
    model_dir = '/home/Y/seeds/models'
    gx2 = get_generator(model_dir, 'Gx2.pth')
    gx4 = get_generator(model_dir, 'Gx4.pth')

    label = 'test'
    data_dir = "/home/Y/seeds/{}".format(label)
    save_dir = "/home/Y/seeds/{}_256x256".format(label)

    os.makedirs(save_dir, exist_ok=True)
    if label == 'train':
        data = get_training_imgs(data_dir, save_dir)
    else:
        data = get_test_imgs(data_dir, save_dir)

    for imgpath, savepath in tqdm(data):
        work(gx2, gx4, imgpath, savepath)
