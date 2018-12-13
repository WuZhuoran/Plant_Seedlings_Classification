# -*- coding:utf-8 -*-

__author__ = 'Yi'

import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import DSDataset, tensorify_img
from torch.utils.data import DataLoader
from torchvision.utils import save_image

os.makedirs('images', exist_ok=True)
os.makedirs('wts', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
# parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
# parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
# parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
# parser.add_argument('--channels', type=int, default=1, help='number of image channels')
# parser.add_argument('--sample_interval', type=int, default=100, help='interval between image sampling')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(128, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, 3, 1, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        # )

        # --x2--
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(128, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #
        #     nn.Upsample(scale_factor=2),
        #
        #     nn.Conv2d(256, 128, 3, 1, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(128, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(64, 32, 3, 1, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        # )

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
        scale_factor = 256 / x.size(2)
        y = F.upsample(x, scale_factor=scale_factor)
        z = self.features(x)
        z = torch.cat([y, z], dim=1)
        return self.merge(z)


# class Discriminator(nn.Module):
#     def __init__(self, imgsize=256):
#         super(Discriminator, self).__init__()
#
#         def discriminator_block(in_filters, out_filters, bn=True):
#             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
#                      nn.LeakyReLU(0.2, inplace=True),
#                      nn.Dropout2d(0.25)]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block
#
#         self.model = nn.Sequential(
#             *discriminator_block(3, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )
#
#         ds_size = imgsize // 2 ** 4
#         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
#
#     def forward(self, img):
#         out = self.model(img)
#         out = out.view(out.size(0), -1)
#         validity = self.adv_layer(out).view(-1)
#         return validity

class Discriminator(nn.Module):
    def __init__(self, imgsize=256):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        ds_size = imgsize // 2 ** 4

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * ds_size ** 2, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x).view(-1)
        return x


def mean_loss(losses):
    return sum(losses) / len(losses)


if __name__ == '__main__':

    root = "/home/Yi/seeds"
    paths = pickle.load(open(root + '/large.paths', 'rb'))

    dataset = DSDataset(paths, input_size=256, ds_ratio=0.25,
                        tensorify=lambda x: tensorify_img(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    # Loss function
    alpha = 0.01
    adversarial_loss = nn.BCEWithLogitsLoss()
    mae_loss = nn.MSELoss(reduction='mean')

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator = nn.DataParallel(generator).cuda()
        discriminator = nn.DataParallel(discriminator).cuda()
        adversarial_loss = adversarial_loss.cuda()
        mae_loss = mae_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, drop_last=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        fool_losses = []
        dist_losses = []
        disc_losses = []

        for i, (real_imgs, input_imgs, label) in enumerate(dataloader):
            # Adversarial ground truths
            valid = Tensor(real_imgs.size(0), ).fill_(1.0)
            fake = Tensor(real_imgs.size(0), ).fill_(0.0)

            # Configure input
            if cuda:
                real_imgs = real_imgs.cuda()
                input_imgs = input_imgs.cuda()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(input_imgs)
            preds = discriminator(gen_imgs)

            # Loss measures generator's ability to fool the discriminator
            fool_loss = adversarial_loss(preds, valid)  # here set target is valid but not fake
            dist_loss = mae_loss(gen_imgs, real_imgs)

            g_loss = dist_loss + alpha * fool_loss
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2.0

            d_loss.backward()
            optimizer_D.step()

            fool_losses.append(fool_loss.item())
            dist_losses.append(dist_loss.item())
            disc_losses.append(d_loss.item())

        print("[Epoch %d/%d] [D loss: %.2f] [fool: %.2f] [dist: %.2f]" % (
            epoch, opt.n_epochs, mean_loss(disc_losses), mean_loss(fool_losses), mean_loss(dist_losses)))

        save_image(gen_imgs.data[:16], 'images/epoch-{}.png'.format(epoch), nrow=4, normalize=True, scale_each=True)
        torch.save(generator.module.state_dict(), 'wts/{}.pth'.format(epoch))
