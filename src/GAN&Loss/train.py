# -*- coding:utf-8 -*-
__author__ = 'Yi'

import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset


def load_txt(filepath, split_sym='\t'):
    from collections import defaultdict

    data = defaultdict(list)
    items = []

    for line in open(filepath).readlines():
        path, label = line.strip().split(split_sym)
        data[label].append(path)
        items.append([path, label])

    return data, items


class TxtDataset(Dataset):
    """
    txt format:
    path cls_name
    """

    def __init__(self, txt_path, transforms, balance=False):
        self.data, self.items = load_txt(txt_path)
        self.transforms = transforms
        self.balance = balance
        self.classes = sorted(list(self.data.keys()))
        self.n_classes = len(self.classes)

    def __getitem__(self, index):
        if self.balance:
            import random

            label = self.classes[index % self.n_classes]
            path = random.choice(self.data[label])

        else:
            path, label = self.items[index]

        pil = Image.open(path).convert('RGB')
        tensor = self.transforms(pil)
        label_ind = self.classes.index(label)

        return tensor, label_ind

    def __len__(self):
        if self.balance:
            return min([len(self.data[x]) for x in self.data]) * len(self.data)

        return len(self.items)


class PathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        pil = Image.open(path).convert('RGB')
        tensor = self.transform(pil)

        return tensor

    def __len__(self):
        return len(self.paths)


def pytorch_topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def eval_network(net, val_dataset, batch_size):
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False, sampler=None)

    net.eval()

    pred = None
    gt = None

    for i, (inputs, labels) in enumerate(val_loader):
        with torch.no_grad():
            inputs = inputs.cuda()  # inputs.to(device)
            out = net(inputs)

            if pred is None:
                pred = out.data
                gt = labels
            else:
                pred = torch.cat((pred, out.data), 0)
                gt = torch.cat((gt, labels), 0)

    y_true = gt.numpy()
    y_pred = pred.argmax(dim=1).cpu().numpy()

    topk = (1,)
    topk_accs = pytorch_topk_accuracy(pred.cpu(), gt, topk=topk)
    for k, acc in zip(topk, topk_accs):
        print("Acc@{}\t{:.2f}".format(k, acc[0]))

    # import numpy as np
    # try:
    #     np.set_printoptions(threshold=np.inf)
    # except:
    #     np.set_printoptions(threshold=np.nan)

    from sklearn.metrics import classification_report, accuracy_score
    print(classification_report(y_true, y_pred))

    return accuracy_score(y_true, y_pred)


def train_network(net, train_dataset, val_dataset,
                  model_save_dir,
                  epochs=20, stepsize=10, batch_size=32, init_lr=0.001):
    import numpy as np

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, sampler=None)

    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.SGD(
    #     filter(lambda p: p.requires_grad, net.parameters()),
    #     init_lr,
    #     momentum=0.9,
    #     weight_decay=1e-4)

    optimizer = torch.optim.Adam(
        net.parameters(),
        init_lr
    )

    n_iters = len(train_loader)
    print("1 epoch = {} iters.".format(n_iters))

    history = {"loss": [], "acc": [], "eval": []}

    for epoch in range(epochs):
        tot_loss = 0
        tot_acc = 0

        # Adjust learning rate
        # if epoch % stepsize == 0:
        #     lr = init_lr * (0.1 ** (epoch // stepsize))
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

        net.train()

        for i, (input, label) in enumerate(train_loader):
            input = input.cuda()
            label = label.cuda()

            # compute output
            out = net(input)

            loss = criterion(out, label)

            pred = out.data.cpu().numpy().argmax(axis=1)
            acc = accuracy_score(label.data.cpu().numpy(), pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.data.mean()
            tot_acc += acc

            history["loss"].append(loss.data.mean())
            history["acc"].append(acc)

            # if i % 1 == 0:
            # print("Iter {iter}:loss={loss:.2f}\tacc={acc:.2f}".format(iter=i, loss=loss, acc=acc))

        val_avg_acc = eval_network(net, val_dataset, batch_size)
        print("Epcch {epoch}: loss={loss:.2f} acc={acc:.2f}".format(epoch=epoch, loss=tot_loss / n_iters,
                                                                    acc=val_avg_acc))

        torch.save(net.module.state_dict(), "{dir}/{name}_{val_acc:.2f}.pth".format(
            dir=model_save_dir, name=epoch, val_acc=val_avg_acc))

        history["eval"].append(val_avg_acc)

    np.vstack([
        history["loss"], history["acc"]
    ]).dump("{}/history.np".format(model_save_dir))
    np.array(history["eval"]).dump("{}/epoch_eval.np".format(model_save_dir))


def init_net(pretrained_model_pth, n_classes, net_arch='densenet121',
             if_strict=False, if_parallel=True):
    pretrained_weights = torch.load(pretrained_model_pth)

    if net_arch == 'densenet121':
        ## Densenet
        print("Densenet121.")
        from torchvision.models.densenet import densenet121
        net = densenet121(pretrained=False, num_classes=n_classes)
        drop_layer = 'classifier'
    elif net_arch == 'resnet50':
        ## Resnet50
        print('Resnet50.')
        from torchvision.models.resnet import resnet50
        net = resnet50(False, num_classes=n_classes)
        drop_layer = 'fc'
    else:
        raise NotImplementedError

    if not if_strict and drop_layer is not None:
        for key in ["{}.bias".format(drop_layer), "{}.weight".format(drop_layer)]:
            pretrained_weights.pop(key)

    net.load_state_dict(pretrained_weights, strict=if_strict)
    if if_parallel:
        net = nn.DataParallel(net)
    return net.cuda()


class Preprocessor(object):
    def __init__(self, input_size=224, mean=None, std=None):
        self.input_size = input_size
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        self.tensorify = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.strong_aug = transforms.Compose([
            transforms.RandomAffine(5, translate=(0.1, 0.1), scale=(0.9, 1.0),
                                    shear=5, resample=False, fillcolor=0),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.input_size, self.input_size)),
            self.tensorify
        ])

        self.weak_aug = transforms.Compose([
            transforms.Resize((self.input_size + 16, self.input_size + 16)),
            transforms.RandomCrop(self.input_size),
            self.tensorify
        ])

        self.just_resize = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            self.tensorify
        ])


def parse_args():
    parser = argparse.ArgumentParser(description='Train classifier.')
    parser.add_argument('--dir', help='dir path of input txt and output weights.')
    parser.add_argument('--net', help='opts: resnet50, densenet121.', default='densenet121')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    output_dir = os.path.join(args.dir, 'weights')
    net_arch = args.net

    input_filepath = lambda x: "{}/{}.txt".format(args.dir, x)

    try:
        os.mkdir(output_dir)
    except:
        pass

    input_size = 224  # 299
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_dataset = TxtDataset(input_filepath('train'), balance=True,
                               transforms=transforms.Compose([
                                   transforms.RandomAffine(30, translate=[0.2, 0.2], scale=None,
                                                           shear=None, resample=False, fillcolor=0),
                                   transforms.Resize((input_size + 8, input_size + 8)),
                                   transforms.RandomCrop(input_size),

                                   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),

                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)
                               ]))

    val_dataset = TxtDataset(input_filepath('test'), balance=False, transforms=transforms.Compose([
        transforms.Resize((input_size, input_size)),
    # eval_network(net, val_dataset, 1024)
