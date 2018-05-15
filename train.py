#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import os
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
from datagen import ListDataset


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

best_loss = float('inf')  # best test loss


def run_train():
    parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'Error: CUDA not found!'
    start_epoch = 0  # start from epoch 0 or last epoch

    # Data
    print('==> Preparing data..')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainset = ListDataset(
        root='./data/voc_all_images',
        list_file='./data/voc12_train.txt',
        train=True, transform=transform, input_size=600)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=8, shuffle=True, num_workers=8,
        collate_fn=trainset.collate_fn)

    testset = ListDataset(
        root='./data/voc_all_images',
        list_file='./data/voc12_val.txt',
        train=False, transform=transform, input_size=600)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=8,
        collate_fn=testset.collate_fn)

    # Model
    net = RetinaNet(num_classes=20)
    net.load_state_dict(torch.load('./model/net.pth'))
    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    net.cuda()

    criterion = FocalLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        net.module.freeze_bn()
        train_loss = 0

        total_batches = int(math.ceil(
            trainloader.dataset.num_samples / trainloader.batch_size))

        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
            inputs = inputs.cuda()
            loc_targets = loc_targets.cuda()
            cls_targets = cls_targets.cuda()

            optimizer.zero_grad()
            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            print('[%d| %d/%d] train_loss: %.3f | avg_loss: %.3f' %
                  (epoch, batch_idx, total_batches,
                   loss.data, train_loss / (batch_idx + 1)))

    # Test
    def test(epoch):
        print('\nTest')
        net.eval()
        test_loss = 0

        total_batches = int(math.ceil(
            testloader.dataset.num_samples / testloader.batch_size))

        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
            inputs = inputs.cuda()
            loc_targets = loc_targets.cuda()
            cls_targets = cls_targets.cuda()

            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            test_loss += loss.data
            print('[%d| %d/%d] test_loss: %.3f | avg_loss: %.3f' %
                  (epoch, batch_idx, total_batches,
                   loss.data, test_loss / (batch_idx + 1)))

        # Save checkpoint
        global best_loss
        test_loss /= len(testloader)
        if test_loss < best_loss:
            print('Saving..')
            state = {
                'net': net.module.state_dict(),
                'loss': test_loss,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_loss = test_loss

    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch)


if __name__ == '__main__':
    run_train()
