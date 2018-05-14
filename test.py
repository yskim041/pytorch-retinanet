#!/usr/bin/env python

import sys

import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw


def run_test():
    print('Loading model..')
    net = RetinaNet()

    ckpt = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(ckpt['net'])
    net.eval()
    net.cuda()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    print('Loading image..')
    img = Image.open('./image/000050.jpg')
    w = h = 500
    img = img.resize((w,h))

    print('Predicting..')
    x = transform(img)
    x = x.unsqueeze(0)
    with torch.no_grad():
        loc_preds, cls_preds = net(x.cuda())

        print('Decoding..')
        encoder = DataEncoder()
        boxes, labels = encoder.decode(
            loc_preds.cpu().data.squeeze(),
            cls_preds.cpu().data.squeeze(),
            (w,h))

        draw = ImageDraw.Draw(img)
        for box in boxes:
           draw.rectangle(list(box), outline='red')
        img.show()


if __name__ == '__main__':
    run_test()
