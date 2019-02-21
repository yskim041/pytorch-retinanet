from __future__ import print_function
from __future__ import division

import sys
import os
import numpy as np
import random

import torch
import torch.utils.data as data

from PIL import Image, ImageEnhance

sys.path.append('../')
from retinanet_utils.encoder import DataEncoder
from retinanet_utils.transform import resize, random_flip, random_crop, center_crop
from retinanet_utils.utils import load_label_map
from retinanet_utils.pt_utils import one_hot_embedding
from config import config


class ListDataset(data.Dataset):
    def __init__(self,
                 img_dir=config.img_dir,
                 depth_dir=config.depth_dir,
                 list_filename=config.train_list_filename,
                 label_map_filename=config.label_map_filename,
                 train=True,
                 transform=None,
                 input_size=config.img_res):

        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.input_channels = 0
        if config.use_rgb:
            self.input_channels += 3
        if config.use_depth:
            self.input_channels += 1

        self.label_map = load_label_map(label_map_filename)

        self.img_filenames = list()
        self.boxes = list()
        self.labels = list()

        self.encoder = DataEncoder()

        with open(list_filename) as f:
            lines = f.readlines()
            f.close()

        self.num_samples = 0
        isize = 6
        for line in lines:
            splited = line.strip().split()

            this_img_filename = splited[0]
            if this_img_filename.startswith('sample'):
                continue

            if config.excluded_item:
                id_tag = this_img_filename.split('+')[1].split('-')[0]
                if id_tag == config.excluded_item:
                    continue

            num_boxes = (len(splited) - 1) // isize
            box = list()
            label = list()
            for bidx in range(num_boxes):
                cls = int(splited[5 + isize * bidx])
                label.append(cls)
                xmin = float(splited[1 + isize * bidx])
                ymin = float(splited[2 + isize * bidx])
                xmax = float(splited[3 + isize * bidx])
                ymax = float(splited[4 + isize * bidx])
                box.append([xmin, ymin, xmax, ymax])

            if len(box) > 0:
                self.img_filenames.append(this_img_filename)
                self.boxes.append(torch.Tensor(box))
                self.labels.append(torch.LongTensor(label))
                self.num_samples += 1

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]

        if config.use_depth:
            img = Image.open(os.path.join(self.depth_dir, img_filename))
            im_arr = np.array(img)
            im_arr[im_arr > 1000] = 0
            im_arr = (im_arr - np.mean(im_arr)) / 500 * 255
            img = Image.fromarray(im_arr)
        else:
            img = Image.open(os.path.join(self.img_dir, img_filename))
            if img.mode != 'RGB':
                img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # Data augmentation
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size, size))
            if random.random() > 0.5:
                img = ImageEnhance.Color(img).enhance(random.uniform(0, 1))
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.4, 2.5))
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.4, 2.0))
                img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.4, 1.5))

        else:
            img, boxes = resize(img, boxes, (size, size))
            # img, boxes = center_crop(img, boxes, (size, size))

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(
            num_imgs,
            self.input_channels,
            h, w)

        loc_targets = list()
        cls_targets = list()
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            this_loc, this_cls = self.encoder.encode(
                boxes[i], labels[i], input_size=(w, h))
            loc_targets.append(this_loc)
            cls_targets.append(this_cls)

        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


def test():
    print('[listdataset] test')
    ds = ListDataset(
        list_filename=config.test_list_filename,
        train=False)
    import IPython; IPython.embed()


if __name__ == '__main__':
    test()
