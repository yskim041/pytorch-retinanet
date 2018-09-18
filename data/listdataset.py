from __future__ import print_function
from __future__ import division

import sys
import os

import torch
import torch.utils.data as data

from PIL import Image

sys.path.append('../')
from retinanet_utils.encoder import DataEncoder
from retinanet_utils.transform import resize, random_flip, random_crop, center_crop
from retinanet_utils.utils import load_label_map
from retinanet_utils.pt_utils import one_hot_embedding
from config import config


class ListDataset(data.Dataset):
    def __init__(self,
                 img_dir=config.img_dir,
                 list_filename=config.train_list_filename,
                 label_map_filename=config.label_map_filename,
                 train=True,
                 transform=None,
                 input_size=config.img_res):

        self.img_dir = img_dir
        self.train = train
        self.transform = transform
        self.input_size = input_size

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

            self.img_filenames.append(this_img_filename)

            num_boxes = (len(splited) - 1) // isize
            box = list()
            label = list()
            for bidx in range(num_boxes):
                xmin = float(splited[1 + isize * bidx])
                ymin = float(splited[2 + isize * bidx])
                xmax = float(splited[3 + isize * bidx])
                ymax = float(splited[4 + isize * bidx])
                cls = int(splited[5 + isize * bidx])
                box.append([xmin, ymin, xmax, ymax])
                label.append(cls)

            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

            self.num_samples += 1

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
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
        inputs = torch.zeros(num_imgs, 3, h, w)

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
