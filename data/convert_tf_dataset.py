#!/usr/bin/env python

import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
import random

try:
    import cPickle as pickle
except ImportError:
    import pickle


img_base_dir = '/mnt/hard_data/Data/foods/data/images'
ann_base_dir = '/mnt/hard_data/Data/foods/data/annotations/xmls'
label_map_filename = '/mnt/hard_data/Data/foods/data/food_label_map.pbtxt'

label_map_inv = None


def update_xml(filepath, filename, is_edit=False):
    print('>>> {}'.format(filename))
    tree = ET.parse(filepath)
    root = tree.getroot()

    for node in root:
        if node.tag == 'folder':
            print(node.tag, node.text)
            node.text = '.'
            print('-->   ', node.text)
        elif node.tag == 'filename':
            print(node.tag, node.text)
            node.text = '%s.jpg' % filename[:-4]
            print('--> ', node.text)
        elif node.tag == 'path':
            print(node.tag, node.text)
            node.text = os.path.join(img_base_dir, '%s.jpg' % filename[:-4])
            print('--> ', node.text)

    if is_edit:
        tree.write(filepath)


def xml_to_line(filename):
    tree = ET.parse(os.path.join(ann_base_dir, filename))
    root = tree.getroot()

    line = root.find('filename').text
    for node in root:
        if node.tag == 'object':
            line += ' {} {} {} {} {}'.format(
                node[4][0].text, node[4][1].text,
                node[4][2].text, node[4][3].text,
                label_map_inv[node[0].text])

    return line


def load_tf_label_map():
    with open(label_map_filename, 'r') as f:
        content = f.read().splitlines()

    assert content is not None, 'cannot find label map'

    temp = list()
    for line in content:
        line = line.strip()
        if (len(line) > 2 and
                (line.startswith('id') or
                 line.startswith('name'))):
            temp.append(line.split(':')[1].strip())

    global label_map_inv
    label_map_inv = dict()

    label_dict = dict()
    for idx in range(0, len(temp), 2):
        item_id = int(temp[idx])
        item_name = temp[idx + 1][1:-1]
        label_dict[item_id] = item_name
        label_map_inv[item_name] = item_id

    return label_dict


def load_pickled_label_map():
    print('\n--- test saved label map ---')
    with open('./label_map.pkl', 'r') as f:
        label_map = pickle.load(f)

    assert label_map is not None, 'cannot load label map'
    print(label_map)


def convert_tf_dataset():
    print('convert tf_dataset for pytorch-retinanet')

    print('\n--- convert label map ---')
    label_map = load_tf_label_map()
    with open('label_map.pkl', 'w') as f:
        pickle.dump(label_map, f, pickle.HIGHEST_PROTOCOL)
    print('label_map is saved in data/label_map.pkl')
    # load_pickled_label_map()

    ann_filename = 'ann'
    import sys
    if len(sys.argv) == 2:
        ann_filename = sys.argv[1]

    print('\n--- convert annotations ---')
    simplified_lines = list()
    anns = os.listdir(ann_base_dir)
    for this_ann in anns:
        simplified_lines.append(xml_to_line(this_ann))
    # print(simplified_lines)
    random.shuffle(simplified_lines)

    f_ann = open(ann_filename + '.txt', 'w')
    f_ann_train = open(ann_filename + '_train.txt', 'w')
    f_ann_test = open(ann_filename + '_test.txt', 'w')

    line_count = 0
    train_num = int(len(simplified_lines) * 0.9)
    for line in simplified_lines:
        this_line = '{}\n'.format(line)
        f_ann.write(this_line)
        if line_count < train_num:
            f_ann_train.write(this_line)
        else:
            f_ann_test.write(this_line)
        line_count += 1
    f_ann.close()
    f_ann_train.close()
    f_ann_test.close()
    print('annotation is saved in data/{}'.format(ann_filename))

    img_link_name = 'all_images'
    if os.path.exists(img_link_name):
        os.remove(img_link_name)
    os.symlink(img_base_dir, img_link_name)
    print('made symlink to {}'.format(img_base_dir))

    print('\n--- finished ---\n')


if __name__ == '__main__':
    convert_tf_dataset()

