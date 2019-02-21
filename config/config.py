''' general configurations'''

import os


gpu_id = '2, 3'

project_dir = os.path.split(os.getcwd())[0]
project_prefix = 'food_all'

available_models = {
    'fpn50': {'model_name': 'fpn50', 'conv_layer': 'resnet50'},
    'fpn101': {'model_name': 'fpn101', 'conv_layer': 'resnet101'}}

model_key = 'fpn101'

model_name = available_models[model_key]['model_name']
base_conv_layer = available_models[model_key]['conv_layer']

use_rgb = True
use_depth = False

img_res = 600
num_classes = 15

# grapes, cherry_tomatoes, broccoli, cauliflower, honeydew,
# banana, kiwi, strawberry, cantaloupe, carrots, celeries,
# apples, bell_pepper
excluded_item = None

train_batch_size = 10
test_batch_size = 4

dataset_dir = os.path.join(
    project_dir, 'data/food_data/bounding_boxes_all')

label_map_filename = os.path.join(
    dataset_dir, '{}_label_map.pbtxt'.format(project_prefix))
img_dir = os.path.join(dataset_dir, 'images')
depth_dir = os.path.join(dataset_dir, 'depth')

train_list_filename = os.path.join(
    dataset_dir, '{}_ann_train.txt'.format(project_prefix))
test_list_filename = os.path.join(
    dataset_dir, '{}_ann_test.txt'.format(project_prefix))

pretrained_dir = os.path.join(project_dir, 'pretrained')
pretrained_filename = os.path.join(
    pretrained_dir, '{}_net.pth'.format(project_prefix))

color_tag = ''
color_tag += 'rgb' if use_rgb else ''
color_tag += 'd' if use_depth else ''

checkpoint_filename = os.path.join(
    project_dir, 'checkpoint/{}_{}_{}ckpt.pth'.format(
        project_prefix,
        color_tag,
        '' if excluded_item is None else 'wo_' + excluded_item + '_'))
