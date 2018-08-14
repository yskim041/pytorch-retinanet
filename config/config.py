''' general configurations'''

import os


project_dir = os.path.split(os.getcwd())[0]
project_prefix = 'retinanet'

available_models = {
    'fpn50': {'model_name': 'fpn50', 'conv_layer': 'resnet50'},
    'fpn101': {'model_name': 'fpn101', 'conv_layer': 'resnet101'}}

model_key = 'fpn50'

model_name = available_models[model_key]['model_name']
base_conv_layer = available_models[model_key]['conv_layer']

pretrained_dir = os.path.join(project_dir, 'pretrained')
pretrained_filename = os.path.join(
    pretrained_dir, '{}_net.pth'.format(project_prefix))

num_classes = 20

