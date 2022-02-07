# License not provided by author
# Taken from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/image_classification

from abc import ABC

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .model_zoo import MODELS, MODEL_CONFIGS


class ModelAndLoss(nn.Module, ABC):
    def __init__(self, arch, loss, pretrained_weights=None, cuda=True, memory_format=torch.contiguous_format,
                 freeze_inner_layers=False):
        print("[INFO] Creating model '{}'".format(arch))
        super(ModelAndLoss, self).__init__()
        model = build_model(arch[0], arch[1], arch[2])

        if pretrained_weights is not None:
            print(f"[INFO] Using pre-trained model from file '{pretrained_weights}'")
            model.load_state_dict(pretrained_weights)
        if cuda:
            model = model.cuda().to(memory_format=memory_format)
        if freeze_inner_layers:
            for layer in list(model.children())[:-1]:
                layer.requires_grad_(False)

        # Define loss function (criterion) and optimizer
        criterion = loss()
        if cuda:
            criterion = criterion.cuda()

        self.arch = arch
        self.model = model
        self.loss = criterion

    def forward(self, data, label):
        output = self.model(data)
        loss = self.loss(output, label)

        return loss, output

    def distributed(self, gpu_id):
        self.model = DistributedDataParallel(self.model, device_ids=[gpu_id], output_device=gpu_id)

    def load_model_state(self, state):
        if state is not None:
            self.model.load_state_dict(state)


def build_model(model_key, config_key, num_classes, verbose=False):
    model_dict = MODELS[model_key]
    config_dict = MODEL_CONFIGS[config_key]

    layer_factory = model_dict["layer_factory"](model_dict, config_dict)
    if verbose:
        print("[INFO] Model class: {}".format(model_dict))
        print("[INFO] Model config: {}".format(config_dict))
        print("[INFO] Number of classes: {}".format(num_classes))
    model = model_dict["model_factory"](model_dict, layer_factory, num_classes)

    return model
