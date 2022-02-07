#   Copyright 2021 Lennart Behme
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import torch.nn as nn
import torchvision.models

from .resnet_nvidia import BasicBlock, Bottleneck, ResNet, ResNetLayerFactory, SEBottleneck


__all__ = ["ARCH_CHOICES", "CONFIG_CHOICES", "MODELS", "MODEL_CONFIGS"]


def get_pytorch_hub_model(model_dict, layer_factory, num_classes):
    m = model_dict["callable"](num_classes=num_classes)
    return m


PYTORCH_HUB_MODELS = {}
for name in torchvision.models.__dict__:
    if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name]):
        PYTORCH_HUB_MODELS[name] = {
            "model_factory": get_pytorch_hub_model,
            "layer_factory": lambda model, config: None,
            "callable": torchvision.models.__dict__[name]
        }

NVIDIA_MODELS = {
    "resnet18-nvidia": {
        "model_factory": ResNet,
        "layer_factory": ResNetLayerFactory,
        "block": BasicBlock,
        "layers": [2, 2, 2, 2],
        "widths": [64, 128, 256, 512],
        "expansion": 1,
    },
    "resnet34-nvidia": {
        "model_factory": ResNet,
        "layer_factory": ResNetLayerFactory,
        "block": BasicBlock,
        "layers": [3, 4, 6, 3],
        "widths": [64, 128, 256, 512],
        "expansion": 1,
    },
    "resnet50-nvidia": {
        "model_factory": ResNet,
        "layer_factory": ResNetLayerFactory,
        "block": Bottleneck,
        "layers": [3, 4, 6, 3],
        "widths": [64, 128, 256, 512],
        "expansion": 4,
    },
    "resnet101-nvidia": {
        "model_factory": ResNet,
        "layer_factory": ResNetLayerFactory,
        "block": Bottleneck,
        "layers": [3, 4, 23, 3],
        "widths": [64, 128, 256, 512],
        "expansion": 4,
    },
    "resnet152-nvidia": {
        "model_factory": ResNet,
        "layer_factory": ResNetLayerFactory,
        "block": Bottleneck,
        "layers": [3, 8, 36, 3],
        "widths": [64, 128, 256, 512],
        "expansion": 4,
    },
    "resnext101-32x4d-nvidia": {
        "model_factory": ResNet,
        "layer_factory": ResNetLayerFactory,
        "block": Bottleneck,
        "cardinality": 32,
        "layers": [3, 4, 23, 3],
        "widths": [128, 256, 512, 1024],
        "expansion": 2,
    },
    "se-resnext101-32x4d-nvidia": {
        "model_factory": ResNet,
        "layer_factory": ResNetLayerFactory,
        "block": SEBottleneck,
        "cardinality": 32,
        "layers": [3, 4, 23, 3],
        "widths": [128, 256, 512, 1024],
        "expansion": 2,
    }
}

MODEL_CONFIGS = {
    "classic": {
        "conv": nn.Conv2d,
        "conv_init": "fan_out",
        "nonlinearity": "relu",
        "last_bn_0_init": False,
        "activation": lambda: nn.ReLU(inplace=True),
    },
    "fan_in": {
        "conv": nn.Conv2d,
        "conv_init": "fan_in",
        "nonlinearity": "relu",
        "last_bn_0_init": False,
        "activation": lambda: nn.ReLU(inplace=True),
    },
    "grp-fan_in": {
        "conv": nn.Conv2d,
        "conv_init": "fan_in",
        "nonlinearity": "relu",
        "last_bn_0_init": False,
        "activation": lambda: nn.ReLU(inplace=True),
    },
    "grp-fan_out": {
        "conv": nn.Conv2d,
        "conv_init": "fan_out",
        "nonlinearity": "relu",
        "last_bn_0_init": False,
        "activation": lambda: nn.ReLU(inplace=True),
    }
}

MODELS = {**PYTORCH_HUB_MODELS, **NVIDIA_MODELS}
ARCH_CHOICES = list(MODELS.keys())
CONFIG_CHOICES = list(MODEL_CONFIGS.keys())
