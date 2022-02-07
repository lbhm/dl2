# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-2021, Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from abc import ABC

import torch
import torch.nn as nn


class ResNet(nn.Module, ABC):
    def __init__(self, model_dict, layer_factory,  num_classes=1000, fused_se=True):
        block = model_dict["block"]
        expansion = model_dict["expansion"]
        layers = model_dict["layers"]
        widths = model_dict["widths"]

        self.in_planes = 64
        self.fused_se = fused_se
        super(ResNet, self).__init__()
        self.conv1 = layer_factory.conv7x7(3, 64, stride=2)
        self.bn1 = layer_factory.batchnorm(64)
        self.relu = layer_factory.activation()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(layer_factory, block, expansion, widths[0], layers[0])
        self.layer2 = self._make_layer(layer_factory, block, expansion, widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(layer_factory, block, expansion, widths[2], layers[2], stride=2)
        self.layer4 = self._make_layer(layer_factory, block, expansion, widths[3], layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3] * expansion, num_classes)

    def _make_layer(self, layer_factory, block, expansion, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * expansion:
            dconv = layer_factory.conv1x1(self.in_planes, planes * expansion, stride=stride)
            dbn = layer_factory.batchnorm(planes * expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = [
            block(layer_factory, self.in_planes, planes, expansion, stride=stride, downsample=downsample,
                  fused_se=self.fused_se)
        ]
        self.in_planes = planes * expansion
        for i in range(1, blocks):
            layers.append(
                block(layer_factory, self.in_planes, planes, expansion, fused_se=self.fused_se)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetLayerFactory(object):
    def __init__(self, version, config):
        self.conv3x3_cardinality = (1 if "cardinality" not in version.keys() else version["cardinality"])
        self.config = config

    def conv(self, kernel_size, in_planes, out_planes, groups=1, stride=1):
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, groups=groups, stride=stride,
                         padding=int((kernel_size - 1) / 2), bias=False)

        if self.config["nonlinearity"] == "relu":
            nn.init.kaiming_normal_(conv.weight, mode=self.config["conv_init"],
                                    nonlinearity=self.config["nonlinearity"])
        return conv

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, groups=self.conv3x3_cardinality, stride=stride)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride)
        return c

    def batchnorm(self, planes, last_bn=False):
        bn = nn.BatchNorm2d(planes)
        gamma_init_val = 0 if last_bn and self.config["last_bn_0_init"] else 1
        nn.init.constant_(bn.weight, gamma_init_val)
        nn.init.constant_(bn.bias, 0)

        return bn

    def activation(self):
        return self.config["activation"]()


class BasicBlock(nn.Module, ABC):
    def __init__(self, layer_factory, in_planes, planes, expansion, stride=1, se=None, se_squeeze=None, downsample=None,
                 fused_se=None):
        super(BasicBlock, self).__init__()
        self.conv1 = layer_factory.conv3x3(in_planes, planes, stride)
        self.bn1 = layer_factory.batchnorm(planes)
        self.relu = layer_factory.activation()
        self.conv2 = layer_factory.conv3x3(planes, planes * expansion)
        self.bn2 = layer_factory.batchnorm(planes * expansion, last_bn=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)
        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SqueezeAndExcitation(nn.Module, ABC):
    def __init__(self, planes, squeeze):
        super(SqueezeAndExcitation, self).__init__()
        self.squeeze = nn.Linear(planes, squeeze)
        self.expand = nn.Linear(squeeze, planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        out = self.squeeze(out)
        out = self.relu(out)
        out = self.expand(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3)

        return out


class Bottleneck(nn.Module, ABC):
    def __init__(self, layer_factory, in_planes, planes, expansion, stride=1, se=False, se_squeeze=16, downsample=None,
                 fused_se=True):
        super(Bottleneck, self).__init__()
        self.conv1 = layer_factory.conv1x1(in_planes, planes)
        self.bn1 = layer_factory.batchnorm(planes)
        self.conv2 = layer_factory.conv3x3(planes, planes, stride=stride)
        self.bn2 = layer_factory.batchnorm(planes)
        self.conv3 = layer_factory.conv1x1(planes, planes * expansion)
        self.bn3 = layer_factory.batchnorm(planes * expansion, last_bn=True)
        self.relu = layer_factory.activation()
        self.downsample = downsample
        self.stride = stride

        self.fused_se = fused_se
        self.squeeze = (SqueezeAndExcitation(planes * expansion, se_squeeze) if se else None)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.squeeze is None:
            out += residual
        else:
            if self.fused_se:
                out = torch.addcmul(residual, out, self.squeeze(out), value=1)
            else:
                out = residual + out * self.squeeze(out)

        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck, ABC):
    def __init__(self, layer_factory, in_planes, planes, expansion, stride=1, downsample=None, fused_se=True):
        super(SEBottleneck, self).__init__(layer_factory, in_planes, planes, expansion, stride=stride, se=True,
                                           se_squeeze=16, downsample=downsample, fused_se=fused_se)
