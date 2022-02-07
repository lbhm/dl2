# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, label):
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -log_probs.gather(dim=-1, index=label.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class NLLMultiLabelSmooth(nn.Module):
    def __init__(self, smoothing=0.0):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            log_probs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -log_probs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -log_probs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class MixUpWrapper(object):
    def __init__(self, alpha, dataloader):
        self.alpha = alpha
        self.dataloader = dataloader

    def _mixup_loader(self, loader):
        for data, label in loader:
            mixup_data, mixup_label = mixup(self.alpha, data, label)
            yield mixup_data, mixup_label

    def __iter__(self):
        return self._mixup_loader(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def mixup(alpha, data, label):
    with torch.no_grad():
        batch_size = data.size(0)
        c = np.random.beta(alpha, alpha)

        perm = torch.randperm(batch_size).cuda()

        mixup_data = c * data + (1 - c) * data[perm, :]
        mixup_label = c * label + (1 - c) * label[perm, :]
        return mixup_data, mixup_label
