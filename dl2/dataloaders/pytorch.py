# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-2021, Facebook, Inc
# Copyright (c) 2021, Lennart Behme
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

import os
from functools import partial

import numpy as np
import torch
import torchvision.transforms as transforms

from .augmentation.autoaugment import AutoaugmentImageNetPolicy
from .datasets import DATASETS, WORKER_INIT_FNS


class PrefetchedWrapper(object):
    def __init__(self, dataloader, start_epoch, num_classes, one_hot, data_mean, data_std):
        self.dataloader = dataloader
        self.epoch = start_epoch
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.data_mean = data_mean
        self.data_std = data_std

    @staticmethod
    def _get_loader(loader, num_classes, one_hot, data_mean, data_std):
        mean = torch.tensor(data_mean).cuda().view(1, 3, 1, 1)
        std = torch.tensor(data_std).cuda().view(1, 3, 1, 1)

        stream = torch.cuda.Stream()
        first = True

        for next_sample, next_label in loader:
            with torch.cuda.stream(stream):
                next_sample = next_sample.cuda(non_blocking=True).float()
                next_sample = next_sample.sub_(mean).div_(std)

                next_label = next_label.cuda(non_blocking=True)
                if one_hot:
                    next_label = expand_tensor(num_classes, torch.float, next_label)

            if not first:
                yield sample, label
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            sample = next_sample
            label = next_label

        yield sample, label

    def __iter__(self):
        if self.dataloader.sampler is not None and isinstance(
                self.dataloader.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            self.dataloader.sampler.set_epoch(self.epoch)

        self.epoch += 1

        return PrefetchedWrapper._get_loader(
            self.dataloader, self.num_classes, self.one_hot, self.data_mean, self.data_std
        )

    def __len__(self):
        return len(self.dataloader)


def get_pytorch_train_loader(data_path, image_size, batch_size, num_classes, one_hot, data_mean, data_std,
                             dataset_class, interpolation="bilinear", augmentation=None, workers=4,
                             memory_format=torch.contiguous_format, **kwargs):
    train_dir = os.path.join(data_path, "train")
    interpolation = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR
    }[interpolation]

    transforms_list = [
        transforms.RandomResizedCrop(image_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
    ]
    if augmentation == "autoaugment":
        transforms_list.append(AutoaugmentImageNetPolicy())

    if dataset_class == "ImageFolder":
        train_dataset = DATASETS[dataset_class](
            train_dir,
            transforms.Compose(transforms_list)
        )
    else:
        train_dataset = DATASETS[dataset_class](
            train_dir,
            transforms.Compose(transforms_list),
            cache_size=kwargs["cache_size"]
        )

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        worker_init_fn=WORKER_INIT_FNS[dataset_class],
        pin_memory=True,
        collate_fn=partial(fast_collate, memory_format),
        drop_last=True,
        persistent_workers=kwargs["persistent_workers"]
    )

    return (
        PrefetchedWrapper(train_loader, kwargs["start_epoch"], num_classes, one_hot, data_mean, data_std),
        len(train_loader)
    )


def get_pytorch_val_loader(data_path, image_size, batch_size, num_classes, one_hot, data_mean, data_std,
                           dataset_class, interpolation="bilinear", workers=4, crop_padding=32,
                           memory_format=torch.contiguous_format, **kwargs):
    val_dir = os.path.join(data_path, "val")
    interpolation = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR
    }[interpolation]

    transforms_list = [
        transforms.Resize(image_size + crop_padding, interpolation=interpolation),
        transforms.CenterCrop(image_size),
    ]

    if dataset_class == "ImageFolder":
        val_dataset = DATASETS[dataset_class](
            val_dir,
            transforms.Compose(transforms_list)
        )
    else:
        val_dataset = DATASETS[dataset_class](
            val_dir,
            transforms.Compose(transforms_list),
            cache_size=kwargs["cache_size"]
        )

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=workers,
        worker_init_fn=WORKER_INIT_FNS[dataset_class],
        pin_memory=True,
        collate_fn=partial(fast_collate, memory_format),
        drop_last=False,
        persistent_workers=kwargs["persistent_workers"]
    )

    return (
        PrefetchedWrapper(val_loader, 0, num_classes, one_hot, data_mean, data_std),
        len(val_loader)
    )


def fast_collate(memory_format, batch):
    """A custom collate function for PyTorch data loaders."""
    images = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.int64)
    w = images[0].size[0]
    h = images[0].size[1]
    tensor = torch.zeros((len(images), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)

    for i, img in enumerate(images):
        np_array = np.asarray(img, dtype=np.uint8)
        if np_array.ndim < 3:
            np_array = np.expand_dims(np_array, axis=-1)
        np_array = np.rollaxis(np_array, 2)

        tensor[i] += torch.from_numpy(np_array.copy())

    return tensor, targets


def expand_tensor(num_classes, dtype, tensor):
    """A utility function to convert scalars to a one-hot encoding."""
    e = torch.zeros(tensor.size(0), num_classes, dtype=dtype, device=torch.device("cuda"))
    return e.scatter(1, tensor.unsqueeze(1), 1.0)
