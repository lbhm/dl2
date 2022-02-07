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

import torch
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy


@pipeline_def
def create_dali_pipeline(data_dir, num_shards, shard_id, interpolation, crop, size, mean, std, dali_cpu=False,
                         is_training=True):
    interpolation = {
        "bicubic": types.INTERP_CUBIC,
        "bilinear": types.INTERP_LINEAR,
        "triangular": types.INTERP_TRIANGULAR,
    }[interpolation]

    images, labels = fn.readers.file(
        file_root=data_dir,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=is_training,
        pad_last_batch=True,
        prefetch_queue_depth=1,
        name="Reader"
    )

    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

    if is_training:
        images = fn.decoders.image_random_crop(
            images,
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            hw_decoder_load=0.65,
            random_aspect_ratio=[0.75, 4.0 / 3.0],
            random_area=[0.1, 1.0],
            num_attempts=100
        )
        images = fn.resize(
            images,
            device=dali_device,
            size=(crop, crop),
            interp_type=interpolation
        )
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(
            images,
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            hw_decoder_load=0.65,
        )
        images = fn.resize(
            images,
            device=dali_device,
            size=size,
            mode="not_smaller",
            interp_type=interpolation
        )
        mirror = False

    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        crop=(crop, crop),
        mean=mean,
        std=std,
        mirror=mirror
    )
    labels = labels.gpu()

    return images, labels


class DALIWrapper(object):
    def __init__(self, pipeline, num_classes, one_hot, memory_format):
        self.pipeline = pipeline
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.memory_format = memory_format

    @staticmethod
    def _get_wrapper(pipeline, num_classes, one_hot, memory_format):
        for i in pipeline:
            data = i[0]["data"].contiguous(memory_format=memory_format)
            label = torch.reshape(i[0]["label"], [-1]).cuda().long()
            if one_hot:
                label = expand_tensor(num_classes, torch.float, label)
            yield data, label
        pipeline.reset()

    def __iter__(self):
        return DALIWrapper._get_wrapper(self.pipeline, self.num_classes, self.one_hot, self.memory_format)

    def __len__(self):
        return len(self.pipeline)


def get_dali_train_loader(dali_cpu=False):
    def gdtl(data_path, image_size, batch_size, num_classes, one_hot, data_mean, data_std, interpolation="bilinear",
             augmentation=None, workers=4, memory_format=torch.contiguous_format, **kwargs):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            local_rank = rank % torch.cuda.device_count()
        else:
            local_rank = 0
            world_size = 1

        train_dir = os.path.join(data_path, 'train')

        if augmentation is not None:
            raise NotImplementedError(f"Augmentation {augmentation} is not supported for the DALI loader.")

        pipe = create_dali_pipeline(
            batch_size=batch_size,
            num_threads=workers,
            device_id=local_rank,
            seed=12 + local_rank,
            data_dir=train_dir,
            num_shards=world_size,
            shard_id=local_rank,
            interpolation=interpolation,
            crop=image_size,
            size=image_size,
            mean=data_mean,
            std=data_std,
            dali_cpu=dali_cpu,
            is_training=True
        )

        pipe.build()
        train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

        return (
            DALIWrapper(train_loader, num_classes, one_hot, memory_format),
            int(pipe.epoch_size("Reader") / (world_size * batch_size))
        )

    return gdtl


def get_dali_val_loader(dali_cpu=False):
    def gdvl(data_path, image_size, batch_size, num_classes, one_hot, data_mean, data_std, interpolation="bilinear",
             crop_padding=32, workers=4, memory_format=torch.contiguous_format, **kwargs):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            local_rank = rank % torch.cuda.device_count()
        else:
            local_rank = 0
            world_size = 1

        val_dir = os.path.join(data_path, 'val')

        pipe = create_dali_pipeline(
            batch_size=batch_size,
            num_threads=workers,
            device_id=local_rank,
            seed=12 + local_rank,
            data_dir=val_dir,
            num_shards=world_size,
            shard_id=local_rank,
            interpolation=interpolation,
            crop=image_size,
            size=image_size + crop_padding,
            mean=data_mean,
            std=data_std,
            dali_cpu=dali_cpu,
            is_training=False
        )

        pipe.build()
        val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

        return (
            DALIWrapper(val_loader, num_classes, one_hot, memory_format),
            int(pipe.epoch_size("Reader") / (world_size * batch_size))
        )

    return gdvl


def expand_tensor(num_classes, dtype, tensor):
    """A utility function to convert scalars to a one-hot encoding."""
    e = torch.zeros(tensor.size(0), num_classes, dtype=dtype, device=torch.device("cuda"))
    return e.scatter(1, tensor.unsqueeze(1), 1.0)
