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

import torch


class SyntheticDataLoader(object):
    def __init__(self, batch_size, num_classes, num_channels, height, width, one_hot, memory_format, num_synth_samples):
        data = (
            torch.randn(batch_size, num_channels, height, width)
            .contiguous(memory_format=memory_format)
            .cuda()
            .normal_(0, 1.0)
        )

        if one_hot:
            label = torch.empty(batch_size, num_classes).cuda()
            label[:, 0] = 1.0
        else:
            label = torch.randint(0, num_classes, (batch_size,)).cuda()

        self.synthetic_data = data
        self.synthetic_label = label
        self.num_samples = num_synth_samples
        self.batch_size = batch_size

    def __iter__(self):
        count = 0
        while count < self.num_samples:
            yield self.synthetic_data, self.synthetic_label
            count += self.batch_size

    def __len__(self):
        return self.num_samples


def get_synthetic_loader(image_size, batch_size, num_classes, one_hot, memory_format=torch.contiguous_format, **kwargs):
    return (
        SyntheticDataLoader(
            batch_size, num_classes, 3, image_size, image_size, one_hot, memory_format, kwargs["num_synth_samples"]
        ),
        kwargs["num_synth_samples"],
    )
