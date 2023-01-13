# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
from typing import BinaryIO, Tuple

import torch
from compressai.zoo import image as image_zoo
from compressai.zoo import models
from torch.nn.functional import pad

EXTENSIONS = (".ptci",)
METRICS = list(sorted(set([k2 for k1, v in image_zoo.model_urls.items() for k2 in v.keys()])))
METRIC_IDS = {k: i for i, k in enumerate(METRICS)}
MODEL_IDS = {k: i for i, k in enumerate(models.keys())}
INVERSE_METRIC_IDS = {i: k for i, k in enumerate(METRICS)}
INVERSE_MODEL_IDS = {i: k for i, k in enumerate(models.keys())}


def get_header(model_name: str, metric: str, quality: int) -> Tuple[int, int]:
    """Format header information:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    metric = METRIC_IDS[metric]
    code = (metric << 4) | (quality - 1 & 0x0F)
    return MODEL_IDS[model_name], code


def parse_header(header: Tuple[int, int]) -> Tuple[str, str, int]:
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    model_id, code = header
    quality = (code & 0x0F) + 1
    metric = code >> 4
    return INVERSE_MODEL_IDS[model_id], INVERSE_METRIC_IDS[metric], quality


def crop_image(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    _, _, h_pad, w_pad = x.size()
    h, w = size
    padding_left = (w_pad - w) // 2
    padding_right = w_pad - w - padding_left
    padding_top = (h_pad - h) // 2
    padding_bottom = h_pad - h - padding_top
    return pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def pad_image(x: torch.Tensor, p: int = 2 ** 6) -> torch.Tensor:
    _, _, h, w = x.size()
    h_pad = (h + p - 1) // p * p
    w_pad = (w + p - 1) // p * p
    padding_left = (w_pad - w) // 2
    padding_right = w_pad - w - padding_left
    padding_top = (h_pad - h) // 2
    padding_bottom = h_pad - h - padding_top
    return pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def write_bytes(fd: BinaryIO, values: str, fmt: str = ">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd: BinaryIO, n: int, fmt: str = ">{:d}s") -> Tuple[str]:
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_uchars(fd: BinaryIO, values: Tuple[int, ...], fmt: str = ">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uchars(fd: BinaryIO, n: int, fmt: str = ">{:d}B") -> Tuple[int, ...]:
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_uints(fd: BinaryIO, values: Tuple[int, ...], fmt: str = ">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd: BinaryIO, n: int, fmt: str = ">{:d}I") -> Tuple[int, ...]:
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))
