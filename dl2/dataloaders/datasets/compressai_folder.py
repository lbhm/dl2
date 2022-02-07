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

from typing import Callable, Optional

import compressai
import torch
from PIL import Image
from compressai.zoo import models
from torchvision.datasets import DatasetFolder
from torchvision.transforms.functional import to_pil_image

from .utils import EXTENSIONS, crop_image, parse_header, read_bytes, read_uchars, read_uints


def compressai_worker_init_fn(worker_id: int) -> None:
    dataset = torch.utils.data.get_worker_info().dataset
    compressai.set_entropy_coder(dataset.coder)


class CompressAIFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            **kwargs
    ) -> None:
        super().__init__(root, self.decode, EXTENSIONS, transform, target_transform)

        # We assume the same model was used to compress the entire dataset
        with open(self.samples[0][0], "rb") as f:
            model, metric, quality = parse_header(read_uchars(f, 2))
        self.net = models[model](quality=quality, metric=metric, pretrained=True).eval()
        self.coder = "ans"  # we only support one coder for now

    def decode(self, path: str) -> Image.Image:
        """ Loads a Pillow Image from a .ptci file. """
        with open(path, "rb") as f:
            _ = read_uchars(f, 2)  # skip the header because we assume the same header for all samples
            original_size = read_uints(f, 2)
            shape = read_uints(f, 2)
            strings = []
            n_strings = read_uints(f, 1)[0]
            for _ in range(n_strings):
                s = read_bytes(f, read_uints(f, 1)[0])
                strings.append([s])

        with torch.no_grad():
            out = self.net.decompress(strings, shape)
        x_hat = crop_image(out["x_hat"], original_size)
        img = to_pil_image(x_hat.clamp_(0, 1).squeeze())

        if img.mode != "RGB":
            print("Non-RGB")
        return img.convert("RGB")
