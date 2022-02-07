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

import io
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import DatasetFolder

PILLOW_EXTENSIONS = (".jpg", ".jpeg", ".jp2", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def min_io_worker_init_fn(worker_id: int) -> None:
    """ This method is called on each worker subprocess after seeding and before data loading. """
    dataset = torch.utils.data.get_worker_info().dataset

    dataset.cache_shm = SharedMemory(dataset.cache_shm.name)
    dataset.cache = dataset.cache_shm.buf

    dataset.index_shm = SharedMemory(dataset.index_shm.name)
    dataset.index = np.ndarray(
        shape=(len(dataset) * 2 + 1,),
        dtype=np.int64,
        buffer=dataset.index_shm.buf
    )


def pil_loader(img_bytes: io.BytesIO) -> Image.Image:
    """ Loads a Pillow Image object from raw bytes. """
    img = Image.open(img_bytes)
    return img.convert("RGB")


class MinIOFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            **kwargs
    ) -> None :
        """ Instantiates a new ImageFolder object using MinIO. """
        super().__init__(root, pil_loader, PILLOW_EXTENSIONS, transform, target_transform)

        self.cache_size = kwargs["cache_size"] if "cache_size" in kwargs else 1073742000
        self.cache_shm = SharedMemory(create=True, size=self.cache_size)  # shared memory reference to the cache
        self.cache = self.cache_shm.buf  # bytearray holding the actual image data
        self.cache_full = False

        # Shared memory index to reference cached images
        # The index has a (first byte, sample length) pairs for each sample in the dataset
        # and a pointer to the next free byte at the last position
        self.index_shm = SharedMemory(create=True, size=(len(self) * 2 + 1) * 8)
        self.index = np.ndarray(
            shape=(len(self) * 2 + 1,),
            dtype=np.int64,
            buffer=self.index_shm.buf
        )
        self.index.fill(-1)
        self.index[-1] = 0
        self.index_lock = Lock()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """ Retrieves a sample from the dataset by its index. """
        path, target = self.samples[index]

        if self.index[index * 2] != -1:
            # Image saved in cache
            (pointer, sample_size) = self.index[index * 2:index * 2 + 2]
            image_bytes = io.BytesIO(self.cache[pointer:pointer + sample_size][:])
        else:
            # Image not cached
            with open(path, "rb") as f:
                image_bytes = io.BytesIO(f.read())

            if not self.cache_full:
                sample_size = len(image_bytes.getvalue())
                self.index_lock.acquire()
                pointer = self.index[-1]

                if pointer + sample_size <= self.cache_size:
                    # Cache has enough space for sample
                    self.index[-1] += sample_size
                    self.index_lock.release()
                    self.index[2 * index] = pointer
                    self.index[2 * index + 1] = sample_size
                    self.cache[pointer:pointer + sample_size] = image_bytes.getvalue()
                else:
                    # Cache is full
                    self.index_lock.release()
                    self.cache_full = True

        sample = self.loader(image_bytes)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
