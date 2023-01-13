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

from torchvision.datasets import ImageFolder

from .compressai_folder import CompressAIFolder, compressai_worker_init_fn
from .lossyless_folder import LossylessFolder
from .min_io_folder import MinIOFolder, min_io_worker_init_fn

DATASETS = {
    "ImageFolder": ImageFolder,
    "CompressAIFolder": CompressAIFolder,
    "LossylessFolder": LossylessFolder,
    "MinIOFolder": MinIOFolder
}
WORKER_INIT_FNS = {
    "ImageFolder": None,
    "CompressAIFolder": compressai_worker_init_fn,
    "LossylessFolder": None,
    "MinIOFolder": min_io_worker_init_fn
}
DATASET_CHOICES = list(DATASETS.keys())
