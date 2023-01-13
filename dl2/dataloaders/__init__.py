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

from .dali import get_dali_train_loader, get_dali_val_loader
from .pytorch import get_pytorch_train_loader, get_pytorch_val_loader
from .synthetic import get_synthetic_loader

DATA_LOADER_CHOICES = ["dali-cpu", "dali-gpu", "pytorch", "synthetic"]


def get_data_loaders(choice):
    """Returns factory methods for a train and val data loader based on the given choice."""
    if choice == "dali-cpu":
        return get_dali_train_loader(dali_cpu=True), get_dali_val_loader(dali_cpu=True)
    elif choice == "dali-gpu":
        return get_dali_train_loader(dali_cpu=False), get_dali_val_loader(dali_cpu=False)
    elif choice == "pytorch":
        return get_pytorch_train_loader, get_pytorch_val_loader
    elif choice == "synthetic":
        return get_synthetic_loader, get_synthetic_loader
    else:
        raise ValueError(f"Please choose a valid data loader out of {DATA_LOADER_CHOICES}.")
