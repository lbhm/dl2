#   Copyright 2020 Yann Dubois
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

import clip
import torch
from compressai.entropy_models import EntropyBottleneck
from compressai.models.utils import update_registered_buffers
from PIL import Image
from torchvision.datasets import DatasetFolder

from .utils import EXTENSIONS, read_bytes, read_uints

LOSSYLESS_WEIGHTS = "https://github.com/YannDubs/lossyless/releases/download/v1.0/beta{beta:0.0e}_factorized_rate.pt"


class LossylessFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            **kwargs
    ) -> None:
        super().__init__(root, self.decode, EXTENSIONS, transform, target_transform)

        path = LOSSYLESS_WEIGHTS.format(beta=kwargs["beta"])
        self.net = ClipCompressor(torch.hub.load_state_dict_from_url(path, progress=False), device="cpu")

    def decode(self, path: str) -> Image.Image:
        """ Loads a Pillow Image from a .ptci file. """
        with open(path, "rb") as f:
            s = read_bytes(f, read_uints(f, 1)[0])
            return self.net.decompress([s]).cpu().numpy()


class ClipCompressor(torch.nn.Module):
    """Our CLIP compressor.

    Parameters
    ----------
    pretrained_state_dict : dict or str or Path
        State dict of pretrained paths of the compressor. Can also be a path to the weights
        to load.

    device : str
        Device on which to load the model.
    """

    def __init__(
            self,
            pretrained_state_dict,
            device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        model, self.preprocess = clip.load("ViT-B/32", jit=False, device=device)
        self.clip = model.visual

        self.z_dim = 512
        self.side_z_dim = 512 // 5

        # => as if you use entropy coding that uses different scales in each dim
        self.scaling = torch.nn.Parameter(torch.ones(self.z_dim))
        self.biasing = torch.nn.Parameter(torch.zeros(self.z_dim))

        self.entropy_bottleneck = EntropyBottleneck(
            self.z_dim, init_scale=10, filters=[3, 3, 3, 3]
        )

        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            pretrained_state_dict,
        )  # compressai needs special because dynamic sized weights
        self.load_state_dict(pretrained_state_dict, strict=False)
        self.entropy_bottleneck.update()

        self.device = device
        self.to(self.device)
        self.eval()

    def to(self, device):
        self.device = device
        return super().to(device)

    @torch.cuda.amp.autocast(False)  # precision here is important
    def forward(self, X, is_compress=False):
        """Takes a batch of images as input and featurizes it with or without compression.
        Parameters
        ----------
        X : torch.Tensor shape=(batch_size,3,224,224)
            Batch of images, should be normalized (with CLIP normalization).
        is_compress : bool, optional
            Whether to return the compressed features instead of decompressed.
        Return
        ------
        if is_compress:
            byte_str : bytes
        else:
            z_hat : torch.Tensor shape=(batch_size,512)
        """
        with torch.no_grad():
            z = self.clip(X)

            z_in = self.process_z_in(z)

            if is_compress:
                out = self.entropy_bottleneck.compress(z_in)
            else:
                z_hat, _ = self.entropy_bottleneck(z_in)
                out = self.process_z_out(z_hat)

        return out

    def process_z_in(self, z):
        """Preprocessing of representation before entropy bottleneck."""
        z_in = (z.float() + self.biasing) * self.scaling.exp()
        # compressai needs 4 dimension (images) as input
        return z_in.unsqueeze(-1).unsqueeze(-1)

    def process_z_out(self, z_hat):
        """Postprocessing of representation after entropy bottleneck."""
        # back to vector
        z_hat = z_hat.squeeze(-1).squeeze(-1)
        return (z_hat / self.scaling.exp()) - self.biasing

    def compress(self, X):
        """Return comrpessed features (byte string)."""
        return self(X, is_compress=True)

    def decompress(self, byte_str):
        """Decompressed the byte strings."""
        with torch.no_grad():
            z_hat = self.entropy_bottleneck.decompress(byte_str, [1, 1])
            return self.process_z_out(z_hat)
