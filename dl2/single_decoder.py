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

import argparse
import json
import os
import time
from multiprocessing import Pool
from typing import List

import compressai
import numpy as np
import torch
from compressai.models import CompressionModel
from compressai.zoo import models
from dataloaders.datasets.lossyless_folder import LOSSYLESS_WEIGHTS, ClipCompressor
from dataloaders.datasets.utils import (
    crop_image,
    parse_header,
    read_bytes,
    read_uchars,
    read_uints,
)
from PIL import Image
from torchvision.transforms.functional import to_pil_image


def parse_args():
    parser = argparse.ArgumentParser(description="Decompress an image dataset using Python multiprocessing")
    parser.add_argument("input", metavar="IN_DIR", type=str, help="source directory")
    parser.add_argument("-l", "--library", metavar="STR", choices=["compressai", "lossyless", "pillow"],
                        help="compression library to use")
    parser.add_argument("-w", "--workers", metavar="N", type=int, default=os.cpu_count(),
                        help="number of worker processes (default: %(default)s)")
    parser.add_argument("--beta", choices=[0.05, 0.01, 0.1], type=float,
                        help="quality level of the lossyless model used for decompression")
    parser.add_argument("--experiment-name", metavar="NAME", type=str, default=None,
                        help="name of tan optional folder within the workspace in which all logs will be "
                             "saved (default: %(default)s)")
    parser.add_argument("--report-file", metavar="NAME", type=str, default=f"experiment_report.json",
                        help="file name of the log (default: %(default)s)")
    parser.add_argument("--workspace", metavar="DIR", type=str, default="$DL2_HOME/logs",
                        help="path to where the directory with logs will be saved (default: %(default)s)")

    return parser.parse_args()


def decode_c(path: str, net: CompressionModel) -> float:
    dec_start = time.perf_counter()
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
        out = net.decompress(strings, shape)
    x_hat = crop_image(out["x_hat"], original_size)
    img = to_pil_image(x_hat.clamp_(0, 1).squeeze())

    return time.perf_counter() - dec_start


def decode_l(path: str, compressor: ClipCompressor) -> float:
    dec_start = time.perf_counter()
    with open(path, "rb") as f:
        s = read_bytes(f, read_uints(f, 1)[0])
        Z_hat = compressor.decompress([s]).cpu().numpy()

    return time.perf_counter() - dec_start


def decode_p(path: str) -> float:
    dec_start = time.perf_counter()
    img = Image.open(path)
    return time.perf_counter() - dec_start


def decode_chunk(files: List[str]) -> List[float]:
    times = []

    if args.library == "compressai":
        # We assume the same header for all images
        with open(files[0], "rb") as f:
            model, metric, quality = parse_header(read_uchars(f, 2))
        net = models[model](quality=quality, metric=metric, pretrained=True).eval()
        compressai.set_entropy_coder("ans")  # we only support one coder for now

        for file in files:
            times.append(decode_c(file, net))
    elif args.library == "lossyless":
        for file in files:
            times.append(decode_l(file, compressor))
    elif args.library == "pillow":
        for file in files:
            times.append(decode_p(file))
    else:
        raise ValueError(f"Library {args.lib} is not supported.")

    return times


if __name__ == "__main__":
    start = time.perf_counter()

    args = parse_args()
    args.input = os.path.expandvars(args.input)
    args.workspace = os.path.expandvars(args.workspace)
    if args.experiment_name is not None:
        args.workspace = os.path.join(args.workspace, args.experiment_name)
    os.makedirs(args.workspace, exist_ok=True)

    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(args.input) for f in filenames]
    chunks = []
    for i in range(0, args.workers):
        chunks.append(files[i::args.workers])

    # Lossyless has a long model load time
    # CompressAI models do not work when loaded inside __main__
    if args.library == "lossyless":
        path = LOSSYLESS_WEIGHTS.format(beta=args.beta)
        compressor = ClipCompressor(torch.hub.load_state_dict_from_url(path, progress=False), device="cpu")

    with Pool(processes=args.workers) as pool:
        results = pool.map(decode_chunk, chunks)

    total_time = time.perf_counter() - start
    dec_times = np.array([item for sublist in results for item in sublist])

    log = {
        "dataset": args.input,
        "total_time": total_time,
        "dec_times": dec_times.tolist()
    }

    with open(os.path.join(args.workspace, args.report_file), "w") as f:
        json.dump(log, f, indent=4)

    print(f"Decompressed {len(dec_times)} images in {total_time:.2f} seconds.")
    print(f"Average decoding time: {dec_times.mean():.6f} seconds")
