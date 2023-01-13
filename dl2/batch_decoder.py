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

import compressai
import numpy as np
import torch
from compressai.zoo import models
from dataloaders.datasets.lossyless_folder import LOSSYLESS_WEIGHTS, ClipCompressor
from dataloaders.datasets.utils import (
    crop_image,
    parse_header,
    read_bytes,
    read_uchars,
    read_uints,
)
from torchvision.datasets import DatasetFolder
from torchvision.transforms.functional import to_pil_image


def parse_args():
    parser = argparse.ArgumentParser(description="Decompress an image dataset using a PyTorch data loader")
    parser.add_argument("input", metavar="IN_DIR", type=str, help="source directory")
    parser.add_argument("-l", "--library", metavar="STR", choices=["compressai", "lossyless"],
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


def decode_c(net, data_loader, batch_times):
    for data, labels in iter(data_loader):
        batch_start = time.perf_counter()

        strings, shape, original_size = data
        strings = [s[0] for s in strings]
        shape = tuple(i.item() for i in shape)
        original_size = tuple(i.item() for i in original_size)

        with torch.no_grad():
            out = net.decompress(strings, shape)
        x_hat = crop_image(out["x_hat"], original_size)
        img = to_pil_image(x_hat.clamp_(0, 1).squeeze())

        batch_times.append(time.perf_counter() - batch_start)

    return batch_times


def decode_l(compressor, data_loader, batch_times):
    for data, labels in iter(data_loader):
        batch_start = time.perf_counter()

        s = data[0]
        Z_hat = compressor.decompress([s]).cpu().numpy()

        batch_times.append(time.perf_counter() - batch_start)

    return batch_times


def c_loader(path):
    with open(path, "rb") as f:
        _ = read_uchars(f, 2)  # skip the header because we assume the same header for all samples
        original_size = read_uints(f, 2)
        shape = read_uints(f, 2)
        strings = []
        n_strings = read_uints(f, 1)[0]
        for _ in range(n_strings):
            s = read_bytes(f, read_uints(f, 1)[0])
            strings.append([s])

        return strings, shape, original_size


def l_loader(path):
    with open(path, "rb") as f:
        return read_bytes(f, read_uints(f, 1)[0])


if __name__ == "__main__":
    start = time.perf_counter()

    args = parse_args()
    args.input = os.path.expandvars(args.input)
    args.workspace = os.path.expandvars(args.workspace)
    if args.experiment_name is not None:
        args.workspace = os.path.join(args.workspace, args.experiment_name)
    os.makedirs(args.workspace, exist_ok=True)
    batch_times = []

    if args.library == "compressai":
        compressai.set_entropy_coder("ans")   # we only support one coder for now
        # We assume the same header for all images
        for i in os.listdir(args.input):
            for j in os.listdir(os.path.join(args.input, i)):
                for k in os.listdir(os.path.join(args.input, i, j)):
                    with open(os.path.join(args.input, i, j, k), "rb") as f:
                        model, metric, quality = parse_header(read_uchars(f, 2))
                    break
                break
            break

        net = models[model](quality=quality, metric=metric, pretrained=True).eval()
        loader = c_loader
    elif args.library == "lossyless":
        path = LOSSYLESS_WEIGHTS.format(beta=args.beta)
        compressor = ClipCompressor(torch.hub.load_state_dict_from_url(path, progress=False), device="cpu")
        loader = l_loader

    for subset in ["train", "val"]:
        dataset = DatasetFolder(
            os.path.join(args.input, subset),
            loader=loader,
            extensions=(".ptci",)
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # CompressAI does not support batched decoding
            num_workers=args.workers,
            shuffle=False,
            drop_last=False
        )

        if args.library == "compressai":
            decode_c(net, data_loader, batch_times)
        elif args.library == "lossyless":
            decode_l(compressor, data_loader, batch_times)

    total_time = time.perf_counter() - start

    log = {
        "dataset": args.input,
        "total_time": total_time,
        "batch_times": batch_times
    }

    with open(os.path.join(args.workspace, args.report_file), "w") as f:
        json.dump(log, f, indent=4)

    print(f"Decompressed dataset in {total_time:.2f} seconds.")
    print(f"Average batch decoding time: {np.array(batch_times).mean():.6f} seconds")
