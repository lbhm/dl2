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
from torchvision.datasets import ImageFolder
from torchvision import transforms

from dataloaders.datasets.lossyless_folder import ClipCompressor, LOSSYLESS_WEIGHTS
from dataloaders.datasets.utils import get_header, pad_image, METRICS, write_bytes, write_uchars, write_uints


def parse_args():
    parser = argparse.ArgumentParser(description="Compress an image dataset using a PyTorch data loader")
    subparsers = parser.add_subparsers(dest="lib", help="compression library to use")

    compress_ai = subparsers.add_parser("compressai", help="CompressAI")
    setup_common_args(compress_ai)
    lossyless = subparsers.add_parser("lossyless", help="lossyless")
    setup_common_args(lossyless)

    # CompressAI arguments
    compress_ai.add_argument("-a", "--arch", choices=models.keys(), default=list(models.keys())[0],
                             help="DNN architecture to use (default: %(default)s)")
    compress_ai.add_argument("-m", "--metric", choices=METRICS, default=METRICS[0],
                             help="metric trained against (default: %(default)s)")
    compress_ai.add_argument("-q", "--quality", choices=list(range(1, 9)), default=3, type=int,
                             help="quality setting (default: %(default)s)")

    # lossyless arguments
    lossyless.add_argument("--beta", choices=[0.05, 0.01, 0.1], type=float, help="quality level of the compressor")

    return parser.parse_args()


def setup_common_args(parser):
    parser.add_argument("input", metavar="IN_DIR", type=str, help="source directory")
    parser.add_argument("output", metavar="OUT_DIR", type=str, help="destination directory")
    parser.add_argument("-b", "--batch-size", metavar="N", type=int, default=128,
                        help="number of images to compress at once (default: %(default)s)")
    parser.add_argument("-w", "--workers", metavar="N", type=int, default=os.cpu_count(),
                        help="number of worker processes (default: %(default)s)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="computing device to use")
    parser.add_argument("--experiment-name", metavar="NAME", type=str, default=None,
                        help="name of tan optional folder within the workspace in which all logs will be "
                             "saved (default: %(default)s)")
    parser.add_argument("--report-file", metavar="NAME", type=str, default=f"experiment_report.json",
                        help="file name of the log (default: %(default)s)")
    parser.add_argument("--workspace", metavar="DIR", type=str, default="$DL2_HOME/logs",
                        help="path to where the directory with logs will be saved (default: %(default)s)")


class PathImageFolder(ImageFolder):
    """
    Extension of ImageFolder that returns the path of an image to reconstruct the original dataset structure
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


def encode_c(net, header, data_loader, batch_times):
    for data, labels, paths in iter(data_loader):
        batch_start = time.perf_counter()

        _, _, h, w = data.size()
        p = 64  # maximum 6 strides of 2
        data = data.to(args.device)
        data = pad_image(data, p)

        with torch.no_grad():
            out = net.compress(data)

        for i in range(len(out["strings"][0])):
            os.makedirs(os.path.join(args.output, *paths[i].split("/")[-3:-1]), exist_ok=True)

            file_name = os.path.splitext(paths[i].split("/")[-1])[0] + ".ptci"
            with open(os.path.join(args.output, *paths[i].split("/")[-3:-1], file_name), "wb") as f:
                write_uchars(f, header)
                write_uints(f, (h, w))  # original image size
                write_uints(f, (*out["shape"], len(out["strings"])))  # shape and number of encoded latents
                for s in out["strings"]:
                    write_uints(f, (len(s[i]),))
                    write_bytes(f, s[i])

        batch_times.append(time.perf_counter() - batch_start)

    return batch_times


def encode_l(compressor, data_loader, batch_times):
    for data, labels, paths in iter(data_loader):
        batch_start = time.perf_counter()

        Z_bytes = compressor.compress(data.to(compressor.device).half())

        for i, b in enumerate(Z_bytes):
            os.makedirs(os.path.join(args.output, *paths[i].split("/")[-3:-1]), exist_ok=True)

            file_name = os.path.splitext(paths[i].split("/")[-1])[0] + ".ptci"
            with open(os.path.join(args.output, *paths[i].split("/")[-3:-1], file_name), "wb") as f:
                write_uints(f, (len(b),))
                write_bytes(f, b)

        batch_times.append(time.perf_counter() - batch_start)

    return batch_times


if __name__ == "__main__":
    start = time.perf_counter()

    args = parse_args()
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
    args.input = os.path.expandvars(args.input)
    args.output = os.path.expandvars(args.output)
    args.workspace = os.path.expandvars(args.workspace)
    if args.experiment_name is not None:
        args.workspace = os.path.join(args.workspace, args.experiment_name)
    os.makedirs(args.workspace, exist_ok=True)
    batch_times = []

    if args.lib == "compressai":
        compressai.set_entropy_coder("ans")
        net = models[args.arch](quality=args.quality, metric=args.metric, pretrained=True).to(args.device).eval()
        header = get_header(args.arch, args.metric, args.quality)
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    elif args.lib == "lossyless":
        path = LOSSYLESS_WEIGHTS.format(beta=args.beta)
        compressor = ClipCompressor(torch.hub.load_state_dict_from_url(path, progress=False), device=args.device)
        transform = compressor.preprocess

    for subset in ["train", "val"]:
        # Since we have to pass batches of uniform size to the encoder, we perform resize and crop on the data
        dataset = PathImageFolder(
            os.path.join(args.input, subset),
            transform=transform
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )

        if args.lib == "compressai":
            encode_c(net, header, data_loader, batch_times)
        elif args.lib == "lossyless":
            encode_l(compressor, data_loader, batch_times)

    total_time = time.perf_counter() - start

    log = {
        "dataset": args.input,
        "total_time": total_time,
        "batch_times": batch_times
    }

    with open(os.path.join(args.workspace, args.report_file), "w") as f:
        json.dump(log, f, indent=4)

    print(f"Compressed dataset in {total_time:.2f} seconds.")
    print(f"Average batch encoding time: {np.array(batch_times).mean():.6f} seconds")
