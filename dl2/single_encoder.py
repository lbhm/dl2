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
import shutil
import time
from multiprocessing import Pool
from typing import Tuple, List

import compressai
import numpy as np
import pandas as pd
import torch
from PIL import Image
from compressai.models import CompressionModel
from compressai.zoo import image as image_zoo, models
from torchvision.transforms.functional import to_tensor

from dataloaders.datasets.utils import get_header, pad_image, METRICS, write_bytes, write_uchars, write_uints


def parse_args():
    parser = argparse.ArgumentParser(description="Compress an image dataset using Python multiprocessing")
    subparsers = parser.add_subparsers(dest="lib", help="compression library to use")

    compress_ai = subparsers.add_parser("compressai", help="CompressAI")
    setup_common_args(compress_ai)
    pillow = subparsers.add_parser("pillow", help="Pillow")
    setup_common_args(pillow)
    subparsers.add_parser("list-methods", help="list available methods and exit")

    # CompressAI arguments
    compress_ai.add_argument("-a", "--arch", choices=models.keys(), default=list(models.keys())[0],
                             help="DNN architecture to use (default: %(default)s)")
    compress_ai.add_argument("-m", "--metric", choices=METRICS, default=METRICS[0],
                             help="metric trained against (default: %(default)s)")
    compress_ai.add_argument("-q", "--quality", choices=list(range(1, 9)), default=3, type=int,
                             help="quality setting (default: %(default)s)")

    # Pillow arguments
    pillow.add_argument("-f", "--format", metavar="STR", default="JPEG", choices=["JPEG", "JPEG2000", "WebP"],
                        help="compression algorithm to use (JPEG, JPEG2000, WebP), default: JPEG")
    format_specific = pillow.add_argument_group("format-specific arguments")
    format_specific.add_argument("--jpeg2000-mode", metavar="STR", default="rates", choices=["rates", "dB"],
                                 help="JPEG2000 quality mode: rates or dB (default: %(default)s)")
    format_specific.add_argument("--lossless", action="store_true", help="use lossless WebP compression")
    format_specific.add_argument("--quality", metavar="N", type=int, default=80,
                                 help="JPEG or WebP compression quality (default: %(default)s)")
    format_specific.add_argument("--quality-list", metavar="PATH", type=str,
                                 help="path to a CSV file with a filename to current quality mapping of the images to "
                                      "convert (only JPEG)")
    format_specific.add_argument("--webp-method", metavar="N", type=int, default=4,
                                 help="quality/speed trade-off: 0=fast, 6=slower/better (default: %(default)s)")

    return parser.parse_args()


def setup_common_args(parser):
    parser.add_argument("input", metavar="IN_DIR", type=str, help="source directory")
    parser.add_argument("output", metavar="OUT_DIR", type=str, help="destination directory")
    parser.add_argument("-w", "--workers", metavar="N", type=int, default=os.cpu_count(),
                        help="number of worker processes (default: %(default)s)")
    parser.add_argument("--experiment-name", metavar="NAME", type=str, default=None,
                        help="name of tan optional folder within the workspace in which all logs will be "
                             "saved (default: %(default)s)")
    parser.add_argument("--report-file", metavar="NAME", type=str, default=f"experiment_report.json",
                        help="file name of the log (default: %(default)s)")
    parser.add_argument("--workspace", metavar="DIR", type=str, default="$DL2_HOME/logs",
                        help="path to where the directory with logs will be saved (default: %(default)s)")


def list_model_configs():
    print("### CompressAI ###")
    print("(Architecture, training metrics, and quality level)")
    for model, v1 in image_zoo.model_urls.items():
        print(model)
        for metric, v2 in v1.items():
            print(f"\t{metric}: ", end="")
            print(*v2.keys(), sep=", ")

    print("\n### Pillow ###")
    print("JPEG\nJPEG2000\nWebP")


def encode_c(img: Image, output: str, net: CompressionModel, header: Tuple[int, int]) -> Tuple[float, float]:
    enc_start = time.perf_counter()
    x = to_tensor(img.convert("RGB")).unsqueeze(0)
    _, _, h, w = x.size()
    p = 64  # maximum 6 strides of 2
    x = pad_image(x, p)

    with torch.no_grad():
        out = net.compress(x)

    with open(output, "wb") as f:
        write_uchars(f, header)
        write_uints(f, (h, w))  # original image size
        write_uints(f, (*out["shape"], len(out["strings"])))  # shape and number of encoded latents
        for s in out["strings"]:
            write_uints(f, (len(s[0]),))
            write_bytes(f, s[0])

    enc_time = time.perf_counter() - enc_start
    size = os.path.getsize(output)
    bpp = float(size) * 8 / (img.size[0] * img.size[1])

    return enc_time, bpp


def encode_p(img: Image, output: str, fmt: str) -> Tuple[float, float]:
    enc_start = time.perf_counter()
    if img.mode not in ["RGB", "L"]:
        img = img.convert("RGB")
    if fmt == "JPEG":
        output = output + ".jpeg"
        img.save(output, format="JPEG", quality=args.quality)
    elif args.format == "JPEG2000":
        output = output + ".jp2"
        img.save(output, format="JPEG2000", quality_mode=args.jpeg2000_mode)
    elif fmt == "WebP":
        output = output + ".webp"
        img.save(output, format="WebP", lossless=args.lossless, method=args.webp_method, quality=args.quality)
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    enc_time = time.perf_counter() - enc_start
    size = os.path.getsize(output)
    bpp = float(size) * 8 / (img.size[0] * img.size[1])

    return enc_time, bpp


def encode_chunk(files: List[Tuple[str, str]]) -> Tuple[Tuple[float], Tuple[float]]:
    metrics = []

    if args.lib == "compressai":
        compressai.set_entropy_coder("ans")  # we only support one coder for now
        net = models[args.arch](quality=args.quality, metric=args.metric, pretrained=True).eval()
        header = get_header(args.arch, args.metric, args.quality)

        for in_file, out_file in files:
            os.makedirs(os.path.split(out_file)[0], exist_ok=True)
            with Image.open(in_file) as img:
                metrics.append(encode_c(img, os.path.splitext(out_file)[0] + ".ptci", net, header))
    elif args.lib == "pillow":
        if args.format == "JPEG" and args.quality_list:
            args.quality_df = pd.read_csv(os.path.expandvars(args.quality_list), index_col=0)
        else:
            args.quality_df = None

        for in_file, out_file in files:
            os.makedirs(os.path.split(out_file)[0], exist_ok=True)

            # In case of JPEGs, do not recompress images with current_quality < target_quality
            if args.quality_df is not None and args.format == "JPEG" and \
                    args.quality_df.at["/".join(in_file.split("/")[-3:]), "quality"] <= args.quality:
                shutil.copyfile(in_file, out_file)
            else:
                with Image.open(in_file) as img:
                    metrics.append(encode_p(img, os.path.splitext(out_file)[0], args.format))
    else:
        raise ValueError(f"Library {args.lib} is not supported.")

    return tuple(zip(*metrics))


if __name__ == "__main__":
    start = time.perf_counter()

    args = parse_args()
    if args.lib == "list-methods":
        list_model_configs()
        exit(0)

    args.input = os.path.expandvars(args.input)
    args.output = os.path.expandvars(args.output)
    args.workspace = os.path.expandvars(args.workspace)
    if args.experiment_name is not None:
        args.workspace = os.path.join(args.workspace, args.experiment_name)
    os.makedirs(args.workspace, exist_ok=True)

    in_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(args.input) for f in filenames]
    out_files = [os.path.join(args.output, *f.split("/")[-3:]) for f in in_files]
    files = list(zip(in_files, out_files))
    chunks = []
    for i in range(0, args.workers):
        chunks.append(files[i::args.workers])

    with Pool(processes=args.workers) as pool:
        results = pool.map(encode_chunk, chunks)

    total_time = time.perf_counter() - start
    results = list(zip(*results))
    enc_times = np.hstack(results[0])
    bpp = np.hstack(results[1])
    assert len(enc_times) == len(bpp)

    log = {
        "dataset": args.input,
        "total_time": total_time,
        "enc_times": enc_times.tolist(),
        "bpp": bpp.tolist()
    }

    with open(os.path.join(args.workspace, args.report_file), "w") as f:
        json.dump(log, f, indent=4)

    print(f"Compressed {len(enc_times)} images in {total_time:.2f} seconds.")
    print(f"Average encoding time: {enc_times.mean():.6f} seconds")
    print(f"Average bits per pixel (BPP): {bpp.mean():.6f}")
