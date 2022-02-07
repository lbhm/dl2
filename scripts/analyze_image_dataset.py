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
import os
import shutil
import subprocess
from multiprocessing import Pool

from PIL import Image, ImageStat


def analyze_dir(d):
    file_name = d.split("/")[-2] + "_" + d.split("/")[-1] + ".log"
    parent = f"{d.split('/')[-2]}/{d.split('/')[-1]}"

    with open(os.path.join("/tmp/dl2", file_name), "w") as log:
        for file in os.listdir(d):
            output = f"{parent}/{file}"

            if "quality" in args.analyses:
                quality = subprocess\
                    .check_output(["magick", "identify", "-format", "%Q", os.path.join(d, file)])\
                    .decode("utf-8")
                output += f",{quality}"

            if "size" in args.analyses:
                size = os.path.getsize(os.path.join(d, file))
                output += f",{size}"

            if "stats" in args.analyses:
                img = Image.open(os.path.join(d, file))
                mode = img.mode
                img = img.convert("RGB")
                stats = ImageStat.Stat(img)
                mean = stats.mean
                std = stats.stddev
                output += f",{mode},{mean[0]},{mean[1]},{mean[2]},{std[0]},{std[1]},{std[2]}"

            log.write(output + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze characteristics of a dataset")
    parser.add_argument("-s", "--source", metavar="DATA_DIR", required=True, help="dataset directory")
    parser.add_argument("-o", "--out", metavar="OUT_FILE", required=True, help="file to write the output")
    parser.add_argument("-a", "--analyses", metavar="A", required=True, nargs="+", choices=["quality", "size", "stats"],
                        help="analysis types to run, choices: quality (requires magick), size, stats")
    parser.add_argument("-w", "--workers", metavar="N", type=int, default=os.cpu_count(),
                        help="number of worker processes")
    args = parser.parse_args()

    if os.path.isdir("/tmp/dl2"):
        shutil.rmtree("/tmp/dl2")
    os.makedirs("/tmp/dl2")

    dirs = []
    for i in os.listdir(args.source):
        if i != "train" and i != "val":
            continue
        for j in os.listdir(os.path.join(args.source, i)):
            dirs.append(os.path.join(args.source, i, j))

    with Pool(processes=args.workers) as pool:
        pool.map(analyze_dir, dirs)

    headers = "file"
    if "quality" in args.analyses:
        headers += ",quality"
    if "size" in args.analyses:
        headers += ",size"
    if "stats" in args.analyses:
        headers += ",mode,mean_0,mean_1,mean_2,std_0,std_1,std_2"

    subprocess.run(f"echo '{headers}' > {args.out}", shell=True)
    subprocess.run(f"cat /tmp/dl2/*.log >> {args.out}", shell=True)
    shutil.rmtree("/tmp/dl2")
