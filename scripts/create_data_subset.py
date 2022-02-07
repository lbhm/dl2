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
from multiprocessing import Pool


def create_dir(d):
    source, destination = d

    files = sorted(os.listdir(source), key=str.lower)
    if not os.path.isdir(destination):
        os.makedirs(destination)

    # The files within a class are expected to be unordered
    subset = files[:int(len(files) * args.size)]

    for f in subset:
        shutil.copyfile(os.path.join(source, f), os.path.join(destination, f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a subset of a dataset")
    parser.add_argument("source", metavar="SOURCE", type=str, help="source directory")
    parser.add_argument("destination", metavar="DEST", type=str, help="destination directory")
    parser.add_argument("-s", "--size", metavar="S", type=float, default=0.5,
                        help="size of the new dataset as of fraction of the source dataset")
    parser.add_argument("-w", "--workers", metavar="N", type=int, default=os.cpu_count(),
                        help="number of worker processes")
    args = parser.parse_args()

    dirs = []
    for i in os.listdir(args.source):
        if i != "train":
            continue
        for j in os.listdir(os.path.join(args.source, i)):
            dirs.append((os.path.join(args.source, i, j), os.path.join(args.destination, i, j)))

    with Pool(processes=args.workers) as pool:
        pool.map(create_dir, dirs)
