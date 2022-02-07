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

import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch-size",
        metavar="N",
        type=int,
        default=256,
        help="mini-batch size per gpu (default: %(default)s)"
    )
    parser.add_argument(
        "-d", "--data",
        metavar="DIR",
        required=True,
        help="path to dataset"
    )
    parser.add_argument(
        "-e", "--epochs",
        metavar="N",
        type=int,
        default=1,
        help="number of times to iterate over the data (default: %(default)s)"
    )
    parser.add_argument(
        "-w", "--workers",
        metavar="N",
        type=int,
        default=4,
        help="number of data loading workers (default: %(default)s)"
    )
    parser.add_argument(
        "--dali-cpu",
        action="store_true",
        help="only use DALI CPU operators (default: %(default)s)"
    )
    parser.add_argument(
        "--experiment-name",
        metavar="NAME",
        type=str,
        default=None,
        help="name of an optional folder within the workspace in which all logs will be saved "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--memory-format",
        metavar="F",
        type=str,
        choices=["nchw", "nhwc"],
        default="nchw",
        help="memory layout: nchw or nhwc, (default: %(default)s)"
    )
    parser.add_argument(
        "--queue-depth",
        metavar="N",
        type=int,
        default=1,
        help="DALI prefetch queue depth (default: %(default)s)"
    )
    parser.add_argument(
        "--report-file",
        metavar="NAME",
        type=str,
        default=f"experiment_report.json",
        help="file name of the log (default: %(default)s)"
    )
    parser.add_argument(
        "--workspace",
        metavar="DIR",
        type=str,
        default="$DL2_HOME/logs",
        help="path to where the directory with logs will be saved (default: %(default)s)"
    )

    return parser.parse_args()


@pipeline_def
def create_dali_pipeline(data_dir, num_shards, shard_id, dali_cpu=False):
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

    images, labels = fn.readers.file(
        file_root=data_dir,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=True,
        pad_last_batch=True,
        name="Reader"
    )
    images = fn.decoders.image(
        images,
        device=decoder_device,
        output_type=types.RGB,
        device_memory_padding=device_memory_padding,
        host_memory_padding=host_memory_padding,
        hw_decoder_load=0.65,
    )
    images = fn.resize(
        images,
        device=dali_device,
        size=(224, 224),
        interp_type=types.INTERP_LINEAR
    )

    return images, labels


class DALIWrapper(object):
    def __init__(self, pipeline, memory_format):
        self.pipeline = pipeline
        self.memory_format = memory_format

    @staticmethod
    def _get_wrapper(pipeline, memory_format):
        for i in pipeline:
            data = i[0]["data"].contiguous(memory_format=memory_format)
            label = torch.reshape(i[0]["label"], [-1]).cuda().long()
            yield data, label
        pipeline.reset()

    def __iter__(self):
        return DALIWrapper._get_wrapper(self.pipeline, self.memory_format)


def get_dali_loader(
        data_path, batch_size, workers=4, memory_format=torch.contiguous_format, prefetch_queue_depth=1, dali_cpu=False
):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = rank % torch.cuda.device_count()
    else:
        local_rank = 0
        world_size = 1

    train_dir = os.path.join(data_path, 'train')

    pipe = create_dali_pipeline(
        batch_size=batch_size,
        num_threads=workers,
        device_id=local_rank,
        seed=12 + local_rank,
        prefetch_queue_depth=prefetch_queue_depth,
        data_dir=train_dir,
        num_shards=world_size,
        shard_id=local_rank,
        dali_cpu=dali_cpu
    )

    pipe.build()
    loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    return DALIWrapper(loader, memory_format)


if __name__ == '__main__':
    start = time.perf_counter()
    args = parse_args()

    args.data = os.path.expandvars(args.data)
    args.workspace = os.path.expandvars(args.workspace)
    if args.experiment_name is not None:
        args.workspace = os.path.join(args.workspace, args.experiment_name)
    os.makedirs(args.workspace, exist_ok=True)
    memory_format = torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format

    loader = get_dali_loader(
        data_path=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        memory_format=memory_format,
        prefetch_queue_depth=args.queue_depth,
        dali_cpu=args.dali_cpu
    )

    epoch_times = []
    data_times = []
    for e in range(args.epochs):
        epoch_start = time.perf_counter()
        data_iter = enumerate(loader)
        times = []
        iter_end = time.perf_counter()
        for i, (data, label) in data_iter:
            times.append(time.perf_counter() - iter_end)
            iter_end = time.perf_counter()

        data_times.append(times)
        epoch_times.append(time.perf_counter() - epoch_start)

    total_time = time.perf_counter() - start
    log = {
        "data_times": data_times,
        "epoch_times": epoch_times,
        "total_time": total_time
    }

    with open(os.path.join(args.workspace, args.report_file), "w") as f:
        json.dump(log, f, indent=4)

    if args.epochs > 1:
        print(f"Average epoch time except first: {sum(epoch_times[1:]) / len(epoch_times[1:]):.2f}s")
    print(f"Total time: {total_time:.2f}s")
