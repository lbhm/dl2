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

import json
import os
import sys
import time
from argparse import ArgumentParser, REMAINDER
from subprocess import PIPE, Popen


def parse_args():
    parser = ArgumentParser(
        description="Train a model over multiple training runs using a training regimen"
    )
    parser.add_argument(
        "--stages",
        metavar="1,2,3,...",
        required=True,
        type=lambda x: [int(i) for i in x.split(',')],
        help="comma-separated list of epoch durations for each stage of the training regimen "
             "(e.g., 10,30,10 for a total of 50 epochs)"
    )
    parser.add_argument(
        "--datasets",
        metavar="PATH,...",
        required=True,
        type=lambda x: [s for s in x.split(',')],
        help="comma-separated list of paths to the datasets used for training, should be only one path or match the "
             "number of training stages"
    )
    parser.add_argument(
        "--freeze-in-last",
        action="store_true",
        help="freeze all model layers but the last in the last stage of training (default: %(default)s)"
    )
    parser.add_argument(
        "--experiment-name",
        metavar="NAME",
        type=str,
        default=f"experiment_{time.strftime('%Y%m%d_%H%M%S')}",
        help="name of the directory within the workspace in which all logs and checkpoints will be saved "
             "(default: experiment_{current time})"
    )
    parser.add_argument(
        "--workspace",
        metavar="DIR",
        type=str,
        default="$DL2_HOME/logs/training_regimen",
        help="path to where the directory with checkpoints and logs will be saved (default: %(default)s)"
    )
    parser.add_argument(
        "--training-script",
        metavar="PATH",
        required=True,
        type=lambda x: [s for s in x.split(' ')],
        help="absolute path to the training script to be launched in each stage of the regimen, can also be set to "
             "\"-m torch.distributed.run\" to launch distributed training"
    )
    parser.add_argument(
        "--script-args",
        metavar="args",
        required=True,
        type=str,
        nargs=REMAINDER,
        help="all following args are interpreted as input to the training script"
    )

    return parser.parse_args()


if __name__ == '__main__':
    start = time.time()
    args = parse_args()

    # Sanity checks
    assert len(args.stages) > 1
    assert all(val not in args.script_args
               for val in ["-d", "--data", "-n", "--experiment-name", "--no-checkpoints", "--workspace"])
    if len(args.datasets) == 1:
        assert args.freeze_in_last and len(args.stages) == 2
        args.datasets = args.datasets * len(args.stages)
    else:
        assert len(args.datasets) == len(args.stages)

    args.workspace = os.path.expandvars(args.workspace)
    total_epochs = sum(args.stages)
    prior_stage_checkpoint = ""
    log = {}
    for i, stage in enumerate(args.stages):
        stage_start = time.time()
        checkpoint_file = f"checkpoint_{i}.pth.tar"

        if args.freeze_in_last and i + 1 == len(args.stages):
            args.script_args.append("--freeze-inner-layers")

        cmd = [sys.executable, "-u", *args.training_script] + args.script_args + [
            "--checkpoint-filename", checkpoint_file, "--data", args.datasets[i], "--epochs", str(total_epochs),
            "--experiment-name", args.experiment_name, "--report-file", f"experiment_report_stage_{i}.json",
            "--resume", prior_stage_checkpoint, "--run-epochs", str(stage), "--workspace", args.workspace
        ]
        print(f"[INFO] Executing: {cmd}")
        with Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True) as p:
            print(f"===== Stage {i} stdout =====")
            for line in p.stdout:
                print(line, end="")

            print(f"===== Stage {i} stderr =====")
            for line in p.stderr:
                print(line, end="")

        print(f"===== Stage {i} end =====")

        prior_stage_checkpoint = os.path.join(args.workspace, args.experiment_name, checkpoint_file)
        log[f"stage_{i}"] = {
            "time": time.time() - stage_start,
            "epochs": stage,
            "dataset": args.datasets[i]
        }

    log["total_time"] = time.time() - start
    with open(os.path.join(args.workspace, args.experiment_name, "regimen_log.json"), "w") as f:
        json.dump(log, f, indent=4)
