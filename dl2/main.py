# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-2021, Facebook, Inc
# Copyright (c) 2021, Lennart Behme
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import random
import time

import dllogger
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from dataloaders import DATA_LOADER_CHOICES as DL_CHOICES
from dataloaders import get_data_loaders
from dataloaders.datasets import DATASET_CHOICES
from models.model_and_loss import ModelAndLoss
from models.model_zoo import ARCH_CHOICES, CONFIG_CHOICES
from models.optimizers import get_optimizer
from training import train_loop
from utils.logging import Logger, format_step
from utils.lr_policies import LR_POLICY_CHOICES as LR_CHOICES
from utils.lr_policies import get_lr_policy
from utils.smoothing import LabelSmoothing, MixUpWrapper, NLLMultiLabelSmooth


def parse_args():
    parser = argparse.ArgumentParser(description="Image Classification with PyTorch")

    parser.add_argument("-a", "--arch", metavar="STR", default="resnet50", choices=ARCH_CHOICES,
                        help="model architecture: " + " | ".join(ARCH_CHOICES) + " (default: %(default)s)")
    parser.add_argument("-b", "--batch-size", metavar="N", type=int, default=256,
                        help="mini-batch size per gpu (default: %(default)s)")
    parser.add_argument("-d", "--data", metavar="DIR", required=True, help="path to dataset")
    parser.add_argument("-e", "--epochs", metavar="N", type=int, default=4,
                        help="number of total epochs to run (default: %(default)s)")
    parser.add_argument("-l", "--data-loader", metavar="STR", default=DL_CHOICES[0], choices=DL_CHOICES,
                        help="data loader backend: " + " | ".join(DL_CHOICES) + f" (default: %(default)s")
    parser.add_argument("-n", "--experiment-name", metavar="STR", default=None,
                        help="name of an optional folder within the workspace in which all logs and checkpoints will "
                             "be saved")
    parser.add_argument("-p", "--print-freq", metavar="N", type=int, default=100,
                        help="print frequency (default: %(default)s)")
    parser.add_argument("-w", "--workers", metavar="N", type=int, default=4,
                        help="number of data loading workers (default: %(default)s)")
    parser.add_argument("--amp", action="store_true", help="use AMP (automatic mixed precision) mode")
    parser.add_argument("--augmentation", metavar="STR", type=str, default=None, choices=[None, "autoaugment"],
                        help="data augmentation method (default: %(default)s)")
    parser.add_argument("--backup-checkpoints", action="store_true",
                        help="backup checkpoints throughout training, without this flag only best and last "
                             "checkpoints will be stored")
    parser.add_argument("--bn-weight-decay", action="store_true",
                        help="use weight_decay on batch normalization learnable parameters")
    parser.add_argument("--cache-size", metavar="N", type=int, default=1073742000,
                        help="size of a dedicated program-level cache in bytes (default 10 GiB)")
    parser.add_argument("--checkpoint-filename", metavar="FILE", type=str, default="checkpoint.pth.tar")
    parser.add_argument("--dataset-class", metavar="STR", default="ImageFolder", choices=DATASET_CHOICES,
                        help="dataset class to use " + " | ".join(DATASET_CHOICES) + " (default: %(default)s)")
    parser.add_argument("--data-mean", metavar="LIST", type=lambda s: [float(item) for item in s.split(',')],
                        default=[0.481 * 255, 0.457 * 255, 0.408 * 255],
                        help="mean RGB pixel values as a comma-separated list on 0-255 scale "
                             "(default: [122.655, 116.535, 104.04])")
    parser.add_argument("--data-std", metavar="LIST", type=lambda s: [float(item) for item in s.split(',')],
                        default=[0.233 * 255, 0.229 * 255, 0.230 * 255],
                        help="standard deviation of RGB pixel values as a comma-separated list on 0-255 scale "
                             "(default: [59.415, 58.395, 58.65])")
    parser.add_argument("--dynamic-loss-scale", action="store_true",
                        help="use dynamic loss scaling. If supplied, this argument supersedes --static-loss-scale.")
    parser.add_argument("--end-lr", default=0, type=float,
                        help="final learning rate of cosine LR policy (ignored for other policies)")
    parser.add_argument("--freeze-inner-layers", action="store_true",
                        help="freeze all model weights but the last layer")
    parser.add_argument("--image-size", default=224, type=int, help="resolution of images")
    parser.add_argument("--interpolation", metavar="INTERPOLATION", default="bilinear",
                        help="interpolation type for resizing images: bilinear, bicubic or triangular (DALI only)")
    parser.add_argument("--label-smoothing", metavar="N", type=float, default=0.1,
                        help="label smoothing factor (default: %(default)s)")
    parser.add_argument("--lr", "--learning-rate", metavar="N", type=float, default=0.1,
                        help="initial learning rate (default: %(default)s)")
    parser.add_argument("--lr-policy", metavar="POLICY", default=LR_CHOICES[0], choices=LR_CHOICES,
                        help="type of LR policy: " + " | ".join(DL_CHOICES) + f" (default: {LR_CHOICES[0]})")
    parser.add_argument("--memory-format", metavar="FORMAT", type=str, default="nhwc", choices=["nchw", "nhwc"],
                        help="memory layout: nchw or nhwc (default: %(default)s)")
    parser.add_argument("--mixup", metavar="N", type=float, default=0.0, help="mixup alpha (default %(default)s)")
    parser.add_argument("--model-config", metavar="CONF", default="classic", choices=CONFIG_CHOICES,
                        help="model configurations for nvidia models, ignored for PyTorch Hub models: " +
                             " | ".join(CONFIG_CHOICES) + f"(default: %(default)s)")
    parser.add_argument("--momentum", metavar="N", type=float, default=0.875, help="momentum (default: %(default)s)")
    parser.add_argument("--nesterov", action="store_true", help="use nesterov momentum")
    parser.add_argument("--no-checkpoints", action="store_false", dest="save_checkpoints",
                        help="do not store any checkpoints, useful for benchmarking")
    parser.add_argument("--no-persistent-workers", action="store_false", dest="persistent_workers",
                        help="disables persistent workers when using the PyTorch data loader")
    parser.add_argument("--num-classes", metavar="N", type=int, default=1000, help="number of classes in the dataset")
    parser.add_argument("--optimizer", default="sgd", type=str, choices=["sgd", "rmsprop"])
    parser.add_argument("--optimizer-batch-size", metavar="N", type=int, default=-1,
                        help="size of the total batch for simulating bigger batches using gradient accumulation")
    parser.add_argument("--pretrained-weights", metavar="PATH", type=str, default=None, help="load weights from here")
    parser.add_argument("--profile-n", metavar="N", type=int, default=-1,
                        help="profile n batches after the first epoch using the PyTorch profiler")
    parser.add_argument("--report-file", metavar="FILE", type=str, default="experiment_report.json",
                        help="file in which to store JSON experiment report")
    parser.add_argument("--resume", metavar="PATH", type=str, default=None, help="path to latest checkpoint")
    parser.add_argument("--rmsprop-alpha", default=0.9, type=float,
                        help="value of alpha parameter in rmsprop optimizer (default: %(default)s)")
    parser.add_argument("--rmsprop-eps", default=1e-3, type=float,
                        help="value of eps parameter in rmsprop optimizer (default: %(default)s)")
    parser.add_argument("--run-epochs", metavar="N", type=int, default=-1,
                        help="run only N epochs, used for checkpointing runs")
    parser.add_argument("--seed", metavar="N", type=int, default=None, help="random seed used for Numpy and PyTorch")
    parser.add_argument("--skip-training", action="store_true", help="only evaluate checkpoint/model")
    parser.add_argument("--skip-validation", action="store_true", help="do not evaluate")
    parser.add_argument("--static-loss-scale", metavar="N", type=float, default=1,
                        help="use static loss scale, positive power of 2 values can improve amp convergence")
    parser.add_argument("--synth-train-samples", metavar="N", type=int, default=-1,
                        help="number of synthetic train samples to generate per epoch, this is ignored if you don't "
                             "use the synthetic data loader")
    parser.add_argument("--synth-val-samples", metavar="N", type=int, default=-1,
                        help="number of synthetic validation samples to generate per epoch, this is ignored if you "
                             "don't use the synthetic data loader")
    parser.add_argument("--warmup", metavar="N", type=int, default=0, help="number of warmup epochs (default: 0)")
    parser.add_argument("--weight-decay", "--wd", metavar="N", type=float, default=1e-4,
                        help="weight decay (default: %(default)s)")
    parser.add_argument("--workspace", metavar="DIR", type=str, default="$DL2_HOME/logs",
                        help="path to where the directory with checkpoints and logs will be saved "
                             "(default: %(default)s)")

    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_args()

    best_acc1 = 0
    start_epoch = 0
    cudnn.benchmark = True

    # Argument consistency checks
    if args.freeze_inner_layers and not (args.pretrained_weights or args.resume):
        print(f"[ERROR] Freezing model weights without loading existing weights yields no useful outcome.")
        exit(1)
    if args.static_loss_scale != 1.0 and not args.amp:
        print("[WARN] If --amp is not used, static_loss_scale will be ignored.")

    # Setup of the workspace
    args.data = os.path.expandvars(args.data)
    args.workspace = os.path.expandvars(args.workspace)
    if args.experiment_name is not None:
        args.workspace = os.path.join(args.workspace, args.experiment_name)
    os.makedirs(args.workspace, exist_ok=True)

    # Distributed training
    args.distributed = False
    args.world_size = 1
    args.gpu = 0
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    # Random seed
    if args.seed is not None:
        print(f"[INFO] Using random seed = {args.seed}")
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(worker_id):
            np.random.seed(args.seed + args.local_rank + worker_id)
            random.seed(args.seed + args.local_rank + worker_id)
    else:
        def _worker_init_fn(worker_id):
            pass

    # Batch size multiplier for distributed training
    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        total_batch_size = args.world_size * args.batch_size
        if args.optimizer_batch_size % total_batch_size != 0:
            print(f"[WARN] Simulated batch size {args.optimizer_batch_size} is not divisible by "
                  f"actual batch size {total_batch_size}")
        batch_size_multiplier = int(args.optimizer_batch_size / total_batch_size)
        print(f"[INFO] Batch size multiplier: {batch_size_multiplier}")

    # Loading pretrained weights
    pretrained_weights = None
    if args.pretrained_weights:
        args.pretrained_weights = os.path.expandvars(args.pretrained_weights)
        if os.path.isfile(args.pretrained_weights):
            print(f"[INFO] Loading pretrained weights from '{args.pretrained_weights}'")
            pretrained_weights = torch.load(args.pretrained_weights)
            # Temporary fix to allow NGC checkpoint loading
            pretrained_weights = {k.replace("module.", ""): v for k, v in pretrained_weights.items()}
        else:
            print(f"[INFO] No pretrained weights found at '{args.resume}'")

    # Optionally resume from a checkpoint
    if args.resume:
        args.resume = os.path.expandvars(args.resume)
        if os.path.isfile(args.resume):
            print(f"[INFO] Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            model_state = checkpoint["state_dict"]
            optimizer_state = checkpoint["optimizer"]

            if start_epoch >= args.epochs:
                print(f"[ERROR] Launched training for {args.epochs} epochs but the checkpointed model already trained "
                      f"for {start_epoch} epochs.")
                exit(1)
            else:
                print(f"[INFO] Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"[WARN] No checkpoint found at '{args.resume}'. Initializing new model.")
            model_state = None
            optimizer_state = None
    else:
        model_state = None
        optimizer_state = None

    # Initialize loggers
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger = Logger(
            args.print_freq,
            [
                dllogger.StdOutBackend(dllogger.Verbosity.DEFAULT, step_format=format_step, prefix_format=lambda _: ""),
                dllogger.JSONStreamBackend(dllogger.Verbosity.VERBOSE, os.path.join(args.workspace, args.report_file))
            ],
            start_epoch=start_epoch - 1
        )
    else:
        logger = Logger(
            args.print_freq,
            [dllogger.StdOutBackend(dllogger.Verbosity.DEFAULT, step_format=format_step, prefix_format=lambda _: "")],
            start_epoch=start_epoch - 1
        )

    logger.log_parameter(args.__dict__, verbosity=dllogger.Verbosity.DEFAULT)

    # Set the memory format
    memory_format = torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format

    # Create data loaders
    train_loader = None
    val_loader = None
    get_train_loader, get_val_loader = get_data_loaders(args.data_loader)

    if not args.skip_training:
        train_loader, _ = get_train_loader(
            data_path=args.data, image_size=args.image_size, batch_size=args.batch_size, num_classes=args.num_classes,
            one_hot=args.mixup > 0.0, data_mean=args.data_mean, data_std=args.data_std, workers=args.workers,
            interpolation=args.interpolation, memory_format=memory_format, dataset_class=args.dataset_class,
            cache_size=args.cache_size, num_synth_samples=args.synth_train_samples, start_epoch=start_epoch,
            persistent_workers=args.persistent_workers
        )
        if args.mixup != 0.0:
            train_loader = MixUpWrapper(args.mixup, train_loader)

    if not args.skip_validation:
        val_loader, _ = get_val_loader(
            data_path=args.data, image_size=args.image_size, batch_size=args.batch_size, num_classes=args.num_classes,
            one_hot=False, data_mean=args.data_mean, data_std=args.data_std, workers=args.workers,
            interpolation=args.interpolation, memory_format=memory_format, dataset_class=args.dataset_class,
            cache_size=args.cache_size, num_synth_samples=args.synth_val_samples,
            persistent_workers=args.persistent_workers
        )

    # Define the loss
    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)

    # Create model wrapper and optimizer
    model_and_loss = ModelAndLoss(arch=(args.arch, args.model_config, args.num_classes), loss=loss,
                                  pretrained_weights=pretrained_weights, cuda=True, memory_format=memory_format,
                                  freeze_inner_layers=args.freeze_inner_layers)

    optimizer = get_optimizer(
        parameters=list(model_and_loss.model.named_parameters()), lr=args.lr, args=args, state=optimizer_state
    )

    scaler = torch.cuda.amp.GradScaler(
        init_scale=args.static_loss_scale,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=100 if args.dynamic_loss_scale else 1000000000,
        enabled=args.amp
    )

    if args.distributed:
        model_and_loss.distributed(args.gpu)

    model_and_loss.load_model_state(model_state)

    # Define a learning rate policy
    lr_policy = get_lr_policy(
        args.lr_policy, base_lr=args.lr, total_epochs=args.epochs, warmup_epochs=args.warmup, logger=logger,
        end_lr=args.end_lr, steps=[30, 60, 80], decay_factor=0.1, final_multiplier=0.001
    )

    print(f"[INFO] Boostrap complete after {time.time() - start_time} seconds.")

    end_epoch = (start_epoch + args.run_epochs) if args.run_epochs != -1 else args.epochs
    train_loop(
        model_and_loss=model_and_loss, optimizer=optimizer, scaler=scaler, lr_policy=lr_policy,
        train_loader=train_loader, val_loader=val_loader, logger=logger, use_amp=args.amp,
        batch_size_multiplier=batch_size_multiplier, best_acc1=best_acc1, profile_n=args.profile_n,
        start_epoch=start_epoch, end_epoch=end_epoch, skip_training=args.skip_training,
        skip_validation=args.skip_validation, save_checkpoints=args.save_checkpoints and not args.skip_training,
        checkpoint_dir=args.workspace, checkpoint_filename=args.checkpoint_filename,
        backup_checkpoints=args.backup_checkpoints, calc_train_acc=args.mixup == 0.0
    )

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.end()

    print(f"[INFO] Experiment ended after {time.time() - start_time} seconds.")


if __name__ == "__main__":
    main()
