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

import datetime as dt
import os
import shutil
import time

import dllogger
import torch

from torch.cuda.amp import autocast

import utils.logging as logging
import utils.metrics as metrics
from utils.timeout_handler import TimeoutHandler

ACC_METADATA = {"unit": "%", "format": ":.5f"}
SPEED_METADATA = {"unit": "img/s", "format": ":.2f"}
TIME_METADATA = {"unit": "s", "format": ":.5f"}
LOSS_METADATA = {"format": ":.2f"}


def get_train_step(model_and_loss, optimizer, scaler, use_amp=False, batch_size_multiplier=1, calc_acc=True):
    def _step(data, labels, optimizer_step=True):
        with autocast(enabled=use_amp):
            loss, output = model_and_loss(data, labels)
            loss /= batch_size_multiplier
            if calc_acc:
                acc1, acc5 = metrics.accuracy(output, labels, top_k=(1, 5))
            else:
                acc1, acc5 = torch.zeros(1), torch.zeros(1)

            if torch.distributed.is_initialized():
                reduced_loss = reduce_tensor(loss)
                if calc_acc:
                    acc1 = reduce_tensor(acc1)
                    acc5 = reduce_tensor(acc5)
            else:
                reduced_loss = loss

        scaler.scale(loss).backward()

        if optimizer_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        return reduced_loss, acc1, acc5

    return _step


def train(train_loader, model_and_loss, optimizer, scaler, lr_policy, logger, epoch, timeout_handler, use_amp=False,
          profiler=None, batch_size_multiplier=1, calc_acc=True):
    step = get_train_step(model_and_loss, optimizer, scaler, use_amp, batch_size_multiplier, calc_acc)

    model_and_loss.train()
    end = time.perf_counter()

    optimizer.zero_grad()

    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)

    for i, (data, label) in data_iter:
        batch_size = data.size(0)
        data_time = time.perf_counter() - end

        lr_policy(optimizer, i, epoch)
        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss, acc1, acc5 = step(data, label, optimizer_step=optimizer_step)

        iteration_time = time.perf_counter() - end

        if logger is not None:
            logger.log_metric("t.top1", acc1.item(), batch_size)
            logger.log_metric("t.top5", acc5.item(), batch_size)
            logger.log_metric("t.loss", loss.item(), batch_size)
            logger.log_metric("t.data_speed", metrics.calc_speed(batch_size, data_time))
            logger.log_metric("t.compute_speed", metrics.calc_speed(batch_size, iteration_time - data_time))
            logger.log_metric("t.iteration_speed", metrics.calc_speed(batch_size, iteration_time))
            logger.log_metric("t.data_time", data_time)
            logger.log_metric("t.compute_time", iteration_time - data_time)
            logger.log_metric("t.iteration_time", iteration_time)

        end = time.perf_counter()
        if profiler:
            profiler.step()
        if timeout_handler.interrupted:
            break


def get_val_step(model_and_loss, use_amp=False):
    def _step(data, labels):
        with torch.no_grad(), autocast(enabled=use_amp):
            loss, output = model_and_loss(data, labels)
            acc1, acc5 = metrics.accuracy(output, labels, top_k=(1, 5))

            if torch.distributed.is_initialized():
                reduced_loss = reduce_tensor(loss)
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)
            else:
                reduced_loss = loss

        torch.cuda.synchronize()

        return reduced_loss, acc1, acc5

    return _step


def validate(val_loader, model_and_loss, logger, timeout_handler, use_amp=False, profiler=None):
    step = get_val_step(model_and_loss, use_amp)

    model_and_loss.eval()
    end = time.perf_counter()

    data_iter = enumerate(val_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)

    for i, (data, label) in data_iter:
        batch_size = data.size(0)
        data_time = time.perf_counter() - end

        loss, acc1, acc5 = step(data, label)

        iteration_time = time.perf_counter() - end

        if logger is not None:
            logger.log_metric("v.top1", acc1.item(), batch_size)
            logger.log_metric("v.top5", acc5.item(), batch_size)
            logger.log_metric("v.loss", loss.item(), batch_size)
            logger.log_metric("v.data_speed", metrics.calc_speed(batch_size, data_time))
            logger.log_metric("v.compute_speed", metrics.calc_speed(batch_size, iteration_time - data_time))
            logger.log_metric("v.iteration_speed", metrics.calc_speed(batch_size, iteration_time))
            logger.log_metric("v.data_time", data_time)
            logger.log_metric("v.compute_time", iteration_time - data_time)
            logger.log_metric("v.iteration_time", iteration_time)
            logger.log_metric("v.compute_latency", iteration_time - data_time)
            logger.log_metric("v.compute_latency_at95", iteration_time - data_time)
            logger.log_metric("v.compute_latency_at99", iteration_time - data_time)
            logger.log_metric("v.compute_latency_at100", iteration_time - data_time)

        end = time.perf_counter()
        if profiler:
            profiler.step()
        if timeout_handler.interrupted:
            break


def train_loop(model_and_loss, optimizer, scaler, lr_policy, train_loader, val_loader, logger, use_amp,
               batch_size_multiplier=1, best_acc1=0, start_epoch=0, end_epoch=0, profile_n=-1, skip_training=False,
               skip_validation=False, save_checkpoints=True, checkpoint_dir="$DL2_HOME/logs",
               checkpoint_filename="checkpoint.pth.tar", backup_checkpoints=False, calc_train_acc=True):
    print(f"[INFO] Running epochs {start_epoch} to {end_epoch}.")
    with TimeoutHandler() as timeout_handler:
        if logger is not None and not skip_training:
            register_train_metrics(logger)
        if logger is not None and not skip_validation:
            register_val_metrics(logger)

        if profile_n > 0:
            # Skip the first epoch of training and/or validation
            if not skip_training and not skip_validation:
                skip_steps = len(train_loader) + len(val_loader)
            elif not skip_training:
                skip_steps = len(train_loader)
            elif not skip_validation:
                skip_steps = len(val_loader)
            else:
                raise ValueError("Either training or validation must be enabled.")

            if torch.distributed.is_initialized():
                worker_name = f"rank_{torch.distributed.get_rank()}"
            else:
                worker_name = None

            with torch.profiler.profile(
                    schedule=torch.profiler.schedule(
                        wait=0,
                        warmup=1,
                        active=profile_n,
                        repeat=1,
                        skip_first=skip_steps
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        dir_name=checkpoint_dir,
                        worker_name=worker_name
                    )
            ) as profiler:
                pass
        else:
            profiler = None

        for epoch in range(start_epoch, end_epoch):
            if logger is not None:
                logger.start_epoch()
                start_time = dt.datetime.now()

            if not skip_training:
                train(train_loader, model_and_loss, optimizer, scaler, lr_policy, logger, epoch, timeout_handler,
                      use_amp, profiler, batch_size_multiplier,  calc_train_acc)
            if not skip_validation:
                validate(val_loader, model_and_loss, logger, timeout_handler, use_amp, profiler)

            if logger is not None:
                logger.end_epoch(epoch_time=(dt.datetime.now() - start_time).total_seconds())

            if save_checkpoints and (
                    not torch.distributed.is_initialized()
                    or torch.distributed.get_rank() == 0
            ):
                if not skip_validation:
                    is_best = logger.metrics["v.top1"]["meter"].get_epoch() > best_acc1
                    best_acc1 = max(logger.metrics["v.top1"]["meter"].get_epoch(), best_acc1)
                else:
                    is_best = False
                    best_acc1 = 0

                if backup_checkpoints and (epoch < 10 or epoch % 10 == 0):
                    backup_filename = f"checkpoint-epoch-{epoch}.pth.tar"
                else:
                    backup_filename = None

                save_checkpoint(
                    state={
                        "epoch": epoch + 1,
                        "model": model_and_loss.arch,
                        "state_dict": model_and_loss.model.state_dict(),
                        "best_acc1": best_acc1,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=is_best,
                    checkpoint_dir=checkpoint_dir,
                    backup_filename=backup_filename,
                    filename=checkpoint_filename,
                )


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
    return rt


def save_checkpoint(state, is_best, filename, checkpoint_dir, backup_filename=None,
                    is_best_filename="model_best.pth.tar"):
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        base_path = os.path.expandvars(checkpoint_dir)
        filename = os.path.join(base_path, filename)

        print("[INFO] Saving checkpoint at {}".format(filename))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(base_path, is_best_filename))
        if backup_filename is not None:
            shutil.copyfile(filename, os.path.join(base_path, backup_filename))


def register_train_metrics(logger):
    logger.register_metric(
        "t.top1",
        logging.ACC_METER(),
        verbosity=dllogger.Verbosity.DEFAULT,
        metadata=ACC_METADATA,
    )
    logger.register_metric(
        "t.top5",
        logging.ACC_METER(),
        verbosity=dllogger.Verbosity.DEFAULT,
        metadata=ACC_METADATA,
    )
    logger.register_metric(
        "t.loss",
        logging.LOSS_METER(),
        verbosity=dllogger.Verbosity.DEFAULT,
        metadata=LOSS_METADATA,
    )
    logger.register_metric(
        "t.data_speed",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.VERBOSE,
        metadata=SPEED_METADATA,
    )
    logger.register_metric(
        "t.compute_speed",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.VERBOSE,
        metadata=SPEED_METADATA,
    )
    logger.register_metric(
        "t.iteration_speed",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.VERBOSE,
        metadata=SPEED_METADATA,
    )
    logger.register_metric(
        "t.data_time",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.DEFAULT,
        metadata=TIME_METADATA,
    )
    logger.register_metric(
        "t.compute_time",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.DEFAULT,
        metadata=TIME_METADATA,
    )
    logger.register_metric(
        "t.iteration_time",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.VERBOSE,
        metadata=TIME_METADATA,
    )


def register_val_metrics(logger):
    logger.register_metric(
        "v.top1",
        logging.ACC_METER(),
        verbosity=dllogger.Verbosity.DEFAULT,
        metadata=ACC_METADATA,
    )
    logger.register_metric(
        "v.top5",
        logging.ACC_METER(),
        verbosity=dllogger.Verbosity.DEFAULT,
        metadata=ACC_METADATA,
    )
    logger.register_metric(
        "v.loss",
        logging.LOSS_METER(),
        verbosity=dllogger.Verbosity.DEFAULT,
        metadata=LOSS_METADATA,
    )
    logger.register_metric(
        "v.data_speed",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.VERBOSE,
        metadata=SPEED_METADATA,
    )
    logger.register_metric(
        "v.compute_speed",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.VERBOSE,
        metadata=SPEED_METADATA,
    )
    logger.register_metric(
        "v.iteration_speed",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.VERBOSE,
        metadata=SPEED_METADATA,
    )
    logger.register_metric(
        "v.data_time",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.DEFAULT,
        metadata=TIME_METADATA,
    )
    logger.register_metric(
        "v.compute_time",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.DEFAULT,
        metadata=TIME_METADATA,
    )
    logger.register_metric(
        "v.iteration_time",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.VERBOSE,
        metadata=TIME_METADATA,
    )
    logger.register_metric(
        "v.compute_latency",
        logging.PERF_METER(),
        verbosity=dllogger.Verbosity.VERBOSE,
        metadata=TIME_METADATA,
    )
    logger.register_metric(
        "v.compute_latency_at100",
        logging.LAT_100(),
        verbosity=dllogger.Verbosity.VERBOSE,
        metadata=TIME_METADATA,
    )
    logger.register_metric(
        "v.compute_latency_at99",
        logging.LAT_99(),
        verbosity=dllogger.Verbosity.VERBOSE,
        metadata=TIME_METADATA,
    )
    logger.register_metric(
        "v.compute_latency_at95",
        logging.LAT_95(),
        verbosity=dllogger.Verbosity.VERBOSE,
        metadata=TIME_METADATA,
    )
