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

import math

import dllogger
import numpy as np

from .logging import LR_METER

LR_POLICY_CHOICES = ["cosine", "step", "linear", "exponential"]


def get_lr_policy(choice, base_lr, total_epochs, warmup_epochs, logger=None, **kwargs):
    if choice == "step":
        return lr_step_policy(base_lr, warmup_epochs, logger, kwargs["steps"], kwargs["decay_factor"])
    elif choice == "linear":
        return lr_linear_policy(base_lr, total_epochs, warmup_epochs, logger)
    elif choice == "cosine":
        return lr_cosine_policy(base_lr, total_epochs, warmup_epochs, logger)
    elif choice == "exponential":
        return lr_exponential_policy(base_lr, total_epochs, warmup_epochs, logger, kwargs["final_multiplier"])
    else:
        raise ValueError(f"Please choose valid a learning rate policy out of {LR_POLICY_CHOICES}.")


def lr_step_policy(base_lr, warmup_epochs, logger, steps, decay_factor):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return _lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, total_epochs, warmup_epochs, logger):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            e = epoch - warmup_epochs
            es = total_epochs - warmup_epochs
            lr = base_lr * (1 - (e / es))
        return lr

    return _lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, total_epochs, warmup_epochs, logger, end_lr=0):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            e = epoch - warmup_epochs
            es = total_epochs - warmup_epochs
            lr = end_lr + (0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - end_lr))
        return lr

    return _lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(base_lr, total_epochs, warmup_epochs, logger, final_multiplier=0.001, decay_factor=None,
                          decay_step=1):
    """Exponential lr policy. Setting decay factor parameter overrides final_multiplier"""
    es = total_epochs - warmup_epochs

    if decay_factor is not None:
        epoch_decay = decay_factor
    else:
        epoch_decay = np.power(2, np.log2(final_multiplier) / math.floor(es/decay_step))

    def _lr_fn(iteration, epoch):
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            e = epoch - warmup_epochs
            lr = base_lr * (epoch_decay ** math.floor(e/decay_step))
        return lr

    return _lr_policy(_lr_fn, logger=logger)


def _lr_policy(lr_fn, logger):
    if logger is not None:
        logger.register_metric("lr", LR_METER(), verbosity=dllogger.Verbosity.VERBOSE)

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric("lr", lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr
