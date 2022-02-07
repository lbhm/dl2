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

from collections import OrderedDict

import dllogger
import numpy as np

PERF_METER = lambda: Meter(AverageMeter(), AverageMeter(), AverageMeter())
LOSS_METER = lambda: Meter(AverageMeter(), AverageMeter(), MinMeter())
ACC_METER = lambda: Meter(AverageMeter(), AverageMeter(), MaxMeter())
LR_METER = lambda: Meter(LastMeter(), LastMeter(), LastMeter())

LAT_100 = lambda: Meter(QuantileMeter(1), QuantileMeter(1), QuantileMeter(1))
LAT_99 = lambda: Meter(QuantileMeter(0.99), QuantileMeter(0.99), QuantileMeter(0.99))
LAT_95 = lambda: Meter(QuantileMeter(0.95), QuantileMeter(0.95), QuantileMeter(0.95))


class Meter(object):
    def __init__(self, iteration_aggregator, epoch_aggregator, run_aggregator):
        self.iteration_aggregator = iteration_aggregator
        self.epoch_aggregator = epoch_aggregator
        self.run_aggregator = run_aggregator

    def record(self, val, n=1):
        self.iteration_aggregator.record(val, n=n)

    def get_iteration(self):
        v, n = self.iteration_aggregator.get_val()
        return v

    def reset_iteration(self):
        v, n = self.iteration_aggregator.get_data()
        self.iteration_aggregator.reset()
        if v is not None:
            self.epoch_aggregator.record(v, n=n)

    def get_epoch(self):
        v, n = self.epoch_aggregator.get_val()
        return v

    def reset_epoch(self):
        v, n = self.epoch_aggregator.get_data()
        self.epoch_aggregator.reset()
        if v is not None:
            self.run_aggregator.record(v, n=n)

    def get_run(self):
        v, n = self.run_aggregator.get_val()
        return v

    def reset_run(self):
        self.run_aggregator.reset()


class QuantileMeter(object):
    def __init__(self, q):
        self.q = q
        self.vals = []
        self.n = 0

    def reset(self):
        self.vals = []
        self.n = 0

    def record(self, val, n=1):
        if isinstance(val, list):
            self.vals += val
            self.n += len(val)
        else:
            self.vals += [val] * n
            self.n += n

    def get_val(self):
        if not self.vals:
            return None, self.n
        return np.quantile(self.vals, self.q, interpolation="nearest"), self.n

    def get_data(self):
        if len(self.vals):
            return self.vals, self.n
        else:
            return None, self.n


class MaxMeter(object):
    def __init__(self):
        self.max = None
        self.n = 0

    def reset(self):
        self.max = None
        self.n = 0

    def record(self, val, n=1):
        if self.max is None:
            self.max = val
        else:
            self.max = max(self.max, val)
        self.n = n

    def get_val(self):
        return self.max, self.n

    def get_data(self):
        return self.max, self.n


class MinMeter(object):
    def __init__(self):
        self.min = None
        self.n = 0

    def reset(self):
        self.min = None
        self.n = 0

    def record(self, val, n=1):
        if self.min is None:
            self.min = val
        else:
            self.min = min(self.min, val)
        self.n = n

    def get_val(self):
        return self.min, self.n

    def get_data(self):
        return self.min, self.n


class LastMeter(object):
    def __init__(self):
        self.last = None
        self.n = 0

    def reset(self):
        self.last = None
        self.n = 0

    def record(self, val, n=1):
        self.last = val
        self.n = n

    def get_val(self):
        return self.last, self.n

    def get_data(self):
        return self.last, self.n


class AverageMeter(object):
    def __init__(self):
        self.n = 0
        self.val = 0

    def reset(self):
        self.n = 0
        self.val = 0

    def record(self, val, n=1):
        self.n += n
        self.val += val * n

    def get_val(self):
        if self.n == 0:
            return None, 0
        return self.val / self.n, self.n

    def get_data(self):
        if self.n == 0:
            return None, 0
        return self.val / self.n, self.n


class Logger(object):
    def __init__(self, print_interval, backends, start_epoch=-1):
        self.epoch = start_epoch
        self.iteration = -1
        self.val_iteration = -1
        self.metrics = OrderedDict()
        self.backends = backends
        self.print_interval = print_interval
        dllogger.init(backends)

    def register_metric(self, metric_name, meter, verbosity=0, metadata=None):
        metadata = dict() if metadata is None else metadata
        self.metrics[metric_name] = {"meter": meter, "level": verbosity}
        dllogger.metadata(metric_name, metadata)

    @staticmethod
    def log_parameter(data, verbosity=0):
        dllogger.log(step=tuple("P"), data=data, verbosity=verbosity)

    def log_metric(self, metric_name, val, n=1):
        self.metrics[metric_name]["meter"].record(val, n=n)

    def start_iteration(self, val=False):
        if val:
            self.val_iteration += 1
        else:
            self.iteration += 1

    def end_iteration(self, val=False):
        iteration = self.val_iteration if val else self.iteration
        if iteration % self.print_interval == 0:
            metrics = {n: m for n, m in self.metrics.items() if n.startswith("v") == val}
            step = ("T", self.epoch, self.iteration) if not val else ("V", self.epoch, self.val_iteration)

            verbosities = {m["level"] for _, m in metrics.items()}
            for ll in verbosities:
                llm = {n: m for n, m in metrics.items() if m["level"] == ll}
                dllogger.log(step=step, data={n: m["meter"].get_iteration() for n, m in llm.items()}, verbosity=ll)

            for n, m in metrics.items():
                m["meter"].reset_iteration()

            dllogger.flush()

    def start_epoch(self):
        self.epoch += 1
        self.iteration = 0
        self.val_iteration = 0

        for n, m in self.metrics.items():
            m["meter"].reset_epoch()

    def end_epoch(self, epoch_time=None):
        for n, m in self.metrics.items():
            m["meter"].reset_iteration()

        verbosities = {m["level"] for _, m in self.metrics.items()}
        for ll in verbosities:
            llm = {n: m for n, m in self.metrics.items() if m["level"] == ll}
            dllogger.log(step=("E", self.epoch), data={n: m["meter"].get_epoch() for n, m in llm.items()}, verbosity=ll)

        if epoch_time is not None:
            dllogger.log(step=("E", self.epoch), data={"e.total_time": f"{epoch_time:.5f} s"},
                         verbosity=dllogger.Verbosity.DEFAULT)

    def end(self):
        for n, m in self.metrics.items():
            m["meter"].reset_epoch()

        verbosities = {m["level"] for _, m in self.metrics.items()}
        for ll in verbosities:
            llm = {n: m for n, m in self.metrics.items() if m["level"] == ll}
            dllogger.log(step=tuple("S"), data={n: m["meter"].get_run() for n, m in llm.items()}, verbosity=ll)

        for n, m in self.metrics.items():
            m["meter"].reset_epoch()

        dllogger.flush()

    def iteration_generator_wrapper(self, gen, val=False):
        for g in gen:
            self.start_iteration(val=val)
            yield g
            self.end_iteration(val=val)


def format_step(step):
    if step[0] == "S":
        return "Summary:"
    elif step[0] == "P":
        return "[INFO]"
    elif step[0] == "E":
        return f"Epoch [{step[1]}]"
    elif step[0] == "T":
        return f"Train [{step[1]}/{step[2]}]"
    elif step[0] == "V":
        return f"Val [{step[1]}/{step[2]}]"
    else:
        raise ValueError(f"Incorrect step type: {step[0]}")
