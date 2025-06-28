#!/usr/bin/env python3
# Benchmark tool used to collect metrics.

import time
import os
from collections import OrderedDict
import numpy as np
import json

cur_dir = os.path.dirname(os.path.abspath(__file__))

adaptive_strategy_env = os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT')
adaptive_cnt = int(adaptive_strategy_env) if (adaptive_strategy_env
                                              is not None) else 0


def load_metrics_from_config():
    required_metrics = {}
    with open(cur_dir + "/configs.json", 'r') as f:
        configs = json.load(f)
        for key, value in configs.items():
            if os.getenv(key) is None:
                continue
            required_metrics[key] = value
    return required_metrics


required_metrics = load_metrics_from_config()


def get_platform():
    import glob
    import fileinput
    files_driver = glob.glob("/proc/driver/*/*/*/information")
    if not len(files_driver):
        return "Unkonw"
    finput = fileinput.input(files_driver[0])
    for line in finput:
        # MLU device information
        if "Device name" in line:
            return line.split(":")[-1].strip()
        # GPU device information
        elif "Model" in line:
            return line.split(":")[-1].strip()
    finput.close()
    return "Unkonw"


cur_platform = get_platform()


def get_dataset():
    dataset = os.getenv("DATASET_NAME")
    if dataset is None:
        return "unknow"
    return dataset


cur_dataset = get_dataset()


class AggregatorMeter(object):

    def __init__(self, *args):
        self.meters = []
        for meter in args:
            self.meters.append(meter)

    def update(self, val):
        for meter in self.meters:
            meter.update(val)

    def result(self):
        results = [meter.result() for meter in self.meters]
        return results


class MaxMeter(object):

    def __init__(self, name):
        self.name = name
        self.max = None

    def reset(self):
        self.max = None

    def update(self, val):
        if self.max is None:
            self.max = val
        else:
            self.max = max(self.max, val)

    def result(self):
        return (self.name, self.max)

    def __str__(self):
        pass


class AverageMeter(object):

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0

    def update(self, val):
        self.count += 1
        self.val += val

    def result(self):
        if self.count == 0:
            return None
        return (self.name, round(self.val / self.count, 3))

    def __str__(self):
        pass


class VarianceMeter(object):

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.vals = []
        self.count = 0

    def update(self, val):
        self.count += 1
        self.vals += [val]

    def result(self):
        if self.count == 0:
            return None
        return (self.name, round(np.var(self.vals), 6))

    def __str__(self):
        pass


class ElapsedTimer(object):

    def __init__(self, count_down=0):
        self.meter = AggregatorMeter(AverageMeter("batch_time_avg"),
                                     VarianceMeter("batch_time_var"))
        self.count_down = count_down

    def place(self):
        self.last_time_stamp = time.time()

    def clock(self):
        now = time.time()
        elapsed_time = now - self.last_time_stamp
        return elapsed_time

    def record(self):
        elapsed_time = self.clock()
        if self.count_down > 0:
            self.count_down -= 1
            return
        self.meter.update(elapsed_time)

    def data(self):
        return self.meter.result()


class HardwareTimer(object):

    def __init__(self, count_down=0):
        import torch_mlu
        from torch_mlu.core.device.notifier import Notifier
        self.meter = AggregatorMeter(AverageMeter("hardware_time_avg"),
                                     VarianceMeter("hardware_time_var"))
        self.start_notifier = Notifier()
        self.end_notifier = Notifier()
        self.count_down = count_down

    def place(self):
        self.start_notifier.place()

    def clock(self):
        self.end_notifier.place()
        self.end_notifier.synchronize()
        # unit of hardware_time should be second.
        hardware_time = self.start_notifier.hardware_time(
            self.end_notifier) / 1000.0 / 1000.0
        return hardware_time

    def record(self):
        hardware_time = self.clock()
        if self.count_down > 0:
            self.count_down -= 1
            return
        self.meter.update(hardware_time)

    def data(self):
        return self.meter.result()


class MemoryProfiler(object):

    def __init__(self):
        raise Exception("Not Implemented")

    def record(self):
        raise Exception("Not Implemented")


class Dumper(object):

    def __init__(self, name, save_path, target):
        self.name = name
        self.save_path = save_path
        self.target = target

    def exception_handle(self, contents, exception):
        """exception method will be triggered when collected metrics 
           can not meet the requirements in the config.json"""
        if exception == "throughput":
            keys = contents.keys()
            if "batch_time_avg" not in keys or "cards" not in keys or "batch_size" not in keys:
                return "unknow"
            batch_time = contents["batch_time_avg"]
            cards = contents["cards"]
            batch_size = contents["batch_size"]
            return round(batch_size / batch_time * cards, 2)
        else:
            return "unknow"

    def dump(self, contents):
        from datetime import datetime
        date_now = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        output = {"date": date_now}
        for target in self.target:
            if target not in contents.keys():
                output[target] = self.exception_handle(contents, target)
            else:
                output[target] = contents[target]
        output["device"] = cur_platform
        output["dataset"] = cur_dataset
        with open(self.save_path, 'a') as f:
            json.dump(output, f, indent=4)
            f.write("\n")


class MetricCollector(object):

    def __init__(self,
                 enable=True,
                 enable_only_benchmark=False,
                 enable_only_avglog=False,
                 record_elapsed_time=False,
                 record_hardware_time=False,
                 profile_memory=False):

        self._enabled = self.check_enable(enable, enable_only_benchmark,
                                          enable_only_avglog)
        if not self._enabled:
            return
        
        self.record_elapsed_time = record_elapsed_time
        self.record_hardware_time = record_hardware_time
        self.profile_memory = profile_memory

        self._recorders = []
        self._init_recorders()
        
        self._dumpers = []
        self._init_dumpers()
        
        self._metrics = OrderedDict()
        self._insert_metrics = {}


    def check_enable(self, enable, enable_only_benchmark, enable_only_avglog):
        """For that envs like AVG_LOG not appear in user training script, the disablement 
           of MetricCollecotr should meet the conditions as follows."""
        if not enable:
            return False

        # At least on env(AVG_LOG or BENCHMARK_LOG) in the config.json exsits.
        if len(required_metrics) == 0:
            return False

        # If enable_only_benchmark equals True when construct MetricCollector, there must exists
        # env BENCHMARK_LOG, enable_only_avglog the same (hardcode like BENCHMARK_LOG and AVG_LOG is
        # not elegant to appear here, but...).
        if enable_only_benchmark and os.getenv("BENCHMARK_LOG") is None:
            return False
        if enable_only_avglog and os.getenv("AVG_LOG") is None:
            return False

        return True

    def _init_recorders(self):
        if self.record_elapsed_time:
            self._recorders.append(ElapsedTimer(count_down=adaptive_cnt))
        if self.record_hardware_time:
            self._recorders.append(HardwareTimer(count_down=adaptive_cnt))
        if self.profile_memory:
            self._recorders.append(MemoryProfiler())

    def _init_dumpers(self):
        for key, value in required_metrics.items():
            file_path = os.getenv(key)
            self._dumpers.append(Dumper(key, file_path, value))

    def place(self):
        """call all recorder's place method."""
        if not self._enabled:
            return
        for recorder in self._recorders:
            recorder.place()

    def record(self):
        """call all recorder's record method."""
        if not self._enabled:
            return
        for recorder in self._recorders:
            recorder.record()

    def __str__(self):
        return str(self.get_metrics())

    def insert_metrics(self, **kwargs):
        self._insert_metrics = kwargs

    def update_recorder_metrics(self):
        for recorder in self._recorders:
            result = recorder.data()
            if len(result) == 0:
                print(
                    "MetricCollector have not recorded any datas, please ensure \
                       you have called place() and record() method or check if iters \
                       lower than adaptive_cnt : {}.".format(adaptive_cnt))
                return

            if isinstance(result, tuple):
                name, val = result
                self._metrics[name] = val
            elif isinstance(result, list):
                for name, val in result:
                    self._metrics[name] = val
            else:
                raise "Unknow result type of recorder."

    def update_metrics(self):
        if not self._enabled:
            return
        self.update_recorder_metrics()
        for key, value in self._insert_metrics.items():
            self._metrics[key] = value

    def get_metrics(self):
        if not self._enabled:
            return OrderedDict()
        self.update_metrics()
        return self._metrics

    def dump(self):
        if not self._enabled:
            return
        self.update_metrics()
        for dumper in self._dumpers:
            dumper.dump(self._metrics)
