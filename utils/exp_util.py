import argparse
from pathlib import Path
import numpy as np
import sys
import json
import yaml
import random
import pickle
from collections import defaultdict, OrderedDict


def parse_config_json(json_path: Path, args: argparse.Namespace = None):
    """
    Parse a json file and add key:value to args namespace.
    Json file format [ {attr}, {attr}, ... ]
        {attr} = { "_": COMMENT, VAR_NAME: VAR_VALUE }
    """
    if args is None:
        args = argparse.Namespace()

    with json_path.open() as f:
        json_text = f.read()

    try:
        raw_configs = json.loads(json_text)
    except:
        # Do some fixing of the json text
        json_text = json_text.replace("\'", "\"")
        json_text = json_text.replace("None", "null")
        json_text = json_text.replace("False", "false")
        json_text = json_text.replace("True", "true")
        raw_configs = json.loads(json_text)

    if isinstance(raw_configs, dict):
        raw_configs = [raw_configs]
    configs = {}
    for raw_config in raw_configs:
        for rkey, rvalue in raw_config.items():
            if rkey != "_":
                configs[rkey] = rvalue

    if configs is not None:
        for ckey, cvalue in configs.items():
            args.__dict__[ckey] = cvalue
    return args


def parse_config_yaml(yaml_path: Path, args: argparse.Namespace = None, override: bool = True):
    """
    Parse a yaml file and add key:value to args namespace.
    """
    if args is None:
        args = argparse.Namespace()
    with yaml_path.open() as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    if configs is not None:
        if "include_configs" in configs.keys():
            base_config = configs["include_configs"]
            del configs["include_configs"]
            base_config_path = yaml_path.parent / Path(base_config)
            with base_config_path.open() as f:
                base_config = yaml.load(f, Loader=yaml.FullLoader)
            base_config.update(configs)
            configs = base_config
        for ckey, cvalue in configs.items():
            if override or ckey not in args.__dict__.keys():
                args.__dict__[ckey] = cvalue
    return args


def dict_to_args(data: dict):
    args = argparse.Namespace()
    for ckey, cvalue in data.items():
        args.__dict__[ckey] = cvalue
    return args


class ArgumentParserX(argparse.ArgumentParser):
    def __init__(self, base_config_path=None, add_hyper_arg=True, **kwargs):
        super().__init__(**kwargs)
        self.add_hyper_arg = add_hyper_arg
        self.base_config_path = base_config_path
        if self.add_hyper_arg:
            self.add_argument('hyper', type=str, help='Path to the yaml parameter')
        self.add_argument('--exec', type=str, help='Extract code to modify the args')

    def parse_args(self, args=None, namespace=None):
        # Parse arg for the first time to extract args defined in program.
        _args = self.parse_known_args(args, namespace)[0]
        # Add the types needed.
        file_args = argparse.Namespace()
        if self.base_config_path is not None:
            file_args = parse_config_yaml(Path(self.base_config_path), file_args)
        if self.add_hyper_arg:
            if _args.hyper.endswith("json"):
                file_args = parse_config_json(Path(_args.hyper), file_args)
            else:
                file_args = parse_config_yaml(Path(_args.hyper), file_args)
            for ckey, cvalue in file_args.__dict__.items():
                try:
                    self.add_argument('--' + ckey, type=type(cvalue), default=cvalue, required=False)
                except argparse.ArgumentError:
                    continue
        # Parse args fully to extract all useful information
        _args = super().parse_args(args, namespace)
        # After that, execute exec part.
        exec_code = _args.exec
        if exec_code is not None:
            for exec_cmd in exec_code.split(";"):
                exec_cmd = "_args." + exec_cmd.strip()
                exec(exec_cmd)
        return _args


class AverageMeter:
    def __init__(self):
        self.loss_dict = OrderedDict()

    def export(self, f):
        if isinstance(f, str):
            f = open(f, 'wb')
        pickle.dump(self.loss_dict, f)

    def load(self, f):
        if isinstance(f, str):
            f = open(f, 'rb')
        self.loss_dict = pickle.load(f)
        return self

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            loss_val = float(loss_val)
            if np.isnan(loss_val):
                continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: [loss_val]})
            else:
                self.loss_dict[loss_name].append(loss_val)

    def get_mean_loss_dict(self):
        loss_dict = {}
        for loss_name, loss_arr in self.loss_dict.items():
            loss_dict[loss_name] = np.mean(loss_arr)
        return loss_dict

    def get_mean_loss(self):
        mean_loss_dict = self.get_mean_loss_dict()
        if len(mean_loss_dict) == 0:
            return 0.0
        else:
            return sum(mean_loss_dict.values()) / len(mean_loss_dict)

    def get_printable_mean(self):
        text = ""
        all_loss_sum = 0.0
        for loss_name, loss_mean in self.get_mean_loss_dict().items():
            all_loss_sum += loss_mean
            text += "(%s:%.4f) " % (loss_name, loss_mean)
        text += " sum = %.4f" % all_loss_sum
        return text

    def get_newest_loss_dict(self, return_count=False):
        loss_dict = {}
        loss_count_dict = {}
        for loss_name, loss_arr in self.loss_dict.items():
            if len(loss_arr) > 0:
                loss_dict[loss_name] = loss_arr[-1]
                loss_count_dict[loss_name] = len(loss_arr)
        if return_count:
            return loss_dict, loss_count_dict
        else:
            return loss_dict

    def get_printable_newest(self):
        nloss_val, nloss_count = self.get_newest_loss_dict(return_count=True)
        return ", ".join([f"{loss_name}[{nloss_count[loss_name] - 1}]: {nloss_val[loss_name]}"
                          for loss_name in nloss_val.keys()])

    def print_format_loss(self, color=None):
        if hasattr(sys.stdout, "terminal"):
            color_device = sys.stdout.terminal
        else:
            color_device = sys.stdout
        if color == "y":
            color_device.write('\033[93m')
        elif color == "g":
            color_device.write('\033[92m')
        elif color == "b":
            color_device.write('\033[94m')
        print(self.get_printable_mean(), flush=True)
        if color is not None:
            color_device.write('\033[0m')


class RunningAverageMeter:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.loss_dict = OrderedDict()

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            loss_val = float(loss_val)
            if np.isnan(loss_val):
                continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: loss_val})
            else:
                old_mean = self.loss_dict[loss_name]
                self.loss_dict[loss_name] = self.alpha * old_mean + (1 - self.alpha) * loss_val

    def get_loss_dict(self):
        return {k: v for k, v in self.loss_dict.items()}


def init_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    # According to https://pytorch.org/docs/stable/notes/randomness.html,
    # As pytorch run-to-run reproducibility is not guaranteed, and perhaps will lead to performance degradation,
    # We do not use manual seed for training.
    # This would influence stochastic network layers but will not influence data generation and processing w/o pytorch.
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class CombinedChunkLoss:
    def __init__(self):
        self.loss_dict = None
        self.loss_sum_dict = None
        self.clear()

    def add_loss(self, name, val):
        self.loss_dict[name] = val
        self.loss_sum_dict[name] += val.item()

    def update_loss_dict(self, loss_dict: dict):
        for l_name, l_val in loss_dict.items():
            self.add_loss(l_name, l_val)

    def get_total_loss(self):
        # Note: to reduce memory, we need to clear the referenced graph.
        total_loss = sum(self.loss_dict.values())
        self.loss_dict = {}
        return total_loss

    def get_accumulated_loss_dict(self):
        return self.loss_sum_dict

    def clear(self):
        self.loss_dict = {}
        self.loss_sum_dict = defaultdict(float)
