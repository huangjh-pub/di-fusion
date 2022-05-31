import math
import torch
import torch.nn as nn
import logging
from utils import exp_util
from pathlib import Path
import importlib


class Networks:
    def __init__(self):
        self.decoder = None
        self.encoder = None

    def eval(self):
        if self.encoder is not None:
            self.encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()


def load_model(training_hyper_path: str, use_epoch: int = -1):
    """
    Load in the model and hypers used.
    :param training_hyper_path:
    :param use_epoch: if -1, will load the latest model.
    :return: Networks
    """
    training_hyper_path = Path(training_hyper_path)

    if training_hyper_path.name.split(".")[-1] == "json":
        args = exp_util.parse_config_json(training_hyper_path)
        exp_dir = training_hyper_path.parent
        model_paths = exp_dir.glob('model_*.pth.tar')
        model_paths = {int(str(t).split("model_")[-1].split(".pth")[0]): t for t in model_paths}
        assert use_epoch in model_paths.keys(), f"{use_epoch} not found in {sorted(list(model_paths.keys()))}"
        args.checkpoint = model_paths[use_epoch]
    else:
        args = exp_util.parse_config_yaml(Path('configs/training_defaults.yaml'))
        args = exp_util.parse_config_yaml(training_hyper_path, args)
        logging.warning("Loaded a un-initialized model.")
        args.checkpoint = None

    model = Networks()
    net_module = importlib.import_module("network." + args.network_name)
    model.decoder = net_module.Model(args.code_length, **args.network_specs).cuda()
    if args.encoder_name is not None:
        encoder_module = importlib.import_module("network." + args.encoder_name)
        model.encoder = encoder_module.Model(**args.encoder_specs).cuda()
    if args.checkpoint is not None:
        if model.decoder is not None:
            state_dict = torch.load(args.checkpoint)["model_state"]
            model.decoder.load_state_dict(state_dict)
        if model.encoder is not None:
            state_dict = torch.load(Path(args.checkpoint).parent / f"encoder_{use_epoch}.pth.tar")["model_state"]
            model.encoder.load_state_dict(state_dict)

    return model, args


def forward_model(model: nn.Module, network_input: torch.Tensor = None,
                  latent_input: torch.Tensor = None,
                  xyz_input: torch.Tensor = None,
                  loss_func=None, max_sample: int = 2 ** 32,
                  no_detach: bool = False,
                  verbose: bool = False):
    """
    Forward the neural network model. (if loss_func is not None, will also compute the gradient w.r.t. the loss)
    Either network_input or (latent_input, xyz_input) tuple could be provided.
    :param model:           MLP model.
    :param network_input:   (N, 128)
    :param latent_input:    (N, 125)
    :param xyz_input:       (N, 3)
    :param loss_func:
    :param max_sample
    :return: [(N, X)] several values
    """
    if latent_input is not None and xyz_input is not None:
        assert network_input is None
        network_input = torch.cat((latent_input, xyz_input), dim=1)

    assert network_input.ndimension() == 2

    n_chunks = math.ceil(network_input.size(0) / max_sample)
    assert not no_detach or n_chunks == 1

    network_input = torch.chunk(network_input, n_chunks)

    if verbose:
        logging.debug(f"Network input chunks = {n_chunks}, each chunk = {network_input[0].size()}")

    head = 0
    output_chunks = None
    for chunk_i, input_chunk in enumerate(network_input):
        # (N, 1)
        network_output = model(input_chunk)
        if not isinstance(network_output, tuple):
            network_output = [network_output, ]

        if chunk_i == 0:
            output_chunks = [[] for _ in range(len(network_output))]

        if loss_func is not None:
            # The 'graph' in pytorch stores how the final variable is computed to its current form.
            # Under normal situations, we can delete this path right after the gradient is computed because the path
            #   will be re-constructed on next forward call.
            # However, in our case, self.latent_vec is the leaf node requesting the gradient, the specific computation:
            #   vec = self.latent_vec[inds] && cat(vec, xyz)
            #   will be forgotten, too. if we delete the entire graph.
            # Indeed, the above computation is the ONLY part that we do not re-build during the next forwarding.
            # So, we set retain_graph to True.
            # According to https://github.com/pytorch/pytorch/issues/31185, if we delete the head loss immediately
            #   after the backward(retain_graph=True), the un-referenced part graph will be deleted too,
            #   hence keeping only the needed part (a sub-graph). Perfect :)
            loss_func(network_output,
                      torch.arange(head, head + network_output[0].size(0), device=network_output[0].device)
                      ).backward(retain_graph=(chunk_i != n_chunks - 1))
        if not no_detach:
            network_output = [t.detach() for t in network_output]

        for payload_i, payload in enumerate(network_output):
            output_chunks[payload_i].append(payload)
        head += network_output[0].size(0)

    output_chunks = [torch.cat(t, dim=0) for t in output_chunks]
    return output_chunks


def get_samples(r: int, device: torch.device, a: float = 0.0, b: float = None):
    """
    Get samples within a cube, the voxel size is (b-a)/(r-1). range is from [a, b]
    :param r: num samples
    :param a: bound min
    :param b: bound max
    :return: (r*r*r, 3)
    """
    overall_index = torch.arange(0, r ** 3, 1, device=device, dtype=torch.long)
    r = int(r)

    if b is None:
        b = 1. - 1. / r

    vsize = (b - a) / (r - 1)
    samples = torch.zeros(r ** 3, 3, device=device, dtype=torch.float32)
    samples[:, 0] = (overall_index // (r * r)) * vsize + a
    samples[:, 1] = ((overall_index // r) % r) * vsize + a
    samples[:, 2] = (overall_index % r) * vsize + a

    return samples


def pack_samples(sample_indexer: torch.Tensor, count: int,
                 sample_values: torch.Tensor = None):
    """
    Pack a set of samples into batches. Each element in the batch is a random subsampling of the sample_values
    :param sample_indexer: (N, )
    :param count: C
    :param sample_values: (N, L), if None, will return packed_inds instead of packed.
    :return: packed (B, C, L) or packed_inds (B, C), mapping: (B, ).
    """
    from system.ext import pack_batch

    # First shuffle the samples to avoid biased samples.
    shuffle_inds = torch.randperm(sample_indexer.size(0), device=sample_indexer.device)
    sample_indexer = sample_indexer[shuffle_inds]

    mapping, pinds, pcount = torch.unique(sample_indexer, return_inverse=True, return_counts=True)

    n_batch = mapping.size(0)
    packed_inds = pack_batch(pinds, n_batch, count * 2)         # (B, 2C)

    pcount.clamp_(max=count * 2 - 1)
    packed_inds_ind = torch.floor(torch.rand((n_batch, count), device=pcount.device) * pcount.unsqueeze(-1)).long()  # (B, C)

    packed_inds = torch.gather(packed_inds, 1, packed_inds_ind)     # (B, C)
    packed_inds = shuffle_inds[packed_inds]                         # (B, C)

    if sample_values is not None:
        assert sample_values.size(0) == sample_indexer.size(0)
        packed = torch.index_select(sample_values, 0, packed_inds.view(-1)).view(n_batch, count, sample_values.size(-1))
        return packed, mapping
    else:
        return packed_inds, mapping


def groupby_reduce(sample_indexer: torch.Tensor, sample_values: torch.Tensor, op: str = "max"):
    """
    Group-By and Reduce sample_values according to their indices, the reduction operation is defined in `op`.
    :param sample_indexer: (N,). An index, must start from 0 and go to the (max-1), can be obtained using torch.unique.
    :param sample_values: (N, L)
    :param op: have to be in 'max', 'mean'
    :return: reduced values: (C, L)
    """
    C = sample_indexer.max() + 1
    n_samples = sample_indexer.size(0)

    assert n_samples == sample_values.size(0), "Indexer and Values must agree on sample count!"

    if op == 'mean':
        from system.ext import groupby_sum
        values_sum, values_count = groupby_sum(sample_values, sample_indexer, C)
        return values_sum / values_count.unsqueeze(-1)
    elif op == 'sum':
        from system.ext import groupby_sum
        values_sum, _ = groupby_sum(sample_values, sample_indexer, C)
        return values_sum
    else:
        raise NotImplementedError


def fix_weight_norm_pickle(net: torch.nn.Module):
    from torch.nn.utils.weight_norm import WeightNorm
    for mdl in net.modules():
        fix_name = None
        if isinstance(mdl, torch.nn.Linear):
            for k, hook in mdl._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    fix_name = hook.name
        if fix_name is not None:
            delattr(mdl, fix_name)
