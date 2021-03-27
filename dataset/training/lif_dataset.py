from pathlib import Path
import torch
from torch.utils import data
import json
import numpy as np
import random
from utils.motion_util import Quaternion


def perturb_normal(normals, theta_range):
    normal_x_1 = np.stack([-normals[:, 1], normals[:, 0], np.zeros_like(normals[:, 0])], axis=1)
    normal_x_2 = np.stack([-normals[:, 2], np.zeros_like(normals[:, 0]), normals[:, 0]], axis=1)
    normal_x_mask = np.abs(np.abs(normals[:, 2]) - 1.0) > 0.1
    normal_x = np.zeros_like(normals)
    normal_x[normal_x_mask] = normal_x_1[normal_x_mask]
    normal_x[~normal_x_mask] = normal_x_2[~normal_x_mask]
    normal_x /= np.linalg.norm(normal_x, axis=1, keepdims=True)
    normal_y = np.cross(normals, normal_x)

    phi = np.random.rand(normal_x.shape[0], 1) * 2.0 * np.pi
    phi_dir = np.cos(phi) * normal_x + np.sin(phi) * normal_y
    theta = np.random.rand(normal_x.shape[0], 1) * theta_range
    perturbed_normal = np.cos(theta) * normals + np.sin(theta) * phi_dir
    return perturbed_normal


class LifDataset(data.Dataset):
    def __init__(self, data_path, num_sample, num_surface_sample: int = 0, augment_rotation=None,
                 augment_noise=(0.0, 0.0)):
        self.data_path = Path(data_path)
        with (self.data_path / "source.json").open() as f:
            self.data_sources = json.load(f)
        self.num_sample = num_sample
        self.num_surface_sample = num_surface_sample
        self.surface_format = None
        self.augment_rotation = augment_rotation
        self.augment_noise = augment_noise

    def __len__(self):
        return len(self.data_sources)

    def get_raw_data(self, idx):
        sdf_path = self.data_path / "payload" / ("%08d.npz" % idx)
        return np.load(sdf_path)

    def __getitem__(self, idx):
        if idx < 0:
            assert -idx <= len(self)
            idx = len(self) + idx

        # Load data
        lif_raw = self.get_raw_data(idx)
        lif_data = lif_raw["data"]
        lif_surface = None
        if self.num_surface_sample > 0:
            lif_surface = lif_raw["surface"]

        lif_data = torch.from_numpy(lif_data)
        pos_mask = lif_data[:, 3] > 0
        pos_tensor = lif_data[pos_mask]
        neg_tensor = lif_data[~pos_mask]
        half = int(self.num_sample / 2)
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
        samples = torch.cat([sample_pos, sample_neg], 0)

        lif_surface = lif_surface[np.random.choice(lif_surface.shape[0], size=self.num_surface_sample, replace=True), :]

        # Data augmentation
        if self.augment_rotation is not None:
            if self.augment_rotation == '3D':
                rand_rot = Quaternion.random()
            elif self.augment_rotation == 'X':
                rand_rot = Quaternion(axis=[1.0, 0.0, 0.0], degrees=360.0 * random.random())
            elif self.augment_rotation == 'Y':
                base_rot = random.choice([0.0, 90.0, 180.0, 270.0])
                rand_rot = Quaternion(axis=[0.0, 1.0, 0.0], degrees=base_rot + 30.0 * random.random())
            else:
                rand_rot = Quaternion(axis=[0.0, 0.0, 1.0], degrees=360.0 * random.random())
            samples[:, 0:3] = samples[:, 0:3] @ rand_rot.rotation_matrix.T.astype(np.float32)
            lif_surface[:, :3] = lif_surface[:, :3] @ rand_rot.rotation_matrix.T.astype(np.float32)
            lif_surface[:, 3:6] = lif_surface[:, 3:6] @ rand_rot.rotation_matrix.T.astype(np.float32)

        if self.augment_noise[0] > 0.0:
            lif_surface[:, :3] += np.random.randn(lif_surface.shape[0], 3) * self.augment_noise[0]
            lif_surface[:, 3:6] = perturb_normal(lif_surface[:, 3:6], np.deg2rad(self.augment_noise[1]))

        if not isinstance(lif_surface, torch.Tensor):
            lif_surface = torch.from_numpy(lif_surface)
        return samples.float(), lif_surface.float(), idx


class LifCombinedDataset(data.Dataset):
    def __init__(self, *datasets):
        super().__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(ds) for ds in self.datasets])

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            assert -idx <= len(self)
            idx = len(self) + idx
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        dat = self.datasets[dataset_idx][sample_idx]
        dat = dat[:-1]
        return *dat, idx
