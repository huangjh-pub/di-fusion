from pathlib import Path
from jittor.dataset import Dataset
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


class LifDataset(Dataset):
    def __init__(self, data_path, num_sample, num_surface_sample: int = 0, augment_rotation=None,
                 augment_noise=(0.0, 0.0), batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = True
        self.num_workers = 8
        self.drop_last = True

        self.data_path = Path(data_path)
        with (self.data_path / "source.json").open() as f:
            self.data_sources = json.load(f)
        self.num_sample = num_sample
        self.num_surface_sample = num_surface_sample
        self.surface_format = None
        self.augment_rotation = augment_rotation
        self.augment_noise = augment_noise
        self.total_len = len(self.data_sources)

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

        pos_mask = lif_data[:, 3] > 0
        pos_tensor = lif_data[pos_mask]
        neg_tensor = lif_data[~pos_mask]
        half = int(self.num_sample / 2)
        random_pos = (np.random.rand(half) * pos_tensor.shape[0]).astype(int)
        random_neg = (np.random.rand(half) * neg_tensor.shape[0]).astype(int)
        sample_pos = pos_tensor[random_pos]
        sample_neg = neg_tensor[random_neg]
        samples = np.concatenate([sample_pos, sample_neg], 0)

        lif_surface = lif_surface[np.random.choice(lif_surface.shape[0], size=self.num_surface_sample, replace=True), :]

        # Data augmentation
        if self.augment_rotation is not None:
            base_rot = random.choice([0.0, 90.0, 180.0, 270.0])
            rand_rot = Quaternion(axis=[0.0, 1.0, 0.0], degrees=base_rot + 30.0 * random.random())
            samples[:, 0:3] = samples[:, 0:3] @ rand_rot.rotation_matrix.T.astype(np.float32)
            lif_surface[:, :3] = lif_surface[:, :3] @ rand_rot.rotation_matrix.T.astype(np.float32)
            lif_surface[:, 3:6] = lif_surface[:, 3:6] @ rand_rot.rotation_matrix.T.astype(np.float32)

        if self.augment_noise[0] > 0.0:
            lif_surface[:, :3] += np.random.randn(lif_surface.shape[0], 3) * self.augment_noise[0]
            lif_surface[:, 3:6] = perturb_normal(lif_surface[:, 3:6], np.deg2rad(self.augment_noise[1]))

        return samples.astype(np.float32), lif_surface.astype(np.float32), idx
