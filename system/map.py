import copy
import functools
import threading
from pathlib import Path

import open3d as o3d
import torch
import logging
import torch.optim
import numpy as np
import numba
import time
import system.ext
import argparse
import network.utility as net_util
from utils import exp_util
import torch.multiprocessing as mp


@numba.jit
def _get_valid_idx(base_idx: np.ndarray, query_idx: np.ndarray):
    mask = np.zeros((base_idx.shape[0], ), dtype=np.bool_)
    for vi, v in enumerate(base_idx):
        if query_idx[np.searchsorted(query_idx, v)] != v:
            mask[vi] = True
    return mask


class OptimizeProcess(mp.Process):
    def __init__(self, decoder, device):
        super().__init__(daemon=True)
        self.job_queue = mp.Queue()
        self.res_queue = mp.Queue()
        self.old_device = None
        self.device = device
        self.decoder = decoder
        self.busy_flag = mp.Value('b', False)

    def _move_args(self, val):
        if isinstance(val, torch.Tensor):
            self.old_device = val.device
            return val.to(device=self.device)
        else:
            return val

    def is_busy(self):
        return self.busy_flag.value

    def run(self):
        """
        Only used in Async scenario.
        """
        if self.decoder is not None:
            fork_decoder = copy.deepcopy(self.decoder).to(device=self.device)
            self.decoder = None
        run_stream = torch.cuda.Stream(device=self.device)
        while True:
            new_job_kwargs = self.job_queue.get()
            with self.busy_flag.get_lock():
                self.busy_flag.value = True
            # Make sure this is the only element in the queue. This assertion is ensured in Map caller.
            assert self.job_queue.empty()
            # If the device are not the same, copy the tensor and release the original ones.
            # Else, let the main thread keep the ref since it will not die anyway.
            job_kwargs = {
                k: self._move_args(v) for k, v in new_job_kwargs.items()
            }
            del new_job_kwargs
            with torch.cuda.device(self.device):
                with torch.cuda.stream(run_stream):
                    torch.cuda.synchronize(self.device)
                    optim_res = self.do_optimize(decoder=fork_decoder, **job_kwargs)
                    run_stream.synchronize()
            self.res_queue.put(optim_res.to(device=self.old_device, copy=True))
            del job_kwargs
            del optim_res
            with self.busy_flag.get_lock():
                self.busy_flag.value = False

    @staticmethod
    def do_optimize(decoder, args, latent_vecs_unique, latent_id_inv_mapping,
                    gathered_sdf, gathered_relative_xyz):
        latent_vecs_unique.requires_grad_(True)
        optimizer = torch.optim.Adam([latent_vecs_unique], lr=1.0e-2)
        batch_loss = exp_util.CombinedChunkLoss()
        n_samples = latent_id_inv_mapping.size(0)

        def loss_func(net_output, output_inds):
            pd_sdf, pd_std = net_output
            gt_sdf = torch.clamp(gathered_sdf[output_inds], -0.2, 0.2)
            pd_sdf = torch.clamp(pd_sdf.squeeze(-1), -0.2, 0.2)
            pd_std = pd_std.squeeze(-1)

            ll_loss = -torch.distributions.Normal(loc=pd_sdf, scale=pd_std).log_prob(gt_sdf)
            batch_loss.add_loss("ll", ll_loss.sum() / n_samples)

            if args.code_regularization:
                l2_size_loss = torch.sum(torch.norm(latent_vecs_unique, dim=1))
                reg_loss = args.code_reg_lambda * l2_size_loss / n_samples
                batch_loss.add_loss("reg", reg_loss)

            return batch_loss.get_total_loss()

        for optim_iter in range(args.optim_n_iters):
            iter_start_time = time.perf_counter()
            optimizer.zero_grad()
            latent_vecs = latent_vecs_unique[latent_id_inv_mapping]
            _ = net_util.forward_model(decoder, latent_input=latent_vecs, xyz_input=gathered_relative_xyz,
                                       loss_func=loss_func, max_sample=int(1.5e6), verbose=(optim_iter == 0))
            optimizer.step()
            batch_loss.clear()
        latent_vecs_unique.requires_grad_(False)
        return latent_vecs_unique


class MeshExtractCache:
    def __init__(self, device):
        self.vertices = None
        self.vertices_flatten_id = None
        self.vertices_std = None
        self.updated_vec_id = None
        self.device = device
        self.clear_updated_vec()

    def clear_updated_vec(self):
        self.updated_vec_id = torch.empty((0, ), device=self.device, dtype=torch.long)

    def clear_all(self):
        self.vertices = None
        self.vertices_flatten_id = None
        self.vertices_std = None
        self.updated_vec_id = None
        self.clear_updated_vec()


class MapVisuals:
    def __init__(self):
        self.mesh = []
        self.blocks = []
        self.samples = []
        self.uncertainty = []


class OptimResultsSet:
    def __init__(self):
        self.latent_ids = None
        self.new_latent_vecs = None
        self.old_latent_vecs = None
        self.old_latent_obs_counts = None

    def clear(self):
        self.latent_ids = None
        self.new_latent_vecs = None
        self.old_latent_vecs = None
        self.old_latent_obs_counts = None


class DenseIndexedMap:
    def __init__(self, model: net_util.Networks, args: argparse.Namespace,
                 latent_dim: int, device: torch.device, enable_async: bool = False,
                 optimization_device: torch.device = None):
        """
        Initialize a densely indexed latent map.
        For easy manipulation, invalid indices are -1, and occupied indices are >= 0.

        :param model:       neural network models
        :param latent_dim:  size of latent dim
        :param device:      device type of the map (some operations can still on host)
        :param optimization_device  note this does not take effect when using sync mode.
        """
        mp.set_start_method('forkserver', force=True)

        self.model = model
        self.model.eval()
        net_util.fix_weight_norm_pickle(self.model.decoder)

        self.voxel_size = args.voxel_size
        self.n_xyz = np.ceil((np.asarray(args.bound_max) - np.asarray(args.bound_min)) / args.voxel_size).astype(int).tolist()
        logging.info(f"Map size Nx = {self.n_xyz[0]}, Ny = {self.n_xyz[1]}, Nz = {self.n_xyz[2]}")

        self.args = args
        self.bound_min = torch.tensor(args.bound_min, device=device).float()
        self.bound_max = self.bound_min + self.voxel_size * torch.tensor(self.n_xyz, device=device)
        self.latent_dim = latent_dim
        self.device = device
        self.integration_offsets = [torch.tensor(t, device=self.device, dtype=torch.float32) for t in [
            [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]
        ]]
        # Directly modifiable from outside.
        self.extract_mesh_std_range = None

        self.mesh_update_affected = [torch.tensor([t], device=self.device)
                                     for t in [[-1, 0, 0], [1, 0, 0],
                                               [0, -1, 0], [0, 1, 0],
                                               [0, 0, -1], [0, 0, 1]]]
        self.relative_network_offset = torch.tensor([[0.5, 0.5, 0.5]], device=self.device, dtype=torch.float32)

        self.cold_vars = {
            "n_occupied": 0,
            "indexer": torch.ones(np.product(self.n_xyz), device=device, dtype=torch.long) * -1,
            # -- Voxel Attributes --
            # 1. Latent Vector (Geometry)
            "latent_vecs": torch.empty((1, self.latent_dim), dtype=torch.float32, device=device),
            # 2. Position
            "latent_vecs_pos": torch.ones((1, ), dtype=torch.long, device=device) * -1,
            # 3. Confidence on its geometry
            "voxel_obs_count": torch.zeros((1, ), dtype=torch.float32, device=device),
            # 4. Optimized mark
            "voxel_optimized": torch.zeros((1, ), dtype=torch.bool, device=device)
        }
        self.backup_var_names = ["indexer", "latent_vecs", "latent_vecs_pos", "voxel_obs_count"]
        self.backup_vars = {}
        self.modifying_lock = threading.Lock()
        # Allow direct visit by variable
        for p in self.cold_vars.keys():
            setattr(DenseIndexedMap, p, property(
                fget=functools.partial(DenseIndexedMap._get_var, name=p),
                fset=functools.partial(DenseIndexedMap._set_var, name=p)
            ))

        if enable_async:
            self.optimize_process = OptimizeProcess(self.model.decoder,
                                                    optimization_device if optimization_device is not None else self.device)
        else:
            self.optimize_process = OptimizeProcess(None, None)

        # self.optimize_process.start()
        self.optimize_result_set = OptimResultsSet()
        self.meshing_thread = None
        self.meshing_thread_id = -1
        self.meshing_stream = torch.cuda.Stream()
        self.mesh_cache = MeshExtractCache(self.device)
        self.latent_vecs.zero_()

    # def __del__(self):
    #     self.optimize_process.kill()

    def save(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        with path.open('wb') as f:
            torch.save(self.cold_vars, f)

    def load(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        with path.open('rb') as f:
            self.cold_vars = torch.load(f)

    def _get_var(self, name):
        if threading.get_ident() == self.meshing_thread_id and name in self.backup_var_names:
            return self.backup_vars[name]
        else:
            return self.cold_vars[name]

    def _set_var(self, value, name):
        if threading.get_ident() == self.meshing_thread_id and name in self.backup_var_names:
            self.backup_vars[name] = value
        else:
            self.cold_vars[name] = value

    def _inflate_latent_buffer(self, count: int):
        target_n_occupied = self.n_occupied + count
        if self.latent_vecs.size(0) < target_n_occupied:
            new_size = self.latent_vecs.size(0)
            while new_size < target_n_occupied:
                new_size *= 2
            new_vec = torch.empty((new_size, self.latent_dim), dtype=torch.float32, device=self.device)
            new_vec[:self.latent_vecs.size(0)] = self.latent_vecs
            new_vec_pos = torch.ones((new_size, ), dtype=torch.long, device=self.device) * -1
            new_vec_pos[:self.latent_vecs.size(0)] = self.latent_vecs_pos
            new_voxel_conf = torch.zeros((new_size, ), dtype=torch.float32, device=self.device)
            new_voxel_conf[:self.latent_vecs.size(0)] = self.voxel_obs_count
            new_voxel_optim = torch.zeros((new_size, ), dtype=torch.bool, device=self.device)
            new_voxel_optim[:self.latent_vecs.size(0)] = self.voxel_optimized
            new_vec[self.latent_vecs.size(0):].zero_()
            self.latent_vecs = new_vec
            self.latent_vecs_pos = new_vec_pos
            self.voxel_obs_count = new_voxel_conf
            self.voxel_optimized = new_voxel_optim

        new_inds = torch.arange(self.n_occupied, target_n_occupied, device=self.device, dtype=torch.long)
        self.n_occupied = target_n_occupied
        return new_inds

    def _linearize_id(self, xyz: torch.Tensor):
        """
        :param xyz (N, 3) long id
        :return: (N, ) lineraized id to be accessed in self.indexer
        """
        return xyz[:, 2] + self.n_xyz[-1] * xyz[:, 1] + (self.n_xyz[-1] * self.n_xyz[-2]) * xyz[:, 0]

    def _unlinearize_id(self, idx: torch.Tensor):
        """
        :param idx: (N, ) linearized id for access in self.indexer
        :return: xyz (N, 3) id to be indexed in 3D
        """
        return torch.stack([idx // (self.n_xyz[1] * self.n_xyz[2]),
                            (idx // self.n_xyz[2]) % self.n_xyz[1],
                            idx % self.n_xyz[2]], dim=-1)

    def _mark_updated_vec_id(self, new_vec_id: torch.Tensor):
        """
        :param new_vec_id: (B,) updated id (indexed in latent vectors)
        """
        self.mesh_cache.updated_vec_id = torch.cat([self.mesh_cache.updated_vec_id, new_vec_id])
        self.mesh_cache.updated_vec_id = torch.unique(self.mesh_cache.updated_vec_id)

    def allocate_block(self, idx: torch.Tensor):
        """
        :param idx: (N, 3) or (N, ), if the first one, will call linearize id.
        NOTE: this will not check index overflow!
        """
        if idx.ndimension() == 2 and idx.size(1) == 3:
            idx = self._linearize_id(idx)
        new_id = self._inflate_latent_buffer(idx.size(0))
        self.latent_vecs_pos[new_id] = idx
        self.indexer[idx] = new_id

    def _update_optimize_result_set(self, deintegrate_old: bool):
        idx = self.optimize_result_set.latent_ids
        assert idx is not None
        if not deintegrate_old:
            self.latent_vecs[idx] = self.optimize_result_set.new_latent_vecs
        else:
            cur_count = self.voxel_obs_count[idx].unsqueeze(-1)
            original_count = self.optimize_result_set.old_latent_obs_counts.unsqueeze(-1)
            delta_vecs_sum = self.latent_vecs[idx] * cur_count + (self.optimize_result_set.new_latent_vecs -
                                                                  self.optimize_result_set.old_latent_vecs) * original_count
            self.latent_vecs[idx] = delta_vecs_sum / cur_count

        self._mark_updated_vec_id(idx)
        self.voxel_optimized[idx] = True
        self.optimize_result_set.clear()     # trigger gc...

    STATUS_CONF_BIT = 1 << 0    # 1
    STATUS_SURF_BIT = 1 << 1    # 2

    def integrate_keyframe(self, surface_xyz: torch.Tensor, surface_normal: torch.Tensor, do_optimize: bool = False, async_optimize: bool = False):
        """
        :param surface_xyz:  (N, 3) x, y, z
        :param surface_normal: (N, 3) nx, ny, nz
        :param do_optimize: whether to do optimization (this will be slow though)
        :param async_optimize: whether to spawn a separate job to optimize.
            Note: the optimization is based on the point at this function call.
                  optimized result will be updated on the next function call after it's ready.
            Caveat: If two optimization thread are started simultaneously, results may not be accurate.
                    Although we explicitly ban this, user can also trigger this by call the function with async_optimize = True+False.
                    Please use consistent `async_optimize` during a SLAM session.
        :return:
        """
        assert surface_xyz.device == surface_normal.device == self.device, \
            f"Device of map {self.device} and input observation " \
            f"{surface_xyz.device, surface_normal.device} must be the same."

        # This lock prevents meshing thread reading error.
        self.modifying_lock.acquire()

        # -- 0. Update map if optimization thread is ready.
        if not self.optimize_process.res_queue.empty():
            self.optimize_result_set.new_latent_vecs = self.optimize_process.res_queue.get()
            self._update_optimize_result_set(deintegrate_old=True)

        # -- 1. Allocate new voxels --
        surface_xyz_zeroed = surface_xyz - self.bound_min.unsqueeze(0)
        surface_xyz_normalized = surface_xyz_zeroed / self.voxel_size
        surface_grid_id = torch.ceil(surface_xyz_normalized).long() - 1
        surface_grid_id = self._linearize_id(surface_grid_id)

        # Remove the observations where it is sparse.
        unq_mask = None
        if self.args.prune_min_vox_obs > 0:
            _, unq_inv, unq_count = torch.unique(surface_grid_id, return_counts=True, return_inverse=True)
            unq_mask = (unq_count > self.args.prune_min_vox_obs)[unq_inv]
            surface_xyz_normalized = surface_xyz_normalized[unq_mask]
            surface_grid_id = surface_grid_id[unq_mask]
            surface_normal = surface_normal[unq_mask]

        # Identify empty cells, fill the indexer.
        invalid_surface_ind = self.indexer[surface_grid_id] == -1
        if invalid_surface_ind.sum() > 0:
            invalid_flatten_id = torch.unique(surface_grid_id[invalid_surface_ind])
            # We expand this because we want to create some dummy voxels which helps the mesh extraction.
            invalid_flatten_id = self._expand_flatten_id(invalid_flatten_id, ensure_valid=False)
            invalid_flatten_id = invalid_flatten_id[self.indexer[invalid_flatten_id] == -1]
            self.allocate_block(invalid_flatten_id)

        def get_pruned_surface(enabled=True, lin_pos=None):
            # Prune useless surface points for quicker gathering (set to True to enable)
            if enabled:
                encoder_voxel_pos_exp = self._expand_flatten_id(lin_pos, False)
                # encoder_voxel_pos_exp = lin_pos
                exp_indexer = torch.zeros_like(self.indexer)
                exp_indexer[encoder_voxel_pos_exp] = 1
                focus_mask = exp_indexer[surface_grid_id] == 1
                return surface_xyz_normalized[focus_mask], surface_normal[focus_mask]
            else:
                return surface_xyz_normalized, surface_normal

        # -- 2. Get all voxels whose confidence is lower than optimization threshold and encoder them --
        # Find my voxels conditions:
        #   1) Voxel confidence < Threshold
        #   2) Voxel is valid.
        #   3) Not optimized.
        #   4) There is surface points in the [-0.5 - 0.5] range of this voxel.
        map_status = torch.zeros(np.product(self.n_xyz), device=self.device, dtype=torch.short)

        encoder_voxel_pos = self.latent_vecs_pos[torch.logical_and(self.voxel_obs_count < self.args.encoder_count_th,
                                                                   self.latent_vecs_pos >= 0)]
        map_status[encoder_voxel_pos] |= self.STATUS_CONF_BIT
        # self.map_status[surface_grid_id] |= self.STATUS_SURF_BIT

        if encoder_voxel_pos.size(0) > 0:
            pruned_surface_xyz_normalized, pruned_surface_normal = get_pruned_surface(
                enabled=True, lin_pos=encoder_voxel_pos)

            # Gather surface samples for encoder inference
            gathered_surface_latent_inds = []
            gathered_surface_xyzn = []
            for offset in self.integration_offsets:
                _surface_grid_id = torch.ceil(pruned_surface_xyz_normalized + offset) - 1
                for dim in range(3):
                    _surface_grid_id[:, dim].clamp_(0, self.n_xyz[dim] - 1)
                surface_relative_xyz = pruned_surface_xyz_normalized - _surface_grid_id - self.relative_network_offset
                surf_gid = self._linearize_id(_surface_grid_id.long())
                surface_latent_ind = self.indexer[surf_gid]
                in_focus_obs_mask = map_status[surf_gid] >= (self.STATUS_CONF_BIT)
                gathered_surface_latent_inds.append(surface_latent_ind[in_focus_obs_mask])
                gathered_surface_xyzn.append(torch.cat(
                    [surface_relative_xyz[in_focus_obs_mask],
                     pruned_surface_normal[in_focus_obs_mask]
                     ], dim=-1))
            gathered_surface_xyzn = torch.cat(gathered_surface_xyzn)
            gathered_surface_latent_inds = torch.cat(gathered_surface_latent_inds)

            surface_blatent_mapping, pinds, pcounts = torch.unique(gathered_surface_latent_inds, return_inverse=True,
                                                                   return_counts=True)
            pcounts = pcounts.float()

            logging.info(f"{surface_blatent_mapping.size(0)} voxels will be updated by the encoder. "
                         f"Points/Voxel: avg = {pcounts.mean().item()}, "
                         f"min = {pcounts.min().item()}, "
                         f"max = {pcounts.max().item()}")
            with torch.no_grad():
                encoder_latent = self.model.encoder(gathered_surface_xyzn)

            encoder_latent_sum = net_util.groupby_reduce(pinds, encoder_latent, op="sum")   # (C, L)
            encoder_latent_sum += self.latent_vecs[surface_blatent_mapping] * self.voxel_obs_count[surface_blatent_mapping].unsqueeze(-1)
            self.voxel_obs_count[surface_blatent_mapping] += pcounts
            self.latent_vecs[surface_blatent_mapping] = encoder_latent_sum / self.voxel_obs_count[surface_blatent_mapping].unsqueeze(-1)
            self._mark_updated_vec_id(surface_blatent_mapping)

        map_status.zero_()

        # -- 3. Get all voxels whose confidence is higher than optimization threshold and not marked and optimize them.
        # Another important criterion is that current frame must have enough good observation.
        # Find my voxels.
        if do_optimize and (not self.optimize_process.is_busy() and self.optimize_process.res_queue.empty()) and \
                self.args.optim_n_iters > 0:
            optim_voxel_pos = self.latent_vecs_pos[torch.logical_and(self.voxel_obs_count >= self.args.encoder_count_th,
                                                                     ~self.voxel_optimized)]
            optim_voxel_pos = optim_voxel_pos[optim_voxel_pos > 0]

            if optim_voxel_pos.size(0) > 0:
                map_status[optim_voxel_pos] |= self.STATUS_CONF_BIT
                pruned_surface_xyz_normalized, pruned_surface_normal = get_pruned_surface(
                    enabled=True, lin_pos=optim_voxel_pos)

                # Gather surface samples for encoder inference
                gathered_latent_inds = []
                gathered_relative_xyz = []
                gathered_sdf = []

                for offset in self.integration_offsets:
                    _surface_grid_id = torch.ceil(pruned_surface_xyz_normalized + offset) - 1
                    for dim in range(3):
                        _surface_grid_id[:, dim].clamp_(0, self.n_xyz[dim] - 1)
                    surface_relative_xyz = pruned_surface_xyz_normalized - _surface_grid_id - self.relative_network_offset
                    lin_pos = self._linearize_id(_surface_grid_id.long())
                    surface_latent_ind = self.indexer[lin_pos]
                    in_focus_obs_mask = map_status[lin_pos] >= self.STATUS_CONF_BIT
                    gathered_latent_inds.append(surface_latent_ind[in_focus_obs_mask])
                    cur_rel_xyz = surface_relative_xyz[in_focus_obs_mask]
                    cur_normal = pruned_surface_normal[in_focus_obs_mask]
                    cur_sdf = torch.randn(cur_rel_xyz.size(0), device=cur_rel_xyz.device, dtype=torch.float32) * 0.05
                    cur_rel_xyz = cur_rel_xyz + cur_sdf.unsqueeze(-1) * cur_normal
                    gathered_relative_xyz.append(cur_rel_xyz)
                    gathered_sdf.append(cur_sdf)

                gathered_latent_inds = torch.cat(gathered_latent_inds)
                gathered_relative_xyz = torch.cat(gathered_relative_xyz)
                gathered_sdf = torch.cat(gathered_sdf)

                latent_id_subset_uniques, latent_id_inv_mapping = torch.unique(gathered_latent_inds, return_inverse=True)
                latent_vecs_unique = self.latent_vecs[latent_id_subset_uniques]

                optimize_kwargs = {
                    "args": self.args,
                    "latent_vecs_unique": latent_vecs_unique,
                    "latent_id_inv_mapping": latent_id_inv_mapping,
                    "gathered_sdf": gathered_sdf,
                    "gathered_relative_xyz": gathered_relative_xyz,
                }

                self.optimize_result_set.latent_ids = latent_id_subset_uniques
                if not async_optimize:
                    self.optimize_result_set.new_latent_vecs = self.optimize_process.do_optimize(decoder=self.model.decoder, **optimize_kwargs)
                    self._update_optimize_result_set(deintegrate_old=False)
                else:
                    self.optimize_result_set.old_latent_vecs = torch.clone(latent_vecs_unique)
                    self.optimize_result_set.old_latent_obs_counts = self.voxel_obs_count[latent_id_subset_uniques]
                    self.optimize_process.job_queue.put(optimize_kwargs)

            # End if optim_voxel_pos.size(0) > 0
        # End if do_optimize

        self.modifying_lock.release()
        return unq_mask

    def _make_mesh_from_cache(self):
        vertices = self.mesh_cache.vertices.reshape((-1, 3))
        triangles = np.arange(vertices.shape[0]).reshape((-1, 3))

        final_mesh = o3d.geometry.TriangleMesh()
        # The pre-conversion is saving tons of time
        final_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(float))
        final_mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))

        # Assign color:
        if vertices.shape[0] > 0:
            import matplotlib.cm
            vert_color = self.mesh_cache.vertices_std.reshape((-1, )).astype(float)
            if self.extract_mesh_std_range is not None:
                vcolor_min, vcolor_max = self.extract_mesh_std_range
                vert_color = np.clip(vert_color, vcolor_min, vcolor_max)
            else:
                vcolor_min, vcolor_max = vert_color.min(), vert_color.max()
            vert_color = (vert_color - vcolor_min) / (vcolor_max - vcolor_min)
            vert_color = matplotlib.cm.jet(vert_color)[:, :3]
            final_mesh.vertex_colors = o3d.utility.Vector3dVector(vert_color)

        return final_mesh

    def _expand_flatten_id(self, base_flatten_id: torch.Tensor, ensure_valid: bool = True):
        expanded_flatten_id = [base_flatten_id]
        updated_pos = self._unlinearize_id(base_flatten_id)
        for affected_offset in self.mesh_update_affected:
            rs_id = updated_pos + affected_offset
            for dim in range(3):
                rs_id[:, dim].clamp_(0, self.n_xyz[dim] - 1)
            rs_id = self._linearize_id(rs_id)
            if ensure_valid:
                rs_id = rs_id[self.indexer[rs_id] != -1]
            expanded_flatten_id.append(rs_id)
        expanded_flatten_id = torch.unique(torch.cat(expanded_flatten_id))
        return expanded_flatten_id

    def get_sdf(self, xyz: torch.Tensor):
        """
        Get the sdf value of the requested positions with computation graph built.
        :param xyz: (N, 3)
        :return: sdf: (M,), std (M,), valid_mask: (N,) with M elements being 1.
        """
        xyz_normalized = (xyz - self.bound_min.unsqueeze(0)) / self.voxel_size
        with torch.no_grad():
            grid_id = torch.ceil(xyz_normalized.detach()).long() - 1
            sample_latent_id = self.indexer[self._linearize_id(grid_id)]
            sample_valid_mask = sample_latent_id != -1
            # Prune validity by ignore-count.
            valid_valid_mask = self.voxel_obs_count[sample_latent_id[sample_valid_mask]] > self.args.ignore_count_th
            sample_valid_mask[sample_valid_mask.clone()] = valid_valid_mask
            valid_latent = self.latent_vecs[sample_latent_id[sample_valid_mask]]

        valid_xyz_rel = xyz_normalized[sample_valid_mask] - grid_id[sample_valid_mask] - self.relative_network_offset

        sdf, std = net_util.forward_model(self.model.decoder,
                                        latent_input=valid_latent, xyz_input=valid_xyz_rel, no_detach=True)
        return sdf.squeeze(-1), std.squeeze(-1), sample_valid_mask

    def extract_mesh(self, voxel_resolution: int, max_n_triangles: int, fast: bool = True,
                     max_std: float = 2000.0, extract_async: bool = False, no_cache: bool = False,
                     interpolate: bool = True):
        """
        Extract mesh using marching cubes.
        :param voxel_resolution: int, number of sub-blocks within an LIF block.
        :param max_n_triangles: int, maximum number of triangles.
        :param fast: whether to hierarchically extract sdf for speed improvement.
        :param interpolate: whether to interpolate sdf values.
        :param extract_async: if set to True, the function will only return a mesh when
                1) There is a change in the map.
                2) The request is completed.
                otherwise, it will just return None.
        :param no_cache: ignore cached mesh and restart over.
        :return: Open3D mesh.
        """
        if self.meshing_thread is not None:
            if not self.meshing_thread.is_alive():
                self.meshing_thread = None
                self.meshing_thread_id = -1
                self.backup_vars = {}
                return self._make_mesh_from_cache()
            elif not extract_async:
                self.meshing_thread.join()
                return self._make_mesh_from_cache()
            else:
                return None

        with self.modifying_lock:
            if self.mesh_cache.updated_vec_id.size(0) == 0 and not no_cache:
                return self._make_mesh_from_cache() if not extract_async else None
            else:
                # We can start meshing, Yay!
                if no_cache:
                    updated_vec_id = torch.arange(self.n_occupied, device=self.device)
                    self.mesh_cache.clear_all()
                else:
                    updated_vec_id = self.mesh_cache.updated_vec_id
                    self.mesh_cache.clear_updated_vec()
                if extract_async:
                    for b_name in self.backup_var_names:
                        self.backup_vars[b_name] = self.cold_vars[b_name]

        def do_meshing(voxel_resolution):
            torch.cuda.synchronize()
            with torch.cuda.stream(self.meshing_stream):
                focused_flatten_id = self.latent_vecs_pos[updated_vec_id]
                occupied_flatten_id = self._expand_flatten_id(focused_flatten_id)
                occupied_vec_id = self.indexer[occupied_flatten_id]  # (B, )
                # Remove voxels with too low confidence.
                occupied_vec_id = occupied_vec_id[self.voxel_obs_count[occupied_vec_id] > self.args.ignore_count_th]

                vec_id_batch_mapping = torch.ones((occupied_vec_id.max().item() + 1,), device=self.device, dtype=torch.int) * -1
                vec_id_batch_mapping[occupied_vec_id] = torch.arange(0, occupied_vec_id.size(0), device=self.device,
                                                                     dtype=torch.int)
                occupied_latent_vecs = self.latent_vecs[occupied_vec_id]  # (B, 125)
                B = occupied_latent_vecs.size(0)

                # Sample more data.
                sample_a = -(voxel_resolution // 2) * (1. / voxel_resolution)
                sample_b = 1. + (voxel_resolution - 1) // 2 * (1. / voxel_resolution)
                voxel_resolution *= 2

                low_resolution = voxel_resolution // 2 if fast else voxel_resolution
                low_samples = net_util.get_samples(low_resolution, self.device, a=sample_a, b=sample_b) - \
                              self.relative_network_offset # (l**3, 3)
                low_samples = low_samples.unsqueeze(0).repeat(B, 1, 1)  # (B, l**3, 3)
                low_latents = occupied_latent_vecs.unsqueeze(1).repeat(1, low_samples.size(1), 1)  # (B, l**3, 3)

                with torch.no_grad():
                    low_sdf, low_std = net_util.forward_model(self.model.decoder,
                                                        latent_input=low_latents.view(-1, low_latents.size(-1)),
                                                        xyz_input=low_samples.view(-1, low_samples.size(-1)))

                if fast:
                    low_sdf = low_sdf.reshape(B, 1, low_resolution, low_resolution, low_resolution)  # (B, 1, l, l, l)
                    low_std = low_std.reshape(B, 1, low_resolution, low_resolution, low_resolution)
                    high_sdf = torch.nn.functional.interpolate(low_sdf, mode='trilinear',
                                                               size=(voxel_resolution, voxel_resolution, voxel_resolution),
                                                               align_corners=True)
                    high_std = torch.nn.functional.interpolate(low_std, mode='trilinear',
                                                               size=(voxel_resolution, voxel_resolution, voxel_resolution),
                                                               align_corners=True)
                    high_sdf = high_sdf.squeeze(0).reshape(B, voxel_resolution ** 3)  # (B, H**3)
                    high_std = high_std.squeeze(0).reshape(B, voxel_resolution ** 3)

                    high_valid_lifs, high_valid_sbs = torch.where(high_sdf.abs() < 0.05)
                    if high_valid_lifs.size(0) > 0:
                        high_samples = net_util.get_samples(voxel_resolution, self.device, a=sample_a, b=sample_b) - \
                                       self.relative_network_offset  # (H**3, 3)
                        high_latents = occupied_latent_vecs[high_valid_lifs]  # (VH, 125)
                        high_samples = high_samples[high_valid_sbs]  # (VH, 3)

                        with torch.no_grad():
                            high_valid_sdf, high_valid_std = net_util.forward_model(self.model.decoder,
                                                                       latent_input=high_latents,
                                                                       xyz_input=high_samples)
                        high_sdf[high_valid_lifs, high_valid_sbs] = high_valid_sdf.squeeze(-1)
                        high_std[high_valid_lifs, high_valid_sbs] = high_valid_std.squeeze(-1)

                    high_sdf = high_sdf.reshape(B, voxel_resolution, voxel_resolution, voxel_resolution)
                    high_std = high_std.reshape(B, voxel_resolution, voxel_resolution, voxel_resolution)
                else:
                    high_sdf = low_sdf.reshape(B, low_resolution, low_resolution, low_resolution)
                    high_std = low_std.reshape(B, low_resolution, low_resolution, low_resolution)

                high_sdf = -high_sdf
                if interpolate:
                    vertices, vertices_flatten_id, vertices_std = system.ext.marching_cubes_interp(
                        self.indexer.view(self.n_xyz), focused_flatten_id, vec_id_batch_mapping,
                        high_sdf, high_std, max_n_triangles, self.n_xyz, max_std)  # (T, 3, 3), (T, ), (T, 3)
                else:
                    vertices, vertices_flatten_id = system.ext.marching_cubes(
                        self.indexer.view(self.n_xyz), focused_flatten_id, vec_id_batch_mapping,
                        high_sdf, max_n_triangles, self.n_xyz)  # (T, 3, 3), (T, ), (T, 3)
                    vertices_std = torch.zeros((vertices.size(0), 3), dtype=torch.float32, device=vertices.device)

                vertices = vertices * self.voxel_size + self.bound_min
                vertices = vertices.cpu().numpy()
                vertices_std = vertices_std.cpu().numpy()
                # Remove relevant cached vertices and append updated/new ones.
                vertices_flatten_id = vertices_flatten_id.cpu().numpy()
                if self.mesh_cache.vertices is None:
                    self.mesh_cache.vertices = vertices
                    self.mesh_cache.vertices_flatten_id = vertices_flatten_id
                    self.mesh_cache.vertices_std = vertices_std
                else:
                    p = np.sort(np.unique(vertices_flatten_id))
                    valid_verts_idx = _get_valid_idx(self.mesh_cache.vertices_flatten_id, p)
                    self.mesh_cache.vertices = np.concatenate([self.mesh_cache.vertices[valid_verts_idx], vertices], axis=0)
                    self.mesh_cache.vertices_flatten_id = np.concatenate([
                        self.mesh_cache.vertices_flatten_id[valid_verts_idx], vertices_flatten_id
                    ], axis=0)
                    self.mesh_cache.vertices_std = np.concatenate([self.mesh_cache.vertices_std[valid_verts_idx], vertices_std], axis=0)

        if extract_async:
            self.meshing_thread = threading.Thread(target=do_meshing, args=(voxel_resolution, ))
            self.meshing_thread_id = self.meshing_thread.ident
            self.meshing_thread.daemon = True
            self.meshing_thread.start()
        else:
            do_meshing(voxel_resolution)
            return self._make_mesh_from_cache()

    def get_fast_preview_visuals(self):
        occupied_flatten_id = torch.where(self.indexer != -1)[0]  # (B, )
        blk_verts = [self._unlinearize_id(occupied_flatten_id) * self.voxel_size + self.bound_min]
        n_block = blk_verts[0].size(0)
        blk_edges = []
        for vert_offset in [[0.0, 0.0, self.voxel_size], [0.0, self.voxel_size, 0.0],
                            [0.0, self.voxel_size, self.voxel_size], [self.voxel_size, 0.0, 0.0],
                            [self.voxel_size, 0.0, self.voxel_size], [self.voxel_size, self.voxel_size, 0.0],
                            [self.voxel_size, self.voxel_size, self.voxel_size]]:
            blk_verts.append(
                blk_verts[0] + torch.tensor(vert_offset, dtype=torch.float32, device=blk_verts[0].device).unsqueeze(0)
            )
        for vert_edge in [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]:
            blk_edges.append(np.stack([np.arange(n_block, dtype=np.int32) + vert_edge[0] * n_block,
                                       np.arange(n_block, dtype=np.int32) + vert_edge[1] * n_block], axis=1))
        blk_verts = torch.cat(blk_verts, dim=0).cpu().numpy().astype(float)
        blk_wireframe = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(blk_verts),
            lines=o3d.utility.Vector2iVector(np.concatenate(blk_edges, axis=0)))
        from utils import vis_util
        return [
            blk_wireframe,
            vis_util.wireframe_bbox(self.bound_min.cpu().numpy(),
                                    self.bound_max.cpu().numpy(), color_id=4)
        ]

    def get_map_visuals(self, return_blocks: bool = False, return_samples: bool = False, return_uncertainty: bool = False,
                        return_mesh: bool = False,
                        sample_range: list = None, voxel_resolution: int = 8, include_bound: bool = False):
        """
        :param return_blocks: whether to include blocks in the visualization.
        :param return_samples: whether to include sdf samples (at voxel resolution)
            Note: only for debugging purpose, can be removed in the future
        :param return_mesh: whether to extract mesh.
        :param sample_range: [low-sdf, high-sdf]
        :param voxel_resolution: int, number of sub-blocks within an LIF block.
        :param include_bound: bool. whether to return the map bound when return_blocks
        :return:
        """
        from utils import vis_util

        map_visuals = MapVisuals()

        if return_blocks:
            occupied_flatten_id = torch.where(self.indexer != -1)[0]  # (B, )
            blk_xyz = self._unlinearize_id(occupied_flatten_id)
            blk_start = blk_xyz * self.voxel_size + self.bound_min
            blk_start = blk_start.cpu().numpy()
            occupied_flatten_id = occupied_flatten_id.cpu().numpy()

            blk_wireframes = []
            blk_dim = np.asarray([self.voxel_size, self.voxel_size, self.voxel_size])
            for blk_start_i, ofid in zip(blk_start, occupied_flatten_id):
                if ofid in self.debug_show_blocks:
                    blk_wireframes.append(vis_util.wireframe_bbox(blk_start_i, blk_start_i + blk_dim,
                                                                  solid=True, color_id=4))
                else:
                    blk_wireframes.append(vis_util.wireframe_bbox(blk_start_i, blk_start_i + blk_dim, solid=False))

            blk_wireframes.append(vis_util.wireframe_bbox(self.bound_min.cpu().numpy(),
                                                          self.bound_max.cpu().numpy(), color_id=4))

            map_visuals.blocks = vis_util.merged_entities(blk_wireframes)

        if return_mesh:
            map_visuals.mesh = [self.extract_mesh(voxel_resolution, int(1e7), extract_async=False)]

        if return_samples or return_uncertainty:
            occupied_flatten_id = torch.where(self.indexer != -1)[0]  # (B, )
            occupied_vec_id = self.indexer[occupied_flatten_id]  # (B, )
            occupied_vec_id = occupied_vec_id[self.voxel_obs_count[occupied_vec_id] > self.args.ignore_count_th]
            occupied_latent_vecs = self.latent_vecs[occupied_vec_id]  # (B, 125)
            B = occupied_latent_vecs.size(0)

            high_samples = net_util.get_samples(voxel_resolution, self.device) - self.relative_network_offset # (H**3, 3)
            high_samples = high_samples.unsqueeze(0).repeat(B, 1, 1)  # (B, H**3, 3)
            high_latents = occupied_latent_vecs.unsqueeze(1).repeat(1, high_samples.size(1), 1)  # (B, H**3, 125)

            with torch.no_grad():
                high_sdf, high_uncertainty = net_util.forward_model(self.model.decoder,
                                                  latent_input=high_latents.view(-1, high_latents.size(-1)),
                                                  xyz_input=high_samples.view(-1, high_samples.size(-1)))

            high_sdf = high_sdf.reshape(B, voxel_resolution, voxel_resolution, voxel_resolution)
            high_uncertainty = high_uncertainty.reshape(B, voxel_resolution, voxel_resolution, voxel_resolution)

            vis_grid_base = self._unlinearize_id(self.latent_vecs_pos[occupied_vec_id])
            vis_sample_pos = high_samples + vis_grid_base.unsqueeze(1).repeat(1, high_samples.size(1), 1) + self.relative_network_offset
            vis_sample_pos = (vis_sample_pos.reshape(-1, 3) * self.voxel_size + self.bound_min).cpu().numpy()
            high_sdf = high_sdf.reshape(-1).cpu().numpy()
            high_uncertainty = high_uncertainty.reshape(-1).cpu().numpy()

            if sample_range is None:
                vis_high_sdf = (high_sdf - high_sdf.min()) / (high_sdf.max() - high_sdf.min())
                vis_std = (high_uncertainty - high_uncertainty.min()) / (high_uncertainty.max() - high_uncertainty.min())
                print(f"Uncertainty normalized to {high_uncertainty.min().item()} ~ {high_uncertainty.max().item()}")
            else:
                vis_high_sdf = (high_sdf - sample_range[0]) / (sample_range[1] - sample_range[0])
                vis_std = (high_uncertainty - sample_range[0]) / (sample_range[1] - sample_range[0])
                vis_high_sdf = np.clip(vis_high_sdf, 0.0, 1.0)
                vis_std = np.clip(vis_std, 0.0, 1.0)

            if return_samples:
                map_visuals.samples = [vis_util.pointcloud(vis_sample_pos, cfloat=vis_high_sdf)]
            if return_uncertainty:
                map_visuals.uncertainty = [vis_util.pointcloud(vis_sample_pos, cfloat=vis_std)]

        return map_visuals
