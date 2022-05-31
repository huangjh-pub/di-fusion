import copy

import torch
import numpy as np
from system.ext import unproject_depth, remove_radius_outlier, estimate_normals, rgb_odometry, gradient_xy

from dataset.production import FrameIntrinsic
from utils import exp_util
from utils.motion_util import Isometry
import logging


def point_box_filter(points: torch.Tensor, normals: torch.Tensor, voxel_size: float):
    import torch_scatter
    min_bound = torch.min(points, dim=0, keepdim=True).values - voxel_size * 0.5
    max_bound = torch.max(points, dim=0, keepdim=True).values + voxel_size * 0.5
    ref_coord = torch.floor((points - min_bound) / voxel_size).long()
    n_x, n_y, n_z = (torch.floor((max_bound - min_bound) / voxel_size).long() + 16).cpu().numpy().tolist()[0]
    ref_coord = ref_coord[:, 0] + ref_coord[:, 1] * n_x + ref_coord[:, 2] * n_x * n_y
    _, inv_inds = torch.unique(ref_coord, return_inverse=True)
    filtered_pc = torch_scatter.scatter_mean(points, inv_inds, dim=0)
    filtered_normal = torch_scatter.scatter_mean(normals, inv_inds, dim=0)
    return filtered_pc, filtered_normal


class SDFTracker:
    def __init__(self, map, args):
        self.map = map
        self.args = args
        self.sdf_args = exp_util.dict_to_args(self.args.sdf)
        self.rgb_args = exp_util.dict_to_args(self.args.rgb)
        # Temporal information
        self.last_intensity = None
        self.last_depth = None
        self.all_pd_pose = []
        self.last_processed_pc = None       # store it to be used in subsequent integration
        self.cur_gt_pose = None             # For measure iteration error.
        self.last_colored_pcd = None        # store it for texture storage and extraction
        self.n_unstable = 0

    def _make_image_pyramid(self, intensity_img: torch.Tensor, depth_img: torch.Tensor):
        d0_w, d0_h = intensity_img.size(1), intensity_img.size(0)
        d1_w, d1_h = d0_w // 2, d0_h // 2
        d2_w, d2_h = d1_w // 2, d1_h // 2
        d0_intensity = intensity_img.view(1, 1, d0_h, d0_w)
        d0_depth = depth_img.view(1, 1, d0_h, d0_w)
        d1_intensity = torch.nn.functional.interpolate(d0_intensity, (d1_h, d1_w), mode='bilinear')
        d1_depth = torch.nn.functional.interpolate(d0_depth, (d1_h, d1_w), mode='nearest')
        d2_intensity = torch.nn.functional.interpolate(d1_intensity, (d2_h, d2_w), mode='bilinear')
        d2_depth = torch.nn.functional.interpolate(d1_depth, (d2_h, d2_w), mode='nearest')
        d0_gradient = gradient_xy(d0_intensity.squeeze(0).squeeze(0))
        d1_gradient = gradient_xy(d1_intensity.squeeze(0).squeeze(0))
        d2_gradient = gradient_xy(d2_intensity.squeeze(0).squeeze(0))
        return [t.squeeze(0).squeeze(0) for t in [d0_intensity, d1_intensity, d2_intensity]], \
               [t.squeeze(0).squeeze(0) for t in [d0_depth, d1_depth, d2_depth]], \
               [d0_gradient, d1_gradient, d2_gradient]

    @staticmethod
    def _robust_kernel(x: torch.Tensor, kernel_type: str, kernel_param: float):
        if kernel_type == 'huber':
            w = torch.ones_like(x)
            x_abs = torch.abs(x)
            w_mask = x_abs > kernel_param
            w[w_mask] = kernel_param / x_abs[w_mask]
            return w
        elif kernel_type == 'tukey':
            w = torch.zeros_like(x)
            w_mask = torch.abs(x) <= kernel_param
            w[w_mask] = (1 - (x[w_mask] / kernel_param) ** 2) ** 2
            return w
        else:
            raise NotImplementedError

    def track_camera(self, rgb_data: torch.Tensor, depth_data: torch.Tensor,
                     calib: FrameIntrinsic, set_pose: Isometry = None):
        """
        :param rgb_data:    (H, W, 3)       float32
        :param depth_data:  (H, W)          float32
        :param calib:       FrameIntrinsic
        :param set_pose:    Force set pose.
        :return: a pose.
        """
        cur_intensity = torch.mean(rgb_data, dim=-1)
        cur_depth = depth_data
        cur_intensity, cur_depth, cur_dIdxy = self._make_image_pyramid(cur_intensity, cur_depth)
        cur_rgb = rgb_data.permute(2, 0, 1)       # (3, H, W)

        # Process to point cloud.
        pc_scale = self.sdf_args.subsample
        pc_data = torch.nn.functional.interpolate(cur_depth[0].unsqueeze(0).unsqueeze(0),
                                                  scale_factor=pc_scale, mode='nearest',
                                                  recompute_scale_factor=False).squeeze(0).squeeze(0)
        cur_rgb = torch.nn.functional.interpolate(cur_rgb.unsqueeze(0), scale_factor=pc_scale, mode='bilinear', recompute_scale_factor=False).squeeze(0)
        pc_data = unproject_depth(pc_data, calib.fx * pc_scale, calib.fy * pc_scale,
                                  calib.cx * pc_scale, calib.cy * pc_scale)
        pc_data = torch.cat([pc_data, torch.zeros((pc_data.size(0), pc_data.size(1), 1), device=pc_data.device)], dim=-1)
        pc_data = pc_data.reshape(-1, 4)
        cur_rgb = cur_rgb.permute(1, 2, 0)      # (W, H, 3)
        cur_rgb = cur_rgb.reshape(-1, 3)

        nan_mask = ~torch.isnan(pc_data[..., 0])
        pc_data = pc_data[nan_mask]
        cur_rgb = cur_rgb[nan_mask]

        with torch.cuda.device(self.map.device):
            pc_data_valid_mask = remove_radius_outlier(pc_data, 16, 0.05)
            pc_data = pc_data[pc_data_valid_mask]
            cur_rgb = cur_rgb[pc_data_valid_mask]
            normal_data = estimate_normals(pc_data, 16, 0.1, [0.0, 0.0, 0.0])
            normal_valid_mask = ~torch.isnan(normal_data[..., 0])
            normal_data = normal_data[normal_valid_mask]
            cur_rgb = cur_rgb[normal_valid_mask]
            pc_data = pc_data[normal_valid_mask, :3]

        self.last_colored_pcd = [pc_data, cur_rgb]
        pc_data, normal_data = point_box_filter(pc_data, normal_data, 0.02)
        self.last_processed_pc = [pc_data, normal_data]

        if set_pose is not None:
            final_pose = set_pose
        else:
            assert len(self.all_pd_pose) > 0
            lspeed = Isometry()
            final_pose = self.gauss_newton(self.all_pd_pose[-1].dot(lspeed), cur_intensity, cur_depth, cur_dIdxy, pc_data, calib)

        self.last_intensity = cur_intensity
        self.last_depth = cur_depth
        self.all_pd_pose.append(final_pose)
        return final_pose

    def compute_rgb_Hg(self, pyramid_level: int, cur_delta_pose: Isometry, cur_intensity_pyramid: list,
                       cur_depth_pyramid: list, cur_dIdxy_pyramid: list, calib: FrameIntrinsic, no_grad: bool = False):
        cur_R = cur_delta_pose.q.rotation_matrix
        cur_t = cur_delta_pose.t
        K = calib.to_K()
        KRKinv = K @ cur_R @ np.linalg.inv(K)
        Kt = K @ cur_t

        odometry_output = rgb_odometry(self.last_intensity[pyramid_level], self.last_depth[pyramid_level],
                                       cur_intensity_pyramid[pyramid_level], cur_depth_pyramid[pyramid_level],
                                       cur_dIdxy_pyramid[pyramid_level],
                                       [calib.fx, calib.fy, calib.cx, calib.cy],
                                       KRKinv.flatten().tolist(), Kt.flatten().tolist(),
                                       self.rgb_args.min_grad_scale,
                                       self.rgb_args.max_depth_delta, not no_grad)
        if no_grad:
            f_map, = odometry_output
            J_map, JW = None, None
        else:
            f_map, J_map = odometry_output

        f_valid_mask = ~torch.isnan(f_map)
        f_map = f_map[f_valid_mask]     # (M, )
        Wf = f_map

        if J_map is not None:
            J_map = -J_map[f_valid_mask]     # (M, 6), The derivative computed is actually for -xi, so we inverse it.
            JW = J_map

        if self.rgb_args.robust_kernel is not None:
            term_weight = self._robust_kernel(f_map, self.rgb_args.robust_kernel, self.rgb_args.robust_k)
            Wf = Wf * term_weight                                 # (M)
            JW = JW * term_weight.unsqueeze(1) if JW is not None else None

        error_scale = 1. / Wf.size(0) * self.rgb_args.weight
        sum_error = (f_map * Wf).sum().item() * error_scale
        if not no_grad:
            H = torch.einsum('na,nb->nab', JW, J_map).sum(0) * error_scale
            g = (J_map * Wf.unsqueeze(1)).sum(0) * error_scale
            return H.cpu().numpy().astype(float), g.cpu().numpy().astype(float), float(sum_error)
        else:
            return None, None, float(sum_error)

    def compute_sdf_Hg(self, n_iter: int, last_pose: Isometry, cur_delta_pose: Isometry, obs_xyz: torch.Tensor, no_grad: bool = False):
        """
        Get the error function: L(xi) = mu_[T(xi)p] / std_[T(xi)p].detach()
        :param cur_pose:
        :param obs_xyz: (N, 3) in camera space.
        :return: f: (M, )  J: (M, 6)
        """
        cur_obs_xyz = (last_pose.dot(cur_delta_pose)) @ obs_xyz

        cur_obs_xyz.requires_grad_(not no_grad)
        cur_obs_sdf, cur_obs_std, cur_obs_valid_mask = self.map.get_sdf(cur_obs_xyz)
        # print(cur_obs_std.min(), cur_obs_std.max()) # ~0.13
        cur_obs_sdf = cur_obs_sdf / cur_obs_std.detach()

        if no_grad:
            JW = None
        else:
            dsdf_dpos = torch.autograd.grad(cur_obs_sdf, [cur_obs_xyz], grad_outputs=torch.ones_like(cur_obs_sdf),
                                            retain_graph=False, create_graph=False)[0]      # (N, 3)
            del cur_obs_xyz
            dsdf_dpos = dsdf_dpos[cur_obs_valid_mask]       # (M, 3)
            cur_obs_sdf = cur_obs_sdf.detach()
            cur_dxyz = (cur_delta_pose @ obs_xyz)[cur_obs_valid_mask]
            Lt = torch.from_numpy(last_pose.q.rotation_matrix.astype(np.float32).T).cuda()
            Lai = torch.mm(dsdf_dpos, Lt)
            Lbi = torch.cross(cur_dxyz, Lai)
            dsdf_dxi = torch.cat([Lai, Lbi], dim=-1)
            JW = dsdf_dxi

        Wf = cur_obs_sdf
        # print("[Empirical] Robust kernel should be = ", 2.6 * torch.std(Wf))
        if self.sdf_args.robust_kernel is not None:
            term_weight = self._robust_kernel(cur_obs_sdf, self.sdf_args.robust_kernel, self.sdf_args.robust_k)
            Wf = Wf * term_weight                                 # (M)
            JW = JW * term_weight.unsqueeze(1) if JW is not None else None

        error_scale = 1.0 / Wf.size(0)
        sum_error = (cur_obs_sdf * Wf).sum().item() * error_scale

        if no_grad:
            return None, None, float(sum_error)
        else:
            H = torch.einsum('na,nb->nab', JW, dsdf_dxi).sum(0) * error_scale
            g = (dsdf_dxi * Wf.unsqueeze(1)).sum(0) * error_scale
            return H.cpu().numpy().astype(float), g.cpu().numpy().astype(float), float(sum_error)

    def gauss_newton(self, init_pose: Isometry, cur_intensity_pyramid: list,
                     cur_depth_pyramid: list, cur_dIdxy_pyramid: list, obs_xyz: torch.Tensor, calib: FrameIntrinsic):
        last_pose = self.all_pd_pose[-1]
        cur_delta_pose = last_pose.inv().dot(init_pose)
        last_delta_pose = copy.deepcopy(cur_delta_pose)

        i_iter = 0
        for group_iter_config in self.args.iter_config:
            last_energy = np.inf

            from utils.exp_util import AverageMeter
            loss_meter = AverageMeter()
            for i_iter in list(range(group_iter_config["n"])) + [-1]:

                H = np.zeros((6, 6), dtype=float)
                g = np.zeros((6, ), dtype=float)
                cur_energy = 0.0

                for loss_config in group_iter_config["type"]:
                    if loss_config[0] == 'sdf':
                        sdf_H, sdf_g, sdf_energy = self.compute_sdf_Hg(i_iter, last_pose, cur_delta_pose, obs_xyz, i_iter == -1)
                        loss_meter.append_loss({'sdf': sdf_energy})
                        cur_energy += sdf_energy
                        if i_iter != -1:
                            H += sdf_H
                            g += sdf_g
                    if loss_config[0] == 'rgb':
                        pyramid_level = loss_config[1]
                        rgb_H, rgb_g, rgb_energy = self.compute_rgb_Hg(pyramid_level, cur_delta_pose,
                                                                       cur_intensity_pyramid, cur_depth_pyramid,
                                                                       cur_dIdxy_pyramid, calib, i_iter == -1)
                        loss_meter.append_loss({'rgb': rgb_energy})
                        cur_energy += rgb_energy
                        if i_iter != -1:
                            H += rgb_H
                            g += rgb_g
                    if loss_config[0] == 'motion':
                        motion_H, motion_g, motion_energy = self.compute_motion_Hg(cur_delta_pose, i_iter == -1)
                        loss_meter.append_loss({'motion': motion_energy})
                        cur_energy += motion_energy
                        if i_iter != -1:
                            H += motion_H
                            g += motion_g

                if cur_energy > last_energy:
                    cur_delta_pose = last_delta_pose
                    break
                else:
                    last_delta_pose = copy.deepcopy(cur_delta_pose)
                    last_energy = cur_energy

                if i_iter != -1:
                    xi = np.linalg.solve(H, -g)
                    cur_delta_pose = Isometry.from_twist(xi) @ cur_delta_pose
                    # logging.info(f"GN Iter {i_iter} @ Group {group_iter_config}, Loss = {loss_meter.get_printable_newest()}")

        if i_iter >= 10:
            # Safe bar: more number of iterations indicate bad convergence. This may be due to too small regularization,
            #       So we fallback to our default setting after this detection.
            self.n_unstable += 1
            if self.n_unstable >= 3:
                self.rgb_args.weight = max(self.rgb_args.weight, 500.)

        return last_pose.dot(cur_delta_pose)

    def track_camera_points_lm(self, init_pose: Isometry, obs_xyz: torch.Tensor):
        assert obs_xyz.size(1) == 3
        cur_pose = init_pose
        damping = self.args.lm_damping_init
        for i_iter in range(self.args.n_gn_iter):
            f, dsdf_dxi = self.get_error_func(cur_pose, obs_xyz, need_grad=True)          # (M, ), (M, 6)

            print(torch.std(f))

            term_weight = torch.ones_like(f)
            if self.args.robust_kernel:
                obs_sdf_abs = torch.abs(f)        # (M,)
                term_weight = torch.where(obs_sdf_abs <= self.args.robust_k,
                                          term_weight, self.args.robust_k / obs_sdf_abs)

            Wf = f * term_weight                                 # (M, 1)
            H = torch.einsum('nx,ny->xy', dsdf_dxi * term_weight.unsqueeze(1), dsdf_dxi).cpu().numpy()               # (6, 6)
            lambda_DtD = damping * np.diag(np.diag(H))
            g = -(dsdf_dxi * Wf.unsqueeze(1)).sum(0).cpu().numpy()                                      # (6,)
            xi = np.linalg.solve(H + lambda_DtD, g)

            new_pose = Isometry.from_twist(xi) @ cur_pose
            rho_denom = (xi * (lambda_DtD @ xi)).sum() + (xi * g).sum()
            f_new = self.get_error_func(new_pose, obs_xyz, need_grad=False)               # (M, )

            new_weight = torch.ones_like(f_new)
            if self.args.robust_kernel:
                f_abs = torch.abs(f_new)
                new_weight = torch.where(f_abs <= self.args.robust_k, new_weight, self.args.robust_k / f_abs)

            rho = ((f * Wf.squeeze()).sum() - (f_new * new_weight * f_new).sum()).item() / rho_denom

            if rho > self.args.lm_eps4:
                damping /= self.args.lm_ldown
                cur_pose = new_pose
                print(f"LM Iter {i_iter} LM Accepted: {(f ** 2).sum().item()}")
            else:
                damping *= self.args.lm_lup
                print(f"LM Iter {i_iter} LM Rejected: {(f ** 2).sum().item()}")
            damping = min(max(damping, 1.0e-7), 1.0e7)

        return cur_pose
