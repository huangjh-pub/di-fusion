import numpy as np
import functools
from multiprocessing import Pool, Value, Manager
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import subprocess
import json
import shutil
import os
import argparse
import logging
from utils import exp_util


CUDA_SAMPLER_PATH = Path(__file__).resolve().parent.parent.parent / "sampler_cuda" / "bin" / "PreprocessMeshCUDA"
_counter = Value('i', 0)
_bad_counter = Value('i', 0)


def generate_samples(idx: int, args: argparse.ArgumentParser, provider, output_base, source_list):

    mesh_path, vcam, ref_bin_path, sampler_mult = provider[idx]

    # Tmp file for sampling.
    output_tmp_path = output_base / ("%06d.raw" % idx)
    surface_tmp_path = output_base / ("%06d.surf" % idx)
    vcam_file_path = output_base / ("%06d.cam" % idx)

    # Save the camera
    with vcam_file_path.open('wb') as f:
        np.asarray(vcam[0]).flatten().astype(np.float32).tofile(f)
        np.asarray([cam.to_gl_camera().inv().matrix.T for cam in vcam[1]]).reshape(-1, 16).astype(np.float32).tofile(f)

    # Call CUDA sampler
    arg_list_common = [str(CUDA_SAMPLER_PATH),
                       '-m', mesh_path,
                       '-s', str(int(args.sampler_count * sampler_mult * sampler_mult)),
                       '-o', str(output_tmp_path),
                       '-c', str(vcam_file_path),
                       '-r', str(args.sample_method),
                       '--surface', str(surface_tmp_path)]
    arg_list_data = arg_list_common + ['-p', '0.8', '--var', str(args.sampler_var), '-e', str(args.voxel_size * 2.5)]
    if ref_bin_path is not None:
        arg_list_data += ['--ref', ref_bin_path, '--max_ref_dist', str(args.max_ref_dist)]

    is_bad = False
    sampler_pass = args.__dict__.get("sampler_pass", 1)

    data_arr = []
    surface_arr = []
    for sid in range(sampler_pass):
        subproc = subprocess.Popen(arg_list_data, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subproc.wait()
        # Read raw file and convert it to numpy file.
        try:
            cur_data_arr = np.fromfile(str(output_tmp_path), dtype=np.float32).reshape(-1, 4)
            cur_surface_arr = np.fromfile(str(surface_tmp_path), dtype=np.float32).reshape(-1, 6)
            data_arr.append(cur_data_arr)
            surface_arr.append(cur_surface_arr)
            os.unlink(output_tmp_path)
            os.unlink(surface_tmp_path)
        except FileNotFoundError:
            print(' '.join(arg_list_data))
            is_bad = True
            break

    # Do cleaning of sampler.
    os.unlink(vcam_file_path)

    if is_bad:
        provider.clean(idx)
        with _bad_counter.get_lock():
            _bad_counter.value += 1
        return

    data_arr = np.concatenate(data_arr, axis=0) * sampler_mult
    surface_arr = np.concatenate(surface_arr, axis=0)
    surface_arr[:, :3] *= sampler_mult

    # Badly, some surface arr may have NaN normal, we prune them
    surface_arr_nan_row = np.any(np.isnan(surface_arr), axis=1)
    surface_arr = surface_arr[~surface_arr_nan_row]

    # Do LIF splitting.
    voxel_size = args.voxel_size
    data_xyz = data_arr[:, :3]
    data_sdf = data_arr[:, 3]
    surface_xyz = surface_arr[:, :3]

    voxel_centers = np.ceil(data_xyz / voxel_size) - 1
    voxel_centers = np.unique(voxel_centers, axis=0)
    voxel_centers = (voxel_centers + 0.5) * voxel_size
    nbrs = NearestNeighbors(radius=voxel_size * (args.nn_size / 2.0), metric='chebyshev').fit(data_xyz)
    lif_indices = nbrs.radius_neighbors(voxel_centers, return_distance=False)
    nbrs_local = NearestNeighbors(radius=voxel_size * 0.5, metric='chebyshev').fit(data_xyz)
    local_indices = nbrs_local.radius_neighbors(voxel_centers, return_distance=False)
    nbrs_surface = NearestNeighbors(radius=voxel_size * (args.nn_size / 2.0), metric='chebyshev').fit(surface_xyz)
    surface_indices = nbrs_surface.radius_neighbors(voxel_centers, return_distance=False)

    lif_data = []
    lif_data_count = []
    surface_data_count = []
    used_points = 0

    for vox_center, lif_index, local_index, surface_index in zip(voxel_centers, lif_indices, local_indices, surface_indices):
        if local_index.shape[0] < 50 or surface_index.shape[0] < 50:
            continue

        inner_sdf = data_sdf[lif_index]
        num_pos = np.count_nonzero(inner_sdf > 0)
        pos_ratio = num_pos / lif_index.shape[0]
        if pos_ratio < 0.1 or pos_ratio > 0.9:
            continue

        vox_min = vox_center - 0.5 * voxel_size
        vox_max = vox_center + 0.5 * voxel_size

        lif_data_count.append(lif_index.shape[0])
        surface_data_count.append(surface_index.shape[0])
        used_points += local_index.shape[0]

        # Gather and normalize data (so that the center of data is 0)
        output_data_xyzs = data_arr[lif_index]
        output_surface_xyzn = surface_arr[surface_index]
        output_data_xyzs[:, :3] = (output_data_xyzs[:, :3] - vox_center) / (vox_max - vox_min)
        output_surface_xyzn[:, :3] = (output_surface_xyzn[:, :3] - vox_center) / (vox_max - vox_min)
        output_data_xyzs[:, 3] /= voxel_size

        if np.max(output_data_xyzs[:, 3]) > 10.0:
            print("Error", np.max(output_data_xyzs[:, 3]))

        lif_data.append({"min": vox_min,
                         "max": vox_max,
                         "data": output_data_xyzs,
                         "surface": output_surface_xyzn})

    output_lif_base = output_base / "payload"
    output_lif_inds = []
    with _counter.get_lock():
        mesh_idx = _counter.value
        _counter.value += 1
        for lif_id in range(len(lif_data)):
            output_lif_inds.append(len(source_list))
            source_list.append([provider.get_source(idx), mesh_idx, len(output_lif_inds) - 1])
        if len(lif_data_count) > 0:
            print(f"{_counter.value}: + {len(output_lif_inds)} = {len(source_list)}, {used_points} / {data_arr.shape[0]}, "
                  f"mean lif #: {int(np.mean(lif_data_count))}, mean surface #: {int(np.mean(surface_data_count))}")

    # Output mesh
    output_obj_path = output_base / "mesh" / ("%06d.obj" % mesh_idx)
    shutil.copy(mesh_path, output_obj_path)
    provider.clean(idx)

    # Write data.
    for new_lif_id, new_lif_data in zip(output_lif_inds, lif_data):
        np.savez(output_lif_base / ("%08d.npz" % new_lif_id), **new_lif_data)


if __name__ == '__main__':
    from simple_shape import SimpleShapeGenerator
    from shapenet_model import ShapeNetGenerator
    # from indoor_scene import IndoorSceneGenerator
    # from scene_net import SceneNetGenerator
    logging.basicConfig(level=logging.INFO)

    exp_util.init_seed(4)
    mesh_providers = {
        'simple_shape': SimpleShapeGenerator,
        'shapenet_model': ShapeNetGenerator,
    }

    parser = exp_util.ArgumentParserX(add_hyper_arg=True, description='ImplicitSLAM LIF Data Generator.')
    args = parser.parse_args()

    print(args)

    dataset = mesh_providers[args.provider](**args.provider_kwargs)
    output_path = Path(args.output)

    if output_path.exists():
        print("Removing old dataset...")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "mesh").mkdir(exist_ok=True, parents=True)
    (output_path / "payload").mkdir(exist_ok=True, parents=True)

    with (output_path / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    # Shared structures:
    manager = Manager()
    source_list = manager.list()

    if args.nproc > 0:
        p = Pool(processes=args.nproc)
        p.map(functools.partial(generate_samples, args=args, output_base=output_path,
                                provider=dataset, source_list=source_list), range(len(dataset)))
    else:
        for idx in range(len(dataset)):
            generate_samples(idx, args, dataset, output_path, source_list)

    with (output_path / "source.json").open("w") as f:
        json.dump(list(source_list), f, indent=2)

    print(f"Done with {_bad_counter.value} bad shapes")
