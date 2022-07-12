import argparse
import logging
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str, help='path to saved mesh')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    data_path = Path(args.save_path)
    mesh_geom = o3d.io.read_triangle_mesh(str(data_path / "mesh.ply"), print_progress=True)

    # Map texture onto this.
    pcd_path = data_path / "pcd"
    pcd_geom = []
    for ppath in pcd_path.glob("*.ply"):
        pcd_geom.append(o3d.io.read_point_cloud(str(ppath)))

    # The method is build up kd-tree search.
    pcd_points = np.vstack([np.asarray(t.points) for t in pcd_geom])
    pcd_colors = np.vstack([np.asarray(t.colors) for t in pcd_geom])

    time_start = time.time()
    nbrs = NearestNeighbors(n_neighbors=4).fit(pcd_points)
    pcd_dist, pcd_id = nbrs.kneighbors(np.asarray(mesh_geom.vertices), return_distance=True)

    # Weight the colors according to distance.
    pcd_dist = pcd_dist / np.sum(pcd_dist, axis=1, keepdims=True)
    pcd_colors = pcd_colors[pcd_id.ravel()] * np.expand_dims(pcd_dist.ravel(), -1)
    pcd_colors = pcd_colors.reshape([*pcd_dist.shape, 3])
    pcd_colors = np.sum(pcd_colors, axis=1)
    time_end = time.time() - time_start

    print("Elapsed", time_end)

    # Assign color to mesh.
    mesh_geom.vertex_colors = o3d.utility.Vector3dVector(pcd_colors)
    mesh_geom.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(data_path / "map_colored.ply"), mesh_geom)
