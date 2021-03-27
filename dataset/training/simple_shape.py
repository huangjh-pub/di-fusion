import os
import open3d as o3d
import numpy as np
from utils import motion_util
import math


class SimpleShapeGenerator:
    """
    Shape is generated approximately within [-1.0, 1.0] range.
    """

    def __init__(self, n_shapes):
        self.translation_range = [-0.5, 0.5]
        self.cube_scale = [0.1, 1.5]
        self.sphere_scale = [0.1, 1.5]
        self.cylinder_scale = [0.1, 1.5]
        self.n_shapes = n_shapes
        self.data_sources = np.random.randint(0, 3, size=(len(self), ))

    def __len__(self):
        return self.n_shapes

    @staticmethod
    def _get_tmp_filename(data_id, make_dirs=False):
        if not os.path.exists('/tmp/simple_shape') and make_dirs:
            os.makedirs('/tmp/simple_shape')
        return f'/tmp/simple_shape/{data_id}.obj'

    @staticmethod
    def _equidist_point_on_sphere(samples):
        points = []
        phi = math.pi * (3. - math.sqrt(5.))
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2
            radius = math.sqrt(1 - y * y)
            theta = phi * i
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points.append((x, y, z))
        return np.asarray(points)

    @staticmethod
    def _scale_non_uniform(geom, sx, sy, sz):
        trans_mat = np.asarray([
            sx, 0.0, 0.0, 0.0,
            0.0, sy, 0.0, 0.0,
            0.0, 0.0, sz, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]).reshape((4, 4))
        geom.transform(trans_mat)

    def generate_cube(self):
        cube_geom = o3d.geometry.TriangleMesh.create_box(
            width=np.random.uniform(self.cube_scale[0], self.cube_scale[1]),
            height=np.random.uniform(self.cube_scale[0], self.cube_scale[1]),
            depth=np.random.uniform(self.cube_scale[0], self.cube_scale[1])
        )
        cube_geom.translate(-cube_geom.get_center())
        return cube_geom

    def generate_sphere(self):
        sphere_geom = o3d.geometry.TriangleMesh.create_sphere(resolution=100, radius=0.5)
        self._scale_non_uniform(sphere_geom,
            np.random.uniform(self.sphere_scale[0], self.sphere_scale[1]),
            np.random.uniform(self.sphere_scale[0], self.sphere_scale[1]),
            np.random.uniform(self.sphere_scale[0], self.sphere_scale[1]))
        return sphere_geom

    def generate_cylinder(self):
        cylinder_geom = o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=1.0, resolution=100)
        self._scale_non_uniform(cylinder_geom,
            np.random.uniform(self.cylinder_scale[0], self.cylinder_scale[1]),
            np.random.uniform(self.cylinder_scale[0], self.cylinder_scale[1]),
            np.random.uniform(self.cylinder_scale[0], self.cylinder_scale[1]))
        return cylinder_geom

    def get_source(self, data_id):
        return ["Cube", "Sphere", "Cylinder"][self.data_sources[data_id]]

    def __getitem__(self, data_id):
        primitive_type = self.get_source(data_id)
        if primitive_type == "Cube":
            geom = self.generate_cube()
        elif primitive_type == "Sphere":
            geom = self.generate_sphere()
        else:
            geom = self.generate_cylinder()
        tf = motion_util.Isometry.random()
        tf.t = np.random.uniform(self.translation_range[0], self.translation_range[1], size=(3, ))
        geom.transform(tf.matrix)

        # Save Mesh.
        ply_path = self._get_tmp_filename(data_id, True)
        o3d.io.write_triangle_mesh(ply_path, geom)
        vp_camera = self._equidist_point_on_sphere(100)
        max_extent = np.max(np.asarray(geom.vertices), axis=0)
        min_extent = np.min(np.asarray(geom.vertices), axis=0)
        mesh_center = (max_extent + min_extent) / 2.
        vp_camera_scale = np.linalg.norm(max_extent - mesh_center) * 1.1
        vp_camera_scale = max(vp_camera_scale, 0.8)
        vp_camera = vp_camera * vp_camera_scale + mesh_center
        camera_ext = []
        for camera_i in range(vp_camera.shape[0]):
            iso = motion_util.Isometry.look_at(vp_camera[camera_i], mesh_center)
            camera_ext.append(iso)
        camera_int = [vp_camera_scale, 0.0, 2.5]

        return ply_path, [camera_int, camera_ext], None

    def clean(self, data_id):
        ply_path = self._get_tmp_filename(data_id, False)
        if os.path.exists(ply_path):
            os.unlink(ply_path)
        mtl_path = ply_path.replace('.obj', '.mtl')
        if os.path.exists(mtl_path):
            os.unlink(mtl_path)
