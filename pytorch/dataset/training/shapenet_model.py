import os
import math
import json
import random
import logging
import numpy as np
from utils import motion_util
from pathlib import Path


class ShapeNetGenerator:
    """
    Use ShapeNet core to generate data.
    """
    VALID_LIST_PATH = Path(__file__).parent / "shapenet_valid_list.json"

    def __init__(self, shapenet_path, categories, shapes_per_category, scale):
        self.categories = categories
        self.shapes_per_category = shapes_per_category
        self.scale = scale

        # Sample objects
        self.data_sources = []
        self.data_scales = []
        with self.VALID_LIST_PATH.open("r") as f:
            valid_list_data = json.load(f)

        for category_name, category_shape_count, category_scale in zip(self.categories, self.shapes_per_category, self.scale):
            category_path = Path(shapenet_path) / category_name
            if category_name in valid_list_data["ShapeNetV2"].keys():
                logging.info(f"Category {category_name} is found in plist file")
                sampled_objects = valid_list_data["ShapeNetV2"][category_name]
            else:
                logging.info(f"Category {category_name} is not found in plist file")
                sampled_objects = os.listdir(category_path)
            if category_shape_count != -1:
                sampled_objects = random.sample(sampled_objects, category_shape_count)
            self.data_sources += [category_path / s for s in sampled_objects]
            self.data_scales += [category_scale for _ in sampled_objects]

    def __len__(self):
        return len(self.data_sources)

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

    def get_source(self, data_id):
        return str(self.data_sources[data_id])

    def __getitem__(self, idx):
        data_source = self.data_sources[idx]
        data_scale = self.data_scales[idx]
        obj_path = data_source / "models" / "model_normalized.obj"

        vp_camera = self._equidist_point_on_sphere(300)
        camera_ext = []
        for camera_i in range(vp_camera.shape[0]):
            iso = motion_util.Isometry.look_at(vp_camera[camera_i], np.zeros(3,))
            camera_ext.append(iso)
        camera_int = [0.8, 0.0, 2.5]  # (window-size-half, z-min, z-max) under ortho-proj.

        return str(obj_path), [camera_int, camera_ext], None, data_scale

    def clean(self, data_id):
        pass
