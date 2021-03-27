import cv2
import os
import torch
from dataset.production import *
from pyquaternion import Quaternion
from pathlib import Path
from utils import motion_util


class ICLNUIMSequence(RGBDSequence):
    def __init__(self, path: str, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False):
        super().__init__()
        self.path = Path(path)
        self.color_names = sorted([f"rgb/{t}" for t in os.listdir(self.path / "rgb")], key=lambda t: int(t[4:].split(".")[0]))
        self.depth_names = [f"depth/{t}.png" for t in range(len(self.color_names))]
        self.calib = [481.2, 480.0, 319.50, 239.50, 5000.0]
        if first_tq is not None:
            self.first_iso = motion_util.Isometry(q=Quaternion(array=first_tq[3:]), t=np.array(first_tq[:3]))
        else:
            self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0]))

        if end_frame == -1:
            end_frame = len(self.color_names)

        self.color_names = self.color_names[start_frame:end_frame]
        self.depth_names = self.depth_names[start_frame:end_frame]

        if load_gt:
            gt_traj_path = (list(self.path.glob("*.freiburg")) + list(self.path.glob("groundtruth.txt")))[0]
            self.gt_trajectory = self._parse_traj_file(gt_traj_path)
            self.gt_trajectory = self.gt_trajectory[start_frame:end_frame]
            change_iso = self.first_iso.dot(self.gt_trajectory[0].inv())
            self.gt_trajectory = [change_iso.dot(t) for t in self.gt_trajectory]
            assert len(self.gt_trajectory) == len(self.color_names)
        else:
            self.gt_trajectory = None

    def _parse_traj_file(self, traj_path):
        camera_ext = {}
        traj_data = np.genfromtxt(traj_path)
        cano_quat = motion_util.Isometry(q=Quaternion(axis=[0.0, 0.0, 1.0], degrees=180.0))
        for cur_p in traj_data:
            cur_q = Quaternion(imaginary=cur_p[4:7], real=cur_p[-1]).rotation_matrix
            cur_t = cur_p[1:4]
            cur_q[1] = -cur_q[1]
            cur_q[:, 1] = -cur_q[:, 1]
            cur_t[1] = -cur_t[1]
            cur_iso = motion_util.Isometry(q=Quaternion(matrix=cur_q), t=cur_t)
            camera_ext[cur_p[0]] = cano_quat.dot(cur_iso)
        camera_ext[0] = camera_ext[1]
        return [camera_ext[t] for t in range(len(camera_ext))]

    def __len__(self):
        return len(self.color_names)

    def __next__(self):
        if self.frame_id >= len(self):
            raise StopIteration

        depth_img_path = self.path / self.depth_names[self.frame_id]
        rgb_img_path = self.path / self.color_names[self.frame_id]

        # Convert depth image into point cloud.
        depth_data = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)
        depth_data = torch.from_numpy(depth_data.astype(np.float32)).cuda() / self.calib[4]
        rgb_data = cv2.imread(str(rgb_img_path))
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        rgb_data = torch.from_numpy(rgb_data).cuda().float() / 255.

        frame_data = FrameData()
        frame_data.gt_pose = self.gt_trajectory[self.frame_id] if self.gt_trajectory is not None else None
        frame_data.calib = FrameIntrinsic(self.calib[0], self.calib[1], self.calib[2], self.calib[3], self.calib[4])
        frame_data.depth = depth_data
        frame_data.rgb = rgb_data

        self.frame_id += 1
        return frame_data
