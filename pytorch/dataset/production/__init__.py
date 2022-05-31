import numpy as np


class FrameIntrinsic:
    def __init__(self, fx, fy, cx, cy, dscale):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.dscale = dscale

    def to_K(self):
        return np.asarray([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ])


class FrameData:
    def __init__(self):
        self.rgb = None
        self.depth = None
        self.gt_pose = None
        self.calib = None


class RGBDSequence:
    def __init__(self):
        self.frame_id = 0

    def __iter__(self):
        return self

    def __len__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError
