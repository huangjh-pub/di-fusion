import numpy as np
from pyquaternion import Quaternion


def project_orthogonal(rot):
    u, s, vh = np.linalg.svd(rot, full_matrices=True, compute_uv=True)
    rot = u @ vh
    if np.linalg.det(rot) < 0:
        u[:, 2] = -u[:, 2]
        rot = u @ vh
    return rot


class Isometry:
    GL_POST_MULT = Quaternion(degrees=180.0, axis=[1.0, 0.0, 0.0])

    def __init__(self, q=None, t=None):
        if q is None:
            q = Quaternion()
        if t is None:
            t = np.zeros(3)
        if not isinstance(t, np.ndarray):
            t = np.asarray(t)
        assert t.shape[0] == 3 and t.ndim == 1
        self.q = q
        self.t = t

    def __repr__(self):
        return f"Isometry: t = {self.t}, q = {self.q}"

    @property
    def rotation(self):
        return Isometry(q=self.q)

    @property
    def matrix(self):
        mat = self.q.transformation_matrix
        mat[0:3, 3] = self.t
        return mat

    @staticmethod
    def from_matrix(mat, t_component=None, ortho=False):
        assert isinstance(mat, np.ndarray)
        if t_component is None:
            assert mat.shape == (4, 4)
            if ortho:
                mat[:3, :3] = project_orthogonal(mat[:3, :3])
            return Isometry(q=Quaternion(matrix=mat), t=mat[:3, 3])
        else:
            assert mat.shape == (3, 3)
            assert t_component.shape == (3,)
            if ortho:
                mat = project_orthogonal(mat)
            return Isometry(q=Quaternion(matrix=mat), t=t_component)

    @property
    def continuous_repr(self):
        rot = self.q.rotation_matrix[:, 0:2].T.flatten()    # (6,)
        return np.concatenate([rot, self.t])                # (9,)

    @staticmethod
    def from_continuous_repr(rep, gs=True):
        if isinstance(rep, list):
            rep = np.asarray(rep)
        assert isinstance(rep, np.ndarray)
        assert rep.shape == (9,)
        # For rotation, use Gram-Schmidt orthogonalization
        col1 = rep[0:3]
        col2 = rep[3:6]
        if gs:
            col1 /= np.linalg.norm(col1)
            col2 = col2 - np.dot(col1, col2) * col1
            col2 /= np.linalg.norm(col2)
        col3 = np.cross(col1, col2)
        return Isometry(q=Quaternion(matrix=np.column_stack([col1, col2, col3])), t=rep[6:9])

    @property
    def full_repr(self):
        rot = self.q.rotation_matrix.T.flatten()
        return np.concatenate([rot, self.t])

    @staticmethod
    def from_full_repr(rep, ortho=False):
        assert isinstance(rep, np.ndarray)
        assert rep.shape == (12,)
        rot = rep[0:9].reshape(3, 3).T
        if ortho:
            rot = project_orthogonal(rot)
        return Isometry(q=Quaternion(matrix=rot), t=rep[9:12])

    @staticmethod
    def random():
        return Isometry(q=Quaternion.random(), t=np.random.random((3,)))

    def inv(self):
        qinv = self.q.inverse
        return Isometry(q=qinv, t=-(qinv.rotate(self.t)))

    def dot(self, right):
        return Isometry(q=(self.q * right.q), t=(self.q.rotate(right.t) + self.t))

    def to_gl_camera(self):
        return Isometry(q=(self.q * self.GL_POST_MULT), t=self.t)

    @staticmethod
    def look_at(source: np.ndarray, target: np.ndarray, up: np.ndarray = None):
        z_dir = target - source
        z_dir /= np.linalg.norm(z_dir)
        if up is None:
            up = np.asarray([0.0, 1.0, 0.0])
            if np.linalg.norm(np.cross(z_dir, up)) < 1e-6:
                up = np.asarray([1.0, 0.0, 0.0])
        else:
            up /= np.linalg.norm(up)
        x_dir = np.cross(z_dir, up)
        x_dir /= np.linalg.norm(x_dir)
        y_dir = np.cross(z_dir, x_dir)
        R = np.column_stack([x_dir, y_dir, z_dir])
        return Isometry(q=Quaternion(matrix=R), t=source)

    def tangent(self, prev_iso, next_iso):
        t = 0.5 * (next_iso.t - prev_iso.t)
        l1 = Quaternion.log((self.q.inverse * prev_iso.q).normalised)
        l2 = Quaternion.log((self.q.inverse * next_iso.q).normalised)
        e = Quaternion()
        e.q = -0.25 * (l1.q + l2.q)
        e = self.q * Quaternion.exp(e)
        return Isometry(t=t, q=e)

    def __matmul__(self, other):
        if isinstance(other, Isometry):
            return self.dot(other)
        if type(other) != np.ndarray or other.ndim == 1:
            return self.q.rotate(other) + self.t
        else:
            return other @ self.q.rotation_matrix.T + self.t[np.newaxis, :]

    @staticmethod
    def interpolate(source, target, alpha):
        iquat = Quaternion.slerp(source.q, target.q, alpha)
        it = source.t * (1 - alpha) + target.t * alpha
        return Isometry(q=iquat, t=it)
