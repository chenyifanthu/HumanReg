import torch
import collections
import numpy as np
from pyquaternion.quaternion import Quaternion

def list_collate(batch):
    """
    This collation does not stack batch dimension, but instead output only lists.
    """
    elem = None
    for e in batch:
        if e is not None:
            elem = e
            break
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return list_collate([torch.as_tensor(b) if b is not None else None for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: list_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [list_collate(samples) for samples in transposed]
    elif elem is None:
        return batch

    raise NotImplementedError


class DataAugmentor:
    """
    Will apply data augmentation to pairwise point clouds, by applying random transformations
    to the point clouds (or one of them), or adding noise.
    """
    def __init__(self, 
                 centralize: dict = None,
                 together: dict = None,
                 pc2: dict = None):
        self.centralize_args = centralize
        self.together_args = together
        self.pc2_args = pc2

    def process(self, data_dict: dict, rng: np.random.RandomState):
        pcs = data_dict["pcs"]
        assert len(pcs) == 2

        pc1, pc2 = pcs[0], pcs[1]
        if 'flows' in data_dict:
            pc1_virtual = pc2 + data_dict["flows"][1]
            pc2_virtual = pc1 + data_dict["flows"][0]

        if self.centralize_args is not None:
            pc1_center = np.mean(pc1, axis=0)
            pc2_center = np.mean(pc2, axis=0)
            pc1 -= pc1_center
            pc2 -= pc2_center
            if 'flows' in data_dict:
                pc1_virtual -= pc1_center
                pc2_virtual -= pc2_center

        if self.together_args is not None:
            scale = np.diag(rng.uniform(self.together_args.scale_low,
                                        self.together_args.scale_high, 3).astype(np.float32))
            angle = rng.uniform(-self.together_args.degree_range, self.together_args.degree_range) / 180.0 * np.pi
            rot_matrix = np.array([
                [np.cos(angle), np.sin(angle), 0.],
                [-np.sin(angle), np.cos(angle), 0.],
                [0., 0., 1.]
            ], dtype=np.float32)
            matrix = scale.dot(rot_matrix.T)

            pc1 = pc1.dot(matrix)
            pc2 = pc2.dot(matrix)
            if 'flows' in data_dict:
                pc1_virtual = pc1_virtual.dot(matrix)
                pc2_virtual = pc2_virtual.dot(matrix)

        if self.pc2_args is not None:
            angle2 = rng.uniform(-self.pc2_args.degree_range, self.pc2_args.degree_range) / 180.0 * np.pi
            rot_axis = np.array([0.0, 0.0, 1.0])
            matrix2 = Quaternion(axis=rot_axis, radians=angle2).rotation_matrix.astype(np.float32)

            jitter1 = np.clip(self.pc2_args.jitter_sigma * rng.randn(pc1.shape[0], 3),
                              -self.pc2_args.jitter_clip, self.pc2_args.jitter_clip).astype(np.float32)
            jitter2 = np.clip(self.pc2_args.jitter_sigma * rng.randn(pc2.shape[0], 3),
                              -self.pc2_args.jitter_clip, self.pc2_args.jitter_clip).astype(np.float32)

            pc1 = pc1 + jitter1
            pc2 = pc2.dot(matrix2) + jitter2
            if 'flows' in data_dict:
                pc2_virtual = pc2_virtual.dot(matrix2)
            

        data_dict["pcs"] = [pc1, pc2]
        if 'flows' in data_dict:
            data_dict["flows"] = [pc2_virtual - pc1, pc1_virtual - pc2]
            