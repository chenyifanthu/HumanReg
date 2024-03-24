import os

import numpy as np
import MinkowskiEngine as ME

from .common import DataAugmentor
from .load import calculate_flow, calculate_labels
from torch.utils.data import Dataset
    
class Human4dDataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 ped_ids: list,
                 intervals: list = [],
                 voxel_size: float = 0.01,
                 random_seed: int = 0,
                 augmentation: dict = None):
        self.data_dir = data_dir
        self.ped_ids = ped_ids
        self.intervals = intervals
        self.voxel_size = voxel_size
        self.rng = np.random.RandomState(random_seed)
        if augmentation is None: 
            self.augmentor = None
        else:
            self.augmentor = DataAugmentor(**augmentation)
        self.generate_cases()
    
    def generate_cases(self):
        self.cases = []
        self.lengths = [0] * 10
        for ped_id in self.ped_ids:
            length = len(os.listdir(os.path.join(self.data_dir, str(ped_id))))
            self.lengths[ped_id] = length
            for i in range(length):
                if not self.intervals:
                    self.cases.append((ped_id, i))
                else:
                    for j in self.intervals:
                        if i + j < length:
                            self.cases.append((ped_id, i, i + j))
                            
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        data = self.cases[idx]
        if len(data) == 3:
            ped_id, i, j = data
        elif len(data) == 2:
            ped_id, i = data
            j = self.rng.randint(0, self.lengths[ped_id] - 1)
        else:
            raise ValueError("Invalid data")

        datai = np.load(os.path.join(self.data_dir, str(ped_id), f"{i:05d}.npz"))
        dataj = np.load(os.path.join(self.data_dir, str(ped_id), f"{j:05d}.npz"))
        ret = {}
        
        # Load point clouds
        ret["pcs"] = [datai['pc'], dataj['pc']]
        
        # Load labels
        labels1 = calculate_labels(datai['pc'], datai['joints'])
        labels2 = calculate_labels(dataj['pc'], dataj['joints'])
        ret["labels"] = [labels1, labels2]

        # Load flows
        flow12 = calculate_flow(datai['pc'], datai['meshid'], dataj['mesh'])
        flow21 = calculate_flow(dataj['pc'], dataj['meshid'], datai['mesh'])
        ret["flows"] = [flow12, flow21]
        
        if self.augmentor is not None:
            self.augmentor.process(ret, self.rng)
            
        # Quantize point clouds
        quan_coords1 = self.quantize(ret["pcs"][0])
        quan_coords2 = self.quantize(ret["pcs"][1])
        ret["coords"] = [quan_coords1, quan_coords2]
        
        return ret
    
    def quantize(self, points):
        coords = np.floor(points / self.voxel_size)
        inds = ME.utils.sparse_quantize(coords, return_index=True, return_maps_only=True)
        return coords[inds], inds
