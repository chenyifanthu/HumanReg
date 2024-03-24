import os
import json
import numpy as np
import MinkowskiEngine as ME

from torch.utils.data import Dataset
from .common import DataAugmentor

class SingleDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 split: str,
                 random_seed: int = 0,
                 voxel_size: float = 0.01,
                 augmentation: dict = None) -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.data_dir = data_dir
        self.split = split
        self.voxel_size = voxel_size
        if augmentation is None:
            self.augmentor = None
        else:
            self.augmentor = DataAugmentor(**augmentation)
        self.rng = np.random.RandomState(random_seed)
        self.generate_cases()
        
    def generate_cases(self) -> None:
        self.filelist = []
        meta = json.load(open(os.path.join(self.data_dir, "meta.json")))[self.split]
        for file, n_frames in meta:
            self.filelist.append(os.path.join(self.data_dir, 'data', file))
        self.idxs = []
        for i in range(4):
            for j in range(4):
                if i < j:
                    self.idxs.append((i, j))
    
    def __len__(self) -> int:
        return len(self.idxs) * len(self.filelist)
    
    def __getitem__(self, index) -> dict:
        filepath = self.filelist[index // len(self.idxs)]
        i, j = self.idxs[index % len(self.idxs)]
        data = np.load(filepath, allow_pickle=True)
        
        ret = {}
        ret = {"file": filepath, "idxs": [i, j]}
        
        # Load point clouds
        pc1 = data['pcs'][i].astype(np.float32)
        pc2 = data['pcs'][j].astype(np.float32)
        ret["pcs"] = [pc1, pc2]
        
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