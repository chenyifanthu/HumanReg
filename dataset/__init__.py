import torch
torch.manual_seed(0) 
from torch.utils.data import DataLoader
from .common import list_collate
from .human4d import Human4dDataset
from .cape import CAPEDataset
from .single import SingleDataset

def get_dataloader(cfg, mode):
    assert mode in ['train', 'valid', 'test']
    dataset = eval(cfg.name)(**cfg[mode])
    loader = DataLoader(dataset, 
                        batch_size = cfg.batch_size, 
                        shuffle = mode == 'train',
                        num_workers = 10,
                        collate_fn = list_collate)
    return loader

def get_dataset(cfg, mode):
    assert mode in ['train', 'valid', 'test']
    dataset = eval(cfg.name)(**cfg[mode])
    return dataset