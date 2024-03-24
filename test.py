import os; os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import importlib
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from tools.exp import to_target_device
from dataset import get_dataloader, get_dataset

ckpt_path = './weights/finetune-cape512.pth'
model_cfg_path = './configs/desc_net_self.yaml'
data_cfg_path = './configs/cape.yaml'

model_args = OmegaConf.load(model_cfg_path)
data_args = OmegaConf.load(data_cfg_path)
model_args.batch_size = data_args.batch_size = 1

net_module = importlib.import_module("models." + model_args.model).Model
net_model = net_module(model_args)
net_model.load_state_dict(torch.load(ckpt_path)['state_dict'])
print(f"Checkpoint loaded from {ckpt_path}.")
import pdb; pdb.set_trace()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net_model = to_target_device(net_model, device)
net_model.device = device
net_model.eval()
net_model.hparams.is_training = False

# test_loader = get_dataloader(data_args, 'test')
from torch.utils.data import DataLoader
from dataset.common import list_collate
test_set = get_dataset(data_args, 'test')
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=10, collate_fn=list_collate)

pbar = tqdm(test_loader, desc=f'Testing', ncols=120)
with torch.no_grad():
    for batch_idx, batch in enumerate(pbar):
        batch = to_target_device(batch, device)
        net_model.step(batch, 'test')
        # if batch_idx == 20: break

d = net_model.log_cache.loss_dict
epe3d = d['test/epe3d']
AccS = d['test/acc3d_strict']
AccR = d['test/acc3d_relax']
outlier = d['test/outlier']
# print(d['test/acc3d_strict'])
# print(len(d['test/acc3d_strict']))
print("Test metrics:")
print("  + EPE3D: \t %.2f\t+/-\t%.2f" % (np.mean(epe3d) * 100, np.std(epe3d) * 100))
print("  + AccS (%%): \t %.1f\t+/-\t%.1f" % (np.mean(AccS) * 100, np.std(AccS) * 100))
print("  + AccR (%%): \t %.1f\t+/-\t%.1f" % (np.mean(AccR) * 100, np.std(AccR) * 100))
print("  + Outlier: \t %.2f\t+/-\t%.2f" % (np.mean(outlier) * 100, np.std(outlier) * 100))