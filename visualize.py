import os; os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import matplotlib.pyplot as plt

from tqdm import tqdm
from omegaconf import OmegaConf
from tools.exp import to_target_device
from models.desc_net import Model
from dataset.common import list_collate

ckpt_path = './weights/best-cls-flow.pth'
model_cfg_path = './configs/desc_net.yaml'
ped_id = 9
start_idx = 100
n_frames = 10
n_skip = 2

def get_loader(voxel_size=0.01):
    def quantize(points):
        coords = np.floor(points / voxel_size)
        inds = ME.utils.sparse_quantize(coords, return_index=True, return_maps_only=True)
        return coords[inds], inds
    
    datai = np.load(f'./human4d/{ped_id}/{start_idx:05d}.npz')
    meani = np.mean(datai['pc'], axis=0)
    for k in range(1, n_frames):
        j = start_idx + k * n_skip
        dataj = np.load(f'./human4d/{ped_id}/{j:05d}.npz')
        meanj = np.mean(dataj['pc'], axis=0)
        pci = datai['pc'] - meani
        pcj = dataj['pc'] - meanj
        ret = {
            'pcs': [pci, pcj],
            # 'pcs': [datai['pc'], dataj['pc']],
            'coords': [quantize(pci), quantize(pcj)],
        }
        yield list_collate([ret])


model_args = OmegaConf.load(model_cfg_path)
model_args.batch_size = 1

net_model = Model(model_args)
net_model.load_state_dict(torch.load(ckpt_path)['state_dict'])
print(f"Checkpoint loaded from {ckpt_path}.")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net_model = to_target_device(net_model, device)
net_model.device = device
net_model.eval()
net_model.hparams.is_training = False

res_flow_vis = o3d.geometry.PointCloud()
res_cls_vis = o3d.geometry.PointCloud()
cmap = plt.get_cmap('jet')
with torch.no_grad():
    for i, batch in tqdm(enumerate(get_loader()), total=n_frames-1):
        batch = to_target_device(batch, device)
        pred = net_model.step(batch, 'test')[1]
        if i == 0:
            pts0 = batch['pcs'][0][0].cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts0)
            pcd.paint_uniform_color(cmap(0.0)[:3])
            res_flow_vis += pcd
            cls0 = pred[0]['cls0'].cpu().numpy()
            colors = cmap(cls0 / 13)[:,:3]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            res_cls_vis += pcd

        pts1 = batch['pcs'][1][0].cpu().numpy()
        cls1 = pred[0]['cls1'].cpu().numpy()
        flow10 = pred[0]['flow10'].cpu().numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts1+flow10)
        pcd.paint_uniform_color(cmap((i+1)/(n_frames-1))[:3])
        res_flow_vis += pcd
        colors = cmap(cls1 / 13)[:,:3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        res_cls_vis += pcd
        
o3d.io.write_point_cloud('flow_vis.ply', res_flow_vis)
o3d.io.write_point_cloud('cls_vis.ply', res_cls_vis)