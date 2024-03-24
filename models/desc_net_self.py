import torch, cv2
import torch_scatter
import MinkowskiEngine as ME
from torch.nn import Parameter

from dataset.human4d import *
from metric import PairwiseMetric

from typing import *
from collections import defaultdict
from models.spconv import ResUNet
from models.base_model import BaseModel
import numpy as np
from functools import reduce
from tools.point import propagate_features

class Model(BaseModel):
    """
    This model trains the descriptor network. This is the 1st stage of training.
    """
    def __init__(self, hparams):
        super().__init__(hparams)
        self.backbone_args = self.hparams.backbone_args
        self.backbone = ResUNet(self.backbone_args,
                                in_channels=3,
                                out_channels=self.backbone_args.feat_channels,
                                n_classes=self.backbone_args.n_classes,
                                normalize_feature=True,
                                conv1_kernel_size=3)
        self.td = Parameter(torch.tensor(np.float32(self.hparams.td_init)), requires_grad=True)

    def forward(self, batch):
        """
        Forward descriptor network.
            As the backbone quantized point cloud into voxels (by selecting one point for each voxel),
        we also return the selected point indices.
        """
        num_batches = len(batch["pcs"][0])
        num_views = len(batch["pcs"])
        all_coords, all_feats, all_sels = [], [], []
        for batch_idx in range(num_batches):
            for view_idx in range(num_views):
                all_coords.append(batch["coords"][view_idx][0][batch_idx])
                cur_sel = batch["coords"][view_idx][1][batch_idx]
                all_sels.append(cur_sel)
                all_feats.append(batch["pcs"][view_idx][batch_idx][cur_sel])
        coords_batch, feats_batch = ME.utils.sparse_collate(all_coords, all_feats, device=self.device)
        sinput = ME.SparseTensor(feats_batch, coordinates=coords_batch)
        desc_output, cls_output = self.backbone(sinput)

        # Compute loss and metrics
        num_batches = len(batch["pcs"][0])
        losses, metrics, pd = [], [], []
        has_labels = "labels" in batch
        has_flows = "flows" in batch
        for batch_idx in range(num_batches):
            cur_pc0, cur_pc1 = batch["pcs"][0][batch_idx], batch["pcs"][1][batch_idx]
            cur_sel0, cur_sel1 = all_sels[batch_idx * 2 + 0], all_sels[batch_idx * 2 + 1]
            cur_gt0 = cur_gt1 = cur_labels0 = cur_labels1 = None
            if has_flows:
                cur_gt0, cur_gt1 = batch["flows"][0][batch_idx], batch["flows"][1][batch_idx]
            if has_labels:
                cur_labels0, cur_labels1 = batch["labels"][0][batch_idx], batch["labels"][1][batch_idx]
            
            cur_cls0 = cls_output.features_at(batch_idx * 2 + 0)
            cur_cls1 = cls_output.features_at(batch_idx * 2 + 1)
            cur_feat0 = desc_output.features_at(batch_idx * 2 + 0)
            cur_feat1 = desc_output.features_at(batch_idx * 2 + 1)
            
            dist_mat = torch.cdist(cur_feat0, cur_feat1) / torch.maximum(
                torch.tensor(np.float32(self.hparams.td_min), device=self.device), self.td)
            cur_pd0 = torch.softmax(-dist_mat, dim=1) @ cur_pc1[cur_sel1] - cur_pc0[cur_sel0]
            cur_pd1 = torch.softmax(-dist_mat, dim=0).transpose(-1, -2) @ cur_pc0[cur_sel0] - cur_pc1[cur_sel1]
            
            # Compute Loss
            loss_dict = self.compute_self_sup_loss(cur_pc0[cur_sel0], cur_pc1[cur_sel1], cur_pd0, 
                                                   cur_cls0, cur_cls1, self.hparams.self_sup_loss)
            losses.append(loss_dict)
            loss_dict = self.compute_self_sup_loss(cur_pc1[cur_sel1], cur_pc0[cur_sel0], cur_pd1,
                                                   cur_cls1, cur_cls0, self.hparams.self_sup_loss)
            losses.append(loss_dict)
            
            # For testing only
            if not self.hparams.is_training:
                pd_full_cls0 = propagate_features(cur_pc0[cur_sel0], cur_pc0, cur_cls0, batched=False)
                pd_full_cls1 = propagate_features(cur_pc1[cur_sel1], cur_pc1, cur_cls1, batched=False)
                pd_full_flow01 = propagate_features(cur_pc0[cur_sel0], cur_pc0, cur_pd0, batched=False)
                pd_full_flow10 = propagate_features(cur_pc1[cur_sel1], cur_pc1, cur_pd1, batched=False)
                metric = self.compute_metric(pd_full_cls0, pd_full_cls1, pd_full_flow01, pd_full_flow10,
                                             cur_labels0, cur_labels1, cur_gt0, cur_gt1)
                metrics.append(metric)
                
                cls0_pd = torch.max(pd_full_cls0, 1)[1]
                cls1_pd = torch.max(pd_full_cls1, 1)[1]
                pd_full_flow01 = self.refine_flow(cur_pc0, cls0_pd, pd_full_flow01)
                pd_full_flow10 = self.refine_flow(cur_pc1, cls1_pd, pd_full_flow10)
                
                pd_full = {'cls0': cls0_pd, 
                           'cls1': cls1_pd,
                           'flow01': pd_full_flow01.cpu().numpy(), 
                           'flow10': pd_full_flow10.cpu().numpy()
                }
                pd.append(pd_full)
                
        return losses, metrics, pd
    
    def refine_flow(self, pc0, cls0, flow01):
        pc0_warpped = pc0 + flow01
        for i in range(14):
            idx = cls0 == i
            if idx.sum() < 2: continue
            src = pc0[idx]
            dst = pc0_warpped[idx]
            src_mean = src.mean(0)
            dst_mean = dst.mean(0)
            # Compute covariance 
            H = (src - src_mean).T @ (dst - dst_mean)
            u, _, vt = torch.linalg.svd(H)
            rot_pd_T = u @ vt
            t_pd = - src_mean @ rot_pd_T + dst_mean
            flow01[idx] = src @ rot_pd_T + t_pd - src
        return flow01
    
    
    @staticmethod
    def compute_self_sup_loss(pc0, pc1, pd_flow01, lbl_out0, lbl_out1, loss_config):
        pc0_warpped = pc0 + pd_flow01
        dist01 = torch.cdist(pc0_warpped, pc1)
        loss_dict = {}
        k_neigh = loss_config.k_neigh
        
        if loss_config.chamfer_weight > 0.0:
            chamfer01 = torch.min(dist01, dim=-1).values
            chamfer10 = torch.min(dist01, dim=-2).values
            loss_dict['chamfer'] = loss_config.chamfer_weight * (chamfer01.mean() + chamfer10.mean())

        if loss_config.smooth_weight > 0.0:
            dist00 = torch.cdist(pc0, pc0)
            _, kidx0 = torch.topk(dist00, k_neigh, dim=-1, largest=False, sorted=False)
            
            grouped_flow = pd_flow01[kidx0]     # (N, K, 3)
            loss_dict['smooth'] = loss_config.smooth_weight * \
                        (((grouped_flow - pd_flow01.unsqueeze(1)) ** 2).sum(-1).sum(-1) / (k_neigh - 1.0)).mean()
                        
        if loss_config.class_weight > 0.0:
            lbl0 = torch.argmax(lbl_out0, dim=-1) # (N)
            grouped_lbl_gt = lbl0.repeat(k_neigh, 1).T.flatten() # (N * K)
            dist, knn01 = torch.topk(dist01, k_neigh, dim=-1, largest=False, sorted=False)  # (N, K)
            grouped_lbl_pd = lbl_out1[knn01].reshape(-1, 14) # (N * K, 14)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            loss_dict['class'] = loss_config.class_weight * \
                criterion(grouped_lbl_pd, grouped_lbl_gt) / k_neigh
        else:
            lbl0 = torch.argmax(lbl_out0, dim=-1) # (N)
        
        if loss_config.translate_weight > 0.0:
            cnts = torch_scatter.scatter_add(torch.ones_like(lbl0), lbl0, dim=0)
            idx = torch.where(cnts[lbl0] > 1)[0]
            lbl0 = lbl0[idx]
            pc0_mean = torch_scatter.scatter_mean(pc0[idx], lbl0, dim=0)
            pc1_mean = torch_scatter.scatter_mean(pc0_warpped[idx], lbl0, dim=0)
            pts0 = pc0[idx] - pc0_mean[lbl0]
            pts1 = pc0_warpped[idx] - pc1_mean[lbl0]

            n_points = pts0.shape[0]
            n_class = 14
            pts0_scatter = torch.zeros(n_class, n_points, 3).cuda().scatter_(0, lbl0[None, :, None].expand(-1, -1, 3), pts0.unsqueeze(0))
            pts1_scatter = torch.zeros(n_class, n_points, 3).cuda().scatter_(0, lbl0[None, :, None].expand(-1, -1, 3), pts1.unsqueeze(0))
            H = torch.bmm(pts0_scatter.transpose(1, 2), pts1_scatter)
            U, _, Vt = torch.linalg.svd(H)
            rot_pd_T = torch.bmm(U, Vt) # (n_class, 3, 3)
            error = pts1 - torch.bmm(pts0.unsqueeze(1), rot_pd_T[lbl0]).squeeze(1)
            tl_loss_sum = torch.sum(error ** 2)
            loss_dict['translate'] = loss_config.translate_weight * tl_loss_sum / n_points

        return loss_dict
    
    @staticmethod
    def compute_metric(cls0, cls1, fpd01, fpd10,
                       labels0=None, labels1=None,
                       fgt01=None, fgt10=None):
        metric = PairwiseMetric(compute_epe3d=True, compute_acc3d_outlier=True)
        res = defaultdict(list)
        if labels0 is not None:
            _, pred0 = torch.max(cls0, 1)
            correct = (pred0 == labels0).sum().item()
            res['accuracy'].append(correct / pred0.shape[0])
        if labels1 is not None:
            _, pred1 = torch.max(cls1, 1)
            correct = (pred1 == labels1).sum().item()
            res['accuracy'].append(correct / pred1.shape[0])
        if fgt01 is not None:
            m = metric.evaluate(fgt01, fpd01)
            for k, v in m.items():
                res[k].append(v.item())
        if fgt10 is not None:
            m = metric.evaluate(fgt10, fpd10)
            for k, v in m.items():
                res[k].append(v.item())
        for k, v in res.items():
            res[k] = np.mean(v)
        return res

    def step(self, batch, mode):
        assert mode in ['train', 'valid', 'test']
        losses, metrics, pd = self(batch)
        loss_sum = 0.0
        for loss_dict in losses:
            for name, val in loss_dict.items():
                self.log(f'{mode}/{name}-loss', val)
                loss_sum += val
        for metric_dict in metrics:
            for name, val in metric_dict.items():
                self.log(f'{mode}/{name}', val)
        total_loss = loss_sum / len(losses)
        self.log(f'{mode}/total_loss', total_loss)
        return total_loss, pd