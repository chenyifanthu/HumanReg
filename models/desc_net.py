import torch
import MinkowskiEngine as ME
from torch.nn import Parameter
from torch.utils.data import DataLoader

from dataset.human4d import *
from metric import PairwiseMetric

from typing import *
from collections import defaultdict
from models.spconv import ResUNet
from models.base_model import BaseModel
import numpy as np
from tools.point import propagate_features
from sklearn.metrics import confusion_matrix


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
            
            # Compute Metrics
            loss_dict = self.compute_sup_loss(self.hparams.sup_loss, 
                                              cur_cls0, cur_cls1, cur_pd0, cur_pd1, cur_sel0, cur_sel1,
                                              cur_labels0, cur_labels1, cur_gt0, cur_gt1)
            losses.append(loss_dict)
            
            pd_full = {}
            if not self.hparams.is_training:
                pd_full_cls0 = propagate_features(cur_pc0[cur_sel0], cur_pc0, cur_cls0, batched=False)
                pd_full_cls1 = propagate_features(cur_pc1[cur_sel1], cur_pc1, cur_cls1, batched=False)
                pd_full_flow01 = propagate_features(cur_pc0[cur_sel0], cur_pc0, cur_pd0, batched=False)
                pd_full_flow10 = propagate_features(cur_pc1[cur_sel1], cur_pc1, cur_pd1, batched=False)
                metric = self.compute_metric(pd_full_cls0, pd_full_cls1, pd_full_flow01, pd_full_flow10,
                                             cur_labels0, cur_labels1, cur_gt0, cur_gt1)
                metrics.append(metric)
                pd_full = {'cls0': torch.max(pd_full_cls0, 1)[1], 
                           'cls1': torch.max(pd_full_cls1, 1)[1],
                           'flow01': pd_full_flow01.cpu().numpy(), 
                           'flow10': pd_full_flow10.cpu().numpy(),
                }
                pd.append(pd_full)
                
        return losses, metrics, pd
        
    @staticmethod
    def compute_sup_loss(loss_config, cls0, cls1, fpd0, fpd1, sel0, sel1,
                         labels0=None, labels1=None,
                         fgt0=None, fgt1=None) -> Tuple[torch.Tensor, torch.Tensor]:
        res = defaultdict(list)
        cls_criterion = torch.nn.CrossEntropyLoss()
        if labels0 is not None:
            loss = (1 - loss_config.lmda) * cls_criterion(cls0, labels0[sel0])
            res['cls_loss'].append(loss)
        if labels1 is not None:
            loss = (1 - loss_config.lmda) * cls_criterion(cls1, labels1[sel1])
            res['cls_loss'].append(loss)
        if fgt0 is not None:
            loss = loss_config.lmda * torch.linalg.norm(fpd0 - fgt0[sel0], dim=-1).mean()
            res['flow_loss'].append(loss)
        if fgt1 is not None:
            loss = loss_config.lmda * torch.linalg.norm(fpd1 - fgt1[sel1], dim=-1).mean()
            res['flow_loss'].append(loss)
        for k, v in res.items():
            res[k] = torch.stack(v).mean()
        return res
    
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
                self.log(f'{mode}/{name}', val)
                loss_sum += val
        for metric_dict in metrics:
            for name, val in metric_dict.items():
                self.log(f'{mode}/{name}', val)
        total_loss = loss_sum / len(losses)
        self.log(f'{mode}/total_loss', total_loss)
        return total_loss, pd