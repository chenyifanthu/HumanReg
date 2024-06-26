import torch

class PairwiseMetric:
    def __init__(self, batch_mean: bool = False, compute_epe3d: bool = True, compute_acc3d_outlier: bool = False, scene_level: bool = False):
        """
        :param batch_mean: Whether to return an array with size (B, ) or a single scalar (mean)
        :param compute_epe3d: compute EPE3D metric
        :param compute_acc3d_outlier: compute Acc3d-strict, Acc3d-relax and outlier metric
        :param scene_level: whether use the scene threshold as proposed in FlowNet3D.
        """
        self.batch_mean = batch_mean
        self.compute_epe3d = compute_epe3d
        self.compute_acc3d_outlier = compute_acc3d_outlier
        self.scene_level = scene_level

    def evaluate(self, gt_flow: torch.Tensor, pd_flow: torch.Tensor, valid_mask: torch.Tensor = None):
        """
        Compute the pairwise flow metric; batch dimension will not be reduced. (Unit will be the same as input)
        :param gt_flow: (..., N, 3)
        :param pd_flow: (..., N, 3)
        :param valid_mask: (..., N)
        :return: metrics dict.
        """
        result_dict = {}
        assert gt_flow.size(-1) == pd_flow.size(-1) == 3
        assert gt_flow.size(-2) == pd_flow.size(-2)

        n_point = gt_flow.size(-2)
        gt_flow = gt_flow.reshape(-1, n_point, 3)
        pd_flow = pd_flow.reshape(-1, n_point, 3)
        if valid_mask is None:
            valid_mask = torch.ones((gt_flow.size(0), n_point), dtype=bool, device=gt_flow.device)
        else:
            valid_mask = valid_mask.reshape(-1, n_point)

        l2_norm = torch.norm(pd_flow - gt_flow, dim=-1)     # (B, N)

        if self.compute_epe3d:
            result_dict['epe3d'] = (l2_norm * valid_mask).sum(-1) / (valid_mask.sum(-1) + 1e-6)

        if self.compute_acc3d_outlier:
            sf_norm = torch.norm(gt_flow, dim=-1)       # (B, N)
            rel_err = l2_norm / (sf_norm + 1e-4)        # (B, N)

            if self.scene_level:
                acc3d_strict_mask = torch.logical_or(l2_norm < 0.05, rel_err < 0.05).float()
                acc3d_relax_mask = torch.logical_or(l2_norm < 0.1, rel_err < 0.1).float()
                outlier_mask = torch.logical_or(l2_norm > 0.3, rel_err > 0.1).float()
            else:
                # acc3d_strict_mask = torch.logical_or(l2_norm < 0.02, rel_err < 0.05).float()
                # acc3d_relax_mask = torch.logical_or(l2_norm < 0.05, rel_err < 0.1).float()
                # outlier_mask = (rel_err > 0.5).float()
                acc3d_strict_mask = (l2_norm < 0.05).float()
                acc3d_relax_mask = (l2_norm < 0.1).float()
                outlier_mask = (l2_norm > 0.2).float()

            result_dict['acc3d_strict'] = (acc3d_strict_mask * valid_mask).sum(-1) / (valid_mask.sum(-1) + 1e-6)
            result_dict['acc3d_relax'] = (acc3d_relax_mask * valid_mask).sum(-1) / (valid_mask.sum(-1) + 1e-6)
            result_dict['outlier'] = (outlier_mask * valid_mask).sum(-1) / (valid_mask.sum(-1) + 1e-6)

        if self.batch_mean:
            for ckey in list(result_dict.keys()):
                result_dict[ckey] = torch.mean(result_dict[ckey])

        return result_dict
