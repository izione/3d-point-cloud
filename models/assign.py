import torch

from .box_utils import corner_distance_cost

TOPQ = 10  # SimOTA-style: dynamic-k comes from the sum of the top-`TOPQ` quality scores


class DynamicLabelAssigner:
    """FSHNet DCLA-style assignment, extended to the *entire* active-voxel set
    (not a fixed top-5 spatial candidate pool -- see design discussion) and
    using a corner-distance regression cost instead of RWIoU, since RWIoU
    assumes yaw-only rotation and our boxes carry a full quaternion.

    cost = cls_weight * (1 - cls_score) + reg_weight * corner_distance
    "quality" for the dynamic-k estimate is exp(-corner_distance), a bounded
    (0,1] stand-in for the IoU quality score SimOTA/FSHNet use natively.
    """

    def __init__(self, cfg, pc_range, effective_voxel_size):
        self.cls_weight = cfg["CLS_WEIGHT"]
        self.reg_weight = cfg["REG_WEIGHT"]
        self.min_positives = cfg["MIN_POSITIVES"]
        self.pc_range = pc_range
        self.effective_voxel_size = effective_voxel_size

    def voxel_world_centers(self, coords):
        xyz_idx = coords[:, 1:4].float()
        return self.pc_range[:3].to(coords.device) + (xyz_idx + 0.5) * self.effective_voxel_size.to(coords.device)

    def assign(self, pred, coords, gt_boxes_list):
        device = coords.device
        n = coords.shape[0]
        voxel_centers = self.voxel_world_centers(coords)
        pred_center = voxel_centers + pred["offset"]
        pred_size = pred["box"][:, :3].exp()
        pred_quat = pred["box"][:, 3:7]
        cls_score = torch.sigmoid(pred["center_logit"].squeeze(-1))

        pos_mask = torch.zeros(n, dtype=torch.bool, device=device)
        target_center = torch.zeros(n, 3, device=device)
        target_size = torch.zeros(n, 3, device=device)
        target_quat = torch.zeros(n, 4, device=device)
        target_quat[:, 0] = 1.0
        assigned_gt = torch.full((n,), -1, dtype=torch.long, device=device)

        for b, gt_boxes in enumerate(gt_boxes_list):
            if gt_boxes.shape[0] == 0:
                continue
            gt_boxes = gt_boxes.to(device)
            batch_idx = (coords[:, 0] == b).nonzero(as_tuple=True)[0]
            if batch_idx.numel() == 0:
                continue

            for oi in range(gt_boxes.shape[0]):
                gt = gt_boxes[oi]
                m = batch_idx.shape[0]
                gt_center = gt[:3].unsqueeze(0).expand(m, 3)
                gt_size = gt[3:6].unsqueeze(0).expand(m, 3)
                gt_quat = gt[6:10].unsqueeze(0).expand(m, 4)

                reg_cost = corner_distance_cost(
                    pred_center[batch_idx], pred_size[batch_idx], pred_quat[batch_idx],
                    gt_center, gt_size, gt_quat,
                )
                cls_cost = 1.0 - cls_score[batch_idx]
                cost = self.cls_weight * cls_cost + self.reg_weight * reg_cost

                quality = torch.exp(-reg_cost)
                q = min(TOPQ, quality.shape[0])
                topq_vals, _ = torch.topk(quality, q)
                dynamic_k = max(self.min_positives, int(topq_vals.sum().floor().item()))
                dynamic_k = min(dynamic_k, cost.shape[0])

                _, topk_local = torch.topk(cost, dynamic_k, largest=False)
                sel = batch_idx[topk_local]

                pos_mask[sel] = True
                target_center[sel] = gt[:3]
                target_size[sel] = gt[3:6]
                target_quat[sel] = gt[6:10]
                assigned_gt[sel] = oi

        return {
            "pos_mask": pos_mask,
            "target_center": target_center,
            "target_size": target_size,
            "target_quat": target_quat,
            "assigned_gt": assigned_gt,
            "pred_center": pred_center,
            "pred_size": pred_size,
            "pred_quat": pred_quat,
        }
