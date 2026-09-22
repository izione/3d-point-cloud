"""Bipartite matching between a DETR-style decoder's fixed-size query set and
the (variable-size) GT box set for one sample -- the "which query is
responsible for which GT box" step that a set-prediction loss needs before it
can compute per-pair regression terms. Same scipy call already used elsewhere
in this project for a different purpose
(voxelnet_baseline/model/centerpoint_final/refinement_final_ver2/learned_data.py
:: match_detections), here with a classification+L1+rotation cost instead of
that file's IoU+distance cost.
"""
import torch
from scipy.optimize import linear_sum_assignment

from .rotation6d import sixd_to_matrix, matrix_geodesic_loss
from .box_utils import quat_to_rotmat


class HungarianMatcher:
    def __init__(self, cost_cls=1.0, cost_center=5.0, cost_size=2.0, cost_rot=2.0):
        self.cost_cls = cost_cls
        self.cost_center = cost_center
        self.cost_size = cost_size
        self.cost_rot = cost_rot

    @torch.no_grad()
    def match(self, pred, gt_boxes_list):
        """pred: dict of (B,Q,*) tensors -- 'exist_logit'(B,Q,1), 'center'(B,Q,3),
        'log_size'(B,Q,3), 'sixd'(B,Q,6). gt_boxes_list: list[B] of (Mb,10)
        [center(3),size(3),quat(4)]. Returns list[B] of (query_idx, gt_idx) index
        tensors, one pair per matched GT box (Mb pairs; unmatched queries are
        just the ones not appearing in query_idx).

        scipy's linear_sum_assignment is CPU-only, so each sample's cost matrix
        has to cross the device boundary via .cpu() -- and .cpu() is a real
        synchronization point (it blocks until every GPU op queued so far has
        finished), not just a data copy. Interleaving "compute cost on GPU,
        then immediately .cpu() it" per sample forces that stall B times in a
        row, with the GPU sitting idle between each pair while the CPU is
        blocked on the previous one. Queuing every sample's cost-matrix kernels
        first (this method's first loop, no .cpu() calls at all) lets the GPU
        run them back-to-back while Python is still building the list; the
        second loop's .cpu() calls then mostly just pick up already-finished
        results instead of blocking anew each time. Matters more here than a
        single call would suggest: detector_detr.py::loss() calls this once
        per decoder layer (6x with the auxiliary loss), so the naive pattern's
        stalls previously multiplied by both B and the layer count."""
        exist_prob = torch.sigmoid(pred["exist_logit"].squeeze(-1))  # (B,Q)
        pred_size = pred["log_size"].exp()
        pred_R = sixd_to_matrix(pred["sixd"])  # (B,Q,3,3)

        cost_matrices = []
        for b, gt_boxes in enumerate(gt_boxes_list):
            m = gt_boxes.shape[0]
            if m == 0:
                cost_matrices.append(None)
                continue

            gt_center, gt_size, gt_quat = gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6:10]
            gt_R = quat_to_rotmat(gt_quat)  # (M,3,3)

            q_center, q_size, q_R = pred["center"][b], pred_size[b], pred_R[b]  # (Q,*)

            # broadcast every query against every GT box -> (Q,M) cost matrix
            cls_cost = -exist_prob[b][:, None].expand(-1, m)
            center_cost = torch.cdist(q_center, gt_center, p=1)                              # (Q,M)
            size_cost = torch.cdist(torch.log(q_size.clamp(min=1e-3)), torch.log(gt_size.clamp(min=1e-3)), p=1)
            rot_cost = matrix_geodesic_loss(q_R[:, None, :, :].expand(-1, m, -1, -1),
                                             gt_R[None, :, :, :].expand(q_R.shape[0], -1, -1, -1))  # (Q,M)

            cost = (self.cost_cls * cls_cost + self.cost_center * center_cost +
                    self.cost_size * size_cost + self.cost_rot * rot_cost)
            # A destabilized/early-training forward pass can produce inf/nan
            # predictions (e.g. exploding log_size before the optimizer settles),
            # which otherwise makes scipy raise "cost matrix is infeasible" and
            # crash the whole run over a single bad step. Clamp instead -- worst
            # case this step's match is poor, not fatal.
            cost_matrices.append(torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6))

        matches = []
        for cost in cost_matrices:
            if cost is None:
                matches.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
                continue
            query_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())
            matches.append((torch.as_tensor(query_idx, dtype=torch.long),
                             torch.as_tensor(gt_idx, dtype=torch.long)))
        return matches
