import argparse
import math

import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import SonarDiverDataset, collate_fn
from models.detector import DiverDetector
from models.decode import decode_detections
from models.sparse_ops import build_index_grid
from models.box_utils import quat_geodesic_loss

MATCH_DIST_THRESHOLD_M = 1.0  # ~ smaller than the diver's own footprint


@torch.no_grad()
def evaluate(model, loader, device, score_threshold):
    model.eval()
    n_gt, n_det, n_matched = 0, 0, 0
    center_errors, rot_errors_deg = [], []

    for batch in loader:
        pred, stem_coords, gt_boxes_list = model.forward(batch, device)
        # stem grid_size is the stride-downsampled grid, not model.grid_size (that's pre-stem)
        stem_grid_size = tuple(int(stem_coords[:, i].max().item()) + 1 if stem_coords.shape[0] > 0 else 1 for i in (1, 2, 3))
        index_grid = build_index_grid(stem_coords, batch["batch_size"], stem_grid_size, device=device)
        eff_voxel_size = model.voxel_size * model.stem_stride

        dets = decode_detections(pred, stem_coords, index_grid, stem_grid_size, model.pc_range, eff_voxel_size, score_threshold)

        for b, gt_boxes in enumerate(gt_boxes_list):
            gt_boxes = gt_boxes.cpu()
            det = dets[b]
            n_gt += gt_boxes.shape[0]
            n_det += det["center"].shape[0]

            used_det = set()
            for oi in range(gt_boxes.shape[0]):
                gt_center = gt_boxes[oi, :3]
                if det["center"].shape[0] == 0:
                    continue
                dists = (det["center"] - gt_center[None, :]).norm(dim=-1)
                order = torch.argsort(dists)
                for di in order.tolist():
                    if di in used_det:
                        continue
                    if dists[di].item() > MATCH_DIST_THRESHOLD_M:
                        break
                    used_det.add(di)
                    n_matched += 1
                    center_errors.append(dists[di].item())
                    rd = quat_geodesic_loss(det["quat"][di:di + 1], gt_boxes[oi, 6:10].unsqueeze(0))
                    rot_errors_deg.append(math.degrees(rd.item()))
                    break

    recall = n_matched / n_gt if n_gt else float("nan")
    precision = n_matched / n_det if n_det else float("nan")
    mean_center_err = sum(center_errors) / len(center_errors) if center_errors else float("nan")
    mean_rot_err = sum(rot_errors_deg) / len(rot_errors_deg) if rot_errors_deg else float("nan")

    return {
        "n_gt": n_gt, "n_det": n_det, "n_matched": n_matched,
        "recall": recall, "precision": precision,
        "mean_center_error_m": mean_center_err, "mean_rotation_error_deg": mean_rot_err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = yaml.safe_load(open(args.config)) if args.config else ckpt["cfg"]

    model = DiverDetector(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    ds = SonarDiverDataset(cfg, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)

    print(f"evaluating {args.split} split ({len(ds)} frames) from {args.checkpoint} (epoch {ckpt.get('epoch')})")
    metrics = evaluate(model, loader, device, args.score_threshold)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
