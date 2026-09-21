"""Sibling to test.py for the DETR-head variant (models/detector_detr.py).
Reuses every rotation-agnostic helper from test.py as-is (iou_matrix is
volume-only/axis-aligned, so it doesn't care whether the detector represents
rotation as a quaternion or 6D) -- only the per-batch decode + rotation-error
computation differ, since DiverDetectorDETR.decode() returns 'rot_matrix'
(a 3x3 matrix) instead of 'quat'.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import models.slotformer as slotformer
from config_utils import load_config
from data.dataset import SonarDiverDataset, collate_fn
from models.detector_detr import DiverDetectorDETR
from models.rotation6d import matrix_geodesic_loss
from models.box_utils import quat_to_rotmat, axis_aligned_iou_3d
from test import (
    MATCH_DIST_THRESHOLD_M, IOU_THRESHOLDS, iou_matrix, greedy_iou_matches,
    _new_accumulator, _finalize, _frame_score, print_metrics, plot_pr_curves,
)


@torch.no_grad()
def collect_pr_data(model, loader, device):
    model.eval()
    frame_data = []
    total_gt = 0
    for batch in loader:
        pred, gt_boxes_list, _ = model.forward(batch, device)
        dets = model.decode(pred, score_threshold=0.0)
        for b, gt_boxes in enumerate(gt_boxes_list):
            gt_boxes = gt_boxes.cpu()
            det = dets[b]
            total_gt += gt_boxes.shape[0]
            frame_data.append({"scores": det["score"], "ious": iou_matrix(det, gt_boxes)})
    return frame_data, total_gt


@torch.no_grad()
def evaluate(model, loader, device, score_threshold):
    model.eval()
    overall = _new_accumulator()
    per_person = defaultdict(_new_accumulator)
    per_frame = []

    for batch in loader:
        pred, gt_boxes_list, _ = model.forward(batch, device)
        dets = model.decode(pred, score_threshold=score_threshold)

        for b, gt_boxes in enumerate(gt_boxes_list):
            gt_boxes = gt_boxes.cpu()
            det = dets[b]
            frame_id = batch["frame_ids"][b]
            person = Path(frame_id).parts[0]
            accs = (overall, per_person[person])

            for acc in accs:
                acc["n_gt"] += gt_boxes.shape[0]
                acc["n_det"] += det["center"].shape[0]

            frame_matches = []
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
                    gt_R = quat_to_rotmat(gt_boxes[oi, 6:10].unsqueeze(0))
                    rd_deg = math.degrees(matrix_geodesic_loss(det["rot_matrix"][di:di + 1], gt_R).item())
                    iou = axis_aligned_iou_3d(det["center"][di], det["size"][di], gt_boxes[oi, :3], gt_boxes[oi, 3:6])
                    for acc in accs:
                        acc["n_matched"] += 1
                        acc["center_errors"].append(dists[di].item())
                        acc["rot_errors_deg"].append(rd_deg)
                    frame_matches.append({
                        "gt_index": oi, "det_index": di,
                        "center_error_m": dists[di].item(), "rotation_error_deg": rd_deg, "iou": iou,
                    })
                    break

            frame_record = _frame_score(gt_boxes.shape[0], det["center"].shape[0], len(frame_matches), frame_matches)
            frame_record["frame_id"] = frame_id
            frame_record["person"] = person
            per_frame.append(frame_record)

            if det["center"].shape[0] > 0 and gt_boxes.shape[0] > 0:
                ious = iou_matrix(det, gt_boxes)
                for t in IOU_THRESHOLDS:
                    n = greedy_iou_matches(ious, det["score"], t)
                    for acc in accs:
                        acc["n_matched_iou"][t] += n

    overall_metrics = _finalize(overall)
    per_person_metrics = {person: _finalize(acc) for person, acc in sorted(per_person.items())}
    per_frame.sort(key=lambda r: r["f1"])
    return overall_metrics, per_person_metrics, per_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--per_frame_out", default=None)
    parser.add_argument("--pr_curve_out", default=None)
    parser.add_argument("--attention_kind", choices=["softmax", "linear"], default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = load_config(args.config) if args.config else ckpt["cfg"]

    if args.attention_kind:
        slotformer.ATTENTION_KIND = args.attention_kind
    elif ckpt.get("attention_kind"):
        slotformer.ATTENTION_KIND = ckpt["attention_kind"]
    print(f"SlotFormer attention kind: {slotformer.ATTENTION_KIND}")

    model = DiverDetectorDETR(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    ds = SonarDiverDataset(cfg, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)

    print(f"evaluating {args.split} split ({len(ds)} frames) from {args.checkpoint} (epoch {ckpt.get('epoch')})")
    overall_metrics, per_person_metrics, per_frame = evaluate(model, loader, device, args.score_threshold)

    print_metrics("overall", overall_metrics)
    for person, metrics in per_person_metrics.items():
        print_metrics(person, metrics)

    print("\n=== worst 5 frames (by F1) ===")
    for r in per_frame[:5]:
        print(f"  f1={r['f1']:.3f}  gt={r['n_gt']} det={r['n_det']} matched={r['n_matched']}  {r['frame_id']}")
    print("=== best 5 frames (by F1) ===")
    for r in per_frame[-5:][::-1]:
        print(f"  f1={r['f1']:.3f}  gt={r['n_gt']} det={r['n_det']} matched={r['n_matched']}  {r['frame_id']}")

    if args.per_frame_out:
        payload = {
            "meta": {"checkpoint": args.checkpoint, "split": args.split,
                      "score_threshold": args.score_threshold, "match_dist_threshold_m": MATCH_DIST_THRESHOLD_M},
            "frames": per_frame,
        }
        with open(args.per_frame_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {len(per_frame)} per-frame records to {args.per_frame_out}")

    if args.pr_curve_out:
        print("\ncollecting all detections (score_threshold=0) for the PR curve...")
        frame_data, total_gt = collect_pr_data(model, loader, device)
        plot_pr_curves(frame_data, total_gt, args.pr_curve_out)
        print(f"wrote PR curve to {args.pr_curve_out}")


if __name__ == "__main__":
    main()
