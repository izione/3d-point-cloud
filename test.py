import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

import models.slotformer as slotformer
from data.dataset import SonarDiverDataset, collate_fn
from models.detector import DiverDetector
from models.decode import decode_detections
from models.sparse_ops import build_index_grid
from models.box_utils import quat_geodesic_loss, axis_aligned_iou_3d

MATCH_DIST_THRESHOLD_M = 1.0  # ~ smaller than the diver's own footprint
IOU_THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9


def iou_matrix(det, gt_boxes):
    """det: dict with 'center'/'size' (Nd,3)/(Nd,3). gt_boxes: (Ng,10) =
    [center(3), size(3), quat(4)] (quat unused -- volume-only IoU). Returns (Nd,Ng)."""
    nd, ng = det["center"].shape[0], gt_boxes.shape[0]
    ious = torch.zeros(nd, ng)
    for di in range(nd):
        for gi in range(ng):
            ious[di, gi] = axis_aligned_iou_3d(det["center"][di], det["size"][di], gt_boxes[gi, :3], gt_boxes[gi, 3:6])
    return ious


def greedy_iou_matches(ious, scores, threshold):
    """Score-ranked greedy 1:1 matching (standard VOC/COCO-style det eval):
    take detections highest-score-first, match each to its best still-free GT
    if that GT's IoU clears `threshold`. Returns the number of matches (TPs)."""
    nd, ng = ious.shape
    if nd == 0 or ng == 0:
        return 0
    order = torch.argsort(scores, descending=True)
    gt_taken = torch.zeros(ng, dtype=torch.bool)
    n_matched = 0
    for di in order.tolist():
        row = ious[di].clone()
        row[gt_taken] = -1.0
        best_gi = torch.argmax(row).item()
        if row[best_gi].item() >= threshold:
            gt_taken[best_gi] = True
            n_matched += 1
    return n_matched


@torch.no_grad()
def collect_pr_data(model, loader, device):
    """Runs the whole split once with score_threshold=0 (every local-max peak kept,
    not just the ones above some cutoff) and precomputes each frame's full det x gt
    axis-aligned IoU matrix. This is the raw material a real PR curve needs: ranking
    ALL candidates by score and sweeping the confidence cutoff, instead of fixing one
    cutoff upfront the way evaluate()'s IoU sweep does. Returns (frame_data, total_gt)
    where frame_data is a list[{"scores": (Nd,), "ious": (Nd,Ng)}]."""
    model.eval()
    frame_data = []
    total_gt = 0

    for batch in loader:
        pred, stem_coords, gt_boxes_list = model.forward(batch, device)
        stem_grid_size = tuple(int(stem_coords[:, i].max().item()) + 1 if stem_coords.shape[0] > 0 else 1 for i in (1, 2, 3))
        index_grid = build_index_grid(stem_coords, batch["batch_size"], stem_grid_size, device=device)
        eff_voxel_size = model.voxel_size * model.stem_stride

        dets = decode_detections(pred, stem_coords, index_grid, stem_grid_size, model.pc_range, eff_voxel_size, 0.0)

        for b, gt_boxes in enumerate(gt_boxes_list):
            gt_boxes = gt_boxes.cpu()
            det = dets[b]
            total_gt += gt_boxes.shape[0]
            frame_data.append({"scores": det["score"], "ious": iou_matrix(det, gt_boxes)})

    return frame_data, total_gt


def compute_ap(recalls, precisions):
    """VOC2012-style all-point interpolated average precision: replace precision at
    each recall with the max precision at any recall >= it (monotonic envelope), then
    integrate. recalls must be non-decreasing (true for a score-descending PR curve)."""
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [1.0] + list(precisions) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def pr_curve_for_threshold(frame_data, total_gt, iou_threshold):
    """Pools every detection across the whole split into one score-ranked list (the
    standard VOC/COCO AP procedure) and walks it high-score-first, greedily claiming
    each detection's best still-free GT *within its own frame* if that pair's IoU
    clears iou_threshold. Returns (recalls, precisions, ap)."""
    pool = []  # (score, frame_index, det_index)
    for fi, fd in enumerate(frame_data):
        for di, s in enumerate(fd["scores"].tolist()):
            pool.append((s, fi, di))
    pool.sort(key=lambda x: x[0], reverse=True)

    gt_taken = [torch.zeros(fd["ious"].shape[1], dtype=torch.bool) for fd in frame_data]
    recalls, precisions = [], []
    tp, fp = 0, 0
    for _, fi, di in pool:
        ious = frame_data[fi]["ious"][di]
        taken = gt_taken[fi]
        row = ious.clone()
        row[taken] = -1.0
        if row.numel() > 0:
            best_gi = torch.argmax(row).item()
            if row[best_gi].item() >= iou_threshold:
                taken[best_gi] = True
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
        recalls.append(tp / total_gt if total_gt else 0.0)
        precisions.append(tp / (tp + fp))

    ap = compute_ap(recalls, precisions) if recalls else 0.0
    return recalls, precisions, ap


def plot_pr_curves(frame_data, total_gt, out_path):
    """One line per IoU threshold (blue, light->dark with threshold), recall on x,
    precision on y, AP in the legend."""
    import matplotlib.pyplot as plt

    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b"]
    fig, ax = plt.subplots(figsize=(7, 6.2), facecolor="#f5f8f9")
    ax.set_facecolor("#ffffff")
    for t, color in zip(IOU_THRESHOLDS, ramp):
        recalls, precisions, ap = pr_curve_for_threshold(frame_data, total_gt, t)
        ax.plot(recalls, precisions, color=color, linewidth=2, label=f"IoU {t:.1f}  (AP {ap:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve by IoU threshold")
    ax.grid(color="#dde5e7", linewidth=1)
    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name in ("left", "bottom"))
        spine.set_color("#c4ced0")
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.32, 1.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="#f5f8f9", bbox_inches="tight")


def _new_accumulator():
    return {
        "n_gt": 0, "n_det": 0, "n_matched": 0,
        "center_errors": [], "rot_errors_deg": [],
        "n_matched_iou": {t: 0 for t in IOU_THRESHOLDS},
    }


def _finalize(acc):
    n_gt, n_det, n_matched = acc["n_gt"], acc["n_det"], acc["n_matched"]
    center_errors, rot_errors_deg = acc["center_errors"], acc["rot_errors_deg"]
    n_matched_iou = acc["n_matched_iou"]

    iou_pr = {
        t: {
            "precision": n_matched_iou[t] / n_det if n_det else float("nan"),
            "recall": n_matched_iou[t] / n_gt if n_gt else float("nan"),
            "n_matched": n_matched_iou[t],
        }
        for t in IOU_THRESHOLDS
    }

    return {
        "n_gt": n_gt, "n_det": n_det, "n_matched": n_matched,
        "recall": n_matched / n_gt if n_gt else float("nan"),
        "precision": n_matched / n_det if n_det else float("nan"),
        "mean_center_error_m": sum(center_errors) / len(center_errors) if center_errors else float("nan"),
        "mean_rotation_error_deg": sum(rot_errors_deg) / len(rot_errors_deg) if rot_errors_deg else float("nan"),
        "iou_pr": iou_pr,
    }


def _frame_score(n_gt, n_det, n_matched, matches):
    """Per-frame quality summary. `matches` is a list[dict], one entry per matched
    GT-pred pair (see the per-object fields below) -- IoU/errors are object-level,
    not averaged into a single frame number; mean_iou is kept only as a cheap sort key.
    F1 treats an empty (no-GT, no-det) frame as a perfect 1.0 -- there was nothing to
    find and nothing was hallucinated."""
    if n_gt == 0 and n_det == 0:
        precision = recall = f1 = 1.0
    else:
        precision = n_matched / n_det if n_det else 0.0
        recall = n_matched / n_gt if n_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ious = [m["iou"] for m in matches]
    return {
        "n_gt": n_gt, "n_det": n_det, "n_matched": n_matched,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_iou": sum(ious) / len(ious) if ious else None,
        "matches": matches,
    }


@torch.no_grad()
def evaluate(model, loader, device, score_threshold):
    """Returns (overall_metrics, per_person_metrics, per_frame). per_person groups
    by the top-level folder in each sample's frame_id (e.g. "Person1", "Person2").
    per_frame is a list[dict], one entry per frame, sorted worst-to-best by F1."""
    model.eval()
    overall = _new_accumulator()
    per_person = defaultdict(_new_accumulator)
    per_frame = []

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
                    rd_deg = math.degrees(quat_geodesic_loss(det["quat"][di:di + 1], gt_boxes[oi, 6:10].unsqueeze(0)).item())
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

            # -- volume-only (axis-aligned) IoU matching, swept over IOU_THRESHOLDS --
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


def print_metrics(name, metrics):
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k == "iou_pr":
            continue
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("  volume-only IoU sweep (score-ranked greedy matching):")
    print(f"  {'IoU>=':>6} {'precision':>10} {'recall':>10} {'matched':>8}")
    for t, m in metrics["iou_pr"].items():
        print(f"  {t:>6.1f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['n_matched']:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--per_frame_out", default=None,
                         help="if set, writes a JSON array of per-frame scores (sorted worst-to-best by F1) to this path")
    parser.add_argument("--pr_curve_out", default=None,
                         help="if set, writes a PR-curve-per-IoU-threshold PNG to this path (score-ranked over ALL "
                              "detections, not just the ones above --score_threshold -- a separate full pass)")
    parser.add_argument("--attention_kind", choices=["softmax", "linear"], default=None,
                         help="override models.slotformer.ATTENTION_KIND (default: whatever the checkpoint was trained with)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = yaml.safe_load(open(args.config)) if args.config else ckpt["cfg"]

    if args.attention_kind:
        slotformer.ATTENTION_KIND = args.attention_kind
    elif ckpt.get("attention_kind"):
        slotformer.ATTENTION_KIND = ckpt["attention_kind"]  # match how this checkpoint was actually trained
    print(f"SlotFormer attention kind: {slotformer.ATTENTION_KIND}")

    model = DiverDetector(cfg).to(device)
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
            "meta": {
                "checkpoint": args.checkpoint, "split": args.split,
                "score_threshold": args.score_threshold, "match_dist_threshold_m": MATCH_DIST_THRESHOLD_M,
            },
            "frames": per_frame,
        }
        with open(args.per_frame_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {len(per_frame)} per-frame records to {args.per_frame_out}")

    if args.pr_curve_out:
        print("\ncollecting all detections (score_threshold=0) for the PR curve -- a separate full pass...")
        frame_data, total_gt = collect_pr_data(model, loader, device)
        plot_pr_curves(frame_data, total_gt, args.pr_curve_out)
        print(f"wrote PR curve to {args.pr_curve_out}")


if __name__ == "__main__":
    main()
