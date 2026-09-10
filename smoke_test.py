"""smoke_test.py - shape/pipeline sanity check, no real training. Verifies:
dataset loading -> collate -> model forward -> target building -> loss ->
backward -> decode -> eval, all run without shape errors, on a handful of
real frames. Run this FIRST after cloning on a new Colab instance, before
committing to a full `python train.py` run.
"""
import torch
from torch.utils.data import DataLoader

import config
from center_loss import center_voxelnet_loss
from dataset import VoxelDataset, collate_fn, zyaw_flatten
import heatmap_targets as ht
from model import CenterPointVoxelNet
from eval import iou_3d_obb, gt_obb_from_row, pred_obb_from_box
import numpy as np


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = VoxelDataset("train")
    train_ds.samples = train_ds.samples[:4]
    loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"batch: voxel_features={tuple(batch['voxel_features'].shape)} "
          f"coords={tuple(batch['coords'].shape)} batch_size={batch['batch_size']}")

    model = CenterPointVoxelNet().to(device)
    pred = model(batch["voxel_features"].to(device), batch["num_points"].to(device),
                 batch["coords"].to(device))
    for k, v in pred.items():
        print(f"pred[{k}]: {tuple(v.shape)}")

    hm, mask, off, z, dim, rot = [], [], [], [], [], []
    for objects in batch["gt_objects"]:
        t = ht.build_heatmap_targets(zyaw_flatten(objects))
        hm.append(t["heatmap"]); mask.append(t["reg_mask"]); off.append(t["offset"])
        z.append(t["z"]); dim.append(t["dim"]); rot.append(t["rot"])
    target = {
        "heatmap": torch.from_numpy(np.stack(hm)).to(device),
        "reg_mask": torch.from_numpy(np.stack(mask)).to(device),
        "offset": torch.from_numpy(np.stack(off)).to(device),
        "z": torch.from_numpy(np.stack(z)).to(device),
        "dim": torch.from_numpy(np.stack(dim)).to(device),
        "rot": torch.from_numpy(np.stack(rot)).to(device),
    }
    loss, stats = center_voxelnet_loss(
        pred["heatmap"], pred["offset"], pred["z"], pred["dim"], pred["rot"],
        target["heatmap"], target["reg_mask"], target["offset"], target["z"], target["dim"], target["rot"])
    loss.backward()
    print(f"loss={loss.item():.4f} stats={stats}")

    hm_np = torch.sigmoid(pred["heatmap"][0]).detach().cpu().numpy()
    boxes = ht.decode_center_boxes(
        hm_np, pred["offset"][0].permute(1, 2, 0).detach().cpu().numpy(),
        pred["z"][0].permute(1, 2, 0).detach().cpu().numpy(),
        pred["dim"][0].permute(1, 2, 0).detach().cpu().numpy(),
        pred["rot"][0].permute(1, 2, 0).detach().cpu().numpy(), score_thresh=0.0)
    print(f"decoded {len(boxes)} candidate boxes (score_thresh=0.0, untrained model)")

    gt_boxes = batch["gt_boxes"][0].numpy()
    if len(boxes) and len(gt_boxes):
        pc, pd, pR = pred_obb_from_box(boxes[0])
        gc, gd, gR = gt_obb_from_row(gt_boxes[0])
        iou = iou_3d_obb(pc, pd, pR, gc, gd, gR, rng=np.random.default_rng(0))
        print(f"sample 3D IoU (untrained, expect ~0): {iou:.4f}")

    print("smoke test OK")


if __name__ == "__main__":
    main()
