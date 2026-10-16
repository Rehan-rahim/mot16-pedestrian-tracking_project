"""
Faster R-CNN for pedestrian detection.

We start from torchvision's COCO-pretrained Faster R-CNN (ResNet-50 + FPN, v2),
swap its 91-class head for a 2-class head (background, pedestrian), freeze the
backbone and fine-tune the rest on MOT16. This is the "few-shot fine-tuning /
transfer learning" setup described in the project slides.
"""

import os
import time
import math

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2,
                                          FasterRCNN_ResNet50_FPN_V2_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from boxes import iou_matrix
from mot_data import read_seqinfo, frame_path

PERSON_LABEL = 1   # "person" in COCO and "pedestrian" in our 2-class model


# ---------------------------------------------------------------------------
# Building the models
# ---------------------------------------------------------------------------

def build_coco_detector():
    """Off-the-shelf COCO model, used as our 'before fine-tuning' baseline."""
    w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    return fasterrcnn_resnet50_fpn_v2(weights=w, box_detections_per_img=200)


def build_pedestrian_detector(freeze_backbone=True, pretrained=True):
    """COCO weights + new 2-class head. Backbone frozen as the slides suggest."""
    w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    # When loading our own checkpoint we don't need ImageNet weights either
    model = fasterrcnn_resnet50_fpn_v2(weights=w, weights_backbone=None if not pretrained else "DEFAULT",
                                       box_detections_per_img=200)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    return model


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def load_pedestrian_detector(ckpt_path, device):
    model = build_pedestrian_detector(pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own trusted file
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_detector(model, train_loader, device, epochs, ckpt_path,
                   lr=0.005, warmup_iters=300, val_fn=None, log_every=50):
    """
    Plain SGD fine-tuning with:
      * linear warm-up for the first few hundred steps (stops early loss spikes)
      * cosine learning-rate decay afterwards
      * mixed precision (fits comfortably on a free Colab T4)
      * a checkpoint after every epoch, so a Colab disconnect only costs one epoch

    Returns a history dict with per-epoch loss and (optional) validation AP.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    total_iters = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda it: (it + 1) / warmup_iters if it < warmup_iters
        else 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / max(1, total_iters - warmup_iters))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = {"epoch": [], "train_loss": [], "val_ap": []}
    start_epoch = 1

    # Resume if a checkpoint from an earlier (disconnected) session exists
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own trusted file
        if "optimizer" in state:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history = state["history"]
            start_epoch = state["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}")

    model.to(device)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        for i, (images, targets) in enumerate(train_loader, 1):
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if i % log_every == 0:
                print(f"  epoch {epoch} | step {i}/{len(train_loader)} | "
                      f"loss {running / i:.4f} | lr {optimizer.param_groups[0]['lr']:.5f}")

        epoch_loss = running / len(train_loader)
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        msg = f"Epoch {epoch}: train loss {epoch_loss:.4f} ({(time.time() - t0) / 60:.1f} min)"
        if val_fn is not None:
            ap = val_fn(model)
            history["val_ap"].append(ap)
            msg += f" | val AP@0.5 {ap:.3f}"
        print(msg)

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history,
                    "epoch": epoch}, ckpt_path)
    return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def detect(model, img_bgr, device, score_thresh=0.3):
    """Run the detector on one BGR frame. Returns (N, 5): x1, y1, x2, y2, score."""
    model.eval()
    t = TF.to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).to(device)
    with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
        out = model([t])[0]
    keep = (out["labels"] == PERSON_LABEL) & (out["scores"] >= score_thresh)
    boxes = out["boxes"][keep].float().cpu().numpy()
    scores = out["scores"][keep].float().cpu().numpy()
    return np.concatenate([boxes, scores[:, None]], axis=1)


def detect_sequence(model, seq_dir, device, score_thresh=0.3, frame_step=1, verbose=True):
    """
    Detect people in every frame of a sequence.
    Returns ({frame: dets}, frames_per_second).
    """
    info = read_seqinfo(seq_dir)
    frames = list(range(1, info["length"] + 1, frame_step))
    dets = {}
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for i, f in enumerate(frames):
        dets[f] = detect(model, cv2.imread(frame_path(seq_dir, info, f)), device, score_thresh)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {info['name']}: {i + 1}/{len(frames)} frames")
    if device.type == "cuda":
        torch.cuda.synchronize()
    fps = len(frames) / (time.time() - t0)
    return dets, fps


# ---------------------------------------------------------------------------
# Evaluation: Average Precision at IoU 0.5 (PASCAL VOC style)
# ---------------------------------------------------------------------------

def evaluate_detections(dets, gt, ignore=None, iou_thr=0.5, op_score=0.5):
    """
    dets:   {frame: (N, 5)} detector output
    gt:     {frame: (M, 5)} with columns id, x1, y1, x2, y2
    ignore: {frame: (K, 4)} "don't care" regions; a detection on one of them
            is simply skipped instead of being counted as a false positive

    Returns AP@0.5 plus precision / recall at a fixed operating score.
    """
    ignore = ignore or {}
    records = []          # (score, is_true_positive)
    n_gt = 0
    for f, d in dets.items():
        g = gt.get(f, np.zeros((0, 5)))[:, 1:5]
        n_gt += len(g)
        if len(d) == 0:
            continue
        order = np.argsort(-d[:, 4])
        d = d[order]
        iou = iou_matrix(d[:, :4], g)
        ign = ignore.get(f, np.zeros((0, 4)))
        iou_ign = iou_matrix(d[:, :4], ign)
        used = np.zeros(len(g), dtype=bool)
        for j in range(len(d)):
            if len(g):
                cand = np.where(~used)[0]
                if len(cand):
                    best = cand[np.argmax(iou[j, cand])]
                    if iou[j, best] >= iou_thr:
                        used[best] = True
                        records.append((d[j, 4], 1))
                        continue
            if len(ign) and iou_ign[j].max() >= iou_thr:
                continue          # sits on an ignore region, so it doesn't count
            records.append((d[j, 4], 0))

    if not records or n_gt == 0:
        return dict(AP=0.0, precision=0.0, recall=0.0, n_gt=n_gt, curve=(np.array([0.0]), np.array([0.0])))
    records.sort(key=lambda r: -r[0])
    scores = np.array([r[0] for r in records])
    tp = np.array([r[1] for r in records])
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1)

    # All-point interpolated AP
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[1], precision, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    at = scores >= op_score
    p_op = float(tp[at].sum() / max(at.sum(), 1))
    r_op = float(tp[at].sum() / n_gt)
    return dict(AP=round(ap, 4), precision=round(p_op, 4), recall=round(r_op, 4),
                n_gt=n_gt, curve=(recall, precision))
