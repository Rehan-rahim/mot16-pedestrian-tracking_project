"""
Person Re-Identification: a Siamese network trained with triplet loss.

"Siamese" just means the same network (shared weights) embeds both images,
and we compare the two embeddings. We don't classify identities, we only learn
a distance: same person -> close together, different people -> far apart.
That way the model works on people it has never seen (like everyone in the
test video) without any re-training, which is exactly the point made in the
project slides.

Backbone: ImageNet-pretrained ResNet-18 (much stronger than a 3-layer CNN for
the same compute), followed by a 128-d embedding layer.
Loss: batch-hard triplet loss (Hermans et al., 2017).
"""

import os
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from PIL import Image

from config import REID_H, REID_W

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def train_transform():
    """Augmentation so the model doesn't just memorise exact pixels."""
    return T.Compose([
        T.Resize((REID_H, REID_W)),
        T.RandomHorizontalFlip(),
        T.Pad(10),
        T.RandomCrop((REID_H, REID_W)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.5, value="random"),   # simulates partial occlusion
    ])


def eval_transform():
    return T.Compose([T.Resize((REID_H, REID_W)), T.ToTensor(), T.Normalize(MEAN, STD)])


class ReIDFolder(torch.utils.data.Dataset):
    """Reads the crop folders made in notebook 01: one folder = one person."""

    def __init__(self, root, transform, max_per_id=None, min_images=2):
        self.transform = transform
        self.samples = []                       # (path, label, frame)
        self.by_label = defaultdict(list)
        pids = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        label = 0
        for pid in pids:
            files = sorted(f for f in os.listdir(os.path.join(root, pid)) if f.endswith(".jpg"))
            if len(files) < min_images:
                continue
            if max_per_id and len(files) > max_per_id:
                keep = np.linspace(0, len(files) - 1, max_per_id).astype(int)
                files = [files[i] for i in keep]
            for fn in files:
                self.by_label[label].append(len(self.samples))
                self.samples.append((os.path.join(root, pid, fn), label, int(fn[:-4])))
            label += 1
        self.num_ids = label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, frame = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label


class PKSampler(torch.utils.data.Sampler):
    """
    Every batch = P people x K images each. Triplet loss needs this: without
    several images of the same person in a batch there are no positives.
    """

    def __init__(self, dataset, P=16, K=4, batches_per_epoch=200):
        self.ds, self.P, self.K, self.n = dataset, P, K, batches_per_epoch
        self.labels = [l for l, idx in dataset.by_label.items() if len(idx) >= K]

    def __len__(self):
        return self.n

    def __iter__(self):
        for _ in range(self.n):
            batch = []
            for l in random.sample(self.labels, self.P):
                batch += random.sample(self.ds.by_label[l], self.K)
            yield batch


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SiameseReID(nn.Module):
    """
    ResNet-18 backbone -> 128-d embedding -> L2 normalised.

    One small trick from the Re-ID literature: we set the stride of the last
    ResNet stage to 1. Our crops are only 128x64, and without this the final
    feature map would be a tiny 4x2 grid, so we keep a bit more detail (8x4).
    """

    def __init__(self, emb_dim=128, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.resnet18(weights=w)
        backbone.layer4[0].conv1.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])   # up to avgpool
        self.embed = nn.Sequential(nn.Linear(512, emb_dim), nn.BatchNorm1d(emb_dim))

    def forward_once(self, x):
        return F.normalize(self.embed(self.features(x).flatten(1)), dim=1)

    def forward(self, x1, x2=None):
        # Siamese interface: both inputs go through the *same* weights
        if x2 is None:
            return self.forward_once(x1)
        return self.forward_once(x1), self.forward_once(x2)


class ImageNetBaseline(nn.Module):
    """Untrained baseline: plain ImageNet ResNet-18 features, no Re-ID training."""

    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward_once(self, x):
        return F.normalize(self.features(x).flatten(1), dim=1)

    forward = forward_once


def load_reid(ckpt_path, device):
    model = SiameseReID(pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own trusted file
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def batch_hard_triplet_loss(emb, labels, margin=0.3):
    """
    For every image (anchor) in the batch:
      hardest positive = same person, but the one that is furthest away
      hardest negative = different person, but the one that is closest
    loss = max(0, d(anchor, hardest_pos) - d(anchor, hardest_neg) + margin)

    Using the hardest examples (instead of random triplets) makes training
    focus on the confusing cases, e.g. two people both wearing dark jackets.
    """
    dist = torch.cdist(emb, emb)
    same = labels[:, None] == labels[None, :]
    hardest_pos = (dist * same.float()).max(dim=1).values
    hardest_neg = (dist + same.float() * 1e4).min(dim=1).values
    loss = F.relu(hardest_pos - hardest_neg + margin).mean()
    frac_active = (hardest_pos - hardest_neg + margin > 0).float().mean()
    return loss, frac_active


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_reid(model, loader, device, epochs, ckpt_path, lr=3e-4, margin=0.3,
               val_fn=None, val_every=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"epoch": [], "loss": [], "active": [], "val": []}
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = tot_act = 0.0
        t0 = time.time()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            emb = model(imgs)
            loss, active = batch_hard_triplet_loss(emb, labels, margin)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            tot_act += active.item()
        scheduler.step()
        n = len(loader)
        history["epoch"].append(epoch)
        history["loss"].append(tot_loss / n)
        history["active"].append(tot_act / n)
        msg = (f"Epoch {epoch:2d} | triplet loss {tot_loss / n:.4f} | "
               f"active triplets {100 * tot_act / n:.0f}% | {time.time() - t0:.0f}s")
        if val_fn is not None and (epoch % val_every == 0 or epoch == epochs):
            res = val_fn(model)
            # keep only the scalar scores in the history (no big arrays inside the checkpoint)
            history["val"].append((epoch, {k: v for k, v in res.items() if not hasattr(v, "shape")}))
            msg += f" | val rank-1 {res['rank1']:.3f} mAP {res['mAP']:.3f} AUC {res['AUC']:.3f}"
        print(msg)
    torch.save({"model": model.state_dict(), "history": history}, ckpt_path)
    return history


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

_MEAN = np.array(MEAN, dtype=np.float32)
_STD = np.array(STD, dtype=np.float32)


@torch.no_grad()
def embed_crops(model, frame_bgr, boxes, device, batch_size=128):
    """
    Cut the given boxes out of a frame and embed them in one batch.
    Returns an (N, D) float32 numpy array (rows are L2-normalised).
    """
    model.eval()
    if len(boxes) == 0:
        return np.zeros((0, 128), dtype=np.float32)
    H, W = frame_bgr.shape[:2]
    crops = []
    for x1, y1, x2, y2 in np.asarray(boxes)[:, :4]:
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(W, max(x2, x1 + 2))), int(min(H, max(y2, y1 + 2)))
        crop = cv2.resize(frame_bgr[y1:y2, x1:x2], (REID_W, REID_H))
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    x = (np.stack(crops).astype(np.float32) / 255.0 - _MEAN) / _STD
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
    out = []
    for i in range(0, len(x), batch_size):
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            out.append(model.forward_once(x[i:i + batch_size]).float())
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def embed_dataset(model, dataset, device, batch_size=256):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)
    feats = []
    for imgs, _ in loader:
        feats.append(model.forward_once(imgs.to(device)).float().cpu())
    feats = torch.cat(feats).numpy()
    labels = np.array([s[1] for s in dataset.samples])
    frames = np.array([s[2] for s in dataset.samples])
    return feats, labels, frames


# ---------------------------------------------------------------------------
# Evaluation on people the model has never seen (MOT16-09)
# ---------------------------------------------------------------------------

def _auc(pos, neg):
    """ROC-AUC: chance that a random negative pair is further apart than a random positive pair."""
    scores = np.concatenate([-pos, -neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_reid(feats, labels, frames, min_gap=15, n_pairs=5000, seed=0):
    """
    Two views of the same question, "can the embedding tell people apart?":

    1. Retrieval (rank-1, mAP): every crop is a query, all other crops are the
       gallery. Crops of the same person within `min_gap` frames are left out,
       because they are near-identical and would make the task too easy.
    2. Verification (AUC, best accuracy): distances of same-person vs
       different-person pairs, the exact decision the tracker has to make.
    """
    rng = np.random.default_rng(seed)
    dist = np.sqrt(np.clip(2 - 2 * feats @ feats.T, 0, None))

    rank1, aps = [], []
    for q in range(len(feats)):
        junk = (labels == labels[q]) & (np.abs(frames - frames[q]) < min_gap)
        valid = ~junk
        match = (labels == labels[q]) & valid
        if match.sum() == 0:
            continue
        order = np.argsort(dist[q][valid])
        hits = match[valid][order]
        rank1.append(hits[0])
        cum = np.cumsum(hits)
        aps.append(np.mean(cum[hits] / (np.where(hits)[0] + 1)))

    pos, neg = [], []
    idx_by = defaultdict(list)
    for i, l in enumerate(labels):
        idx_by[l].append(i)
    multi = [l for l, v in idx_by.items() if len(v) >= 2]
    all_l = list(idx_by)
    for _ in range(n_pairs):
        l = multi[rng.integers(len(multi))]
        i, j = rng.choice(idx_by[l], 2, replace=False)
        pos.append(dist[i, j])
        a, b = rng.choice(all_l, 2, replace=False)
        neg.append(dist[rng.choice(idx_by[a]), rng.choice(idx_by[b])])
    pos, neg = np.array(pos), np.array(neg)
    ths = np.linspace(0, 2, 201)
    accs = [((pos < t).mean() + (neg >= t).mean()) / 2 for t in ths]
    best = int(np.argmax(accs))
    if not rank1:   # happens only if every person is visible for less than `min_gap` frames
        print("Warning: no valid retrieval queries; rank-1/mAP undefined")
    return dict(rank1=float(np.mean(rank1)) if rank1 else float("nan"),
                mAP=float(np.mean(aps)) if aps else float("nan"), AUC=_auc(pos, neg),
                best_acc=float(accs[best]), best_threshold=float(ths[best]),
                pos_dist=pos, neg_dist=neg)
