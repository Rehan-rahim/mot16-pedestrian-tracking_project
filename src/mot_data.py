"""
Everything to do with reading MOT16: getting the zip, parsing gt.txt,
the PyTorch dataset for detector training, and cutting out person crops
for the Re-ID network.
"""

import os
import glob
import random
import shutil
import zipfile
import subprocess
import configparser
from collections import defaultdict

import cv2
import numpy as np

from config import PEDESTRIAN_CLASS, IGNORE_CLASSES, MOT16_URL, REID_H, REID_W


# ---------------------------------------------------------------------------
# Getting the data onto the Colab machine
# ---------------------------------------------------------------------------

def download_mot16(zip_path, url=MOT16_URL):
    """Download MOT16.zip to Drive once. Later sessions skip this."""
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1e9:
        print("MOT16.zip already on Drive, skipping download.")
        return
    print("Downloading MOT16 (~1.9 GB). This happens only once...")
    subprocess.run(["wget", "-q", "--show-progress", "-O", zip_path, url], check=True)


def unzip_to_local(zip_path, local_dir):
    """
    Unzip onto Colab's local disk. Reading 11k frames straight from Drive is
    painfully slow, while unzipping from Drive takes about a minute.
    """
    marker = os.path.join(local_dir, ".unzipped")
    if os.path.exists(marker):
        return local_dir
    os.makedirs(local_dir, exist_ok=True)
    print(f"Unzipping {os.path.basename(zip_path)} to {local_dir} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(local_dir)
    open(marker, "w").close()
    return local_dir


def find_mot16_root(local_dir):
    """The zip sometimes has an extra MOT16/ level inside. Find train/ wherever it is."""
    hits = glob.glob(os.path.join(local_dir, "**", "train"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"No 'train' folder found under {local_dir}")
    return os.path.dirname(hits[0])


def zip_existing_folder(mot16_folder, zip_path):
    """
    For teammates who already have MOT16 unzipped somewhere on Drive:
    pack it into a single zip so the rest of the pipeline works the same way.
    """
    base = zip_path[:-4]
    print("Zipping existing MOT16 folder, this takes a few minutes...")
    shutil.make_archive(base, "zip", root_dir=os.path.dirname(mot16_folder),
                        base_dir=os.path.basename(mot16_folder))


# ---------------------------------------------------------------------------
# Reading one sequence
# ---------------------------------------------------------------------------

def read_seqinfo(seq_dir):
    """seqinfo.ini tells us fps, resolution and number of frames."""
    cp = configparser.ConfigParser()
    cp.read(os.path.join(seq_dir, "seqinfo.ini"))
    s = cp["Sequence"]
    return dict(name=s["name"], fps=int(s["frameRate"]), length=int(s["seqLength"]),
                width=int(s["imWidth"]), height=int(s["imHeight"]),
                im_dir=s["imDir"], im_ext=s["imExt"])


def frame_path(seq_dir, info, frame):
    return os.path.join(seq_dir, info["im_dir"], f"{frame:06d}{info['im_ext']}")


def load_gt_raw(seq_dir):
    """
    Raw gt.txt as a numpy array. Columns:
    frame, id, left, top, width, height, flag, class, visibility
    (flag = 1 means "evaluate this box", 0 means ignore it)
    """
    return np.loadtxt(os.path.join(seq_dir, "gt", "gt.txt"), delimiter=",", ndmin=2)


def parse_gt(seq_dir, min_visibility=0.0):
    """
    Pedestrians we actually care about, as {frame: array([[id, x1, y1, x2, y2], ...])}.
    We convert to corner format (x1, y1, x2, y2) right away because that is
    what torchvision and our IoU code expect.
    """
    raw = load_gt_raw(seq_dir)
    keep = (raw[:, 7] == PEDESTRIAN_CLASS) & (raw[:, 6] == 1) & (raw[:, 8] >= min_visibility)
    out = defaultdict(list)
    for f, tid, x, y, w, h in raw[keep][:, :6]:
        out[int(f)].append([int(tid), x, y, x + w, y + h])
    return {f: np.array(v, dtype=float) for f, v in out.items()}


def parse_ignore_regions(seq_dir):
    """
    Boxes the official evaluation ignores (static people, reflections, ...).
    If our tracker puts a box on one of these, it should not count as a
    false positive, so the metrics code needs them.
    """
    raw = load_gt_raw(seq_dir)
    cls, flag = raw[:, 7], raw[:, 6]
    ign = np.isin(cls, list(IGNORE_CLASSES)) | ((cls == PEDESTRIAN_CLASS) & (flag == 0))
    out = defaultdict(list)
    for f, _, x, y, w, h in raw[ign][:, :6]:
        out[int(f)].append([x, y, x + w, y + h])
    return {f: np.array(v, dtype=float) for f, v in out.items()}


def sequence_stats(seq_dir):
    info = read_seqinfo(seq_dir)
    gt = parse_gt(seq_dir)
    ids = {int(i) for boxes in gt.values() for i in boxes[:, 0]}
    n_boxes = sum(len(v) for v in gt.values())
    return dict(sequence=info["name"], frames=info["length"],
                resolution=f"{info['width']}x{info['height']}", fps=info["fps"],
                identities=len(ids), boxes=n_boxes,
                boxes_per_frame=round(n_boxes / info["length"], 1))


# ---------------------------------------------------------------------------
# Detector training data (with augmentation, as the slides ask for)
# ---------------------------------------------------------------------------

def _lazy_torch():
    # torch is imported lazily so the parsing helpers above also work
    # in places where torch is not installed.
    import torch
    from torchvision import transforms as T
    import torchvision.transforms.functional as TF
    return torch, T, TF


class MOT16DetectionDataset:
    """
    One item = one video frame + all pedestrian boxes in it.

    `frame_step` lets us skip frames: neighbouring frames are almost identical,
    so taking every 2nd frame halves training time without losing much.

    Augmentation (training only):
      * horizontal flip (boxes are flipped too)
      * colour jitter and Gaussian blur, the exact transforms from the slides
    """

    def __init__(self, mot_root, sequences, frame_step=2, train=True,
                 min_visibility=0.1):
        torch, T, _ = _lazy_torch()
        self.train = train
        self.items = []
        for seq in sequences:
            seq_dir = os.path.join(mot_root, "train", seq)
            info = read_seqinfo(seq_dir)
            gt = parse_gt(seq_dir, min_visibility=min_visibility)
            for f in range(1, info["length"] + 1, frame_step):
                boxes = gt.get(f, np.zeros((0, 5)))[:, 1:5]
                self.items.append((frame_path(seq_dir, info, f), boxes, info))
        self.color_aug = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.3),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        torch, _, TF = _lazy_torch()
        path, boxes, info = self.items[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        image = TF.to_tensor(img)
        boxes = boxes.copy()

        # Keep boxes inside the image and drop degenerate ones
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            boxes = boxes[(boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)]

        if self.train:
            if random.random() < 0.5:
                image = image.flip(-1)
                if len(boxes):
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            image = self.color_aug(image)

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes_t,
            "labels": torch.ones((len(boxes_t),), dtype=torch.int64),   # 1 = pedestrian
            "image_id": torch.tensor(idx),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]),
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    """Detection models take a list of images (they can be different sizes)."""
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Re-ID crops: same person = same folder
# ---------------------------------------------------------------------------

def extract_reid_crops(mot_root, sequences, out_dir, frame_step=3,
                       min_visibility=0.5, min_height=50):
    """
    Cut every visible pedestrian out of the frames and save it as a small image.
    Folder name = "<sequence>_<id>", so two images in the same folder are the
    same person (a positive pair) and images from different folders are
    different people (a negative pair).

    We skip tiny and heavily occluded people, because a crop that is 70%
    someone else's back teaches the network the wrong thing.
    """
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for seq in sequences:
        seq_dir = os.path.join(mot_root, "train", seq)
        info = read_seqinfo(seq_dir)
        gt = parse_gt(seq_dir, min_visibility=min_visibility)
        n_seq = 0
        for f in range(1, info["length"] + 1, frame_step):
            if f not in gt:
                continue
            img = cv2.imread(frame_path(seq_dir, info, f))
            H, W = img.shape[:2]
            for tid, x1, y1, x2, y2 in gt[f]:
                if y2 - y1 < min_height:
                    continue
                x1, y1 = max(int(x1), 0), max(int(y1), 0)
                x2, y2 = min(int(x2), W), min(int(y2), H)
                if x2 - x1 < 10 or y2 - y1 < 20:
                    continue
                crop = cv2.resize(img[y1:y2, x1:x2], (REID_W, REID_H))
                pid_dir = os.path.join(out_dir, f"{seq}_{int(tid):04d}")
                os.makedirs(pid_dir, exist_ok=True)
                cv2.imwrite(os.path.join(pid_dir, f"{f:06d}.jpg"), crop)
                n_seq += 1
        total += n_seq
        print(f"  {seq}: {n_seq} crops")
    n_ids = len([d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))])
    print(f"Total: {total} crops of {n_ids} people -> {out_dir}")
    return total, n_ids


# ---------------------------------------------------------------------------
# One-call helpers used at the top of every notebook
# ---------------------------------------------------------------------------

def prepare_mot16(paths, existing_folder=None):
    """
    Make sure MOT16 is on this Colab machine and return its root folder.

    First time:  downloads MOT16.zip into <project>/data on Drive
                 (or zips a copy you already have, if `existing_folder` is set)
    Every time:  unzips it to /content/MOT16 (fast local disk)
    """
    if not os.path.exists(paths.mot16_zip):
        if existing_folder:
            zip_existing_folder(existing_folder, paths.mot16_zip)
        else:
            download_mot16(paths.mot16_zip)
    unzip_to_local(paths.mot16_zip, paths.local_mot16)
    root = find_mot16_root(paths.local_mot16)
    print("MOT16 ready at", root)
    return root


def prepare_reid_crops(paths):
    """Unzip the Re-ID crops made in notebook 01 to local disk."""
    if os.path.isdir(paths.reid_train) and os.path.isdir(paths.reid_val):
        return paths.local_reid                      # already unzipped this session
    if not os.path.exists(paths.reid_zip):
        raise FileNotFoundError("reid_crops.zip not found. Run notebook 01 first.")
    print("Unzipping Re-ID crops ...")
    with zipfile.ZipFile(paths.reid_zip) as z:
        z.extractall(os.path.dirname(paths.local_reid))
    return paths.local_reid
