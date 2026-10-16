"""
Glue between the models and the tracker: run detection + Re-ID over a whole
sequence once, time each stage, and cache the result on Drive. After that the
tracker (and all the ablation experiments) run in seconds without touching
the GPU again.
"""

import os
import pickle
import time

import cv2
import torch

from detector import detect
from mot_data import read_seqinfo, frame_path
from reid import embed_crops


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def process_sequence(det_model, reid_model, seq_dir, device, score_thresh=0.3,
                     cache_path=None, verbose=True):
    """
    Returns {"dets": {frame: (N,5)}, "feats": {frame: (N,D)} or None,
             "timing": {...frames per second for each stage...}}
    """
    if cache_path and os.path.exists(cache_path):
        print("Loaded cached detections:", os.path.basename(cache_path))
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    info = read_seqinfo(seq_dir)
    dets, feats = {}, ({} if reid_model is not None else None)
    t_det = t_reid = 0.0
    for f in range(1, info["length"] + 1):
        img = cv2.imread(frame_path(seq_dir, info, f))
        _sync(device); t0 = time.time()
        dets[f] = detect(det_model, img, device, score_thresh)
        _sync(device); t1 = time.time()
        if reid_model is not None:
            feats[f] = embed_crops(reid_model, img, dets[f][:, :4], device)
        _sync(device); t2 = time.time()
        t_det += t1 - t0
        t_reid += t2 - t1
        if verbose and f % 100 == 0:
            print(f"  {info['name']}: {f}/{info['length']} frames")

    n = info["length"]
    timing = {"frames": n, "detector_fps": n / t_det,
              "reid_fps": (n / t_reid) if reid_model is not None else None,
              "boxes_per_frame": sum(len(d) for d in dets.values()) / n}
    out = {"dets": dets, "feats": feats, "timing": timing}
    if cache_path:
        with open(cache_path, "wb") as fh:
            pickle.dump(out, fh)
    return out
