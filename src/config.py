"""
Central place for every path and constant used in the project.

Every notebook calls `get_paths(PROJECT_DIR)` once at the top, so if the
folder ever moves (or a teammate keeps it somewhere else on Drive), only
that one line in the notebook has to change.
"""

import os
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------
# MOT16 gives us 7 training videos (with labels) and 7 test videos (no labels).
# We keep MOT16-09 completely out of training so we have an honest validation
# video to measure the detector, the Re-ID model and the tracker on.
ALL_TRAIN_SEQS = ["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09",
                  "MOT16-10", "MOT16-11", "MOT16-13"]
VAL_SEQ = "MOT16-09"
TRAIN_SEQS = [s for s in ALL_TRAIN_SEQS if s != VAL_SEQ]

# The test video we submit. MOT16-01 has a static camera and a moderate crowd,
# which makes it the most realistic choice for a clean single-target demo.
TEST_SEQ = "MOT16-01"

# ---------------------------------------------------------------------------
# MOT16 ground-truth conventions
# ---------------------------------------------------------------------------
PEDESTRIAN_CLASS = 1
# Classes the official benchmark treats as "don't care": a prediction that
# lands on one of these is neither rewarded nor punished.
# 2 = person on vehicle, 7 = static person, 8 = distractor, 12 = reflection
IGNORE_CLASSES = {2, 7, 8, 12}

MOT16_URL = "https://motchallenge.net/data/MOT16.zip"

# Re-ID crop size (height x width). People are tall and thin, so 2:1 is the
# standard choice in person re-identification papers.
REID_H, REID_W = 128, 64


def get_paths(project_dir, local_root="/content"):
    """Build every folder path from the project root and create the folders."""
    p = SimpleNamespace()
    p.project = project_dir
    p.src = os.path.join(project_dir, "src")

    # Big stuff that never goes to GitHub
    p.data = os.path.join(project_dir, "data")
    p.weights = os.path.join(project_dir, "weights")
    p.outputs = os.path.join(project_dir, "outputs")
    p.detections = os.path.join(p.outputs, "detections")
    p.tracks = os.path.join(p.outputs, "tracks")
    p.videos = os.path.join(p.outputs, "videos")

    # Small results that DO go to GitHub and into the report
    p.results = os.path.join(project_dir, "results")
    p.figures = os.path.join(p.results, "figures")
    p.tables = os.path.join(p.results, "tables")
    p.results_tracks = os.path.join(p.results, "tracks")

    # Zip files kept on Drive, unzipped to fast local Colab disk each session
    p.mot16_zip = os.path.join(p.data, "MOT16.zip")
    p.reid_zip = os.path.join(p.data, "reid_crops.zip")

    p.local_mot16 = os.path.join(local_root, "MOT16")
    p.local_reid = os.path.join(local_root, "reid_crops")
    p.reid_train = os.path.join(p.local_reid, "train")
    p.reid_val = os.path.join(p.local_reid, "val")

    # Model files
    p.detector_ckpt = os.path.join(p.weights, "detector_frcnn_mot16.pth")
    p.reid_ckpt = os.path.join(p.weights, "reid_resnet18_triplet.pth")

    for d in [p.data, p.weights, p.detections, p.tracks, p.videos,
              p.figures, p.tables, p.results_tracks]:
        os.makedirs(d, exist_ok=True)
    return p
