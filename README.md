# Pedestrian Multi-Object Tracking on MOT16

Course project for **COMP-SCI 5567 Deep Learning**, University of Missouri–Kansas City (Fall 2026).
**Team:** Rehan Ali · Bosia N’dri · Tony Nguyen

We track pedestrians through crowded street videos so that every person keeps **one consistent ID** from the moment they appear until they leave, even when someone walks in front of them.

**Result videos:** see the [`outputs/videos/`](outputs/videos) folder
- `MOT16-01_target.mp4`: our target pedestrian, tracked in all 361 frames he is visible with a single ID
- `MOT16-01_all_tracks.mp4`: every tracked person in the test video
- `MOT16-09_tracking.mp4`: the held-out validation video

## How it works

```
frame -> Faster R-CNN -> boxes -> Siamese Re-ID -> embeddings -> Tracker -> boxes + IDs
         (fine-tuned)            (ResNet-18,                    (Kalman filter, Hungarian
                                  triplet loss)                  matching, motion gating,
                                                                 offline rules)
```

1. **Detection.** torchvision Faster R-CNN (ResNet-50 FPN v2), COCO-pretrained, with a new 2-class head (background, pedestrian). The backbone is frozen; the RPN and heads are fine-tuned on MOT16 with flip, colour-jitter and blur augmentation.
2. **Re-identification.** A Siamese network (shared ImageNet-pretrained ResNet-18, 128-d embedding) trained with batch-hard triplet loss on 20,146 person crops of 362 people cut from the MOT16 ground truth. It learns a distance rather than fixed identities, so it works on people it has never seen.
3. **Tracking.** A Kalman filter predicts each person's next position. Detections are assigned to tracks with the Hungarian algorithm on a cost that mixes appearance distance and box overlap, restricted to nearby, similar-looking pairs. Offline rules end a track when the person leaves the picture and fill detector drop-outs of up to 5 frames.

## Results

All numbers are on **MOT16-09**, which none of the models saw during training.

| Component | Metric | Baseline | Ours |
|---|---|---|---|
| Detector | AP@0.5 | 0.741 (COCO pretrained) | **0.780** |
| Detector | Precision / Recall @0.5 | 0.664 / 0.751 | **0.743 / 0.769** |
| Re-ID | Rank-1 on unseen people | 0.812 (ImageNet features) | **0.914** |
| Re-ID | mAP on unseen people | 0.490 | **0.728** |
| Tracker | IDF1 | 47.7 (IoU only) | **60.3** |
| Tracker | ID switches | 71 | **35** |

- **Target pedestrian (MOT16-01 test video):** one ID in all 361 frames he is visible.
- **Speed:** 8.6 FPS end to end on a free Google Colab T4 GPU (the detector takes 89% of the time).
- The tracker's appearance threshold was chosen on MOT16-09 itself, so the tracking numbers are slightly optimistic.

Full tables and figures: [`results/`](results).

## Repository layout

```
notebooks/
  01_Setup_Data.ipynb         dataset, statistics, augmentation, Re-ID crops
  02_Detector_Finetune.ipynb  Faster R-CNN fine-tuning, COCO vs fine-tuned
  03_ReID_Siamese.ipynb       Siamese network + triplet loss, evaluated on unseen people
  04_Tracking_Final.ipynb     full pipeline, final videos, speed
  05_Evaluation.ipynb         ablation, parameter study, per-person analysis, LaTeX tables
src/
  config.py     paths, dataset split, constants
  mot_data.py   MOT16 parsing, detection dataset with augmentation, crop extraction
  detector.py   Faster R-CNN build / train / inference / AP evaluation
  reid.py       Siamese ResNet-18, P×K sampler, batch-hard triplet loss, Re-ID metrics
  tracker.py    Kalman filter, Hungarian matching, gating, offline rules, MOT file I/O
  metrics.py    MOTA, IDF1, MT/ML (MOTChallenge protocol with ignore regions)
  pipeline.py   runs detector + Re-ID over a sequence and caches the result
  viz.py        H.264 video rendering, target highlighting, figures
results/        tables, figures and MOT-format track files
report/         LaTeX source of the report
outputs/videos/ final tracking videos (the rest of outputs/ stays on Drive)
```

Not included (too large, and model weights are not part of the submission): the MOT16 dataset, trained model weights, and cached detections.

## Running it

1. Put this folder in Google Drive as `MyDrive/DL_Project_MOT`.
2. Open the notebooks in Google Colab with a **T4 GPU** runtime and run them in order: 01 → 02 and 03 (independent) → 04 → 05.
3. Notebook 01 downloads [MOT16](https://motchallenge.net/data/MOT16/) automatically. Detector training resumes by itself if Colab disconnects.

About 2 hours in total on a free Colab T4.

## Dataset split

MOT16-02, 04, 05, 10, 11 and 13 for training; MOT16-09 held out for validation; MOT16-01 (test set, no labels) for the submitted video.

## References

Ren et al., Faster R-CNN (NeurIPS 2015) · He et al., ResNet (CVPR 2016) · Schroff et al., FaceNet (CVPR 2015) · Hermans et al., In Defense of the Triplet Loss (2017) · Bewley et al., SORT (ICIP 2016) · Wojke et al., DeepSORT (ICIP 2017) · Zhang et al., ByteTrack (ECCV 2022) · Milan et al., MOT16 (2016) · Bernardin & Stiefelhagen, CLEAR MOT (2008) · Ristani et al., IDF1 (ECCVW 2016)
