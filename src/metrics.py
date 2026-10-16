"""
Tracking metrics, written from the definitions so we know exactly what they mean:

  MOTA  (Bernardin & Stiefelhagen, 2008)
        1 - (missed people + false boxes + ID switches) / number of GT boxes.
        Mostly a detection-quality score.
  IDF1  (Ristani et al., 2016)
        How long each real person keeps one consistent predicted ID.
        This is the number that best matches the project rubric
        ("track the target across frames without losing it").
  MT / ML
        Mostly Tracked (>= 80% of a person's life covered) / Mostly Lost (<= 20%).

Like the official MOTChallenge evaluation, predictions that land on
"don't care" regions (static people, reflections, ...) are removed first.
Our numbers should be very close to the official TrackEval/devkit ones but
are not guaranteed to match to the last decimal, and we say so in the report.
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from boxes import iou_matrix


def remove_ignored(pred, gt, ignore, iou_thr=0.5):
    """Drop predicted boxes that are matched to an ignore region rather than a real pedestrian."""
    out = {}
    for f, p in pred.items():
        g = gt.get(f, np.zeros((0, 5)))[:, 1:5]
        ign = ignore.get(f, np.zeros((0, 4)))
        if len(ign) == 0 or len(p) == 0:
            out[f] = p
            continue
        both = np.vstack([g, ign]) if len(g) else ign
        iou = iou_matrix(p[:, 1:5], both)
        rows, cols = linear_sum_assignment(-iou)
        drop = {r for r, k in zip(rows, cols) if iou[r, k] >= iou_thr and k >= len(g)}
        out[f] = p[[i for i in range(len(p)) if i not in drop]]
    return out


def clear_mot(gt, pred, iou_thr=0.5):
    """gt: {frame: [id,x1,y1,x2,y2]}, pred: {frame: [id,x1,y1,x2,y2,(score)]}."""
    last_match = {}                 # gt id -> pred id it was last matched to
    tp = fp = fn = idsw = 0
    iou_sum = 0.0
    n_gt = 0
    covered = defaultdict(int)      # gt id -> frames it was matched
    life = defaultdict(int)         # gt id -> frames it exists
    frames = sorted(set(gt) | set(pred))
    for f in frames:
        g = gt.get(f, np.zeros((0, 5)))
        p = pred.get(f, np.zeros((0, 6)))
        n_gt += len(g)
        for gid in g[:, 0]:
            life[int(gid)] += 1
        if len(g) == 0:
            fp += len(p)
            continue
        if len(p) == 0:
            fn += len(g)
            continue
        iou = iou_matrix(g[:, 1:5], p[:, 1:5])
        gids, pids = g[:, 0].astype(int), p[:, 0].astype(int)
        matched_g, matched_p, pairs = set(), set(), []

        # 1) keep last frame's pairings when they are still valid (CLEAR-MOT rule)
        for i, gid in enumerate(gids):
            if gid in last_match:
                js = np.where(pids == last_match[gid])[0]
                if len(js) and js[0] not in matched_p and iou[i, js[0]] >= iou_thr:
                    pairs.append((i, js[0]))
                    matched_g.add(i)
                    matched_p.add(js[0])

        # 2) optimal assignment for everything else
        rest_g = [i for i in range(len(g)) if i not in matched_g]
        rest_p = [j for j in range(len(p)) if j not in matched_p]
        if rest_g and rest_p:
            sub = iou[np.ix_(rest_g, rest_p)]
            rows, cols = linear_sum_assignment(-sub)
            for r, k in zip(rows, cols):
                if sub[r, k] >= iou_thr:
                    i, j = rest_g[r], rest_p[k]
                    if gids[i] in last_match and last_match[gids[i]] != pids[j]:
                        idsw += 1
                    pairs.append((i, j))

        for i, j in pairs:
            last_match[gids[i]] = pids[j]
            covered[gids[i]] += 1
            iou_sum += iou[i, j]
        tp += len(pairs)
        fn += len(g) - len(pairs)
        fp += len(p) - len(pairs)

    ratios = [covered[k] / life[k] for k in life]
    return dict(MOTA=1 - (fn + fp + idsw) / max(n_gt, 1),
                MOTP=iou_sum / max(tp, 1),
                TP=tp, FP=fp, FN=fn, IDSW=idsw, GT=n_gt,
                MT=int(sum(r >= 0.8 for r in ratios)),
                ML=int(sum(r <= 0.2 for r in ratios)),
                GT_IDs=len(life))


def identity_metrics(gt, pred, iou_thr=0.5):
    """IDF1 / IDP / IDR via the best one-to-one matching of whole trajectories."""
    gt_ids = sorted({int(i) for v in gt.values() for i in v[:, 0]})
    pr_ids = sorted({int(i) for v in pred.values() for i in v[:, 0]})
    if not gt_ids or not pr_ids:
        return dict(IDF1=0.0, IDP=0.0, IDR=0.0, pred_IDs=len(pr_ids))
    gi = {k: n for n, k in enumerate(gt_ids)}
    pi = {k: n for n, k in enumerate(pr_ids)}
    overlap = np.zeros((len(gt_ids), len(pr_ids)))
    n_gt = sum(len(v) for v in gt.values())
    n_pr = sum(len(v) for v in pred.values())
    for f in set(gt) & set(pred):
        g, p = gt[f], pred[f]
        if len(g) == 0 or len(p) == 0:
            continue
        iou = iou_matrix(g[:, 1:5], p[:, 1:5])
        for a, b in zip(*np.where(iou >= iou_thr)):
            overlap[gi[int(g[a, 0])], pi[int(p[b, 0])]] += 1
    rows, cols = linear_sum_assignment(-overlap)
    idtp = overlap[rows, cols].sum()
    idfp, idfn = n_pr - idtp, n_gt - idtp
    return dict(IDF1=2 * idtp / max(2 * idtp + idfp + idfn, 1),
                IDP=idtp / max(n_pr, 1), IDR=idtp / max(n_gt, 1), pred_IDs=len(pr_ids))


def evaluate_tracking(gt, pred, ignore=None, iou_thr=0.5):
    """Everything in one dict, rounded for tables."""
    if ignore:
        pred = remove_ignored(pred, gt, ignore, iou_thr)
    res = clear_mot(gt, pred, iou_thr)
    res.update(identity_metrics(gt, pred, iou_thr))
    for k in ["MOTA", "MOTP", "IDF1", "IDP", "IDR"]:
        res[k] = round(100 * res[k], 1)
    return res
