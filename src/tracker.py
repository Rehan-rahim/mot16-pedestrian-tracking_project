"""
The tracker: turns per-frame detections into tracks with stable IDs.

It follows the DeepSORT idea, with a couple of later improvements:

  1. Motion   - a Kalman filter predicts where each person will be in the next
                frame (constant-velocity model on centre x, centre y, aspect, height).
  2. Matching - detections and tracks are matched all at once with the
                Hungarian algorithm (not greedily one by one). The cost mixes
                appearance (Re-ID embedding distance) and overlap (1 - IoU).
  3. Gating   - a pair is only allowed if the detection is near where the
                track should be. This is what stops an ID from jumping across
                the screen to a different person who happens to look similar.
  4. Second pass on low-confidence boxes (ByteTrack idea): partly hidden
                people often get a low detector score; we still use those boxes
                to keep an existing track alive, but never to start a new one.
  5. Offline extras - we process the whole video first, so we can:
                * split a track where the person walked out of the picture
                  (otherwise the old ID can latch onto a newcomer)
                * fill very short detector drop-outs by linear interpolation
                * drop tiny tracks that are almost always false positives

Every one of these can be switched off in TrackerConfig, which is how the
ablation table in notebook 05 is produced.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from boxes import iou_matrix, xyxy_to_cxcyah, cxcyah_to_xyxy

INF_COST = 1e5


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class TrackerConfig:
    # Detection confidence
    # Defaults are the final settings, chosen by measuring on MOT16-09 (see notebook 04/05)
    high_thresh: float = 0.7        # boxes above this can start / update tracks
    low_thresh: float = 0.6         # boxes between low and high only keep existing tracks alive

    # Which components are on (for the ablation study)
    use_kalman: bool = True
    use_appearance: bool = True
    use_low_score: bool = True
    interpolate: bool = True
    split_at_exits: bool = True

    # Matching
    w_app: float = 0.6              # weight of appearance vs IoU in the cost
    app_thresh: float = 0.25        # max cosine distance for "same person"
    iou_strong: float = 0.5         # a big overlap can override a weak appearance match
    iou_only_thresh: float = 0.3    # minimum IoU when appearance is switched off
    gate_base: float = 0.75         # allowed centre jump, in units of the person's height ...
    gate_growth: float = 0.1        # ... growing a bit for every frame the track was missing
    gate_cap: float = 2.5

    # Track life cycle
    min_hits: int = 3               # detections needed before a track is "real"
    max_age: int = 60               # frames a lost track is kept (2 s at 30 fps)
    feat_budget: int = 30           # appearance memory per track
    feat_update_score: float = 0.7  # only clean, confident boxes update the appearance

    # Offline post-processing
    max_gap: int = 5                # only very short detector drop-outs are filled
    min_track_len: int = 8          # shorter tracks are dropped as noise
    exit_margin: int = 40           # px from the image border that counts as "leaving"
    exit_min_gap: int = 5           # a track missing this long after touching the border has left

    def as_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Kalman filter (same formulation as SORT / DeepSORT)
# ---------------------------------------------------------------------------

class KalmanBoxFilter:
    """
    State = [cx, cy, a, h, vx, vy, va, vh]. Measurement = [cx, cy, a, h].
    Noise is scaled by the box height, so big (near) people are allowed to
    move more pixels per frame than small (far) people.
    """

    _w_pos, _w_vel = 1 / 20, 1 / 160

    def __init__(self):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self._w_pos * h, 2 * self._w_pos * h, 1e-2, 2 * self._w_pos * h,
               10 * self._w_vel * h, 10 * self._w_vel * h, 1e-5, 10 * self._w_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self._w_pos * h, self._w_pos * h, 1e-2, self._w_pos * h,
               self._w_vel * h, self._w_vel * h, 1e-5, self._w_vel * h]
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + np.diag(np.square(std))
        return mean, cov

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self._w_pos * h, self._w_pos * h, 1e-1, self._w_pos * h]))
        S = self.H @ cov @ self.H.T + R
        K = np.linalg.solve(S, (cov @ self.H.T).T).T
        mean = mean + K @ (z - self.H @ mean)
        cov = cov - K @ S @ K.T
        return mean, cov


# ---------------------------------------------------------------------------
# One track = one person
# ---------------------------------------------------------------------------

class Track:
    _kf = KalmanBoxFilter()

    def __init__(self, uid, box, score, feat, frame, cfg):
        self.uid = uid
        self.cfg = cfg
        self.hits = 1
        self.time_since_update = 0
        self.box = np.asarray(box[:4], dtype=float)       # last observed box
        self.pred_box = self.box.copy()
        self.mean, self.cov = self._kf.initiate(xyxy_to_cxcyah(self.box))
        self.feats = [] if feat is None else [feat]
        self.history = [(frame, self.box.copy(), float(score))]

    @property
    def confirmed(self):
        return self.hits >= self.cfg.min_hits

    def predict(self):
        if self.cfg.use_kalman:
            if self.mean[3] + self.mean[7] <= 0:      # height would go negative
                self.mean[7] = 0
            self.mean, self.cov = self._kf.predict(self.mean, self.cov)
            self.pred_box = cxcyah_to_xyxy(self.mean)
        else:
            self.pred_box = self.box.copy()           # "no motion model" = stay where last seen
        self.time_since_update += 1

    def update(self, box, score, feat, frame, update_feat=True):
        self.box = np.asarray(box[:4], dtype=float)
        self.mean, self.cov = self._kf.update(self.mean, self.cov, xyxy_to_cxcyah(self.box))
        self.hits += 1
        self.time_since_update = 0
        self.history.append((frame, self.box.copy(), float(score)))
        if feat is not None and update_feat:
            self.feats.append(feat)
            self.feats = self.feats[-self.cfg.feat_budget:]


# ---------------------------------------------------------------------------
# The tracker
# ---------------------------------------------------------------------------

class Tracker:
    def __init__(self, cfg=None):
        self.cfg = cfg or TrackerConfig()
        self.tracks, self.finished = [], []
        self._next_uid = 1

    # -- cost matrices -------------------------------------------------------

    def _appearance_distance(self, tracks, feats):
        """Cosine distance to the closest stored appearance of each track."""
        d = np.ones((len(tracks), len(feats)))
        for i, t in enumerate(tracks):
            if t.feats:
                d[i] = (1 - np.stack(t.feats) @ feats.T).min(axis=0)
        return d

    def _gate(self, tracks, boxes):
        """True where the detection is close enough to the track's predicted position."""
        c = self.cfg
        pred = np.array([t.pred_box for t in tracks])
        pc = (pred[:, :2] + pred[:, 2:]) / 2
        dc = (boxes[:, :2] + boxes[:, 2:4]) / 2
        h = np.maximum(pred[:, 3] - pred[:, 1], 1.0)
        dist = np.linalg.norm(pc[:, None] - dc[None], axis=2) / h[:, None]
        tsu = np.array([t.time_since_update for t in tracks])
        allowed = np.minimum(c.gate_base + c.gate_growth * (tsu - 1), c.gate_cap)
        return dist < allowed[:, None]

    def _match(self, tracks, dets, feats, iou_only=False, iou_min=None):
        if not tracks or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        c = self.cfg
        iou = iou_matrix([t.pred_box for t in tracks], dets[:, :4])
        if iou_only or not c.use_appearance or feats is None:
            thr = iou_min if iou_min is not None else c.iou_only_thresh
            cost = np.where(iou >= thr, 1 - iou, INF_COST)
        else:
            app = self._appearance_distance(tracks, feats)
            valid = self._gate(tracks, dets) & ((app < c.app_thresh) | (iou >= c.iou_strong))
            cost = np.where(valid, c.w_app * app + (1 - c.w_app) * (1 - iou), INF_COST)
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, k) for r, k in zip(rows, cols) if cost[r, k] < INF_COST]
        mr, mc = {r for r, _ in matches}, {k for _, k in matches}
        return (matches, [i for i in range(len(tracks)) if i not in mr],
                [j for j in range(len(dets)) if j not in mc])

    # -- main step -----------------------------------------------------------

    def update(self, frame, dets, feats=None):
        """dets: (N, 5) x1, y1, x2, y2, score. feats: (N, D) Re-ID embeddings or None."""
        c = self.cfg
        dets = np.asarray(dets, dtype=float).reshape(-1, 5)
        for t in self.tracks:
            t.predict()

        high = dets[:, 4] >= c.high_thresh
        low = (~high) & (dets[:, 4] >= c.low_thresh)
        hi_idx, lo_idx = np.where(high)[0], np.where(low)[0]
        hi_feats = feats[hi_idx] if (feats is not None and c.use_appearance) else None

        # Pass 1: every track vs confident detections (appearance + motion)
        matches, un_tracks, un_dets = self._match(self.tracks, dets[hi_idx], hi_feats)
        for ti, dj in matches:
            j = hi_idx[dj]
            f = feats[j] if feats is not None else None
            self.tracks[ti].update(dets[j], dets[j, 4], f, frame,
                                   update_feat=dets[j, 4] >= c.feat_update_score)

        # Pass 2: tracks seen in the previous frame vs low-confidence boxes (IoU only)
        if c.use_low_score and len(lo_idx):
            recent = [i for i in un_tracks if self.tracks[i].time_since_update == 1]
            m2, _, _ = self._match([self.tracks[i] for i in recent], dets[lo_idx], None,
                                   iou_only=True, iou_min=0.5)
            matched2 = set()
            for ri, dj in m2:
                ti, j = recent[ri], lo_idx[dj]
                self.tracks[ti].update(dets[j], dets[j, 4], None, frame, update_feat=False)
                matched2.add(ti)
            un_tracks = [i for i in un_tracks if i not in matched2]

        # Start new tracks from confident detections nobody claimed
        for dj in un_dets:
            j = hi_idx[dj]
            f = feats[j] if feats is not None else None
            self.tracks.append(Track(self._next_uid, dets[j], dets[j, 4], f, frame, c))
            self._next_uid += 1

        # Retire tracks: tentative ones die on the first miss, confirmed ones after max_age
        alive = []
        for t in self.tracks:
            dead = (t.time_since_update > 0 and not t.confirmed) or t.time_since_update > c.max_age
            if dead:
                if t.confirmed:
                    self.finished.append(t)
            else:
                alive.append(t)
        self.tracks = alive

    def results(self, image_size=None):
        """
        Final offline output: {frame: array([[id, x1, y1, x2, y2, score], ...])}.

        Offline post-processing, in this order:
          1. split a track where the person walked out of the picture
             (needs image_size = (width, height))
          2. fill very short gaps inside each piece by interpolation
          3. drop pieces shorter than min_track_len
        IDs are renumbered 1..N in order of first appearance so they look tidy.
        Interpolated boxes get score -1 so they can be told apart.
        """
        c = self.cfg
        done = [t for t in self.finished + self.tracks if t.confirmed]
        done.sort(key=lambda t: t.history[0][0])
        out = {}
        new_id = 0
        for t in done:
            hist = sorted(t.history, key=lambda x: x[0])
            if c.split_at_exits and image_size is not None:
                pieces = split_at_exits(hist, image_size, c.exit_margin, c.exit_min_gap)
            else:
                pieces = [hist]
            for piece in pieces:
                if c.interpolate:
                    piece = interpolate_history(piece, c.max_gap)
                if len(piece) < c.min_track_len:
                    continue
                new_id += 1
                for f, box, s in piece:
                    out.setdefault(f, []).append([new_id, *box, s])
        return {f: np.array(v) for f, v in sorted(out.items())}


def interpolate_history(history, max_gap):
    """Fill missing frames inside a track with straight-line boxes (offline only)."""
    hist = sorted(history, key=lambda x: x[0])
    filled = [hist[0]]
    for (f0, b0, _), (f1, b1, s1) in zip(hist[:-1], hist[1:]):
        gap = f1 - f0
        if 1 < gap <= max_gap + 1:
            for k in range(1, gap):
                a = k / gap
                filled.append((f0 + k, (1 - a) * b0 + a * b1, -1.0))
        filled.append((f1, b1, s1))
    return filled


def split_at_exits(history, image_size, margin=40, min_gap=5):
    """
    A person who walks out of the picture has left. If a track was last seen
    touching the image border and only continues several frames later, the
    continuation is someone else who happened to appear near the predicted
    position (we saw exactly this in MOT16-01: the target left on the right
    edge and 34 frames later his old track latched onto a different person).
    """
    width, height = image_size
    pieces, current = [], [history[0]]
    for prev, nxt in zip(history[:-1], history[1:]):
        x1, y1, x2, y2 = prev[1]
        at_edge = x1 < margin or y1 < margin or x2 > width - margin or y2 > height - margin
        if at_edge and nxt[0] - prev[0] > min_gap:
            pieces.append(current)
            current = []
        current.append(nxt)
    pieces.append(current)
    return pieces


def run_tracker(dets, feats=None, cfg=None, image_size=None):
    """
    Run the tracker over a whole sequence of cached detections.
    image_size = (width, height) enables the "left the picture" rule.
    """
    tracker = Tracker(cfg)
    for f in sorted(dets):
        tracker.update(f, dets[f], None if feats is None else feats.get(f))
    return tracker.results(image_size)


# ---------------------------------------------------------------------------
# Reading / writing MOTChallenge result files
# ---------------------------------------------------------------------------

def save_mot(results, path):
    """frame, id, x, y, w, h, score, -1, -1, -1 (the official MOTChallenge format)."""
    with open(path, "w") as fh:
        for f in sorted(results):
            for tid, x1, y1, x2, y2, s in results[f]:
                fh.write(f"{f},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                         f"{max(s, 0):.3f},-1,-1,-1\n")


def load_mot(path):
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    out = {}
    for row in raw:
        f, tid, x, y, w, h, s = row[:7]
        out.setdefault(int(f), []).append([tid, x, y, x + w, y + h, s])
    return {f: np.array(v) for f, v in out.items()}


def track_lengths(results):
    """{id: number of frames} sorted longest first. Handy for picking the demo target."""
    counts = {}
    for rows in results.values():
        for r in rows:
            counts[int(r[0])] = counts.get(int(r[0]), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
