"""
Drawing boxes, writing videos and making figures for the slides / report.

Videos are written straight to H.264 through ffmpeg. OpenCV's default
'mp4v' files often refuse to play in browsers and PowerPoint, which is the
last thing we want during the final presentation.
"""

import colorsys
import os
import subprocess
from base64 import b64encode
from collections import defaultdict

import cv2
import numpy as np

from mot_data import read_seqinfo, frame_path


def id_color(tid):
    """Stable, well-separated colour per ID (golden-ratio hue spacing). BGR for OpenCV."""
    h = (int(tid) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


class VideoWriter:
    """Pipe raw frames into ffmpeg -> browser/PowerPoint friendly MP4."""

    def __init__(self, path, fps, size):
        w, h = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast", path],
            stdin=subprocess.PIPE)
        self.size = size

    def write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def _label(img, text, x, y, color, scale=0.6, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick)


def render_tracks(seq_dir, results, out_path, highlight_id=None, max_frames=None,
                  scale=1.0, trail=45, title=None):
    """
    Draw tracking results onto the video.

    highlight_id=None -> every person in their own colour (multi-object view)
    highlight_id=k    -> person k in thick red with a motion trail, everyone
                         else thin grey (the single-target view the
                         submission asks for)
    """
    info = read_seqinfo(seq_dir)
    W, H = int(info["width"] * scale), int(info["height"] * scale)
    writer = VideoWriter(out_path, info["fps"], (W, H))
    n = info["length"] if max_frames is None else min(max_frames, info["length"])
    trails = defaultdict(list)
    target_seen = 0
    for f in range(1, n + 1):
        img = cv2.imread(frame_path(seq_dir, info, f))
        for row in results.get(f, []):
            tid = int(row[0])
            x1, y1, x2, y2 = map(int, row[1:5])
            interp = len(row) > 5 and row[5] < 0
            trails[tid].append(((x1 + x2) // 2, y2))
            is_target = highlight_id is not None and tid == highlight_id
            if highlight_id is not None and not is_target:
                cv2.rectangle(img, (x1, y1), (x2, y2), (170, 170, 170), 1)
                continue
            color = (0, 0, 255) if is_target else id_color(tid)
            thick = 4 if is_target else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick,
                          lineType=cv2.LINE_AA if not interp else cv2.LINE_4)
            _label(img, f"TARGET ID {tid}" if is_target else f"ID {tid}", x1, y1 - 2, color,
                   0.8 if is_target else 0.55)
            pts = trails[tid][-trail:]
            for p, q in zip(pts[:-1], pts[1:]):
                cv2.line(img, p, q, color, 3 if is_target else 2)
            if is_target:
                target_seen += 1
        banner = title or info["name"]
        cv2.putText(img, f"{banner} | frame {f}/{info['length']}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv2.putText(img, f"{banner} | frame {f}/{info['length']}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        writer.write(img)
    writer.close()
    print(f"saved {out_path}" + (f" (target visible in {target_seen} frames)" if highlight_id else ""))
    return out_path


def show_video(path, width=800):
    """Show a *small* video inside the notebook. Keep previews short, or the notebook balloons."""
    from IPython.display import HTML
    size_mb = os.path.getsize(path) / 1e6
    if size_mb > 25:
        print(f"{size_mb:.0f} MB is too big to embed; open it from Drive instead: {path}")
        return None
    data = b64encode(open(path, "rb").read()).decode()
    return HTML(f'<video width={width} controls><source src="data:video/mp4;base64,{data}" '
                f'type="video/mp4"></video>')


def target_filmstrip(seq_dir, results, target_id, n=8, out_path=None):
    """Crops of the target at evenly spaced moments: a nice figure for the report."""
    import matplotlib.pyplot as plt
    info = read_seqinfo(seq_dir)
    frames = [f for f in sorted(results) if target_id in results[f][:, 0].astype(int)]
    pick = [frames[i] for i in np.linspace(0, len(frames) - 1, n).astype(int)]
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 4))
    for ax, f in zip(axes, pick):
        row = results[f][results[f][:, 0].astype(int) == target_id][0]
        x1, y1, x2, y2 = map(int, row[1:5])
        img = cv2.imread(frame_path(seq_dir, info, f))
        crop = img[max(y1, 0):y2, max(x1, 0):x2]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Target ID {target_id} across {info['name']}")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
