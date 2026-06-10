#!/usr/bin/env python3
"""
Refine YOLO person detections using tracking-by-detection on per-frame txt files.

Input format per line:
    conf cx cy w h   (all normalized to [0..1])

Output format (same):
    conf cx cy w h

Assumptions:
- Input directory contains files strictly named: frame_000000.txt, frame_000001.txt, ...
- Files exist for ALL frames. If a frame is missing -> error.
- Works for video results (sequential frames).
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger("refine_detections")

FRAME_RE = re.compile(r"^frame_(\d{6})\.txt$")


# ----------------------------- Geometry utils -----------------------------


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """
    xywh: [...,4] where (cx,cy,w,h), normalized
    returns xyxy: [...,4] where (x1,y1,x2,y2), normalized
    """
    cx, cy, w, h = xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return np.stack([cx, cy, w, h], axis=-1)


def iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """
    a_xyxy: [A,4], b_xyxy: [B,4]
    returns IoU: [A,B]
    """
    if a_xyxy.size == 0 or b_xyxy.size == 0:
        return np.zeros((a_xyxy.shape[0], b_xyxy.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a_xyxy[:, 0:1], a_xyxy[:, 1:2], a_xyxy[:, 2:3], a_xyxy[:, 3:4]
    bx1, by1, bx2, by2 = b_xyxy[:, 0], b_xyxy[:, 1], b_xyxy[:, 2], b_xyxy[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    a_area = np.maximum(0.0, (ax2 - ax1)) * np.maximum(0.0, (ay2 - ay1))
    b_area = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))

    union = a_area + b_area - inter_area + 1e-9
    return (inter_area / union).astype(np.float32)


def diou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """
    Distance-IoU (DIoU) metric matrix in [-1..1], higher is better.
    """
    if a_xyxy.size == 0 or b_xyxy.size == 0:
        return np.zeros((a_xyxy.shape[0], b_xyxy.shape[0]), dtype=np.float32)

    iou = iou_matrix(a_xyxy, b_xyxy)

    # centers
    a = a_xyxy
    b = b_xyxy
    acx = (a[:, 0:1] + a[:, 2:3]) / 2.0
    acy = (a[:, 1:2] + a[:, 3:4]) / 2.0
    bcx = (b[:, 0] + b[:, 2]) / 2.0
    bcy = (b[:, 1] + b[:, 3]) / 2.0

    # center distance squared
    rho2 = (acx - bcx) ** 2 + (acy - bcy) ** 2  # [A,B]

    # enclosing box diagonal squared
    c_x1 = np.minimum(a[:, 0:1], b[:, 0])
    c_y1 = np.minimum(a[:, 1:2], b[:, 1])
    c_x2 = np.maximum(a[:, 2:3], b[:, 2])
    c_y2 = np.maximum(a[:, 3:4], b[:, 3])
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + 1e-9

    diou = iou - (rho2 / c2)
    return diou.astype(np.float32)


def nms_xywh(dets_xywh: np.ndarray, confs: np.ndarray, iou_thr: float) -> np.ndarray:
    """
    dets_xywh: [N,4] (cx,cy,w,h)
    confs: [N]
    returns indices kept
    """
    n = dets_xywh.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)

    xyxy = xywh_to_xyxy(dets_xywh)
    order = np.argsort(-confs)
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(xyxy[i : i + 1], xyxy[rest]).reshape(-1)
        rest = rest[ious <= iou_thr]
        order = rest

    return np.array(keep, dtype=np.int64)


def greedy_match(
    score: np.ndarray, thr: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    score: [T,D] higher is better
    thr: minimal acceptable score for matching
    Returns:
      matches: list of (t_idx, d_idx)
      unmatched_t: track indices
      unmatched_d: det indices
    """
    T, D = score.shape
    if T == 0:
        return [], [], list(range(D))
    if D == 0:
        return [], list(range(T)), []

    score_work = score.copy()
    matches: List[Tuple[int, int]] = []
    used_t = set()
    used_d = set()

    while True:
        t, d = np.unravel_index(np.argmax(score_work), score_work.shape)
        best = float(score_work[t, d])
        if best < thr:
            break
        if t in used_t or d in used_d:
            score_work[t, d] = -1e9
            continue
        matches.append((int(t), int(d)))
        used_t.add(int(t))
        used_d.add(int(d))
        score_work[t, :] = -1e9
        score_work[:, d] = -1e9

    unmatched_t = [i for i in range(T) if i not in used_t]
    unmatched_d = [j for j in range(D) if j not in used_d]
    return matches, unmatched_t, unmatched_d


# ----------------------------- Tracker -----------------------------


@dataclass
class Track:
    tid: int
    # last two states for constant-velocity prediction
    last_xywh: np.ndarray  # (4,)
    prev_xywh: Optional[np.ndarray] = None

    time_since_update: int = 0
    age: int = 0  # frames since created
    hits: int = 0  # number of matched detections

    measured: Dict[int, Tuple[np.ndarray, float]] = field(
        default_factory=dict
    )  # frame -> (xywh, conf)

    def predict(self) -> np.ndarray:
        """
        Constant velocity prediction in xywh space.
        """
        self.age += 1
        self.time_since_update += 1

        if self.prev_xywh is None:
            return self.last_xywh.copy()

        v = self.last_xywh - self.prev_xywh
        pred = self.last_xywh + v
        return pred

    def update(self, frame_idx: int, det_xywh: np.ndarray, det_conf: float) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.prev_xywh = self.last_xywh.copy()
        self.last_xywh = det_xywh.copy()
        self.measured[frame_idx] = (det_xywh.copy(), float(det_conf))

    def first_last_measured(self) -> Tuple[int, int]:
        ks = sorted(self.measured.keys())
        return ks[0], ks[-1]


# ----------------------------- IO -----------------------------


def parse_frame_idx(path: Path) -> int:
    m = FRAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Bad filename (expected frame_000000.txt): {path.name}")
    return int(m.group(1))


def load_frame_dets(path: Path, strict: bool = True) -> np.ndarray:
    """
    returns array [N,5] columns: conf,cx,cy,w,h
    """
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    for ln, line in enumerate(txt.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            msg = (
                f"Invalid line in {path.name}:{ln}: expected 5 floats, got {len(parts)}"
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            continue
        try:
            vals = [float(p) for p in parts]
        except Exception:
            msg = f"Invalid float in {path.name}:{ln}: {line}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            continue
        rows.append(vals)

    if not rows:
        return np.zeros((0, 5), dtype=np.float32)

    arr = np.asarray(rows, dtype=np.float32)
    return arr


def ensure_contiguous_frames(frame_files: List[Path]) -> List[Path]:
    indexed = [(parse_frame_idx(p), p) for p in frame_files]
    indexed.sort(key=lambda x: x[0])

    if not indexed:
        raise ValueError("No frame_*.txt files found")

    # must start at 0 and be contiguous
    first = indexed[0][0]
    if first != 0:
        raise ValueError(f"First frame index must be 0, got {first}")

    for i, (idx, p) in enumerate(indexed):
        if idx != i:
            raise ValueError(
                f"Missing frame file: expected frame_{i:06d}.txt, got {p.name}"
            )

    return [p for _, p in indexed]


# ----------------------------- Refinement -----------------------------


@dataclass
class RefineConfig:
    # matching
    match_metric: str = "diou"  # iou | diou
    match_thr: float = 0.30
    max_age: int = 15

    # track filtering
    min_det_hits: int = 3
    min_track_len: int = (
        0  # 0 = disabled, else length in frames between first and last measured
    )

    # filling gaps
    fill_gaps: bool = True
    max_gap_fill: int = 3

    extrapolate: bool = False
    max_gap_extrap: int = 2

    # smoothing
    smooth: bool = True
    smooth_window: int = 5  # odd recommended

    # NMS
    nms_in_frame: bool = True
    nms_iou_in: float = 0.70

    nms_out_frame: bool = True
    nms_iou_out: float = 0.70

    # confidence handling
    conf_decay: float = 0.90
    min_conf_out: float = 0.0

    # strict parsing
    strict: bool = True


def moving_average(arr: np.ndarray, win: int) -> np.ndarray:
    """
    Simple centered moving average per column.
    arr: [N,4]
    """
    if win <= 1 or arr.shape[0] <= 2:
        return arr
    win = int(win)
    if win < 1:
        return arr
    # allow even, but odd is better; we just do variable window at borders
    out = np.zeros_like(arr)
    n = arr.shape[0]
    half = win // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = arr[a:b].mean(axis=0)
    return out


def build_track_outputs(
    track: Track,
    num_frames: int,
    cfg: RefineConfig,
) -> Dict[int, Tuple[np.ndarray, float]]:
    """
    Returns frame -> (xywh, conf) for this track after fill/extrap/smooth.
    """
    measured_frames = sorted(track.measured.keys())
    if not measured_frames:
        return {}

    out: Dict[int, Tuple[np.ndarray, float]] = {}

    # Insert measured
    for f in measured_frames:
        xywh, conf = track.measured[f]
        out[f] = (xywh.copy(), float(conf))

    # Fill gaps by interpolation between measured points
    if cfg.fill_gaps and cfg.max_gap_fill > 0:
        for f1, f2 in zip(measured_frames[:-1], measured_frames[1:]):
            gap = f2 - f1 - 1
            if gap <= 0:
                continue
            if gap > cfg.max_gap_fill:
                continue

            b1, c1 = track.measured[f1]
            b2, c2 = track.measured[f2]
            base_conf = float(min(c1, c2))
            for t in range(1, gap + 1):
                alpha = t / (gap + 1.0)
                b = (1 - alpha) * b1 + alpha * b2
                dist_to_nearest = min(t, gap + 1 - t)
                conf = base_conf * (cfg.conf_decay**dist_to_nearest)
                ff = f1 + t
                if 0 <= ff < num_frames:
                    out[ff] = (b.astype(np.float32), float(conf))

    # Extrapolate after last measured
    if cfg.extrapolate and cfg.max_gap_extrap > 0:
        last_f = measured_frames[-1]
        b_last, c_last = track.measured[last_f]

        if len(measured_frames) >= 2:
            prev_f = measured_frames[-2]
            b_prev, _ = track.measured[prev_f]
            dt = max(1, last_f - prev_f)
            v = (b_last - b_prev) / dt
        else:
            v = np.zeros((4,), dtype=np.float32)

        for t in range(1, cfg.max_gap_extrap + 1):
            ff = last_f + t
            if ff >= num_frames:
                break
            b = b_last + v * t
            conf = float(c_last) * (cfg.conf_decay**t)
            out[ff] = (b.astype(np.float32), float(conf))

    # Smooth per contiguous segments (where we have detections for consecutive frames)
    if cfg.smooth and cfg.smooth_window > 1:
        frames_sorted = sorted(out.keys())
        if frames_sorted:
            seg_start = 0
            while seg_start < len(frames_sorted):
                seg_end = seg_start
                while (
                    seg_end + 1 < len(frames_sorted)
                    and frames_sorted[seg_end + 1] == frames_sorted[seg_end] + 1
                ):
                    seg_end += 1

                seg_frames = frames_sorted[seg_start : seg_end + 1]
                seg_boxes = np.stack([out[f][0] for f in seg_frames], axis=0)
                seg_boxes_s = moving_average(seg_boxes, cfg.smooth_window)

                for f, b in zip(seg_frames, seg_boxes_s):
                    conf = out[f][1]  # keep conf
                    out[f] = (b.astype(np.float32), float(conf))

                seg_start = seg_end + 1

    # Clip / sanitize
    cleaned: Dict[int, Tuple[np.ndarray, float]] = {}
    for f, (b, c) in out.items():
        if not (0 <= f < num_frames):
            continue
        b = b.astype(np.float32)
        # avoid negative sizes
        b[2] = max(float(b[2]), 1e-6)
        b[3] = max(float(b[3]), 1e-6)
        b[0:2] = clip01(b[0:2])
        b[2:4] = np.clip(b[2:4], 1e-6, 1.0)
        if c >= cfg.min_conf_out:
            cleaned[f] = (b, float(c))
    return cleaned


def refine(input_dir: Path, output_dir: Path, cfg: RefineConfig) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = list(input_dir.glob("frame_*.txt"))
    frame_files = ensure_contiguous_frames(frame_files)
    num_frames = len(frame_files)
    logger.info("Found %d frames.", num_frames)

    # Online tracking
    tracks: List[Track] = []
    finished: List[Track] = []
    next_tid = 1

    for frame_idx, path in enumerate(tqdm(frame_files, desc="Tracking")):
        arr = load_frame_dets(path, strict=cfg.strict)  # [N,5]
        if arr.shape[0] > 0:
            confs = arr[:, 0]
            det_xywh = arr[:, 1:5]

            # per-frame NMS (input)
            if cfg.nms_in_frame and det_xywh.shape[0] > 1:
                keep = nms_xywh(det_xywh, confs, cfg.nms_iou_in)
                confs = confs[keep]
                det_xywh = det_xywh[keep]

        else:
            confs = np.zeros((0,), dtype=np.float32)
            det_xywh = np.zeros((0, 4), dtype=np.float32)

        # Predict tracks
        pred_xywh = []
        for tr in tracks:
            pred = tr.predict()
            pred_xywh.append(pred)
        pred_xywh = (
            np.stack(pred_xywh, axis=0)
            if pred_xywh
            else np.zeros((0, 4), dtype=np.float32)
        )

        # Match
        if tracks and det_xywh.shape[0] > 0:
            a_xyxy = xywh_to_xyxy(pred_xywh)
            b_xyxy = xywh_to_xyxy(det_xywh)

            if cfg.match_metric == "iou":
                score = iou_matrix(a_xyxy, b_xyxy)
            elif cfg.match_metric == "diou":
                score = diou_matrix(a_xyxy, b_xyxy)
            else:
                raise ValueError(f"Unknown match metric: {cfg.match_metric}")

            matches, unmatched_t, unmatched_d = greedy_match(score, cfg.match_thr)
        else:
            matches = []
            unmatched_t = list(range(len(tracks)))
            unmatched_d = list(range(det_xywh.shape[0]))

        # Update matched
        for t_i, d_i in matches:
            tr = tracks[t_i]
            tr.update(frame_idx, det_xywh[d_i], float(confs[d_i]))

        # Create new tracks for unmatched detections
        for d_i in unmatched_d:
            tr = Track(tid=next_tid, last_xywh=det_xywh[d_i].copy())
            tr.update(frame_idx, det_xywh[d_i], float(confs[d_i]))
            tracks.append(tr)
            next_tid += 1

        # Age unmatched tracks and finish old ones
        alive: List[Track] = []
        for i, tr in enumerate(tracks):
            # note: tr.time_since_update already incremented in predict()
            if tr.time_since_update > cfg.max_age:
                finished.append(tr)
            else:
                alive.append(tr)
        tracks = alive

    finished.extend(tracks)

    logger.info("Total tracks: %d", len(finished))

    # Filter tracks
    kept_tracks: List[Track] = []
    for tr in finished:
        if tr.hits < cfg.min_det_hits:
            continue
        if cfg.min_track_len > 0:
            f1, f2 = tr.first_last_measured()
            if (f2 - f1 + 1) < cfg.min_track_len:
                continue
        kept_tracks.append(tr)

    logger.info("Tracks kept after filtering: %d", len(kept_tracks))

    # Build outputs per track, merge per frame
    frame_to_dets: List[List[Tuple[float, np.ndarray]]] = [
        [] for _ in range(num_frames)
    ]  # list of (conf, xywh)

    for tr in kept_tracks:
        out = build_track_outputs(tr, num_frames=num_frames, cfg=cfg)
        for f, (xywh, conf) in out.items():
            frame_to_dets[f].append((conf, xywh))

    # Write output files for ALL frames (including empty)
    for frame_idx, path in enumerate(tqdm(frame_files, desc="Writing")):
        out_path = output_dir / path.name
        dets = frame_to_dets[frame_idx]
        if not dets:
            out_path.write_text("", encoding="utf-8")
            continue

        confs = np.array([c for c, _ in dets], dtype=np.float32)
        xywh = np.stack([b for _, b in dets], axis=0).astype(np.float32)

        # final NMS (output)
        if cfg.nms_out_frame and xywh.shape[0] > 1:
            keep = nms_xywh(xywh, confs, cfg.nms_iou_out)
            confs = confs[keep]
            xywh = xywh[keep]

        # sort by conf desc
        order = np.argsort(-confs)
        confs = confs[order]
        xywh = xywh[order]

        lines = []
        for conf, (cx, cy, w, h) in zip(confs, xywh):
            lines.append(
                f"{float(conf):.6f} {float(cx):.6f} {float(cy):.6f} {float(w):.6f} {float(h):.6f}"
            )
        out_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    logger.info("Done. Output written to: %s", str(output_dir))


# ----------------------------- CLI -----------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Refine YOLO detections using tracking (offline, from txt files)."
    )
    p.add_argument(
        "--input-dir", type=Path, required=True, help="Folder with frame_000000.txt ..."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder to write refined frame_*.txt (will be created).",
    )

    # matching / tracker
    p.add_argument("--match-metric", choices=["iou", "diou"], default="diou")
    p.add_argument("--match-thr", type=float, default=0.30)
    p.add_argument("--max-age", type=int, default=15)

    # filtering
    p.add_argument("--min-det-hits", type=int, default=3)
    p.add_argument("--min-track-len", type=int, default=0)

    # gap fill / extrap
    p.add_argument("--fill-gaps", action="store_true", default=True)
    p.add_argument("--no-fill-gaps", dest="fill_gaps", action="store_false")
    p.add_argument("--max-gap-fill", type=int, default=3)

    p.add_argument("--extrapolate", action="store_true", default=False)
    p.add_argument("--max-gap-extrap", type=int, default=2)

    # smoothing
    p.add_argument("--smooth", action="store_true", default=True)
    p.add_argument("--no-smooth", dest="smooth", action="store_false")
    p.add_argument("--smooth-window", type=int, default=5)

    # NMS
    p.add_argument("--nms-in", action="store_true", default=True)
    p.add_argument("--no-nms-in", dest="nms_in", action="store_false")
    p.add_argument("--nms-iou-in", type=float, default=0.70)

    p.add_argument("--nms-out", action="store_true", default=True)
    p.add_argument("--no-nms-out", dest="nms_out", action="store_false")
    p.add_argument("--nms-iou-out", type=float, default=0.70)

    # conf handling
    p.add_argument("--conf-decay", type=float, default=0.90)
    p.add_argument("--min-conf-out", type=float, default=0.0)

    # parsing
    p.add_argument("--strict", action="store_true", default=True)
    p.add_argument("--no-strict", dest="strict", action="store_false")

    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    args = build_argparser().parse_args()

    cfg = RefineConfig(
        match_metric=args.match_metric,
        match_thr=float(args.match_thr),
        max_age=int(args.max_age),
        min_det_hits=int(args.min_det_hits),
        min_track_len=int(args.min_track_len),
        fill_gaps=bool(args.fill_gaps),
        max_gap_fill=int(args.max_gap_fill),
        extrapolate=bool(args.extrapolate),
        max_gap_extrap=int(args.max_gap_extrap),
        smooth=bool(args.smooth),
        smooth_window=int(args.smooth_window),
        nms_in_frame=bool(args.nms_in),
        nms_iou_in=float(args.nms_iou_in),
        nms_out_frame=bool(args.nms_out),
        nms_iou_out=float(args.nms_iou_out),
        conf_decay=float(args.conf_decay),
        min_conf_out=float(args.min_conf_out),
        strict=bool(args.strict),
    )

    refine(args.input_dir, args.output_dir, cfg)


if __name__ == "__main__":
    main()
