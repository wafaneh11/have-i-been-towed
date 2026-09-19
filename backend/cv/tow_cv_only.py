#!/usr/bin/env python3
"""
Standalone computer-vision pipeline for identifying the license plate of a
vehicle being towed and sending an accepted plate to the backend.

CV pipeline:
1. Detect license plates with FastALPR / YOLO.
2. OCR each detected plate.
3. Track plates across video frames.
4. Estimate camera/tow-truck motion from background optical flow.
5. Identify plates that remain rigid relative to the camera while the truck moves.
6. Vote across repeated OCR reads to reduce single-frame OCR errors.
7. Save the best snapshot for each accepted plate.

Example:
    python tow_cv_only.py tow_test.mp4 --debug
"""

import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fast_alpr import ALPR


def mean_conf(conf) -> float:
    """fast-alpr can return one float or a per-character list."""
    if conf is None:
        return 0.0
    if isinstance(conf, (list, tuple)) or hasattr(conf, "__len__"):
        conf = list(conf)
        return float(sum(conf) / len(conf)) if conf else 0.0
    return float(conf)


def parse_roi(text):
    """Parse 'x1,y1,x2,y2' as fractions of the frame (0-1)."""
    try:
        x1, y1, x2, y2 = (float(v) for v in text.split(","))
    except ValueError:
        sys.exit("--roi must be four numbers: x1,y1,x2,y2 (fractions 0-1)")

    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        sys.exit("--roi needs 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")

    return x1, y1, x2, y2


def choose_targets(results, target, frame_w, frame_h, roi, min_width_frac):
    """
    Decide which plates in one frame matter.

    roi:
        Ignore plates whose center is outside this region.

    min_width_frac:
        Ignore plates narrower than this fraction of the frame.

    target:
        "all"     -> keep all remaining plates
        "largest" -> keep the largest plate
        "center"  -> keep the plate nearest the center
        "towed"   -> handled as "all" before tracking
    """
    kept = []

    for r in results:
        if r.ocr is None or not r.ocr.text:
            continue

        b = r.detection.bounding_box
        cx = (b.x1 + b.x2) / 2 / frame_w
        cy = (b.y1 + b.y2) / 2 / frame_h

        if roi and not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
            continue

        if (b.x2 - b.x1) / frame_w < min_width_frac:
            continue

        kept.append((r, b, cx, cy))

    if not kept or target == "all":
        return [r for r, *_ in kept]

    if target == "largest":
        best = max(
            kept,
            key=lambda k: (k[1].x2 - k[1].x1) * (k[1].y2 - k[1].y1),
        )
    else:
        mx = (roi[0] + roi[2]) / 2 if roi else 0.5
        my = (roi[1] + roi[3]) / 2 if roi else 0.5
        best = min(
            kept,
            key=lambda k: (k[2] - mx) ** 2 + (k[3] - my) ** 2,
        )

    return [best[0]]


def box_iou(a, b):
    """Intersection-over-union between two bounding boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy

    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - inter
    )

    return inter / union if union > 0 else 0.0


class EgoMotion:
    """
    Estimate how the background/world moves between analyzed frames.

    This approximates camera/tow-truck motion using tracked corner points,
    while masking the regions around detected plates/cars.
    """

    def __init__(self, max_width=640):
        self.prev = None
        self.max_width = max_width

    def step(self, frame, plate_boxes):
        """Return median background shift (dx, dy) in original pixels."""
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_width / w)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if scale < 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        prev, self.prev = self.prev, gray

        if prev is None:
            return None

        mask = np.full(gray.shape, 255, np.uint8)

        for x1, y1, x2, y2 in plate_boxes:
            pw = (x2 - x1) * scale
            ph = (y2 - y1) * scale
            cx = (x1 + x2) / 2 * scale
            cy = (y1 + y2) / 2 * scale

            mask[
                max(0, int(cy - 6 * ph)) : int(cy + 3 * ph),
                max(0, int(cx - 3 * pw)) : int(cx + 3 * pw),
            ] = 0

        pts = cv2.goodFeaturesToTrack(prev, 200, 0.01, 8, mask=mask)

        if pts is None or len(pts) < 15:
            return None

        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            prev,
            gray,
            pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )

        ok = st.ravel() == 1

        if ok.sum() < 15:
            return None

        d = (nxt - pts)[ok].reshape(-1, 2)

        return (
            float(np.median(d[:, 0]) / scale),
            float(np.median(d[:, 1]) / scale),
        )


class Track:
    """One physical plate followed across frames."""

    def __init__(self, tid, box, frame_no):
        self.id = tid
        self.boxes = [box]
        self.last_box = box
        self.last_frame = frame_no
        self.first_frame = frame_no

        self.reads = {}

        self.moving_steps = 0
        self.rigid_steps = 0
        self.parked_steps = 0

        self.win_n = 0
        self.win_bx = 0.0
        self.win_by = 0.0
        self.win_start = None

    def add_box(self, box, frame_no):
        self.boxes.append(box)
        self.last_box = box
        self.last_frame = frame_no

    def add_read(self, text, conf, frame, box):
        entry = self.reads.setdefault(
            text,
            {
                "confs": [],
                "best_conf": 0.0,
                "best_frame": None,
            },
        )

        entry["confs"].append(conf)

        if conf > entry["best_conf"] and frame is not None:
            snap = frame.copy()

            cv2.rectangle(
                snap,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                snap,
                f"{text} {conf:.2f}",
                (int(box[0]), max(25, int(box[1]) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            entry["best_conf"] = conf
            entry["best_frame"] = snap

    @property
    def votes(self):
        return sum(len(e["confs"]) for e in self.reads.values())

    @property
    def top_text(self):
        if not self.reads:
            return "?"

        return max(
            self.reads.items(),
            key=lambda kv: len(kv[1]["confs"]),
        )[0]

    def motion(self):
        """
        Approximate how much the plate wanders in the image,
        expressed in plate-widths.
        """
        b = np.array(self.boxes, dtype=float)
        w = b[:, 2] - b[:, 0]
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        mw = w.mean() or 1.0

        return float(
            (np.hypot(cx.std(), cy.std()) + w.std()) / mw
        )


class TrackSet:
    """Simple IoU/center-distance tracker for license plates."""

    def __init__(
        self,
        max_gap_frames,
        step=1,
        move_thr=0.5,
        rigid_ratio=0.3,
        win_steps=20,
    ):
        self.tracks = []
        self.max_gap = max_gap_frames
        self.step = step
        self.win_steps = win_steps
        self.move_thr = move_thr
        self.rigid_ratio = rigid_ratio

    def match(self, box, frame_no, taken):
        best = None
        best_score = 0.0

        for t in self.tracks:
            if t.id in taken or frame_no - t.last_frame > self.max_gap:
                continue

            score = box_iou(box, t.last_box)

            if score < 0.1:
                w = max(
                    box[2] - box[0],
                    t.last_box[2] - t.last_box[0],
                )

                dx = (
                    (box[0] + box[2]) / 2
                    - (t.last_box[0] + t.last_box[2]) / 2
                )
                dy = (
                    (box[1] + box[3]) / 2
                    - (t.last_box[1] + t.last_box[3]) / 2
                )

                score = (
                    0.1
                    if (dx * dx + dy * dy) ** 0.5 < 0.6 * w
                    else 0.0
                )

            if score > best_score:
                best = t
                best_score = score

        return best

    def update(self, frame_no, items, frame=None, ego=None):
        """
        items:
            list of (box, text, conf, valid)

        ego:
            background motion (dx, dy) or None
        """
        taken = set()

        for box, text, conf, valid in sorted(
            items,
            key=lambda i: -(i[0][2] - i[0][0]),
        ):
            t = self.match(box, frame_no, taken)

            if t is None:
                t = Track(
                    len(self.tracks) + 1,
                    box,
                    frame_no,
                )
                self.tracks.append(t)

            else:
                if (
                    ego is not None
                    and frame_no - t.last_frame == self.step
                ):
                    if t.win_n == 0:
                        lb = t.last_box
                        t.win_start = (
                            (lb[0] + lb[2]) / 2,
                            (lb[1] + lb[3]) / 2,
                        )
                        t.win_bx = 0.0
                        t.win_by = 0.0

                    t.win_n += 1
                    t.win_bx += ego[0]
                    t.win_by += ego[1]

                    if t.win_n >= self.win_steps:
                        w = max(box[2] - box[0], 1.0)

                        bg = math.hypot(
                            t.win_bx,
                            t.win_by,
                        )

                        moved = math.hypot(
                            (box[0] + box[2]) / 2 - t.win_start[0],
                            (box[1] + box[3]) / 2 - t.win_start[1],
                        )

                        if bg / w >= self.move_thr:
                            t.moving_steps += t.win_n

                            if moved < self.rigid_ratio * bg:
                                t.rigid_steps += t.win_n
                        else:
                            t.parked_steps += t.win_n

                        t.win_n = 0
                else:
                    t.win_n = 0

                t.add_box(box, frame_no)

            taken.add(t.id)

            if valid:
                t.add_read(
                    text,
                    conf,
                    frame,
                    box,
                )


def select_towed(
    tracks,
    fps,
    every,
    min_seconds,
    min_rigid_frac,
    min_votes,
    keep,
):
    """
    A plate is considered "towed" only if:

    1. The truck/camera was moving for at least min_seconds.
    2. The plate stayed fixed relative to the camera for at least
       min_rigid_frac of that moving time.
    3. The plate has at least min_votes valid OCR reads.
    """
    rows = []
    qualified = []

    for t in tracks:
        if t.votes < min_votes:
            continue

        moving_s = t.moving_steps * every / fps
        parked_s = t.parked_steps * every / fps

        frac = (
            t.rigid_steps / t.moving_steps
            if t.moving_steps
            else 0.0
        )

        ok = (
            moving_s >= min_seconds
            and frac >= min_rigid_frac
        )

        rows.append(
            (t, moving_s, parked_s, frac, ok)
        )

        if ok:
            qualified.append(
                (moving_s, t)
            )

    qualified.sort(
        key=lambda q: -q[0]
    )

    chosen = [
        t for _, t in qualified[:keep]
    ]

    rivals = [
        t for _, t in qualified[keep:]
    ]

    return chosen, rivals, rows


def merge_reads(tracks):
    """Merge OCR reads across tracks."""
    merged = {}

    for t in tracks:
        for text, e in t.reads.items():
            m = merged.setdefault(
                text,
                {
                    "confs": [],
                    "best_conf": 0.0,
                    "best_frame": None,
                },
            )

            m["confs"].extend(
                e["confs"]
            )

            if e["best_conf"] > m["best_conf"]:
                m["best_conf"] = e["best_conf"]
                m["best_frame"] = e["best_frame"]

    return merged


def read_video(
    path,
    every,
    min_conf,
    min_len,
    max_len,
    debug,
    roi=None,
    target="all",
    min_width_frac=0.0,
    move_thr=0.5,
    rigid_ratio=0.3,
):
    """
    Run plate detection/OCR/tracking over a video.

    Returns:
        reads, total_frames, tracks, fps
    """
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
    )

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        sys.exit(
            f"Could not open video: {path}"
        )

    fps = (
        cap.get(cv2.CAP_PROP_FPS)
        or 30.0
    )

    tracks = TrackSet(
        max_gap_frames=max(
            every * 10,
            int(fps * 5),
        ),
        step=every,
        move_thr=move_thr,
        rigid_ratio=rigid_ratio,
        win_steps=max(
            5,
            int(round(fps / every)),
        ),
    )

    ego = EgoMotion()
    frame_no = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_no += 1

        if frame_no % every:
            continue

        fh, fw = frame.shape[:2]

        # Towed mode must see all plates before deciding which track is rigid.
        pick = (
            "all"
            if target == "towed"
            else target
        )

        items = []

        results = alpr.predict(frame)

        for r in choose_targets(
            results,
            pick,
            fw,
            fh,
            roi,
            min_width_frac,
        ):
            text = re.sub(
                r"[^A-Z0-9]",
                "",
                r.ocr.text.upper(),
            )

            conf = mean_conf(
                r.ocr.confidence
            )

            b = (
                r.detection.bounding_box
            )

            box = (
                float(b.x1),
                float(b.y1),
                float(b.x2),
                float(b.y2),
            )

            if debug:
                print(
                    f"frame {frame_no}: "
                    f"{text} ({conf:.2f})"
                )

            valid = (
                min_len <= len(text) <= max_len
                and conf >= min_conf
            )

            items.append(
                (box, text, conf, valid)
            )

        shift = (
            ego.step(
                frame,
                [i[0] for i in items],
            )
            if target == "towed"
            else None
        )

        tracks.update(
            frame_no,
            items,
            frame,
            ego=shift,
        )

    cap.release()

    return (
        merge_reads(tracks.tracks),
        frame_no,
        tracks.tracks,
        fps,
    )


def hamming1(a: str, b: str) -> bool:
    """True if two equal-length strings differ by at most one character."""
    return (
        len(a) == len(b)
        and sum(
            x != y
            for x, y in zip(a, b)
        ) <= 1
    )


def pick_plates(reads, min_votes):
    """
    Keep plates seen at least min_votes times.

    Drop weak one-character OCR variants of a much stronger plate.
    """
    ranked = sorted(
        reads.items(),
        key=lambda kv: len(
            kv[1]["confs"]
        ),
        reverse=True,
    )

    accepted = []

    for text, data in ranked:
        votes = len(
            data["confs"]
        )

        if votes < min_votes:
            continue

        if any(
            hamming1(text, a)
            and votes
            < 0.3
            * len(
                reads[a]["confs"]
            )
            for a in accepted
        ):
            continue

        accepted.append(
            text
        )

    return accepted


def save_results(
    reads,
    plates,
    snapshot_dir,
):
    """Save the best annotated frame for each accepted plate."""
    out_dir = Path(snapshot_dir)
    out_dir.mkdir(
        exist_ok=True
    )

    for text in plates:
        data = reads[text]
        votes = len(
            data["confs"]
        )

        confidence = (
            sum(data["confs"])
            / votes
        )

        snap = (
            out_dir
            / f"{text}_{int(datetime.now().timestamp())}.jpg"
        )

        if data["best_frame"] is not None:
            cv2.imwrite(
                str(snap),
                data["best_frame"],
            )

        print(
            f"\nPlate {text}: "
            f"{votes} votes, "
            f"avg confidence {confidence:.3f}, "
            f"snapshot {snap}"
        )



def send_plate_to_backend(
    backend_url,
    plate,
    confidence,
    truck_id,
    destination,
):
    """Send one accepted tow record to the TowTrace backend."""
    payload = json.dumps(
        {
            "plate": plate,
            "detected_at": datetime.now().astimezone().isoformat(),
            "truck_id": truck_id,
            "destination": destination,
            "status": "towed",
            "confidence": round(float(confidence), 4),
            "image_url": None,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        backend_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8", errors="replace")
            print(
                f"Sent tow record for {plate} to backend "
                f"(HTTP {response.status})."
            )
            if body:
                print(f"Backend response: {body}")
            return True

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(
            f"Backend rejected tow record for {plate}: "
            f"HTTP {exc.code} {exc.reason}"
        )
        if body:
            print(f"Backend response: {body}")
        return False

    except urllib.error.URLError as exc:
        print(
            f"Could not send tow record for {plate} to backend: "
            f"{exc.reason}"
        )
        return False


def main():
    p = argparse.ArgumentParser(
        description="Standalone tow-truck license-plate CV pipeline."
    )

    p.add_argument(
        "video",
        help="Path to the video file.",
    )

    p.add_argument(
        "--every",
        type=int,
        default=5,
        help="Analyze every Nth frame.",
    )

    p.add_argument(
        "--min-conf",
        type=float,
        default=0.80,
        help="Ignore OCR reads below this confidence.",
    )

    p.add_argument(
        "--min-votes",
        type=int,
        default=2,
        help="Number of reads required to trust a plate.",
    )

    p.add_argument(
        "--min-len",
        type=int,
        default=5,
    )

    p.add_argument(
        "--max-len",
        type=int,
        default=8,
    )

    p.add_argument(
        "--roi",
        help=(
            "Only analyze plates inside x1,y1,x2,y2 "
            "fractions of the frame."
        ),
    )

    p.add_argument(
        "--target",
        choices=[
            "all",
            "largest",
            "center",
            "towed",
        ],
        default="towed",
        help=(
            "Which plate(s) to keep. "
            "'towed' uses motion analysis."
        ),
    )

    p.add_argument(
        "--min-towed-seconds",
        type=float,
        default=3.0,
        help=(
            "In towed mode, minimum moving time "
            "with the plate fixed."
        ),
    )

    p.add_argument(
        "--min-rigid-frac",
        type=float,
        default=0.15,
        help=(
            "In towed mode, fraction of moving time "
            "the plate must stay rigid."
        ),
    )

    p.add_argument(
        "--move-thr",
        type=float,
        default=0.5,
        help=(
            "Background shift per window, in plate-widths, "
            "that counts as vehicle/camera motion."
        ),
    )

    p.add_argument(
        "--rigid-ratio",
        type=float,
        default=0.3,
        help=(
            "Maximum plate motion relative to background "
            "motion for the plate to count as rigid."
        ),
    )

    p.add_argument(
        "--max-towed",
        type=int,
        default=1,
        help="Maximum number of towed vehicles to keep.",
    )

    p.add_argument(
        "--allow-ambiguous",
        action="store_true",
        help=(
            "Keep the best candidate even if multiple "
            "tracks look towed."
        ),
    )

    p.add_argument(
        "--min-plate-width",
        type=float,
        default=0.0,
        help=(
            "Ignore plates narrower than this fraction "
            "of frame width."
        ),
    )

    p.add_argument(
        "--backend-url",
        default="http://127.0.0.1:8000/tows",
        help=(
            "TowTrace backend endpoint. A request is made only "
            "when a plate is accepted."
        ),
    )

    p.add_argument(
        "--truck-id",
        default="TRUCK-01",
        help="Tow truck identifier stored with the detection.",
    )

    p.add_argument(
        "--destination",
        default="Demo Tow Yard",
        help="Tow destination stored with the detection.",
    )

    p.add_argument(
        "--snapshot-dir",
        default="snapshots",
    )

    p.add_argument(
        "--save-roi-preview",
        action="store_true",
        help=(
            "Save roi_preview.jpg with the ROI drawn, "
            "then exit."
        ),
    )

    p.add_argument(
        "--debug",
        action="store_true",
        help="Print every OCR read.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run CV normally but do not send an accepted plate "
            "to the backend."
        ),
    )

    a = p.parse_args()

    roi = (
        parse_roi(a.roi)
        if a.roi
        else None
    )

    if a.save_roi_preview:
        cap = cv2.VideoCapture(
            str(a.video)
        )

        ok, frame = cap.read()
        cap.release()

        if not ok:
            sys.exit(
                f"Could not read a frame from {a.video}"
            )

        if roi:
            h, w = frame.shape[:2]

            cv2.rectangle(
                frame,
                (
                    int(roi[0] * w),
                    int(roi[1] * h),
                ),
                (
                    int(roi[2] * w),
                    int(roi[3] * h),
                ),
                (0, 255, 255),
                3,
            )

        cv2.imwrite(
            "roi_preview.jpg",
            frame,
        )

        print(
            "Saved roi_preview.jpg."
        )

        return

    reads, total_frames, tracks, fps = read_video(
        a.video,
        a.every,
        a.min_conf,
        a.min_len,
        a.max_len,
        a.debug,
        roi=roi,
        target=a.target,
        min_width_frac=a.min_plate_width,
        move_thr=a.move_thr,
        rigid_ratio=a.rigid_ratio,
    )

    if a.target == "towed":
        chosen, rivals, rows = select_towed(
            tracks,
            fps,
            a.every,
            a.min_towed_seconds,
            a.min_rigid_frac,
            a.min_votes,
            a.max_towed,
        )

        print(
            f"\nNeed: truck moving >= "
            f"{a.min_towed_seconds:.0f}s "
            f"with plate fixed >= "
            f"{a.min_rigid_frac:.0%} "
            f"of that time."
        )

        print(
            f"  {'plate':<10} "
            f"{'moving(s)':>10} "
            f"{'parked(s)':>10} "
            f"{'fixed%':>7}"
        )

        for t, mv, pk, fr, ok in rows:
            marker = (
                "   <- TOWED"
                if t in chosen
                else ""
            )

            print(
                f"  {t.top_text:<10} "
                f"{mv:>10.1f} "
                f"{pk:>10.1f} "
                f"{fr:>7.0%}"
                f"{marker}"
            )

        if not rows:
            print(
                "  (no plate was tracked)"
            )

        elif not chosen:
            total_moving = sum(
                r[1]
                for r in rows
            )

            if (
                total_moving
                < a.min_towed_seconds
            ):
                print(
                    "Not decided: the truck was not "
                    "seen moving long enough."
                )
            else:
                print(
                    "No plate stayed fixed while "
                    "the truck moved."
                )

        if rivals and not a.allow_ambiguous:
            names = ", ".join(
                t.top_text
                for t in [
                    *chosen,
                    *rivals,
                ]
            )

            print(
                "AMBIGUOUS: more than one plate "
                f"looks towed ({names}). "
                "No final plate selected."
            )

            chosen = []

        reads = merge_reads(
            chosen
        )

    plates = pick_plates(
        reads,
        a.min_votes,
    )

    print(
        f"\nAnalyzed "
        f"{total_frames // a.every} "
        f"frames, "
        f"{len(reads)} distinct reads."
    )

    if not plates:
        print(
            "No plate reached the vote threshold."
        )
        return

    save_results(
        reads,
        plates,
        a.snapshot_dir,
    )

    # Backend rule:
    #   accepted plate -> POST it
    #   no accepted plate -> no request at all
    if not a.dry_run:
        for plate in plates:
            plate_data = reads[plate]
            avg_confidence = (
                sum(plate_data["confs"]) / len(plate_data["confs"])
            )
            send_plate_to_backend(
                a.backend_url,
                plate,
                avg_confidence,
                a.truck_id,
                a.destination,
            )


if __name__ == "__main__":
    main()
