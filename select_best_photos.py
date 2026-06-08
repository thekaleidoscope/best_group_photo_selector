#!/usr/bin/env python3
"""
Best Group Photo Selector
=========================
Scans an engagement event photo folder, groups near-duplicate shots,
scores every photo on sharpness, exposure, face quality, blink detection,
and composition, then copies the best photo(s) from each group to a
selected_best/ folder without touching the originals.

Setup
-----
pip install pillow opencv-python-headless imagehash exifread numpy pandas \
            tqdm mediapipe insightface onnxruntime

Optional (CLIP aesthetic scoring):
pip install open_clip_torch torch torchvision

Optional (HEIC support):
pip install pillow-heif

Usage
-----
python select_best_photos.py /path/to/photos
python select_best_photos.py /path/to/photos --top-k 2 --time-window 10
python select_best_photos.py /path/to/photos --output /path/to/output --top-k 3
python select_best_photos.py /path/to/photos --no-contact-sheets --no-cache
"""

import argparse
import csv
import json
import logging
import math
import os
import pickle
import shutil
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import exifread
import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── optional imports ──────────────────────────────────────────────────────────

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

try:
    import mediapipe as mp
    _mp_face = mp.solutions.face_detection
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except Exception:
    INSIGHTFACE_AVAILABLE = False

try:
    import open_clip
    import torch
    CLIP_AVAILABLE = True
except Exception:
    CLIP_AVAILABLE = False


# ── configuration dataclass ───────────────────────────────────────────────────

@dataclass
class Config:
    input_folder: str = ""
    output_folder: str = ""
    top_k: int = 1
    time_window: int = 10          # seconds between shots to consider same burst
    hash_threshold: int = 10       # max hamming distance (0–64) for phash similarity
    min_group_size: int = 1        # groups smaller than this are skipped
    similarity_threshold: float = 0.0  # 0–1 extra similarity gate (currently via hash)
    max_analysis_dim: int = 1024   # resize to this for CV analysis (not for copy)
    thumbnail_dim: int = 220       # contact sheet thumbnail size
    contact_sheets: bool = True
    use_cache: bool = True
    min_quality_threshold: float = 0.30  # solo photos below this are skipped

    # scoring weights (must sum to 1.0)
    w_sharpness: float = 0.30
    w_face: float = 0.20
    w_eye_open: float = 0.15
    w_exposure: float = 0.15
    w_composition: float = 0.10
    w_aesthetic: float = 0.10


# ── photo metadata / score ─────────────────────────────────────────────────────

@dataclass
class PhotoRecord:
    path: str
    filename: str
    capture_time: Optional[datetime] = None
    phash: Optional[str] = None
    resolution: Tuple[int, int] = (0, 0)
    # scores (0–1)
    sharpness_score: float = 0.0
    exposure_score: float = 0.0
    face_score: float = 0.0
    eye_open_score: float = 0.5    # default neutral when detection unavailable
    composition_score: float = 0.5
    aesthetic_score: float = 0.5
    final_score: float = 0.0
    # diagnostics
    face_count: int = 0
    blink_detected: bool = False
    error: str = ""
    group_id: int = -1
    selected: bool = False
    selection_reason: str = ""


# ── EXIF helpers ──────────────────────────────────────────────────────────────

_EXIF_DATETIME_TAGS = [
    "EXIF DateTimeOriginal",
    "EXIF DateTimeDigitized",
    "Image DateTime",
]
_EXIF_DATETIME_FMT = "%Y:%m:%d %H:%M:%S"


def read_exif_time(path: str) -> Optional[datetime]:
    try:
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, stop_tag="EXIF DateTimeOriginal", details=False)
        for tag in _EXIF_DATETIME_TAGS:
            if tag in tags:
                return datetime.strptime(str(tags[tag]), _EXIF_DATETIME_FMT)
    except Exception:
        pass
    # fallback: file mtime
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return None


# ── image loading ─────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif", ".bmp"}


def load_image_rgb(path: str, max_dim: int = 1024) -> Optional[np.ndarray]:
    """Load image, optionally downscale for analysis, return RGB uint8 array."""
    try:
        pil = Image.open(path)
        pil = pil.convert("RGB")
        w, h = pil.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return np.array(pil)
    except Exception as e:
        log.debug("Cannot load %s: %s", path, e)
        return None


# ── quality scoring ───────────────────────────────────────────────────────────

def score_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance → 0–1 score using a soft sigmoid."""
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Typical sharp engagement photo ~300–2000; blurry ~0–50
    score = 1.0 - math.exp(-lap_var / 400.0)
    return float(np.clip(score, 0.0, 1.0))


def score_exposure(gray: np.ndarray) -> float:
    """
    Prefer mean brightness 90–170 (out of 255).
    Penalise very dark (<60) or overexposed (>210).
    Also reward low clipping fraction.
    """
    mean_val = float(gray.mean())
    overexposed_frac = float((gray > 250).mean())
    underexposed_frac = float((gray < 10).mean())

    # bell curve centred at 130
    brightness_score = math.exp(-((mean_val - 130) ** 2) / (2 * 50**2))
    clipping_penalty = overexposed_frac * 2 + underexposed_frac
    score = brightness_score * (1.0 - min(clipping_penalty, 1.0))
    return float(np.clip(score, 0.0, 1.0))


def score_composition_basic(img_rgb: np.ndarray, face_boxes: List) -> float:
    """
    Simple heuristic:
    - Reward faces near the horizontal centre third.
    - Penalise faces cut at edges.
    - Reward balanced horizontal distribution.
    """
    h, w = img_rgb.shape[:2]
    if not face_boxes:
        # no faces: use centre-of-mass of detected edges as a proxy
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        ys, xs = np.where(edges)
        if len(xs) == 0:
            return 0.5
        cx = xs.mean() / w
        return float(1.0 - abs(cx - 0.5) * 2)  # 1.0 if centred

    score_parts = []
    for box in face_boxes:
        x1, y1, x2, y2 = box
        face_cx = (x1 + x2) / 2 / w
        face_cy = (y1 + y2) / 2 / h

        # penalise faces very close to image edges (within 5% margin)
        edge_margin = 0.05
        at_edge = (x1 < edge_margin * w or x2 > (1 - edge_margin) * w or
                   y1 < edge_margin * h or y2 > (1 - edge_margin) * h)
        edge_penalty = 0.3 if at_edge else 0.0

        # horizontal rule-of-thirds reward
        thirds = [1/3, 1/2, 2/3]
        horiz_score = max(math.exp(-((face_cx - t)**2) / 0.04) for t in thirds)

        # reward upper half positioning (faces usually above mid)
        vert_score = 1.0 if face_cy < 0.65 else max(0.0, 1.0 - (face_cy - 0.65) * 5)

        score_parts.append(max(0.0, (horiz_score * 0.5 + vert_score * 0.5) - edge_penalty))

    return float(np.clip(np.mean(score_parts), 0.0, 1.0))


# ── face detection ────────────────────────────────────────────────────────────

_insightface_app = None
_mediapipe_detector = None
_opencv_face_cascade = None


def _get_opencv_cascade():
    global _opencv_face_cascade
    if _opencv_face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _opencv_face_cascade = cv2.CascadeClassifier(cascade_path)
    return _opencv_face_cascade


def detect_faces_opencv(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    cascade = _get_opencv_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return []
    return [(x, y, x + w, y + h) for (x, y, w, h) in faces]


def detect_faces_mediapipe(img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    global _mediapipe_detector
    if _mediapipe_detector is None:
        _mediapipe_detector = _mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    h, w = img_rgb.shape[:2]
    results = _mediapipe_detector.process(img_rgb)
    if not results.detections:
        return []
    boxes = []
    for det in results.detections:
        bb = det.location_data.relative_bounding_box
        x1 = int(bb.xmin * w)
        y1 = int(bb.ymin * h)
        x2 = int((bb.xmin + bb.width) * w)
        y2 = int((bb.ymin + bb.height) * h)
        boxes.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
    return boxes


def detect_faces_insightface(img_rgb: np.ndarray) -> Tuple[List, List]:
    """Returns (boxes, face_objects) for insightface."""
    global _insightface_app
    if _insightface_app is None:
        _insightface_app = FaceAnalysis(providers=["CPUExecutionProvider"])
        _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    faces = _insightface_app.get(bgr)
    if not faces:
        return [], []
    boxes = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        boxes.append((x1, y1, x2, y2))
    return boxes, faces


def estimate_eye_open_score(img_rgb: np.ndarray, face_boxes: List) -> float:
    """
    Use OpenCV eye cascade as a proxy: if eyes detected in a face region,
    that face is likely not blinking. Returns fraction of faces with open eyes.
    """
    if not face_boxes:
        return 0.5  # neutral — cannot determine

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    open_count = 0

    for (x1, y1, x2, y2) in face_boxes:
        # look only in the upper half of the face for eyes
        mid_y = (y1 + y2) // 2
        roi = gray[y1:mid_y, x1:x2]
        if roi.size == 0:
            continue
        eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
        if len(eyes) >= 1:
            open_count += 1

    return float(open_count / len(face_boxes))


def run_face_analysis(img_rgb: np.ndarray, cfg: Config) -> Dict:
    """
    Returns dict with:
      face_boxes, face_count, face_score, eye_open_score, blink_detected
    """
    face_boxes = []
    face_score = 0.5
    eye_open_score = 0.5
    blink_detected = False
    face_count = 0

    try:
        if INSIGHTFACE_AVAILABLE:
            boxes, iface_faces = detect_faces_insightface(img_rgb)
            face_boxes = boxes
            face_count = len(boxes)
            if face_count > 0:
                # InsightFace provides det_score per face
                det_scores = [float(f.det_score) for f in iface_faces if hasattr(f, "det_score")]
                face_score = min(1.0, np.mean(det_scores) * face_count / max(face_count, 1))
        elif MEDIAPIPE_AVAILABLE:
            face_boxes = detect_faces_mediapipe(img_rgb)
            face_count = len(face_boxes)
            if face_count > 0:
                h, w = img_rgb.shape[:2]
                img_area = h * w
                face_areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in face_boxes]
                face_score = min(1.0, sum(face_areas) / img_area * 8)
        else:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            face_boxes = detect_faces_opencv(gray)
            face_count = len(face_boxes)
            if face_count > 0:
                h, w = gray.shape
                img_area = h * w
                face_areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in face_boxes]
                face_score = min(1.0, sum(face_areas) / img_area * 8)

        if face_count > 0:
            eye_open_score = estimate_eye_open_score(img_rgb, face_boxes)
            blink_detected = eye_open_score < 0.4
        else:
            # no faces is a soft penalty but not catastrophic
            face_score = 0.2
            eye_open_score = 0.5

    except Exception as e:
        log.debug("Face analysis error: %s", e)

    return {
        "face_boxes": face_boxes,
        "face_count": face_count,
        "face_score": float(np.clip(face_score, 0.0, 1.0)),
        "eye_open_score": float(np.clip(eye_open_score, 0.0, 1.0)),
        "blink_detected": blink_detected,
    }


# ── CLIP aesthetic scoring ────────────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def clip_aesthetic_score(img_rgb: np.ndarray) -> float:
    """
    Use CLIP cosine similarity against positive/negative aesthetic prompts.
    Returns 0–1.
    """
    global _clip_model, _clip_preprocess, _clip_tokenizer
    try:
        if _clip_model is None:
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            _clip_model.eval()

        pil = Image.fromarray(img_rgb)
        img_tensor = _clip_preprocess(pil).unsqueeze(0)

        positive = ["a beautiful photo", "professional photography", "sharp clear photo",
                    "well-lit portrait", "happy couple engagement photo"]
        negative = ["blurry photo", "bad photo", "dark photo", "overexposed photo",
                    "eyes closed", "blinking person"]

        pos_tokens = _clip_tokenizer(positive)
        neg_tokens = _clip_tokenizer(negative)

        with torch.no_grad():
            img_feat = _clip_model.encode_image(img_tensor)
            pos_feat = _clip_model.encode_text(pos_tokens)
            neg_feat = _clip_model.encode_text(neg_tokens)

            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
            neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)

            pos_sim = (img_feat @ pos_feat.T).mean().item()
            neg_sim = (img_feat @ neg_feat.T).mean().item()

        score = (pos_sim - neg_sim + 0.3) / 0.6
        return float(np.clip(score, 0.0, 1.0))
    except Exception as e:
        log.debug("CLIP scoring failed: %s", e)
        return 0.5


# ── perceptual hash ───────────────────────────────────────────────────────────

def compute_phash(path: str) -> Optional[str]:
    try:
        with Image.open(path) as pil:
            pil = pil.convert("RGB")
            h = imagehash.phash(pil, hash_size=16)
            return str(h)
    except Exception:
        return None


def phash_distance(h1: str, h2: str) -> int:
    try:
        return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)
    except Exception:
        return 999


# ── grouping ──────────────────────────────────────────────────────────────────

def group_photos(records: List[PhotoRecord], cfg: Config) -> List[List[PhotoRecord]]:
    """
    Two-pass grouping:
    1. Sort by capture time.
    2. Within a time window, merge records with phash distance <= threshold.
    Returns list of groups (each group is a list of PhotoRecords).
    """
    # sort by time (None times go to end)
    sorted_recs = sorted(
        records,
        key=lambda r: (r.capture_time is None, r.capture_time or datetime.min)
    )

    groups: List[List[PhotoRecord]] = []
    used = [False] * len(sorted_recs)

    for i, rec in enumerate(sorted_recs):
        if used[i]:
            continue
        group = [rec]
        used[i] = True

        for j in range(i + 1, len(sorted_recs)):
            if used[j]:
                continue
            other = sorted_recs[j]

            # time gate
            if rec.capture_time and other.capture_time:
                delta = abs((other.capture_time - rec.capture_time).total_seconds())
                if delta > cfg.time_window:
                    # since sorted, subsequent will only be further in time
                    break
            elif rec.capture_time or other.capture_time:
                # one has time, other doesn't — skip time grouping for this pair
                pass

            # visual similarity gate
            if rec.phash and other.phash:
                dist = phash_distance(rec.phash, other.phash)
                if dist <= cfg.hash_threshold:
                    group.append(other)
                    used[j] = True
            else:
                # no hash available — group by time only if within window
                if rec.capture_time and other.capture_time:
                    delta = abs((other.capture_time - rec.capture_time).total_seconds())
                    if delta <= cfg.time_window:
                        group.append(other)
                        used[j] = True

        groups.append(group)

    return groups


# ── main scoring pipeline ─────────────────────────────────────────────────────

def analyse_photo(path: str, cfg: Config) -> PhotoRecord:
    rec = PhotoRecord(path=path, filename=os.path.basename(path))

    # EXIF time
    rec.capture_time = read_exif_time(path)

    # perceptual hash
    rec.phash = compute_phash(path)

    # load for analysis
    img_rgb = load_image_rgb(path, max_dim=cfg.max_analysis_dim)
    if img_rgb is None:
        rec.error = "failed to load image"
        return rec

    try:
        pil_check = Image.open(path)
        rec.resolution = pil_check.size
        pil_check.close()
    except Exception:
        pass

    # grayscale for quality metrics
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    rec.sharpness_score = score_sharpness(gray)
    rec.exposure_score = score_exposure(gray)

    # face analysis
    face_info = run_face_analysis(img_rgb, cfg)
    rec.face_count = face_info["face_count"]
    rec.face_score = face_info["face_score"]
    rec.eye_open_score = face_info["eye_open_score"]
    rec.blink_detected = face_info["blink_detected"]

    # composition
    rec.composition_score = score_composition_basic(img_rgb, face_info["face_boxes"])

    # aesthetic
    if CLIP_AVAILABLE:
        small = load_image_rgb(path, max_dim=224)
        if small is not None:
            rec.aesthetic_score = clip_aesthetic_score(small)
    else:
        # simple proxy: high sharpness + good exposure → decent aesthetic
        rec.aesthetic_score = (rec.sharpness_score * 0.6 + rec.exposure_score * 0.4)

    # weighted final score
    rec.final_score = (
        cfg.w_sharpness    * rec.sharpness_score +
        cfg.w_face         * rec.face_score +
        cfg.w_eye_open     * rec.eye_open_score +
        cfg.w_exposure     * rec.exposure_score +
        cfg.w_composition  * rec.composition_score +
        cfg.w_aesthetic    * rec.aesthetic_score
    )

    return rec


# ── selection ─────────────────────────────────────────────────────────────────

def select_from_group(group: List[PhotoRecord], cfg: Config) -> List[PhotoRecord]:
    if len(group) == 1:
        rec = group[0]
        if rec.final_score >= cfg.min_quality_threshold:
            rec.selected = True
            rec.selection_reason = "only photo in group, passed quality threshold"
        else:
            rec.selection_reason = f"only photo in group, failed quality threshold ({rec.final_score:.2f})"
        return group

    sorted_group = sorted(group, key=lambda r: r.final_score, reverse=True)
    top_k = min(cfg.top_k, len(sorted_group))
    for idx, rec in enumerate(sorted_group):
        if idx < top_k:
            rec.selected = True
            rec.selection_reason = f"rank {idx+1} in group (score={rec.final_score:.3f})"
        else:
            rec.selection_reason = f"rank {idx+1} in group, not in top-{top_k}"
    return sorted_group


# ── contact sheets ────────────────────────────────────────────────────────────

def make_contact_sheet(group: List[PhotoRecord], group_id: int, output_dir: Path, cfg: Config):
    """Generate a side-by-side thumbnail grid for a group."""
    thumb_size = cfg.thumbnail_dim
    padding = 8
    label_h = 52
    cols = min(len(group), 6)
    rows = math.ceil(len(group) / cols)

    sheet_w = cols * (thumb_size + padding) + padding
    sheet_h = rows * (thumb_size + label_h + padding) + padding
    sheet = Image.new("RGB", (sheet_w, sheet_h), (240, 240, 240))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        font_bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
        font_bold = font

    sorted_group = sorted(group, key=lambda r: r.final_score, reverse=True)

    for idx, rec in enumerate(sorted_group):
        col = idx % cols
        row = idx // cols
        x = padding + col * (thumb_size + padding)
        y = padding + row * (thumb_size + label_h + padding)

        try:
            thumb = Image.open(rec.path).convert("RGB")
            thumb.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            # center in cell
            off_x = x + (thumb_size - thumb.width) // 2
            off_y = y + (thumb_size - thumb.height) // 2
            sheet.paste(thumb, (off_x, off_y))
        except Exception:
            draw.rectangle([x, y, x + thumb_size, y + thumb_size], fill=(200, 200, 200))
            draw.text((x + 5, y + 5), "error", fill=(100, 100, 100), font=font)

        # green border if selected
        border_color = (50, 200, 50) if rec.selected else (180, 180, 180)
        border_w = 3 if rec.selected else 1
        draw.rectangle([x - border_w, y - border_w,
                        x + thumb_size + border_w, y + thumb_size + border_w],
                       outline=border_color, width=border_w)

        # label
        label_y = y + thumb_size + 2
        name = rec.filename[:22] + "…" if len(rec.filename) > 22 else rec.filename
        mark = "★ " if rec.selected else ""
        draw.text((x, label_y), f"{mark}{name}", fill=(30, 30, 30),
                  font=font_bold if rec.selected else font)
        draw.text((x, label_y + 14), f"score: {rec.final_score:.3f}", fill=(60, 60, 60), font=font)
        draw.text((x, label_y + 26), f"sharp:{rec.sharpness_score:.2f} "
                                      f"exp:{rec.exposure_score:.2f} "
                                      f"face:{rec.face_score:.2f}",
                  fill=(80, 80, 80), font=font)
        eye_txt = "BLINK" if rec.blink_detected else f"eyes:{rec.eye_open_score:.2f}"
        eye_col = (220, 50, 50) if rec.blink_detected else (80, 80, 80)
        draw.text((x, label_y + 38), eye_txt, fill=eye_col, font=font)

    out_dir = output_dir / "review_groups"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / f"group_{group_id:04d}.jpg"
    sheet.save(sheet_path, "JPEG", quality=88)


# ── caching ───────────────────────────────────────────────────────────────────

def cache_path(input_folder: str) -> Path:
    return Path(input_folder) / ".photo_selector_cache.pkl"


def load_cache(input_folder: str) -> Dict[str, PhotoRecord]:
    cp = cache_path(input_folder)
    if cp.exists():
        try:
            with open(cp, "rb") as fh:
                data = pickle.load(fh)
            log.info("Loaded cache with %d entries.", len(data))
            return data
        except Exception as e:
            log.warning("Cache load failed (%s), starting fresh.", e)
    return {}


def save_cache(input_folder: str, cache: Dict[str, PhotoRecord]):
    cp = cache_path(input_folder)
    try:
        with open(cp, "wb") as fh:
            pickle.dump(cache, fh)
    except Exception as e:
        log.warning("Cache save failed: %s", e)


def cache_key(path: str) -> str:
    stat = os.stat(path)
    return f"{path}|{stat.st_size}|{stat.st_mtime}"


# ── CSV report ────────────────────────────────────────────────────────────────

def write_csv_report(all_records: List[PhotoRecord], output_folder: Path):
    report_path = output_folder / "selection_report.csv"
    fieldnames = [
        "filename", "group_id", "selected", "final_score",
        "sharpness_score", "exposure_score", "face_score",
        "eye_open_score", "composition_score", "aesthetic_score",
        "face_count", "blink_detected", "resolution", "capture_time",
        "selection_reason", "error",
    ]
    with open(report_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted(all_records, key=lambda r: (r.group_id, -r.final_score)):
            writer.writerow({
                "filename": rec.filename,
                "group_id": rec.group_id,
                "selected": rec.selected,
                "final_score": f"{rec.final_score:.4f}",
                "sharpness_score": f"{rec.sharpness_score:.4f}",
                "exposure_score": f"{rec.exposure_score:.4f}",
                "face_score": f"{rec.face_score:.4f}",
                "eye_open_score": f"{rec.eye_open_score:.4f}",
                "composition_score": f"{rec.composition_score:.4f}",
                "aesthetic_score": f"{rec.aesthetic_score:.4f}",
                "face_count": rec.face_count,
                "blink_detected": rec.blink_detected,
                "resolution": f"{rec.resolution[0]}x{rec.resolution[1]}",
                "capture_time": rec.capture_time.isoformat() if rec.capture_time else "",
                "selection_reason": rec.selection_reason,
                "error": rec.error,
            })
    log.info("CSV report written to %s", report_path)


# ── main pipeline ─────────────────────────────────────────────────────────────

def discover_photos(input_folder: str) -> List[str]:
    paths = []
    for root, _dirs, files in os.walk(input_folder):
        for f in sorted(files):
            ext = Path(f).suffix.lower()
            if ext in SUPPORTED_EXTS:
                full = os.path.join(root, f)
                if not f.startswith("."):
                    paths.append(full)
    return paths


def run(cfg: Config):
    input_path = Path(cfg.input_folder).resolve()
    if not input_path.is_dir():
        log.error("Input folder does not exist: %s", input_path)
        sys.exit(1)

    if cfg.output_folder:
        output_path = Path(cfg.output_folder).resolve()
    else:
        output_path = input_path.parent / "selected_best"
    output_path.mkdir(parents=True, exist_ok=True)

    log.info("Input  : %s", input_path)
    log.info("Output : %s", output_path)
    log.info("Backend: insightface=%s  mediapipe=%s  CLIP=%s  HEIC=%s",
             INSIGHTFACE_AVAILABLE, MEDIAPIPE_AVAILABLE, CLIP_AVAILABLE, HEIC_SUPPORT)

    # discover
    photo_paths = discover_photos(str(input_path))
    log.info("Found %d photos.", len(photo_paths))
    if not photo_paths:
        log.warning("No supported photos found. Exiting.")
        return

    # load / build cache
    cache: Dict[str, PhotoRecord] = load_cache(str(input_path)) if cfg.use_cache else {}

    # analyse
    records: List[PhotoRecord] = []
    skipped_errors = []

    for path in tqdm(photo_paths, desc="Analysing photos", unit="img"):
        key = cache_key(path)
        if key in cache:
            records.append(cache[key])
            continue

        rec = analyse_photo(path, cfg)
        if rec.error:
            skipped_errors.append((path, rec.error))
        records.append(rec)
        cache[key] = rec

    if cfg.use_cache:
        save_cache(str(input_path), cache)

    if skipped_errors:
        log.warning("Skipped %d files due to errors:", len(skipped_errors))
        for p, e in skipped_errors[:10]:
            log.warning("  %s — %s", os.path.basename(p), e)

    # group
    log.info("Grouping photos (time_window=%ds, hash_threshold=%d)…",
             cfg.time_window, cfg.hash_threshold)
    groups = group_photos(records, cfg)
    log.info("Formed %d groups from %d photos.", len(groups), len(records))

    # filter small groups
    if cfg.min_group_size > 1:
        before = len(groups)
        groups = [g for g in groups if len(g) >= cfg.min_group_size]
        log.info("Kept %d/%d groups (min_group_size=%d).", len(groups), before, cfg.min_group_size)

    # assign group IDs and select
    all_records: List[PhotoRecord] = []
    selected_records: List[PhotoRecord] = []

    for gid, group in enumerate(tqdm(groups, desc="Selecting best", unit="group")):
        for rec in group:
            rec.group_id = gid

        result = select_from_group(group, cfg)
        all_records.extend(result)
        selected_records.extend([r for r in result if r.selected])

        if cfg.contact_sheets:
            try:
                make_contact_sheet(result, gid, output_path, cfg)
            except Exception as e:
                log.debug("Contact sheet error for group %d: %s", gid, e)

    # copy selected
    log.info("Copying %d selected photos to %s…", len(selected_records), output_path)
    copy_errors = 0
    for rec in tqdm(selected_records, desc="Copying files", unit="file"):
        dst = output_path / rec.filename
        # avoid overwrites if same filename in different subdirs
        if dst.exists():
            stem = Path(rec.filename).stem
            suffix = Path(rec.filename).suffix
            dst = output_path / f"{stem}_g{rec.group_id}{suffix}"
        try:
            shutil.copy2(rec.path, dst)
        except Exception as e:
            log.error("Failed to copy %s: %s", rec.filename, e)
            copy_errors += 1

    # write CSV
    write_csv_report(all_records, output_path)

    # summary
    print("\n" + "=" * 60)
    print(f"  Total photos scanned : {len(records)}")
    print(f"  Groups formed        : {len(groups)}")
    print(f"  Photos selected      : {len(selected_records)}")
    print(f"  Copy errors          : {copy_errors}")
    print(f"  Output folder        : {output_path}")
    if cfg.contact_sheets:
        print(f"  Contact sheets       : {output_path / 'review_groups'}")
    print(f"  Report               : {output_path / 'selection_report.csv'}")
    print("=" * 60)

    if skipped_errors:
        print(f"\n  WARNING: {len(skipped_errors)} files were skipped due to errors.")
        print("  Check the CSV report for details.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Select the best photos from an engagement event folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python select_best_photos.py /path/to/photos
  python select_best_photos.py /path/to/photos --top-k 2
  python select_best_photos.py /path/to/photos --output /tmp/picks --time-window 8
  python select_best_photos.py /path/to/photos --top-k 3 --hash-threshold 15
  python select_best_photos.py /path/to/photos --no-contact-sheets --no-cache
""",
    )
    parser.add_argument("input_folder", nargs="?", help="Path to folder containing photos")
    parser.add_argument("--output", "-o", dest="output_folder", default="",
                        help="Output folder (default: selected_best/ next to input)")
    parser.add_argument("--top-k", "-k", type=int, default=1,
                        help="How many photos to select per group (default: 1)")
    parser.add_argument("--time-window", type=int, default=10,
                        help="Max seconds between shots to consider same burst (default: 10)")
    parser.add_argument("--hash-threshold", type=int, default=10,
                        help="Max phash hamming distance for visual similarity (0–64, default: 10)")
    parser.add_argument("--min-group-size", type=int, default=1,
                        help="Skip groups with fewer photos than this (default: 1)")
    parser.add_argument("--min-quality", type=float, default=0.30,
                        help="Minimum quality score for solo photos (default: 0.30)")
    parser.add_argument("--no-contact-sheets", action="store_true",
                        help="Skip generating contact sheet thumbnails")
    parser.add_argument("--no-cache", action="store_true",
                        help="Do not use or write analysis cache")
    parser.add_argument("--max-dim", type=int, default=1024,
                        help="Max image dimension for analysis (default: 1024)")

    # weight overrides
    parser.add_argument("--w-sharpness", type=float, default=0.30)
    parser.add_argument("--w-face", type=float, default=0.20)
    parser.add_argument("--w-eye-open", type=float, default=0.15)
    parser.add_argument("--w-exposure", type=float, default=0.15)
    parser.add_argument("--w-composition", type=float, default=0.10)
    parser.add_argument("--w-aesthetic", type=float, default=0.10)

    args = parser.parse_args()

    if not args.input_folder:
        args.input_folder = input("Enter the path to the photo folder: ").strip()

    cfg = Config(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        top_k=args.top_k,
        time_window=args.time_window,
        hash_threshold=args.hash_threshold,
        min_group_size=args.min_group_size,
        min_quality_threshold=args.min_quality,
        contact_sheets=not args.no_contact_sheets,
        use_cache=not args.no_cache,
        max_analysis_dim=args.max_dim,
        w_sharpness=args.w_sharpness,
        w_face=args.w_face,
        w_eye_open=args.w_eye_open,
        w_exposure=args.w_exposure,
        w_composition=args.w_composition,
        w_aesthetic=args.w_aesthetic,
    )

    # normalise weights
    total = (cfg.w_sharpness + cfg.w_face + cfg.w_eye_open +
             cfg.w_exposure + cfg.w_composition + cfg.w_aesthetic)
    if abs(total - 1.0) > 0.01:
        log.warning("Weights sum to %.3f, normalising.", total)
        cfg.w_sharpness /= total
        cfg.w_face /= total
        cfg.w_eye_open /= total
        cfg.w_exposure /= total
        cfg.w_composition /= total
        cfg.w_aesthetic /= total

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)