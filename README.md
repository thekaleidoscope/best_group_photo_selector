# Best Group Photo Selector

Scans an engagement event photo folder, groups near-duplicate shots, scores every photo on sharpness, exposure, face quality, blink detection, and composition, then copies the best photo(s) from each group to a `selected_best/` folder without touching the originals.

## Files

- `select_best_photos.py` — the full pipeline script
- `pyproject.toml` — dependencies managed by [uv](https://docs.astral.sh/uv/)

## Install and run

```bash
# install dependencies
uv sync

# basic usage — picks 1 best photo per group
uv run python select_best_photos.py /path/to/photos

# keep top 2 per group, wider time window
uv run python select_best_photos.py /path/to/photos --top-k 2 --time-window 15

# custom output folder, skip contact sheets
uv run python select_best_photos.py /path/to/photos --output ~/Desktop/picks --no-contact-sheets
```

## Pipeline

| Stage | Detail |
|-------|--------|
| **Discover** | Walks the input folder for `.jpg .jpeg .png .heic .webp .tiff` |
| **Analyse** | Sharpness (Laplacian variance), exposure, face detection, eye-open check, composition, aesthetic score |
| **Group** | Sort by EXIF capture time → merge shots within `--time-window` seconds that are also visually similar (pHash ≤ `--hash-threshold`) |
| **Score** | Weighted formula: 30% sharpness + 20% face + 15% eyes + 15% exposure + 10% composition + 10% aesthetic |
| **Select** | Top-K from each group; solo photos pass only if score ≥ `--min-quality` |
| **Copy** | `shutil.copy2` to `selected_best/` — originals untouched |
| **Report** | `selection_report.csv` with all scores + reason for every photo |
| **Contact sheets** | `review_groups/group_XXXX.jpg` — thumbnail grid per group with green borders on picks, scores annotated |
| **Cache** | `.photo_selector_cache.pkl` in the input folder — reruns skip already-analysed files |

## Face and aesthetic backends

Face detection is auto-detected and used in priority order: InsightFace → MediaPipe → OpenCV Haar cascade.

CLIP aesthetic scoring is optional and activates automatically if `open-clip-torch` is installed. Without it, the aesthetic score falls back to a sharpness + exposure proxy.

## Optional dependencies

Uncomment the relevant blocks in `pyproject.toml` under `[project.optional-dependencies]` and run `uv sync --extra <name>` to enable:

- `pillow-heif` — HEIC/HEIF support for iPhone photos
- `insightface` + `onnxruntime` — higher-quality face analysis (requires a C++ toolchain)
- `open-clip-torch` + `torch` + `torchvision` — CLIP-based aesthetic scoring

## CLI reference

```
usage: select_best_photos.py [input_folder] [options]

options:
  --output, -o          Output folder (default: selected_best/ next to input)
  --top-k, -k           Photos to select per group (default: 1)
  --time-window         Max seconds between shots for same burst (default: 10)
  --hash-threshold      Max pHash hamming distance for visual similarity (default: 10)
  --min-group-size      Skip groups with fewer photos than this (default: 1)
  --min-quality         Minimum score for solo photos (default: 0.30)
  --no-contact-sheets   Skip generating contact sheet thumbnails
  --no-cache            Do not use or write the analysis cache
  --max-dim             Max image dimension for analysis (default: 1024)
  --w-sharpness         Weight for sharpness score (default: 0.30)
  --w-face              Weight for face quality score (default: 0.20)
  --w-eye-open          Weight for eye-open score (default: 0.15)
  --w-exposure          Weight for exposure score (default: 0.15)
  --w-composition       Weight for composition score (default: 0.10)
  --w-aesthetic         Weight for aesthetic score (default: 0.10)
```