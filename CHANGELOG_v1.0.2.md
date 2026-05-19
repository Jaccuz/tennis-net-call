# tennis-net-call v1.0.2 — Changelog

**Release date:** 2026-05-19

## Overview

Complete redesign of detection and calibration logic to address the false-positive / false-negative issues documented in v1.0.1. The net-line crossing approach is replaced by a **zone-based crossing detector**, and automatic calibration now uses a **court-first three-step algorithm** to correctly locate the net across different court backgrounds (eliminates fence / building confusion).

---

## Breaking Changes

- `ShotJudge` is removed. Replaced by `ZoneJudge`.
- `CourtCalibrator` / `NET_LINE_RATIO` constants are removed. Zone is now fully auto-detected per video.
- `main.py` / `ball_tracker.py` / `shot_judge.py` are not changed — the redesign lives entirely in `test_video.py`.

---

## New: Zone-Based Shot Detection (`ZoneJudge`)

Instead of watching a horizontal net line, a **good-shot zone rectangle** is drawn immediately above the net:

```
  ┌──────┐  ← zone top  (frame_height × 5%)
  │ ZONE │
  └──────┘  ← zone bottom = net top Y
  left = net_x    right = net_x + ZONE_WIDTH (50 px @ 320×180)
```

**Good shot:** ball enters from the **left boundary** and exits from the **right boundary**.  
**Net:** ball enters from the left boundary and drops out of the **bottom boundary**.

### Anti-false-positive mechanisms

| Mechanism | Parameter | Purpose |
|-----------|-----------|---------|
| Left-presence counter | `MIN_LEFT_FRAMES = 3` | Ball must appear left of zone for 3+ consecutive frames before entry is accepted. Prevents single-frame shadow spikes from triggering. |
| Entry velocity gate | `MIN_ENTRY_VX = 4.0 px/frame` | Ball must be moving rightward at ≥4 px/frame at entry. Filters slow-drifting shadows. |
| Right-side reset | — | If ball first appears to the right of the zone (opponent's ball), `left_frames` is reset to 0, so it can never satisfy `MIN_LEFT_FRAMES`. |
| Cooldown | `COOLDOWN = 30 frames` | No re-trigger within 30 frames after a result. |
| Kalman bridge limit | `lost_frames ≤ 3` | Kalman position used for judging only when actual detection was lost for ≤3 frames. |

---

## New: Court-First Three-Step Calibration (`detect_zone_auto`)

Previous versions searched for the white net band across the full vertical ROI, causing fences and building walls to be mistaken for the net. The new approach anchors the search on the **court surface color**.

### Step 1 — Sample court color (`_sample_court_color`)
- Reads the first 10 frames, crops the bottom-right 30%×35% patch (closest court surface to camera).
- Returns the median HSV value. Works on blue, green, and red clay courts automatically.

### Step 2 — Find court upper boundary per frame (`_court_top_y`)
- For each calibration frame, scans rows in the right ROI for pixels matching the sampled court color (±H22, ±S60, ±V65).
- Returns the highest row where ≥15% of columns are court-colored — this is the court-to-net boundary.
- Fences and buildings appear **above** this boundary and are therefore never included in the search window.

### Step 3 — Detect net white band in constrained window
- Y search is clamped to `[court_top_y − 25, court_top_y + 15]` instead of the full ROI.
- Within this window: scores each row by `white_count × dark_count_below` (white band + dark mesh directly below is the net's unique signature).
- X detection: vectorized column scan finds the leftmost column satisfying both white (row) and dark (below) conditions.
- Multi-frame voting via `_cluster_vote()` (tolerance Y=5, X=6) for stability.

### Calibration parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `CALIB_FRAMES` | 120 | Frames scanned at startup (~2 s @ 60 fps) |
| `NET_WHITE_THRESH` | 185 | Brightness threshold for white band |
| `NET_DARK_THRESH` | 70 | Darkness threshold for net mesh |
| `NET_DARK_OFFSET` | 6 px | Gap between white band bottom and dark check start |
| `NET_DARK_WINDOW` | 20 px | Height of dark check region |
| `NET_DARK_MIN_RATIO` | 0.22 | Minimum dark-pixel ratio to confirm net mesh |
| `NET_MIN_WHITE_COL` | 0.12 | Minimum fraction of white columns to accept a row |
| `ZONE_WIDTH` | 50 px | Zone width in processing resolution (320×180) |
| `DEFAULT_ZONE` | `(210, 9, 260, 95)` | Fallback if calibration fails |

---

## New: Calibration Debug Frame (`save_debug_frame`)

After calibration, the first video frame is saved with the detected zone annotated to `demo/zone_debug.jpg`. This allows remote verification without an interactive display.

Annotations include:
- Semi-transparent green zone fill
- Blue "ENTER" left boundary line
- Green "EXIT" right boundary line
- White crossing arrow
- `net_top_y` and zone coordinates in the footer

---

## HSV Tuning

| | v1.0.1 | v1.0.2 |
|-|--------|--------|
| `HSV_LOWER` | `[20, 70, 45]` | `[20, 90, 80]` |
| `HSV_UPPER` | `[50, 255, 255]` | `[50, 255, 255]` |
| Min area | 3 | 10 |
| Min circularity | 0.4 | 0.5 |

Higher S_min (90) and V_min (80) suppress tree shadow false detections (shadows are dark, V < 80, and desaturated, S < 90).

---

## Diagnostic Tool: `debug_frames.py`

New standalone script for offline frame-level debugging:

- Extracts frames ±10 around a list of problem frame numbers
- Annotates each frame: raw HSV detection (red circle), Kalman smoothed position (green circle), velocity arrow
- Prints per-frame: `det=(x,y)`, `y`, `vx`, `vy`, `side`
- Saves HSV mask thumbnail in the bottom-right corner
- Output directory: `debug_frames/`

Usage: `python3 debug_frames.py [video_path]`

---

## v1.0.1 Known Issues — Status

| Issue | Status in v1.0.2 |
|-------|-----------------|
| False positives from opponent balls | **Fixed** — right-side reset + `MIN_LEFT_FRAMES` prevent opponent balls from ever satisfying entry conditions |
| Missed fast shots (Kalman lag) | **Fixed** — zone crossing does not depend on Kalman velocity; raw detection position used for judging |
| HSV false negatives in shadows | **Improved** — higher S/V thresholds reduce shadow hits; zone geometry further filters spurious blobs |
| Fence / building detected as net | **Fixed** — court-first calibration excludes anything above the court surface boundary |

---

## git

- `c60c60a` — v1.0.1: simple last_side direction filter, 25-frame cooldown, no vy check
- `(this commit)` — v1.0.2: zone-based detector, court-first three-step calibration, HSV tightened, debug_frames.py
