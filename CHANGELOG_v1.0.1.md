# tennis-net-call v1.0.1 — Changelog & Known Issues

## Release Summary
Direction-aware shot detection from net-post camera angle. HSV ball tracking with Kalman filter, simple last_side direction gating, WSLg visualization + Windows TTS.

---

## Test Video

| Property | Value |
|----------|-------|
| File | `demo/1_analyzed.avi` (73 MB) |
| Resolution | 3840×2160 (4K) |
| Codec | MPEG4 |
| FPS | 60 |
| Duration | 20.4s, 1224 frames |
| Net position | ~39-46% from top |

Run: `python3 test_video.py demo/1_analyzed.avi`

---

## Configuration (test_video.py lines 22-47)

| Parameter | Value | Notes |
|-----------|-------|-------|
| DISPLAY_WIDTH/HEIGHT | 960×540 | WSLg window size |
| PROCESS_WIDTH/HEIGHT | 320×180 | Detection resolution |
| NET_LINE_RATIO | 0.38 | Net line at 38% from top |
| SKIP_FRAMES | 2 | Process 1 of every 3 frames (~20fps effective) |
| HSV_LOWER | [20, 70, 45] | Wide H for lighting shifts, S>70 filters court noise |
| HSV_UPPER | [50, 255, 255] | |
| PLAYER_SIDE | "bottom" | Player is below net line, hitting upward |
| FLASH_FRAMES_GOOD | 45 | "GOOD SHOT!" flash duration |
| FLASH_FRAMES_NET | 30 | "NET" flash duration |
| TRAIL_LENGTH | 15 | Ball trail history |
| Circularity threshold | >0.4 | Filters out elongated court artifacts |

---

## Architecture

### BallTracker (Kalman)
- 4-state Kalman: x, y, vx, vy
- processNoiseCov = 1.0 (responsive), measurementNoiseCov = 0.1 (trusts detections)
- Trajectory: only appends actual detected positions, NOT pure predictions (prevents drift)
- On detection loss: predicts position for tracking continuation but does NOT add to trail
- Lost >8 frames → full re-initialization

### ShotJudge (Direction Logic)
- `last_side` tracks which side of net the ball was on last frame
- `last_side` is NOT cleared on detection loss → survives brief occlusions
- On net-line crossing (`current_side != last_side`):
  - If `last_side == PLAYER_SIDE` → "GOOD SHOT"
  - If `last_side != PLAYER_SIDE` → ignored (opponent's ball)
- Cooldown: 25 frames (~1.25s) to prevent double-triggers
- No vy (vertical velocity) check — Kalman velocity estimates are too unreliable

### HSV Detection
- detect_ball_hsv(): HSV mask → erode → dilate → contours
- Top 5 contours by area → circularity > 0.4 → area 3-2000
- Returns centroid of best candidate or None

### Display
- Yellow dashed net line across display window
- Blue gradient ball trail (avoids clashing with yellow net)
- Green circle on current ball position
- "GOOD SHOT!" / "NET" flash overlay on detection
- Bottom status bar: shot counts + frame counter

---

## Known Issues (v1.0.1)

### 1. False positives from opponent balls (frame 408 scenario)
**Symptom:** Opponent's ball traveling downward across net triggers false "GOOD SHOT".
**Root cause:** If the ball is first detected below the net line (bounce, detection delay), `last_side` gets set to `"bottom"` (player side). When it crosses back up, it triggers.
**Current mitigation:** None in this version. The `origin_side` + `reported` approach was tried but caused regression (missed real shots in fast video).
**Plan for v1.0.2:** Track ball trajectory origin from first N frames, or use a "confirmed side" with minimum frame count before locking.

### 2. Missed detections on fast shots (frame 900+ scenario)
**Symptom:** Player strikes ball hard; Kalman prediction lags behind; ball crosses net before tracked position updates; shot not reported.
**Root cause:** Kalman filter smoothing causes prediction to trail behind fast-moving balls. By the time the predicted position crosses the net line, the ball is already far past it and ShotJudge may miss the crossing.
**Contributing factor:** Video playback speed (accelerated) makes this worse.
**Current mitigation:** Relaxed Kalman params (processNoiseCov=1.0), low measurementNoiseCov=0.1.
**Plan for v1.0.2:** Consider using raw detection position for crossing check in parallel with Kalman track, or increase Kalman responsiveness further.

### 3. HSV false negatives in shadow / motion blur
**Symptom:** Ball not detected during fast motion or in shadows.
**Current mitigation:** Wide H range (20-50), low V floor (45), circularity filter.
**Limitation:** Tennis ball color varies significantly with lighting; no single HSV range is perfect.

### 4. Opponent-side balls reported on bounce-back
**Symptom:** Opponent ball crosses net into player side, bounces, then goes back up → may be detected as player shot on the return crossing.
**Current mitigation:** Cooldown (25 frames) prevents immediate re-trigger, but if the ball stays on player side long enough (25+ frames), `last_side` could be player-side at next crossing.
**Plan:** The `origin_side` approach (lock side on first detection, report once) is conceptually correct but needs careful handling of detection gaps.

---

## Failed Approaches (for reference)

| Approach | Why it failed |
|----------|--------------|
| `origin_side` + `reported` flag | Detection gaps in fast video caused origin_side to reset, missing real shots. The flag prevented re-trigger but also prevented detection after brief occlusions. |
| `lost` counter in ShotJudge (20-frame grace) | Added complexity without fixing the core issue. Simple `last_side` preservation is sufficient for this video speed. |
| Kalman `vy < 0` check in ShotJudge | Kalman velocity estimates are noisy/unreliable at high speed. Many real shots were filtered out. |

---

## What Works Well
- Direction filtering correctly ignores most opponent balls
- Kalman smooths trajectory well at moderate ball speeds
- HSV + circularity filter catches the ball reliably in good lighting
- WSLg display is smooth, TTS fires correctly on Windows host
- Pure ASCII labels avoid all OpenCV Unicode font issues

---

## git
- `c60c60a` — v1.0.1: simple last_side direction filter, 25-frame cooldown, no vy check
